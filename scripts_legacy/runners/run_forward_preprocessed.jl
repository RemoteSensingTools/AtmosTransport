#!/usr/bin/env julia
# ===========================================================================
# Forward transport reading PRE-COMPUTED mass fluxes from NetCDF/binary
#
# The mass fluxes (am, bm, cm, m) were computed by preprocess_mass_fluxes.jl
# and saved to disk. This script:
#   1. Reads am/bm/cm/m per met window from the preprocessed file(s)
#   2. Copies to GPU (single H2D transfer per window)
#   3. Injects EDGAR emissions once per window (constant field)
#   4. Runs Strang-split advection sub-steps (pure GPU, zero allocation)
#
# Supports monthly-sharded mass flux files: set MASSFLUX_DIR to a directory
# containing massflux_era5_YYYYMM_*.nc (or .bin) files. The script chains
# through them in chronological order.
#
# Usage:
#   USE_GPU=true USE_FLOAT32=true julia --project=. scripts/run_forward_preprocessed.jl
#
# Environment variables:
#   USE_GPU       — "true" for GPU execution (default: false)
#   USE_FLOAT32   — "true" for Float32 precision (default: false)
#   DT            — advection sub-step [s] (read from preprocessed file if not set)
#   MASSFLUX_FILE — path to single preprocessed mass-flux file (legacy mode)
#   MASSFLUX_DIR  — directory with monthly-sharded mass flux files (preferred)
#   EDGAR_FILE    — path to EDGAR emission NetCDF
#   OUTDIR        — output directory
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Convection
using AtmosTransport.Diffusion
using AtmosTransport.Parameters
using AtmosTransport.Sources: load_edgar_co2, GriddedEmission, M_AIR, M_CO2
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients
using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend
using NCDatasets
using Dates
using Printf
using Mmap
using JSON3

const USE_GPU      = parse(Bool, get(ENV, "USE_GPU", "false"))
const USE_FLOAT32  = parse(Bool, get(ENV, "USE_FLOAT32", "false"))
const USE_DIFFUSION = parse(Bool, get(ENV, "USE_DIFFUSION", "true"))
const KZ_MAX       = parse(Float64, get(ENV, "KZ_MAX", "50.0"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

const FT = USE_FLOAT32 ? Float32 : Float64

const MASSFLUX_FILE = expanduser(get(ENV, "MASSFLUX_FILE",
    "~/data/metDrivers/era5/massflux_era5_202406_$(USE_FLOAT32 ? "float32" : "float64").nc"))
const MASSFLUX_DIR  = expanduser(get(ENV, "MASSFLUX_DIR", ""))
const EDGAR_FILE = expanduser(get(ENV, "EDGAR_FILE",
    "~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc"))
const OUTDIR = expanduser(get(ENV, "OUTDIR",
    "~/data/output/era5_edgar_preprocessed_$(USE_FLOAT32 ? "f32" : "f64")"))

"""
Discover monthly mass-flux shard files in a directory.
Prefers .bin over .nc for each month. Returns sorted list of file paths.
"""
function find_massflux_shards(dir::String, ft_tag::String)
    all_files = readdir(dir; join=true)
    # Group by YYYYMM key
    months = Dict{String, String}()
    for f in all_files
        bn = basename(f)
        m = match(r"massflux_era5_(\d{6})_" * ft_tag, bn)
        m === nothing && continue
        month_key = m[1]
        if endswith(bn, ".bin")
            months[month_key] = f  # .bin always wins
        elseif endswith(bn, ".nc") && !haskey(months, month_key)
            months[month_key] = f
        end
    end
    return [months[k] for k in sort(collect(keys(months)))]
end

# ---------------------------------------------------------------------------
# GPU emission injection kernel
# ---------------------------------------------------------------------------
@kernel function _emit_surface_kernel!(c, @Const(flux_2d), @Const(area_j),
                                       @Const(ps_2d), ΔA_sfc, ΔB_sfc,
                                       g, dt_window, mol_ratio, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        f = flux_2d[i, j]
        if f != zero(FT)
            Δp = ΔA_sfc + ΔB_sfc * ps_2d[i, j]
            m_air = Δp * area_j[j] / g
            ΔM = f * dt_window * area_j[j]
            c[i, j, Nz] += ΔM * mol_ratio / m_air
        end
    end
end

# ---------------------------------------------------------------------------
# GPU diagnostic kernels (column-mean, surface slice, total mass)
# ---------------------------------------------------------------------------
@kernel function _column_mean_kernel!(c_col, @Const(c), @Const(m), Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        sum_cm = zero(FT)
        sum_m  = zero(FT)
        for k in 1:Nz
            sum_cm += c[i, j, k] * m[i, j, k]
            sum_m  += m[i, j, k]
        end
        c_col[i, j] = sum_m > zero(FT) ? sum_cm / sum_m : zero(FT)
    end
end

@kernel function _copy_surface_kernel!(c_sfc, @Const(c), Nz)
    i, j = @index(Global, NTuple)
    @inbounds c_sfc[i, j] = c[i, j, Nz]
end

function compute_diagnostics_gpu!(c_col_dev, c_sfc_dev, c, m, Nz)
    backend = get_backend(c)
    Nx, Ny = size(c, 1), size(c, 2)
    k1! = _column_mean_kernel!(backend, 256)
    k1!(c_col_dev, c, m, Nz; ndrange=(Nx, Ny))
    k2! = _copy_surface_kernel!(backend, 256)
    k2!(c_sfc_dev, c, Nz; ndrange=(Nx, Ny))
    synchronize(backend)
end

function apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev,
                                 A_coeff, B_coeff, Nz, g, dt_window)
    FT_e = eltype(c)
    ΔA = FT_e(A_coeff[Nz + 1] - A_coeff[Nz])
    ΔB = FT_e(B_coeff[Nz + 1] - B_coeff[Nz])
    mol = FT_e(1e6 * M_AIR / M_CO2)
    backend = get_backend(c)
    Nx, Ny = size(c, 1), size(c, 2)
    k! = _emit_surface_kernel!(backend, 256)
    k!(c, flux_dev, area_j_dev, ps_dev, ΔA, ΔB, FT_e(g), FT_e(dt_window), mol, Nz;
       ndrange=(Nx, Ny))
    synchronize(backend)
end

# ===========================================================================
# Fast binary mass-flux reader (mmap-based, zero-copy)
# ===========================================================================
const BINARY_HEADER_SIZE = 4096

struct MassFluxBinaryReader{FT}
    data   :: Vector{FT}        # mmap'd flat vector over entire file
    io     :: IOStream
    Nx     :: Int
    Ny     :: Int
    Nz     :: Int
    Nt     :: Int
    n_m    :: Int
    n_am   :: Int
    n_bm   :: Int
    n_cm   :: Int
    n_ps   :: Int
    elems_per_window :: Int
    lons   :: Vector{FT}
    lats   :: Vector{FT}
    dt_seconds      :: FT
    half_dt_seconds :: FT
    steps_per_met   :: Int
    level_top       :: Int
    level_bot       :: Int
end

function MassFluxBinaryReader(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, BINARY_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), BINARY_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nx = Int(hdr.Nx); Ny = Int(hdr.Ny); Nz = Int(hdr.Nz); Nt = Int(hdr.Nt)
    n_m  = Int(hdr.n_m)
    n_am = Int(hdr.n_am)
    n_bm = Int(hdr.n_bm)
    n_cm = Int(hdr.n_cm)
    n_ps = Int(hdr.n_ps)
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps
    total_elems = elems_per_window * Nt

    # mmap the data region (skip header)
    seek(io, BINARY_HEADER_SIZE)
    data = Mmap.mmap(io, Vector{FT}, total_elems, BINARY_HEADER_SIZE)

    lons = FT.(collect(hdr.lons))
    lats = FT.(collect(hdr.lats))

    MassFluxBinaryReader{FT}(
        data, io, Nx, Ny, Nz, Nt,
        n_m, n_am, n_bm, n_cm, n_ps, elems_per_window,
        lons, lats,
        FT(hdr.dt_seconds), FT(hdr.half_dt_seconds),
        Int(hdr.steps_per_met_window), Int(hdr.level_top), Int(hdr.level_bot))
end

function load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu,
                      reader::MassFluxBinaryReader, win::Int)
    off = (win - 1) * reader.elems_per_window
    o = off
    copyto!(m_cpu,  1, reader.data, o + 1, reader.n_m);  o += reader.n_m
    copyto!(am_cpu, 1, reader.data, o + 1, reader.n_am); o += reader.n_am
    copyto!(bm_cpu, 1, reader.data, o + 1, reader.n_bm); o += reader.n_bm
    copyto!(cm_cpu, 1, reader.data, o + 1, reader.n_cm); o += reader.n_cm
    copyto!(ps_cpu, 1, reader.data, o + 1, reader.n_ps)
    return nothing
end

Base.close(r::MassFluxBinaryReader) = close(r.io)

"""
Copy binary file to local fast storage if not already cached.
Returns the path to use (local if cached, original otherwise).
"""
function ensure_local_cache(src_path::String)
    cache_dir = get(ENV, "LOCAL_CACHE_DIR", "/var/tmp/massflux_cache")
    try
        mkpath(cache_dir)
    catch
        @warn "Cannot create local cache dir $cache_dir; using original path"
        return src_path
    end
    dst = joinpath(cache_dir, basename(src_path))
    if isfile(dst) && filesize(dst) == filesize(src_path)
        @info "  Using local cache: $dst"
        return dst
    end
    @info "  Copying to local NVMe cache: $dst ($(round(filesize(src_path)/1e9, digits=2)) GB)..."
    t0 = time()
    cp(src_path, dst; force=true)
    @info @sprintf("  Cache copy done in %.1fs", time() - t0)
    return dst
end

# ===========================================================================
# Open a single mass-flux file (binary or NetCDF) and return metadata.
# Returns (source, lons, lats, Nx, Ny, Nz, Nt, dt, steps_per_met, level_top, level_bot)
# where source is either MassFluxBinaryReader or NCDataset.
# ===========================================================================
function open_massflux_source(filepath::String)
    if endswith(filepath, ".bin")
        use_cache = parse(Bool, get(ENV, "USE_LOCAL_CACHE", "false"))
        bin_path = use_cache ? ensure_local_cache(filepath) : filepath
        reader = MassFluxBinaryReader(bin_path, FT)
        return (reader, reader.lons, reader.lats, reader.Nx, reader.Ny, reader.Nz,
                reader.Nt, reader.dt_seconds, reader.steps_per_met,
                reader.level_top, reader.level_bot)
    else
        ds = NCDataset(filepath)
        lons = FT.(ds["lon"][:])
        lats = FT.(ds["lat"][:])
        return (ds, lons, lats, length(lons), length(lats),
                ds.dim["lev"], ds.dim["time"],
                FT(ds.attrib["dt_seconds"]),
                Int(ds.attrib["steps_per_met_window"]),
                Int(ds.attrib["level_top"]), Int(ds.attrib["level_bot"]))
    end
end

function load_window_from_source!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu,
                                  source, win::Int)
    if source isa MassFluxBinaryReader
        load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu, source, win)
    else
        m_cpu  .= FT.(source["m"][:, :, :, win])
        am_cpu .= FT.(source["am"][:, :, :, win])
        bm_cpu .= FT.(source["bm"][:, :, :, win])
        cm_cpu .= FT.(source["cm"][:, :, :, win])
        ps_cpu .= FT.(source["ps"][:, :, win])
    end
end

function close_source(source)
    close(source)
end

# ===========================================================================
# Main simulation
# ===========================================================================
function run_forward()
    @info "=" ^ 70
    @info "AtmosTransport — Forward Run from Preprocessed Mass Fluxes"
    @info "=" ^ 70

    isfile(EDGAR_FILE) || error("EDGAR file not found: $EDGAR_FILE")

    # --- Discover mass flux file(s) ---
    ft_tag = USE_FLOAT32 ? "float32" : "float64"
    massflux_files = if !isempty(MASSFLUX_DIR) && isdir(MASSFLUX_DIR)
        shards = find_massflux_shards(MASSFLUX_DIR, ft_tag)
        isempty(shards) && error("No massflux shard files found in $MASSFLUX_DIR")
        @info "  Multi-month mode: $(length(shards)) shard files in $MASSFLUX_DIR"
        shards
    else
        # Legacy single-file mode: also check for .bin companion
        bin_candidate = replace(MASSFLUX_FILE, r"\.nc$" => ".bin")
        primary = isfile(bin_candidate) ? bin_candidate : MASSFLUX_FILE
        isfile(primary) || error("Preprocessed file not found: $primary")
        [primary]
    end

    for (i, f) in enumerate(massflux_files)
        @info "  [$i] $(basename(f))"
    end

    # Read metadata from the first file
    first_source, lons, lats, Nx, Ny, Nz, _, dt_file, steps_per_met,
        level_top, level_bot = open_massflux_source(massflux_files[1])
    close_source(first_source)

    DT = haskey(ENV, "DT") ? parse(FT, ENV["DT"]) : dt_file
    total_windows = 0
    for f in massflux_files
        src, _, _, _, _, _, nt, _, _, _, _ = open_massflux_source(f)
        total_windows += nt
        close_source(src)
    end
    total_steps = total_windows * steps_per_met

    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz"
    @info "  Total windows: $total_windows across $(length(massflux_files)) file(s)"
    @info "  DT=$(DT)s, steps/window=$steps_per_met, total_steps=$total_steps"
    @info "  Levels: $level_top-$level_bot"
    @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"

    # --- Build grid ---
    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=level_top:level_bot)
    A_full, B_full = load_vertical_coefficients(config; FT)
    A_coeff = A_full[level_top:level_bot+1]
    B_coeff = B_full[level_top:level_bot+1]

    params = load_parameters(FT)
    pp = params.planet

    arch = USE_GPU ? GPU() : CPU()
    AT = array_type(arch)

    Δlon = lons[2] - lons[1]
    grid = LatitudeLongitudeGrid(arch;
        FT, size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    # --- Load EDGAR ---
    @info "\n--- Loading EDGAR CO2 emissions ---"
    edgar_source = load_edgar_co2(EDGAR_FILE, grid; year=2022)
    total_flux = sum(edgar_source.flux[i, j] * cell_area(i, j, grid)
                     for j in 1:Ny, i in 1:Nx)
    @info @sprintf("  Total emission: %.1f kg/s (%.2f Gt/yr)",
                   total_flux, total_flux * 365.25 * 24 * 3600 / 1e12)

    # --- Pre-allocate device arrays ---
    flux_dev   = AT(edgar_source.flux)
    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    area_j_dev = AT(area_j_cpu)

    m_dev  = AT(zeros(FT, Nx, Ny, Nz))
    m_ref  = AT(zeros(FT, Nx, Ny, Nz))
    am_dev = AT(zeros(FT, Nx + 1, Ny, Nz))
    bm_dev = AT(zeros(FT, Nx, Ny + 1, Nz))
    cm_dev = AT(zeros(FT, Nx, Ny, Nz + 1))
    ps_dev = AT(zeros(FT, Nx, Ny))
    c      = AT(zeros(FT, Nx, Ny, Nz))
    tracers = (; c)
    ws = allocate_massflux_workspace(m_dev, am_dev, bm_dev, cm_dev)

    c_col_dev = AT(zeros(FT, Nx, Ny))
    c_sfc_dev = AT(zeros(FT, Nx, Ny))

    # --- Diffusion setup ---
    diff_scheme = USE_DIFFUSION ? BoundaryLayerDiffusion(FT(KZ_MAX)) : NoDiffusion()
    diff_ws = if USE_DIFFUSION
        DiffusionWorkspace(grid, FT(KZ_MAX), DT, c)
    else
        nothing
    end

    # CPU staging buffers (reused every window)
    m_cpu  = Array{FT}(undef, Nx, Ny, Nz)
    am_cpu = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm_cpu = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm_cpu = Array{FT}(undef, Nx, Ny, Nz + 1)
    ps_cpu = Array{FT}(undef, Nx, Ny)

    @info "  All arrays pre-allocated ($(USE_GPU ? "GPU" : "CPU"))"
    @info "  Diffusion: $(USE_DIFFUSION ? "ON (Kz_max=$KZ_MAX)" : "OFF")"

    # --- Output ---
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "era5_edgar_preprocessed.nc")
    rm(outfile; force=true)
    snapshot_every = max(1, round(Int, 3600 / DT))

    NCDataset(outfile, "c") do ds_out
        defDim(ds_out, "lon", Nx)
        defDim(ds_out, "lat", Ny)
        defDim(ds_out, "time", Inf)
        defVar(ds_out, "lon", Float32, ("lon",))[:] = Float32.(lons)
        defVar(ds_out, "lat", Float32, ("lat",))[:] = Float32.(lats)
        defVar(ds_out, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since simulation start"))
        defVar(ds_out, "co2_surface", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm"))
        defVar(ds_out, "co2_column_mean", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm"))
        defVar(ds_out, "mass_total", Float64, ("time",))

        ds_out["time"][1] = 0.0
        ds_out["co2_surface"][:, :, 1] = zeros(Float32, Nx, Ny)
        ds_out["co2_column_mean"][:, :, 1] = zeros(Float32, Nx, Ny)
        ds_out["mass_total"][1] = 0.0

        snapshot_idx = 1
        step = 0
        global_win = 0
        wall_start = time()

        @info "\n--- Starting simulation ---"

        for (file_idx, mf_path) in enumerate(massflux_files)
            source, _, _, _, _, _, Nt_file, _, _, _, _ = open_massflux_source(mf_path)
            @info @sprintf("  Opening shard %d/%d: %s (%d windows)",
                           file_idx, length(massflux_files), basename(mf_path), Nt_file)

            for win in 1:Nt_file
                global_win += 1
                t_win = time()

                load_window_from_source!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu,
                                         source, win)

                copyto!(m_ref, m_cpu)
                copyto!(m_dev, m_ref)
                copyto!(am_dev, am_cpu)
                copyto!(bm_dev, bm_cpu)
                copyto!(cm_dev, cm_cpu)
                copyto!(ps_dev, ps_cpu)

                t_load = time() - t_win

                dt_window = FT(DT * steps_per_met)
                apply_emissions_window!(tracers.c, flux_dev, area_j_dev,
                                         ps_dev, A_coeff, B_coeff, Nz,
                                         grid.gravity, dt_window)

                t_adv = time()
                for sub in 1:steps_per_met
                    step += 1
                    copyto!(m_dev, m_ref)
                    strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                                           grid, true, ws; cfl_limit = FT(0.95))
                    if USE_DIFFUSION && diff_ws !== nothing
                        diffuse_gpu!(tracers, diff_ws)
                    end
                end
                t_adv_done = time() - t_adv

                if step % snapshot_every < steps_per_met || step <= steps_per_met
                    compute_diagnostics_gpu!(c_col_dev, c_sfc_dev, tracers.c, m_ref, Nz)
                    c_sfc = Array(c_sfc_dev)
                    c_col = Array(c_col_dev)
                    sim_hours = step * Float64(DT) / 3600.0

                    snapshot_idx += 1
                    ds_out["time"][snapshot_idx] = sim_hours
                    ds_out["co2_surface"][:, :, snapshot_idx] = Float32.(c_sfc)
                    ds_out["co2_column_mean"][:, :, snapshot_idx] = Float32.(c_col)
                    ds_out["mass_total"][snapshot_idx] = Float64(sum(tracers.c))

                    elapsed = round(time() - wall_start, digits=1)
                    rate = global_win > 1 ? round((time() - wall_start) / (global_win - 1), digits=2) : 0.0

                    @info @sprintf(
                        "  Win %d/%d (day %.1f): load=%.2fs adv=%.2fs | sfc_max=%.3f mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
                        global_win, total_windows, sim_hours/24,
                        t_load, t_adv_done,
                        maximum(c_sfc), sum(tracers.c)/length(tracers.c),
                        elapsed, rate)
                end
            end

            close_source(source)
        end

        wall_total = round(time() - wall_start, digits=1)
        c_min = Float64(minimum(tracers.c))
        c_max = Float64(maximum(tracers.c))
        c_mean = Float64(sum(tracers.c)) / length(tracers.c)
        c_final = Array(tracers.c)
        n_neg = count(x -> x < 0, c_final)

        @info "\n" * "=" ^ 70
        @info "Simulation complete!"
        @info "=" ^ 70
        @info "  Steps: $step, windows: $global_win, days: $(step * Float64(DT) / 3600 / 24)"
        @info "  Wall time: $(wall_total)s ($(round(wall_total/global_win, digits=2))s/window)"
        @info "  Files: $(length(massflux_files))"
        @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"
        @info @sprintf("  Tracer: min=%.6f, max=%.4f, mean=%.6f ppm",
                       c_min, c_max, c_mean)
        @info "  Negative cells: $n_neg / $(length(c_final))"
        @info "  Output: $outfile ($snapshot_idx snapshots)"
        @info "=" ^ 70
    end
end

run_forward()
