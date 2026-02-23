#!/usr/bin/env julia
# ===========================================================================
# Double-buffered forward transport from PRE-COMPUTED mass fluxes
#
# Same physics as run_forward_preprocessed.jl but with a ping-pong buffer
# strategy: while the GPU advects window N, a background task pre-loads
# window N+1 from disk into CPU staging buffers. After GPU finishes,
# the staged data is uploaded to the alternate GPU buffer set and the
# buffers are swapped.
#
# Usage:
#   USE_GPU=true USE_FLOAT32=true julia --project=. scripts/run_forward_preprocessed_dbuf.jl
#
# Environment variables:  (same as run_forward_preprocessed.jl)
#   USE_GPU, USE_FLOAT32, DT, MASSFLUX_FILE, MASSFLUX_DIR, EDGAR_FILE, OUTDIR
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Convection
using AtmosTransportModel.Diffusion
using AtmosTransportModel.Parameters
using AtmosTransportModel.Sources: load_edgar_co2, GriddedEmission, M_AIR, M_CO2
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate,
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
    "~/data/output/era5_edgar_preprocessed_dbuf_$(USE_FLOAT32 ? "f32" : "f64")"))

# ---------------------------------------------------------------------------
# Reuse helpers from single-buffer version (binary reader, etc.)
# ---------------------------------------------------------------------------
const BINARY_HEADER_SIZE = 4096

struct MassFluxBinaryReader{FT}
    data   :: Vector{FT}
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

function find_massflux_shards(dir::String, ft_tag::String)
    all_files = readdir(dir; join=true)
    months = Dict{String, String}()
    for f in all_files
        bn = basename(f)
        m = match(r"massflux_era5_(\d{6})_" * ft_tag, bn)
        m === nothing && continue
        month_key = m[1]
        if endswith(bn, ".bin")
            months[month_key] = f
        elseif endswith(bn, ".nc") && !haskey(months, month_key)
            months[month_key] = f
        end
    end
    return [months[k] for k in sort(collect(keys(months)))]
end

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

close_source(source) = close(source)

# ---------------------------------------------------------------------------
# GPU kernels (identical to single-buffer version)
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

# ---------------------------------------------------------------------------
# Double-buffer container for GPU-side met fields
# ---------------------------------------------------------------------------
struct MetBuf{A3, A2}
    m_ref :: A3   # reference air mass (Nx, Ny, Nz)
    m_dev :: A3   # working air mass   (Nx, Ny, Nz)
    am    :: A3   # x mass flux        (Nx+1, Ny, Nz)
    bm    :: A3   # y mass flux        (Nx, Ny+1, Nz)
    cm    :: A3   # z mass flux        (Nx, Ny, Nz+1)
    ps    :: A2   # surface pressure   (Nx, Ny)
end

function MetBuf(AT, ::Type{FT}, Nx, Ny, Nz) where FT
    MetBuf(
        AT(zeros(FT, Nx, Ny, Nz)),
        AT(zeros(FT, Nx, Ny, Nz)),
        AT(zeros(FT, Nx + 1, Ny, Nz)),
        AT(zeros(FT, Nx, Ny + 1, Nz)),
        AT(zeros(FT, Nx, Ny, Nz + 1)),
        AT(zeros(FT, Nx, Ny)))
end

struct CPUStagingBuf{FT}
    m  :: Array{FT, 3}
    am :: Array{FT, 3}
    bm :: Array{FT, 3}
    cm :: Array{FT, 3}
    ps :: Array{FT, 2}
end

function CPUStagingBuf(::Type{FT}, Nx, Ny, Nz) where FT
    CPUStagingBuf{FT}(
        Array{FT}(undef, Nx, Ny, Nz),
        Array{FT}(undef, Nx + 1, Ny, Nz),
        Array{FT}(undef, Nx, Ny + 1, Nz),
        Array{FT}(undef, Nx, Ny, Nz + 1),
        Array{FT}(undef, Nx, Ny))
end

function upload!(buf::MetBuf, cpu::CPUStagingBuf)
    copyto!(buf.m_ref, cpu.m)
    copyto!(buf.m_dev, cpu.m)
    copyto!(buf.am, cpu.am)
    copyto!(buf.bm, cpu.bm)
    copyto!(buf.cm, cpu.cm)
    copyto!(buf.ps, cpu.ps)
    return nothing
end

# ===========================================================================
# Main simulation — double-buffered pipeline
# ===========================================================================
function run_forward_dbuf()
    @info "=" ^ 70
    @info "AtmosTransportModel — Preprocessed Forward Run (DOUBLE-BUFFERED)"
    @info "=" ^ 70

    isfile(EDGAR_FILE) || error("EDGAR file not found: $EDGAR_FILE")

    ft_tag = USE_FLOAT32 ? "float32" : "float64"
    massflux_files = if !isempty(MASSFLUX_DIR) && isdir(MASSFLUX_DIR)
        shards = find_massflux_shards(MASSFLUX_DIR, ft_tag)
        isempty(shards) && error("No massflux shard files found in $MASSFLUX_DIR")
        @info "  Multi-month mode: $(length(shards)) shard files in $MASSFLUX_DIR"
        shards
    else
        bin_candidate = replace(MASSFLUX_FILE, r"\.nc$" => ".bin")
        primary = isfile(bin_candidate) ? bin_candidate : MASSFLUX_FILE
        isfile(primary) || error("Preprocessed file not found: $primary")
        [primary]
    end

    for (i, f) in enumerate(massflux_files)
        @info "  [$i] $(basename(f))"
    end

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
    @info "  Pipeline: DOUBLE-BUFFERED (ping-pong)"

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

    @info "\n--- Loading EDGAR CO2 emissions ---"
    edgar_source = load_edgar_co2(EDGAR_FILE, grid; year=2022)
    total_flux = sum(edgar_source.flux[i, j] * cell_area(i, j, grid)
                     for j in 1:Ny, i in 1:Nx)
    @info @sprintf("  Total emission: %.1f kg/s (%.2f Gt/yr)",
                   total_flux, total_flux * 365.25 * 24 * 3600 / 1e12)

    flux_dev   = AT(edgar_source.flux)
    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    area_j_dev = AT(area_j_cpu)

    # Double GPU buffers
    buf_A = MetBuf(AT, FT, Nx, Ny, Nz)
    buf_B = MetBuf(AT, FT, Nx, Ny, Nz)

    # Double CPU staging buffers
    cpu_A = CPUStagingBuf(FT, Nx, Ny, Nz)
    cpu_B = CPUStagingBuf(FT, Nx, Ny, Nz)

    c       = AT(zeros(FT, Nx, Ny, Nz))
    tracers = (; c)

    # Single shared workspace (only one advection runs at a time)
    ws = allocate_massflux_workspace(buf_A.m_dev, buf_A.am, buf_A.bm, buf_A.cm)

    c_col_dev = AT(zeros(FT, Nx, Ny))
    c_sfc_dev = AT(zeros(FT, Nx, Ny))

    diff_scheme = USE_DIFFUSION ? BoundaryLayerDiffusion(FT(KZ_MAX)) : NoDiffusion()
    diff_ws = if USE_DIFFUSION
        DiffusionWorkspace(grid, FT(KZ_MAX), DT, c)
    else
        nothing
    end

    mem_per_buf = sizeof(FT) * (2*Nx*Ny*Nz + (Nx+1)*Ny*Nz + Nx*(Ny+1)*Nz + Nx*Ny*(Nz+1) + Nx*Ny)
    @info @sprintf("  Buffer memory: %.1f MB (2 × %.1f MB GPU + 2 × %.1f MB CPU)",
                   4*mem_per_buf/1e6, mem_per_buf/1e6, mem_per_buf/1e6)
    @info "  Diffusion: $(USE_DIFFUSION ? "ON (Kz_max=$KZ_MAX)" : "OFF")"

    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "era5_edgar_preprocessed_dbuf.nc")
    rm(outfile; force=true)
    snapshot_every = max(1, round(Int, 3600 / DT))

    # Timing accumulators
    cum_t_cpuload  = 0.0
    cum_t_upload   = 0.0
    cum_t_advect   = 0.0
    cum_t_overlap  = 0.0

    # In-memory snapshot accumulation (written to NetCDF after simulation)
    snap_times     = Float64[0.0]
    snap_sfc       = Matrix{Float32}[zeros(Float32, Nx, Ny)]
    snap_col       = Matrix{Float32}[zeros(Float32, Nx, Ny)]
    snap_mass      = Float64[0.0]

    step = 0
    global_win = 0
    wall_start = time()

    @info "\n--- Starting double-buffered simulation ---"

    curr_gpu = buf_A
    next_gpu = buf_B
    curr_cpu = cpu_A
    next_cpu = cpu_B

    for (file_idx, mf_path) in enumerate(massflux_files)
        source, _, _, _, _, _, Nt_file, _, _, _, _ = open_massflux_source(mf_path)
        @info @sprintf("  Opening shard %d/%d: %s (%d windows)",
                       file_idx, length(massflux_files), basename(mf_path), Nt_file)

        for win in 1:Nt_file
            global_win += 1
            t_win = time()

            if win == 1
                t0 = time()
                load_window_from_source!(curr_cpu.m, curr_cpu.am, curr_cpu.bm,
                                         curr_cpu.cm, curr_cpu.ps, source, 1)
                cum_t_cpuload += time() - t0
                t0 = time()
                upload!(curr_gpu, curr_cpu)
                USE_GPU && CUDA.synchronize()
                cum_t_upload += time() - t0
            end

            dt_window = FT(DT * steps_per_met)

            apply_emissions_window!(tracers.c, flux_dev, area_j_dev,
                                     curr_gpu.ps, A_coeff, B_coeff, Nz,
                                     grid.gravity, dt_window)

            preload_task = nothing
            if win < Nt_file
                preload_task = Threads.@spawn begin
                    load_window_from_source!(next_cpu.m, next_cpu.am,
                                             next_cpu.bm, next_cpu.cm,
                                             next_cpu.ps, $source, $(win + 1))
                end
            end

            t_adv = time()
            for sub in 1:steps_per_met
                step += 1
                copyto!(curr_gpu.m_dev, curr_gpu.m_ref)
                strang_split_massflux!(tracers, curr_gpu.m_dev, curr_gpu.am,
                                       curr_gpu.bm, curr_gpu.cm,
                                       grid, true, ws; cfl_limit = FT(0.95))
                if USE_DIFFUSION && diff_ws !== nothing
                    diffuse_gpu!(tracers, diff_ws)
                end
            end
            USE_GPU && CUDA.synchronize()
            t_adv_done = time() - t_adv
            cum_t_advect += t_adv_done

            m_ref_advected = curr_gpu.m_ref

            if preload_task !== nothing
                t_wait = time()
                wait(preload_task)
                t_cpu_remaining = time() - t_wait
                cum_t_overlap += max(0.0, t_adv_done - t_cpu_remaining)

                t0 = time()
                upload!(next_gpu, next_cpu)
                USE_GPU && CUDA.synchronize()
                cum_t_upload += time() - t0

                curr_gpu, next_gpu = next_gpu, curr_gpu
                curr_cpu, next_cpu = next_cpu, curr_cpu
            end

            t_win_total = time() - t_win

            if step % snapshot_every < steps_per_met || step <= steps_per_met
                compute_diagnostics_gpu!(c_col_dev, c_sfc_dev, tracers.c,
                                         m_ref_advected, Nz)
                c_sfc = Array(c_sfc_dev)
                c_col = Array(c_col_dev)
                sim_hours = step * Float64(DT) / 3600.0

                push!(snap_times, sim_hours)
                push!(snap_sfc,  Float32.(c_sfc))
                push!(snap_col,  Float32.(c_col))
                push!(snap_mass, Float64(sum(tracers.c)))

                elapsed = round(time() - wall_start, digits=1)
                rate = global_win > 1 ? round((time() - wall_start) / (global_win - 1), digits=2) : 0.0

                @info @sprintf(
                    "  Win %d/%d (day %.1f): win=%.2fs adv=%.2fs | sfc_max=%.3f mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
                    global_win, total_windows, sim_hours/24,
                    t_win_total, t_adv_done,
                    maximum(c_sfc), sum(tracers.c)/length(tracers.c),
                    elapsed, rate)
            end
        end

        close_source(source)
    end

    wall_compute = time() - wall_start
    c_min = Float64(minimum(tracers.c))
    c_max = Float64(maximum(tracers.c))
    c_mean = Float64(sum(tracers.c)) / length(tracers.c)
    c_final = Array(tracers.c)
    n_neg = count(x -> x < 0, c_final)
    n_snaps = length(snap_times)

    @info "\n" * "=" ^ 70
    @info "Simulation complete! (DOUBLE-BUFFERED)"
    @info "=" ^ 70
    @info "  Steps: $step, windows: $global_win, days: $(step * Float64(DT) / 3600 / 24)"
    @info "  Compute time: $(round(wall_compute, digits=1))s ($(round(wall_compute/global_win, digits=2))s/window)"
    @info "  Files: $(length(massflux_files))"
    @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"
    @info ""
    @info "  --- Timing breakdown ---"
    @info "    CPU preload (disk→RAM): $(round(cum_t_cpuload, digits=1))s  ($(round(100*cum_t_cpuload/wall_compute, digits=1))%)"
    @info "    H→D upload:            $(round(cum_t_upload, digits=1))s  ($(round(100*cum_t_upload/wall_compute, digits=1))%)"
    @info "    Advection+diffusion:   $(round(cum_t_advect, digits=1))s  ($(round(100*cum_t_advect/wall_compute, digits=1))%)"
    @info "    Overlap saved:         $(round(cum_t_overlap, digits=1))s"
    sum_phases = cum_t_cpuload + cum_t_upload + cum_t_advect
    @info "    Sum of phases:         $(round(sum_phases, digits=1))s  (overhead=$(round(wall_compute - sum_phases, digits=1))s)"
    @info ""
    @info @sprintf("  Tracer: min=%.6f, max=%.4f, mean=%.6f ppm",
                   c_min, c_max, c_mean)
    @info "  Negative cells: $n_neg / $(length(c_final))"

    # --- Write all snapshots to NetCDF (post-simulation) ---
    t_nc = time()
    @info "\n--- Writing $n_snaps snapshots to NetCDF ---"
    NCDataset(outfile, "c") do ds_out
        defDim(ds_out, "lon", Nx)
        defDim(ds_out, "lat", Ny)
        defDim(ds_out, "time", n_snaps)
        defVar(ds_out, "lon", Float32, ("lon",);
               attrib = Dict("units" => "degrees_east"))[:] = Float32.(lons)
        defVar(ds_out, "lat", Float32, ("lat",);
               attrib = Dict("units" => "degrees_north"))[:] = Float32.(lats)
        v_time = defVar(ds_out, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since 2024-06-01 00:00:00"))
        v_sfc  = defVar(ds_out, "co2_surface", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm", "long_name" => "Surface CO2 from EDGAR"))
        v_col  = defVar(ds_out, "co2_column_mean", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm", "long_name" => "Column-mean CO2"))
        v_mass = defVar(ds_out, "mass_total", Float64, ("time",))

        for i in 1:n_snaps
            v_time[i]       = snap_times[i]
            v_sfc[:, :, i]  = snap_sfc[i]
            v_col[:, :, i]  = snap_col[i]
            v_mass[i]       = snap_mass[i]
        end
    end
    t_nc_done = time() - t_nc
    wall_total = wall_compute + t_nc_done

    @info @sprintf("  NetCDF write: %.1fs (%d snapshots)", t_nc_done, n_snaps)
    @info @sprintf("  Total wall time: %.1fs (compute) + %.1fs (I/O) = %.1fs",
                   wall_compute, t_nc_done, wall_total)
    @info "  Output: $outfile"
    @info "=" ^ 70
end

run_forward_dbuf()
