#!/usr/bin/env julia
# ===========================================================================
# GEOS-FP C720 Cubed-Sphere Forward Transport — GPU
#
# Two data ingestion modes:
#   MODE 1 (fast): Preprocessed flat binary via mmap (set PREPROCESSED_DIR)
#   MODE 2 (fallback): Raw GEOS-FP NetCDF files (set GEOSFP_DATA_DIR)
#
# Preprocessed binary files are created by preprocess_geosfp_cs.jl.
# EDGAR binary is created by preprocess_edgar_cs.jl.
#
# Usage:
#   # Fast mode (preprocessed):
#   PREPROCESSED_DIR=~/data/geosfp_cs/preprocessed \
#     USE_GPU=true julia --project=. scripts/run_forward_geosfp_cs_gpu.jl
#
#   # Fallback mode (raw NetCDF):
#   USE_GPU=true julia --project=. scripts/run_forward_geosfp_cs_gpu.jl
#
# Environment variables:
#   USE_GPU, USE_FLOAT32, DT
#   PREPROCESSED_DIR  — directory with preprocessed .bin met files (fast mode)
#   EDGAR_BIN         — preprocessed EDGAR binary (fast mode)
#   GEOSFP_DATA_DIR   — raw NetCDF data (fallback mode)
#   EDGAR_FILE        — raw EDGAR NetCDF (fallback mode)
#   GEOSFP_START, GEOSFP_END — date range
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Grids: fill_panel_halos!, allocate_cubed_sphere_field
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.Sources: M_AIR, M_CO2
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              read_geosfp_cs_timestep, to_haloed_panels,
                              cgrid_to_staggered_panels
using KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend
using NCDatasets
using Dates
using Printf
using Mmap
using JSON3

const _HAS_CAIROMAKIE = try
    @eval using CairoMakie
    true
catch
    false
end

const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "true"))
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "true"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

const FT = USE_FLOAT32 ? Float32 : Float64
const DT = parse(FT, get(ENV, "DT", "900"))

const PREPROCESSED_DIR = expanduser(get(ENV, "PREPROCESSED_DIR", ""))
const GEOSFP_DIR = expanduser(get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs")))
const EDGAR_BIN  = expanduser(get(ENV, "EDGAR_BIN", ""))
const EDGARFILE  = expanduser(get(ENV, "EDGAR_FILE",
                       joinpath(homedir(), "data", "emissions", "edgar_v8",
                                "v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc")))
const OUTDIR     = expanduser(get(ENV, "OUTDIR",
                       joinpath(homedir(), "data", "output", "geosfp_cs_gpu")))

const START_DATE = Date(get(ENV, "GEOSFP_START", "2024-06-01"))
const END_DATE   = Date(get(ENV, "GEOSFP_END", "2024-06-05"))

const SAVE_OUTPUT = parse(Bool, get(ENV, "SAVE_OUTPUT", "true"))

const CS_HEADER_SIZE = 8192
const EDGAR_HEADER_SIZE = 4096

# ===========================================================================
# Cubed-Sphere Binary Reader (mmap-based, zero-copy)
# ===========================================================================
struct CSBinaryReader{FT}
    data   :: Vector{FT}
    io     :: IOStream
    Nc     :: Int
    Nz     :: Int
    Hp     :: Int
    Nt     :: Int
    n_delp_panel :: Int
    n_am_panel   :: Int
    n_bm_panel   :: Int
    elems_per_window :: Int
end

function CSBinaryReader(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, CS_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), CS_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc); Nz = Int(hdr.Nz); Hp = Int(hdr.Hp); Nt = Int(hdr.Nt)
    n_delp = Int(hdr.n_delp_panel)
    n_am   = Int(hdr.n_am_panel)
    n_bm   = Int(hdr.n_bm_panel)
    elems  = Int(hdr.elems_per_window)
    total  = elems * Nt

    seek(io, CS_HEADER_SIZE)
    data = Mmap.mmap(io, Vector{FT}, total, CS_HEADER_SIZE)

    CSBinaryReader{FT}(data, io, Nc, Nz, Hp, Nt, n_delp, n_am, n_bm, elems)
end

Base.close(r::CSBinaryReader) = close(r.io)

function load_cs_window!(delp_cpu::NTuple{6}, am_cpu::NTuple{6}, bm_cpu::NTuple{6},
                          reader::CSBinaryReader, win::Int)
    off = (win - 1) * reader.elems_per_window
    o = off
    for p in 1:6
        n = reader.n_delp_panel
        copyto!(vec(delp_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_am_panel
        copyto!(vec(am_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_bm_panel
        copyto!(vec(bm_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    return nothing
end

# ===========================================================================
# EDGAR Binary Reader
# ===========================================================================
function load_edgar_binary(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, EDGAR_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), EDGAR_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc)
    n_panel = Int(hdr.n_per_panel)

    flux_panels = ntuple(6) do p
        arr = Array{FT}(undef, Nc, Nc)
        read!(io, arr)
        arr
    end
    close(io)
    return flux_panels
end

# ===========================================================================
# Find preprocessed binary files
# ===========================================================================
function find_preprocessed_files(dir, start_date, end_date, ft_tag)
    files = String[]
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        fp = joinpath(dir, "geosfp_cs_$(datestr)_$(ft_tag).bin")
        isfile(fp) && push!(files, fp)
    end
    return files
end

# ===========================================================================
# Find raw NetCDF files (fallback)
# ===========================================================================
function find_geosfp_cs_files(datadir, start_date, end_date)
    files = String[]
    for date in start_date:Day(1):end_date
        daydir = joinpath(datadir, Dates.format(date, "yyyymmdd"))
        isdir(daydir) || continue
        for f in sort(readdir(daydir))
            if contains(f, "tavg_1hr_ctm_c0720_v72") && endswith(f, ".nc4")
                push!(files, joinpath(daydir, f))
            end
        end
    end
    return files
end

# ===========================================================================
# EDGAR fallback: regrid from NetCDF
# ===========================================================================
function regrid_edgar_to_cs(edgar_raw::Matrix{FT},
                            edgar_lons::Vector{FT},
                            edgar_lats::Vector{FT},
                            grid::CubedSphereGrid{FT}) where FT
    Nc = grid.Nc
    R  = FT(grid.radius)
    Δlon = edgar_lons[2] - edgar_lons[1]
    Δlat = edgar_lats[2] - edgar_lats[1]
    sec_per_yr = FT(365.25 * 24 * 3600)
    Nlon_e = length(edgar_lons)
    Nlat_e = length(edgar_lats)

    flux_kgm2s = Matrix{FT}(undef, Nlon_e, Nlat_e)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(edgar_lats[j]) - Δlat / 2
        φ_n = FT(edgar_lats[j]) + Δlat / 2
        cell_area_e = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        flux_kgm2s[i, j] = FT(edgar_raw[i, j]) * FT(1000) / (sec_per_yr * cell_area_e)
    end

    flux_panels = ntuple(6) do p
        pf = zeros(FT, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            lon = mod(grid.λᶜ[p][i, j] + 180, 360) - 180
            lat = grid.φᶜ[p][i, j]
            ii = clamp(round(Int, (lon - edgar_lons[1]) / Δlon) + 1, 1, Nlon_e)
            jj = clamp(round(Int, (lat - edgar_lats[1]) / Δlat) + 1, 1, Nlat_e)
            pf[i, j] = flux_kgm2s[ii, jj]
        end
        pf
    end
    return flux_panels
end

# ---------------------------------------------------------------------------
# GPU emission kernel
# ---------------------------------------------------------------------------
@kernel function _emit_cs_kernel!(rm, @Const(flux), @Const(area), dt_window, mol_ratio, Hp)
    i, j = @index(Global, NTuple)
    @inbounds begin
        f = flux[i, j]
        if f != zero(eltype(rm))
            rm[Hp + i, Hp + j, size(rm, 3)] += f * dt_window * area[i, j] * mol_ratio
        end
    end
end

function apply_emissions_gpu!(rm_panels::NTuple{6}, flux_dev::NTuple{6},
                               area_dev::NTuple{6}, dt_window::FT,
                               Nc::Int, Hp::Int) where FT
    mol_ratio = FT(1e6 * M_AIR / M_CO2)
    backend = get_backend(rm_panels[1])
    k! = _emit_cs_kernel!(backend, 256)
    for p in 1:6
        k!(rm_panels[p], flux_dev[p], area_dev[p], dt_window, mol_ratio, Hp;
           ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

# ---------------------------------------------------------------------------
# GPU diagnostic kernels for cubed-sphere
# ---------------------------------------------------------------------------
@kernel function _cs_column_mean_kernel!(c_col, @Const(rm), @Const(m), Hp, Nz)
    i, j = @index(Global, NTuple)
    FT_k = eltype(rm)
    @inbounds begin
        sum_cm = zero(FT_k)
        sum_m  = zero(FT_k)
        for k in 1:Nz
            mk = m[Hp + i, Hp + j, k]
            sum_cm += rm[Hp + i, Hp + j, k]
            sum_m  += mk
        end
        c_col[i, j] = sum_m > zero(FT_k) ? sum_cm / sum_m : zero(FT_k)
    end
end

@kernel function _cs_stats_kernel!(max_out, sum_out, @Const(rm), @Const(m), Hp, Nc, Nz)
    p = @index(Global)
    FT_k = eltype(rm[1])
    @inbounds begin
        local_max = zero(FT_k)
        local_sum = zero(FT_k)
        for k in 1:Nz, j in 1:Nc, i in 1:Nc
            mk = m[p][Hp + i, Hp + j, k]
            c = mk > zero(FT_k) ? rm[p][Hp + i, Hp + j, k] / mk : zero(FT_k)
            local_max = max(local_max, c)
            local_sum += c
        end
        max_out[p] = local_max
        sum_out[p] = local_sum
    end
end

function compute_cs_column_mean_gpu!(c_col_panels, rm_panels, m_panels,
                                      Nc, Nz, Hp)
    backend = get_backend(rm_panels[1])
    k! = _cs_column_mean_kernel!(backend, 256)
    for p in 1:6
        k!(c_col_panels[p], rm_panels[p], m_panels[p], Hp, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

"""
Regrid cubed-sphere panel data (6 × Nc×Nc) to a lat-lon grid via nearest-neighbor.
Returns a (Nlon × Nlat) matrix.
"""
function regrid_cs_to_latlon(panels_cpu::NTuple{6, Matrix{FT}},
                              grid::CubedSphereGrid{FT};
                              Nlon=720, Nlat=361) where FT
    lons_out = range(FT(-180), FT(180) - FT(360)/Nlon, length=Nlon)
    lats_out = range(FT(-90), FT(90), length=Nlat)
    Nc = grid.Nc
    out = zeros(FT, Nlon, Nlat)
    count = zeros(Int, Nlon, Nlat)

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            lon = mod(grid.λᶜ[p][i, j] + FT(180), FT(360)) - FT(180)
            lat = grid.φᶜ[p][i, j]
            ii = clamp(round(Int, (lon - lons_out[1]) / (lons_out[2] - lons_out[1])) + 1, 1, Nlon)
            jj = clamp(round(Int, (lat - lats_out[1]) / (lats_out[2] - lats_out[1])) + 1, 1, Nlat)
            out[ii, jj] += panels_cpu[p][i, j]
            count[ii, jj] += 1
        end
    end

    for idx in eachindex(out)
        if count[idx] > 0
            out[idx] /= count[idx]
        end
    end
    return out, collect(lons_out), collect(lats_out)
end

# ---------------------------------------------------------------------------
# Upload met data from CPU to GPU
# ---------------------------------------------------------------------------
function upload_met_to_gpu!(delp_gpu, am_gpu, bm_gpu,
                             delp_cpu, am_cpu, bm_cpu)
    for p in 1:6
        copyto!(delp_gpu[p], delp_cpu[p])
        copyto!(am_gpu[p], am_cpu[p])
        copyto!(bm_gpu[p], bm_cpu[p])
    end
end

# ===========================================================================
# Main simulation
# ===========================================================================
function run_geosfp_cs_gpu()
    @info "=" ^ 70
    @info "AtmosTransport -- GEOS-FP C720 Cubed-Sphere GPU Forward Run"
    @info "=" ^ 70

    ft_tag = USE_FLOAT32 ? "float32" : "float64"
    use_preprocessed = !isempty(PREPROCESSED_DIR) && isdir(PREPROCESSED_DIR)

    # --- Discover data files ---
    if use_preprocessed
        bin_files = find_preprocessed_files(PREPROCESSED_DIR, START_DATE, END_DATE, ft_tag)
        isempty(bin_files) && error("No preprocessed .bin files found in $PREPROCESSED_DIR")
        @info "  Mode: PREPROCESSED BINARY (mmap)"
        @info "  Found $(length(bin_files)) daily binary files"

        r0 = CSBinaryReader(bin_files[1], FT)
        Nc, Nz, Hp = r0.Nc, r0.Nz, r0.Hp
        total_windows = sum(CSBinaryReader(f, FT).Nt for f in bin_files)
        close(r0)
    else
        nc_files = find_geosfp_cs_files(GEOSFP_DIR, START_DATE, END_DATE)
        isempty(nc_files) && error("No GEOS-FP files found in $GEOSFP_DIR")
        @info "  Mode: RAW NetCDF (slow)"
        @info "  Found $(length(nc_files)) hourly NetCDF files"

        ts0 = read_geosfp_cs_timestep(nc_files[1]; FT)
        Nc, Nz = ts0.Nc, ts0.Nz
        Hp = 3
        total_windows = length(nc_files)
    end

    @info "  Grid: C$Nc, Nz=$Nz, Hp=$Hp"
    @info "  Total windows: $total_windows"
    @info "  Architecture: $(USE_GPU ? "GPU" : "CPU"), FT=$FT, DT=$DT"

    config = default_met_config("geosfp")
    vc = build_vertical_coordinate(config; FT)
    params = load_parameters(FT)
    pp = params.planet

    arch = USE_GPU ? GPU() : CPU()
    AT = array_type(arch)

    grid = CubedSphereGrid(arch; FT, Nc,
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    ref_panel = AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz))
    gc = build_geometry_cache(grid, ref_panel)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)

    rm_panels = allocate_cubed_sphere_field(grid, Nz)
    m_panels  = allocate_cubed_sphere_field(grid, Nz)

    delp_gpu = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    am_gpu   = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    bm_gpu   = ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6)
    cm_gpu   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz + 1)), 6)

    # CPU staging buffers for binary mode
    delp_cpu = ntuple(_ -> Array{FT}(undef, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    am_cpu   = ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), 6)
    bm_cpu   = ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), 6)

    c_col_panels = ntuple(_ -> AT(zeros(FT, Nc, Nc)), 6)

    mem_panel = sizeof(FT) * (Nc + 2Hp)^2 * Nz / 1e6
    @info @sprintf("  GPU memory per haloed panel: %.1f MB", mem_panel)

    # --- Load EDGAR emissions ---
    @info "\n--- Loading EDGAR CO2 emissions ---"
    edgar_bin_path = !isempty(EDGAR_BIN) ? EDGAR_BIN :
        joinpath(homedir(), "data", "emissions", "edgar_v8", "edgar_cs_c$(Nc)_float32.bin")

    if isfile(edgar_bin_path)
        @info "  Loading preprocessed EDGAR binary: $edgar_bin_path"
        t0 = time()
        flux_panels_cpu = load_edgar_binary(edgar_bin_path, FT)
        @info @sprintf("  EDGAR loaded in %.2fs", time() - t0)
    else
        @info "  Preprocessed EDGAR not found, regridding from NetCDF..."
        isfile(EDGARFILE) || error("EDGAR file not found: $EDGARFILE")
        ds = NCDataset(EDGARFILE)
        edgar_lons = FT.(ds["lon"][:])
        edgar_lats = FT.(ds["lat"][:])
        edgar_flux = FT.(replace(ds["emissions"][:, :], missing => zero(FT)))
        close(ds)
        flux_panels_cpu = regrid_edgar_to_cs(edgar_flux, edgar_lons, edgar_lats, grid)
        @info "  EDGAR regridded to C$Nc"
    end

    flux_dev = ntuple(p -> AT(flux_panels_cpu[p]), 6)
    area_dev = gc.area
    @info "  EDGAR uploaded to GPU"

    met_interval = FT(3600)
    steps_per_met = max(1, round(Int, met_interval / DT))
    dt_window = DT * steps_per_met

    @info @sprintf("  Met windows: %d, sub-steps/window: %d, DT=%.0fs",
                   total_windows, steps_per_met, DT)

    mkpath(OUTDIR)
    step = 0
    wall_start = time()
    cum_t_load = 0.0
    cum_t_upload = 0.0
    cum_t_advect = 0.0
    global_win = 0

    @info "\n--- Starting GPU simulation ---"

    if use_preprocessed
        # ===================== BINARY MODE =====================
        for bin_file in bin_files
            reader = CSBinaryReader(bin_file, FT)
            @info "  Opening: $(basename(bin_file)) ($(reader.Nt) windows)"

            for w in 1:reader.Nt
                global_win += 1

                t0 = time()
                load_cs_window!(delp_cpu, am_cpu, bm_cpu, reader, w)
                t_load = time() - t0
                cum_t_load += t_load

                t0 = time()
                upload_met_to_gpu!(delp_gpu, am_gpu, bm_gpu, delp_cpu, am_cpu, bm_cpu)
                USE_GPU && CUDA.synchronize()
                t_upload = time() - t0
                cum_t_upload += t_upload

                for p in 1:6
                    compute_air_mass_panel!(m_panels[p], delp_gpu[p],
                                            gc.area[p], gc.gravity, Nc, Nz, Hp)
                end
                for p in 1:6
                    compute_cm_panel!(cm_gpu[p], am_gpu[p], bm_gpu[p], gc.bt, Nc, Nz)
                end

                apply_emissions_gpu!(rm_panels, flux_dev, area_dev, dt_window, Nc, Hp)

                t0 = time()
                for sub in 1:steps_per_met
                    step += 1
                    strang_split_massflux!(rm_panels, m_panels,
                                           am_gpu, bm_gpu, cm_gpu,
                                           grid, true, ws)
                end
                USE_GPU && CUDA.synchronize()
                t_adv = time() - t0
                cum_t_advect += t_adv

                if global_win % 24 == 0 || global_win == 1 || global_win == total_windows
                    _print_diagnostics_gpu(c_col_panels, rm_panels, m_panels,
                                            Nc, Nz, Hp, global_win, total_windows,
                                            wall_start, t_load, t_upload, t_adv)
                end
            end
            close(reader)
        end
    else
        # ===================== NetCDF MODE =====================
        for w in 1:total_windows
            global_win += 1

            t0 = time()
            ts = read_geosfp_cs_timestep(nc_files[w]; FT, convert_to_kgs=true)
            delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)
            am_nc, bm_nc = cgrid_to_staggered_panels(mfxc, mfyc)
            t_load = time() - t0
            cum_t_load += t_load

            t0 = time()
            upload_met_to_gpu!(delp_gpu, am_gpu, bm_gpu, delp_haloed, am_nc, bm_nc)
            USE_GPU && CUDA.synchronize()
            t_upload = time() - t0
            cum_t_upload += t_upload

            for p in 1:6
                compute_air_mass_panel!(m_panels[p], delp_gpu[p],
                                        gc.area[p], gc.gravity, Nc, Nz, Hp)
            end
            for p in 1:6
                compute_cm_panel!(cm_gpu[p], am_gpu[p], bm_gpu[p], gc.bt, Nc, Nz)
            end

            apply_emissions_gpu!(rm_panels, flux_dev, area_dev, dt_window, Nc, Hp)

            t0 = time()
            for sub in 1:steps_per_met
                step += 1
                strang_split_massflux!(rm_panels, m_panels,
                                       am_gpu, bm_gpu, cm_gpu,
                                       grid, true, ws)
            end
            USE_GPU && CUDA.synchronize()
            t_adv = time() - t0
            cum_t_advect += t_adv

            if global_win % 24 == 0 || global_win == 1 || global_win == total_windows
                _print_diagnostics_gpu(c_col_panels, rm_panels, m_panels,
                                        Nc, Nz, Hp, global_win, total_windows,
                                        wall_start, t_load, t_upload, t_adv)
            end
        end
    end

    wall_compute = time() - wall_start
    @info "\n" * "=" ^ 70
    @info "Simulation complete! (GEOS-FP C$(Nc) Cubed-Sphere GPU)"
    @info "=" ^ 70
    @info "  Mode: $(use_preprocessed ? "PREPROCESSED BINARY" : "RAW NetCDF")"
    @info "  Steps: $step, windows: $total_windows"
    @info @sprintf("  Total: %.1fs (%.2fs/window)", wall_compute, wall_compute / total_windows)
    @info ""
    @info "  --- Timing breakdown ---"
    @info @sprintf("    Data load:    %7.1fs  (%4.1f%%)", cum_t_load, 100*cum_t_load/wall_compute)
    @info @sprintf("    H->D upload:  %7.1fs  (%4.1f%%)", cum_t_upload, 100*cum_t_upload/wall_compute)
    @info @sprintf("    Advection:    %7.1fs  (%4.1f%%)", cum_t_advect, 100*cum_t_advect/wall_compute)
    @info "=" ^ 70

    if SAVE_OUTPUT
        @info "\n--- Saving final column-mean snapshot ---"
        t0 = time()

        compute_cs_column_mean_gpu!(c_col_panels, rm_panels, m_panels, Nc, Nz, Hp)
        panels_cpu = ntuple(p -> Array(c_col_panels[p]), 6)

        @info "  Regridding cubed-sphere to lat-lon..."
        col_mean_ll, lons_out, lats_out = regrid_cs_to_latlon(panels_cpu, grid)

        outfile = joinpath(OUTDIR, "geosfp_cs_snapshot.nc")
        rm(outfile; force=true)
        NCDataset(outfile, "c") do ds_out
            Nlon_o = length(lons_out)
            Nlat_o = length(lats_out)
            defDim(ds_out, "lon", Nlon_o)
            defDim(ds_out, "lat", Nlat_o)
            defVar(ds_out, "lon", Float32, ("lon",);
                   attrib = Dict("units" => "degrees_east"))[:] = Float32.(lons_out)
            defVar(ds_out, "lat", Float32, ("lat",);
                   attrib = Dict("units" => "degrees_north"))[:] = Float32.(lats_out)
            defVar(ds_out, "co2_column_mean", Float32, ("lon", "lat");
                   attrib = Dict("units" => "ppm",
                                 "long_name" => "Column-mean CO2 from EDGAR (GEOS-FP C$(Nc) transport)"))[:, :] = Float32.(col_mean_ll)
        end

        @info @sprintf("  Snapshot saved in %.1fs: %s", time() - t0, outfile)

        if _HAS_CAIROMAKIE
            png_path = joinpath(homedir(), "co2_geosfp_c$(Nc)_snapshot.png")
            try
                shift = findfirst(>=(FT(180)), lons_out)
                if shift !== nothing
                    lons_sh = vcat(lons_out[shift:end] .- FT(360), lons_out[1:shift-1])
                    data_sh = vcat(col_mean_ll[shift:end, :], col_mean_ll[1:shift-1, :])
                else
                    lons_sh = lons_out
                    data_sh = col_mean_ll
                end

                fig = Figure(size=(1400, 600), fontsize=14)
                ax = Axis(fig[1, 1],
                    title = "Column-Mean CO₂ Enhancement — GEOS-FP C$(Nc), EDGAR emissions\n$(START_DATE) to $(END_DATE)",
                    xlabel = "Longitude", ylabel = "Latitude",
                    aspect = DataAspect())
                hm = heatmap!(ax, Float64.(lons_sh), Float64.(lats_out), Float64.(data_sh),
                    colormap = :YlOrRd, colorrange = (0, 7))
                Colorbar(fig[1, 2], hm, label = "CO₂ enhancement [ppm]")
                xlims!(ax, -180, 180)
                ylims!(ax, -90, 90)
                save(png_path, fig, px_per_unit=2)
                @info "  Plot saved: $png_path"
            catch e
                @warn "  CairoMakie plot failed: $e"
            end
        else
            @info "  CairoMakie not available; use the NetCDF output for plotting"
        end
    end
end

function _print_diagnostics_lightweight(rm_panels, m_panels, Nc, Nz, Hp,
                                        win, total, wall_start, t_load, t_upload, t_adv)
    max_c = zero(FT)
    sum_c = zero(FT)
    n_cells = 6 * Nc * Nc * Nz
    for p in 1:6
        rm_p = rm_panels[p]
        m_p  = m_panels[p]
        for k in 1:Nz, j in (Hp+1):(Hp+Nc), i in (Hp+1):(Hp+Nc)
            c = m_p[i,j,k] > 0 ? rm_p[i,j,k] / m_p[i,j,k] : zero(FT)
            max_c = max(max_c, c)
            sum_c += c
        end
    end
    mean_c = sum_c / n_cells
    day = div(win - 1, 24) + 1
    elapsed = round(time() - wall_start, digits=1)
    rate = win > 1 ? round((time() - wall_start) / (win - 1), digits=2) : 0.0
    @info @sprintf(
        "  Day %d, win %d/%d: load=%.2fs up=%.2fs adv=%.2fs | max=%.1f mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
        day, win, total, t_load, t_upload, t_adv, max_c, mean_c, elapsed, rate)
end

function _print_diagnostics_gpu(c_col_panels, rm_panels, m_panels,
                                 Nc, Nz, Hp, win, total, wall_start,
                                 t_load, t_upload, t_adv)
    compute_cs_column_mean_gpu!(c_col_panels, rm_panels, m_panels, Nc, Nz, Hp)
    max_col = zero(FT)
    mean_col = zero(FT)
    for p in 1:6
        cp = Array(c_col_panels[p])
        max_col = max(max_col, maximum(cp))
        mean_col += sum(cp)
    end
    mean_col /= (6 * Nc * Nc)

    day = div(win - 1, 24) + 1
    elapsed = round(time() - wall_start, digits=1)
    rate = win > 1 ? round((time() - wall_start) / (win - 1), digits=2) : 0.0
    @info @sprintf(
        "  Day %d, win %d/%d: load=%.2fs up=%.2fs adv=%.2fs | col_max=%.2f col_mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
        day, win, total, t_load, t_upload, t_adv, max_col, mean_col, elapsed, rate)
end

run_geosfp_cs_gpu()
