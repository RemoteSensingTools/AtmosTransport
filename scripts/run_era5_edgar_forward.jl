#!/usr/bin/env julia
# ===========================================================================
# Forward transport of EDGAR CO2 emissions driven by ERA5 model-level winds
#
# Architecture:
#   Phase 1 — Pre-compute mass fluxes (am, bm, cm) for each met window (TM5-style)
#   Phase 2 — Time-step: inject emissions once per window, then advect sub-steps
#
# Mass fluxes depend only on (u, v, ps) and are static per met window.
# Pre-computing them separately means:
#   - Adjoint runs reuse the same fluxes without recomputation
#   - Sensitivity experiments skip the preprocessing entirely
#   - Future: cache to disk for offline preprocessing
#
# Uses TM5-faithful mass-flux advection with:
#   - ERA5 L88 model levels (hybrid sigma-pressure coordinates)
#   - Per-column surface pressure from log(sp)
#   - Continuity-consistent vertical mass flux (cm)
#   - EDGAR v8.0 CO2 annual emissions regridded to model grid
#   - CFL-adaptive subcycled Strang splitting
#   - Zero-allocation inner loop (MassFluxWorkspace)
#   - Unified CPU/GPU via KernelAbstractions
#
# Usage:
#   julia --project=. scripts/run_era5_edgar_forward.jl
#
# Environment variables:
#   USE_GPU      — "true" for GPU execution (default: false)
#   USE_FLOAT32  — "true" for Float32 precision (default: false)
#   DT           — advection sub-step [s] (default: 900)
#   LEVEL_TOP    — topmost model level  (default: 50)
#   LEVEL_BOT    — bottommost level     (default: 137)
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

const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "false"))
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "false"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

const FT         = USE_FLOAT32 ? Float32 : Float64
const DT         = parse(FT, get(ENV, "DT", "900"))
const LEVEL_TOP  = parse(Int, get(ENV, "LEVEL_TOP", "50"))
const LEVEL_BOT  = parse(Int, get(ENV, "LEVEL_BOT", "137"))
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT

const DATADIRS  = [expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607"),
                   expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240608_20240630")]
const EDGARFILE = expanduser("~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc")
const OUTDIR    = expanduser("~/data/output/era5_edgar_june2024")

# ---------------------------------------------------------------------------
# ERA5 model-level reader (N→S latitudes flipped to S→N)
# ---------------------------------------------------------------------------
function load_era5_timestep(filepath::String, tidx::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u     = FT.(ds["u"][:, :, :, tidx])[:, end:-1:1, :]
        v     = FT.(ds["v"][:, :, :, tidx])[:, end:-1:1, :]
        lnsp  = FT.(ds["lnsp"][:, :, tidx])[:, end:-1:1]
        return u, v, exp.(lnsp)
    finally
        close(ds)
    end
end

function get_era5_grid_info(filepath::String, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        lons = FT.(ds["longitude"][:])
        lats = FT.(ds["latitude"][:])
        levs = ds["model_level"][:]
        Nt   = length(ds["valid_time"][:])
        return lons, reverse(lats), levs, length(lons), length(lats), length(levs), Nt
    finally
        close(ds)
    end
end

# ---------------------------------------------------------------------------
# Stagger cell-center winds to faces (CPU-only, no grid array access)
# ---------------------------------------------------------------------------
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u_stag[Nx + 1, :, :] .= u_stag[1, :, :]

    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0
    v_stag[:, Ny + 1, :] .= 0
    return nothing
end

# ---------------------------------------------------------------------------
# GPU emission injection kernel (constant field, called once per met window)
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

function apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev,
                                 A_coeff, B_coeff, Nz, g, dt_window)
    FT = eltype(c)
    ΔA = FT(A_coeff[Nz + 1] - A_coeff[Nz])
    ΔB = FT(B_coeff[Nz + 1] - B_coeff[Nz])
    mol = FT(1e6 * M_AIR / M_CO2)
    backend = get_backend(c)
    Nx, Ny = size(c, 1), size(c, 2)
    k! = _emit_surface_kernel!(backend, 256)
    k!(c, flux_dev, area_j_dev, ps_dev, ΔA, ΔB, FT(g), FT(dt_window), mol, Nz;
       ndrange=(Nx, Ny))
    synchronize(backend)
end

# ---------------------------------------------------------------------------
# GPU diagnostic kernels (column-mean, surface slice)
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

# ---------------------------------------------------------------------------
# Mass diagnostic (CPU)
# ---------------------------------------------------------------------------
function total_mass(c, grid, ps_2d, A_coeff, B_coeff)
    FT = eltype(c)
    Nx, Ny, Nz = size(c)
    g = grid.gravity
    mass = zero(FT)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        area = cell_area(i, j, grid)
        Δp = (A_coeff[k + 1] - A_coeff[k]) + (B_coeff[k + 1] - B_coeff[k]) * ps_2d[i, j]
        mass += c[i, j, k] * area * Δp / g
    end
    return mass
end

# ---------------------------------------------------------------------------
# Collect daily file paths
# ---------------------------------------------------------------------------
function find_era5_files(datadirs::Vector{String})
    files = String[]
    for datadir in datadirs
        isdir(datadir) || continue
        for f in readdir(datadir)
            if startswith(f, "era5_ml_") && endswith(f, ".nc") && !contains(f, "_tmp")
                push!(files, joinpath(datadir, f))
            end
        end
    end
    sort!(files; by = basename)
    return files
end

# ===========================================================================
# Main simulation
# ===========================================================================
function run_era5_edgar()
    @info "=" ^ 70
    @info "AtmosTransportModel — ERA5 + EDGAR CO2 Forward Transport"
    @info "=" ^ 70

    files = find_era5_files(DATADIRS)
    isempty(files) && error("No ERA5 files found in $(DATADIRS)")
    isfile(EDGARFILE) || error("EDGAR file not found: $EDGARFILE")

    @info "  ERA5 files: $(length(files)) daily files"
    @info "  Level range: $(LEVEL_TOP)-$(LEVEL_BOT) ($(length(LEVEL_RANGE)) levels)"
    @info "  DT = $(DT)s, FT = $FT"

    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=LEVEL_RANGE)
    A_full, B_full = load_vertical_coefficients(config; FT)
    A_coeff = A_full[LEVEL_TOP:LEVEL_BOT+1]
    B_coeff = B_full[LEVEL_TOP:LEVEL_BOT+1]
    Nz_vc = length(LEVEL_RANGE)

    lons, lats, levs, Nx, Ny, Nz, Nt_per_file = get_era5_grid_info(files[1], FT)
    @assert Nz == Nz_vc "File has $Nz levels but expected $Nz_vc"
    Δlon = lons[2] - lons[1]
    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt/file=$Nt_per_file"

    params = load_parameters(FT)
    pp = params.planet

    arch = USE_GPU ? GPU() : CPU()
    AT = array_type(arch)
    @info "  Architecture: $(USE_GPU ? "GPU (CUDA)" : "CPU")"

    grid = LatitudeLongitudeGrid(arch;
        FT, size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    @info "\n--- Loading EDGAR CO2 emissions ---"
    edgar_source = load_edgar_co2(EDGARFILE, grid; year=2022)
    total_flux = sum(edgar_source.flux[i, j] * cell_area(i, j, grid)
                     for j in 1:Ny, i in 1:Nx)
    @info "  Total EDGAR emission rate: $(round(total_flux, digits=1)) kg/s " *
          "($(round(total_flux * 365.25 * 24 * 3600 / 1e12, digits=2)) Gt/yr)"

    # --- Timing setup ---
    met_interval = FT(21600)   # 6 hours between ERA5 snapshots
    steps_per_met = max(1, round(Int, met_interval / DT))
    total_met_windows = Nt_per_file * length(files)
    total_steps = total_met_windows * steps_per_met
    half_dt = DT / 2
    dt_window = DT * steps_per_met  # total time per met window

    @info "\n--- Simulation setup ---"
    @info "  Total steps: $total_steps ($(length(files)) days)"
    @info "  Steps per met window: $steps_per_met, dt_window=$(dt_window)s"
    @info "  Emissions: injected ONCE per met window (constant field)"
    @info "  Mass fluxes: pre-computed per window (TM5 dynam0 style)"

    # --- Pre-allocate ALL arrays (zero-alloc inner loop) ---
    flux_dev   = AT(edgar_source.flux)
    area_j_cpu = FT[cell_area(1, j, grid) for j in 1:Ny]
    area_j_dev = AT(area_j_cpu)

    Δp_cpu   = Array{FT}(undef, Nx, Ny, Nz)
    u_cpu    = Array{FT}(undef, Nx + 1, Ny, Nz)
    v_cpu    = Array{FT}(undef, Nx, Ny + 1, Nz)

    Δp       = AT(zeros(FT, Nx, Ny, Nz))
    u_dev    = AT(zeros(FT, Nx + 1, Ny, Nz))
    v_dev    = AT(zeros(FT, Nx, Ny + 1, Nz))
    ps_dev   = AT(zeros(FT, Nx, Ny))
    m        = AT(zeros(FT, Nx, Ny, Nz))
    am       = AT(zeros(FT, Nx + 1, Ny, Nz))
    bm       = AT(zeros(FT, Nx, Ny + 1, Nz))
    cm       = AT(zeros(FT, Nx, Ny, Nz + 1))
    ws       = allocate_massflux_workspace(m, am, bm, cm)
    c        = AT(zeros(FT, Nx, Ny, Nz))
    tracers  = (; c)

    # Diffusion workspace (GPU Thomas solver — eliminates D2H/H2D per sub-step)
    diff_ws = DiffusionWorkspace(grid, FT(50), DT, c)

    # GPU diagnostic buffers
    c_col_dev = AT(zeros(FT, Nx, Ny))
    c_sfc_dev = AT(zeros(FT, Nx, Ny))

    @info "  All arrays pre-allocated ($(USE_GPU ? "GPU" : "CPU"), zero-alloc inner loop)"

    # --- Output ---
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "era5_edgar_$(USE_FLOAT32 ? "f32" : "f64").nc")
    rm(outfile; force=true)
    @info "  Output: $outfile"

    snapshot_interval = max(1, round(Int, 3600 / DT))  # ~hourly

    NCDataset(outfile, "c") do ds
        defDim(ds, "lon", Nx); defDim(ds, "lat", Ny); defDim(ds, "time", Inf)
        defVar(ds, "lon", Float32, ("lon",);
               attrib = Dict("units" => "degrees_east"))[:] = Float32.(lons)
        defVar(ds, "lat", Float32, ("lat",);
               attrib = Dict("units" => "degrees_north"))[:] = Float32.(lats)
        defVar(ds, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since 2024-06-01 00:00:00"))
        defVar(ds, "co2_surface", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm", "long_name" => "Surface CO2 from EDGAR"))
        defVar(ds, "co2_column_mean", Float32, ("lon", "lat", "time");
               attrib = Dict("units" => "ppm", "long_name" => "Column-mean CO2"))
        defVar(ds, "mass_total", Float64, ("time",))
        defVar(ds, "emitted_total_kg", Float64, ("time",))

        ds["time"][1] = 0.0
        ds["co2_surface"][:, :, 1] = zeros(Float32, Nx, Ny)
        ds["co2_column_mean"][:, :, 1] = zeros(Float32, Nx, Ny)
        ds["mass_total"][1] = 0.0
        ds["emitted_total_kg"][1] = 0.0

        snapshot_idx = 1
        step = 0
        met_window = 0
        cumulative_emitted_kg = 0.0
        wall_start = time()

        @info "\n--- Starting simulation ---"

        for (day_idx, filepath) in enumerate(files)
            @info "\nDay $day_idx: $(basename(filepath))"

            for tidx in 1:Nt_per_file
                met_window += 1

                # ===== PHASE 1: Pre-compute mass fluxes (TM5 dynam0) =====
                t_phase1 = time()

                u_cc, v_cc, ps = load_era5_timestep(filepath, tidx, FT)
                stagger_winds!(u_cpu, v_cpu, u_cc, v_cc, Nx, Ny, Nz)
                Advection._build_Δz_3d!(Δp_cpu, grid, ps)

                copyto!(Δp, Δp_cpu)
                copyto!(u_dev, u_cpu)
                copyto!(v_dev, v_cpu)
                copyto!(ps_dev, ps)

                compute_air_mass!(m, Δp, grid)
                compute_mass_fluxes!(am, bm, cm, u_dev, v_dev, grid, Δp, half_dt)

                t_phase1_done = time() - t_phase1

                # ===== Inject emissions ONCE for this met window =====
                apply_emissions_window!(tracers.c, flux_dev, area_j_dev,
                                         ps_dev, A_coeff, B_coeff, Nz,
                                         grid.gravity, dt_window)
                cumulative_emitted_kg += total_flux * Float64(dt_window)

                # ===== PHASE 2: Advection + diffusion sub-steps =====
                t_phase2 = time()
                for sub in 1:steps_per_met
                    step += 1
                    strang_split_massflux!(tracers, m, am, bm, cm,
                                           grid, true, ws; cfl_limit = FT(0.95))
                    diffuse_gpu!(tracers, diff_ws)
                end
                t_phase2_done = time() - t_phase2

                # --- Diagnostics (every snapshot_interval steps) ---
                if step % snapshot_interval < steps_per_met || step <= steps_per_met
                    compute_diagnostics_gpu!(c_col_dev, c_sfc_dev, tracers.c, Δp, Nz)
                    c_sfc = Array(c_sfc_dev)
                    c_col = Array(c_col_dev)
                    sim_hours = step * Float64(DT) / 3600.0

                    snapshot_idx += 1
                    ds["time"][snapshot_idx] = sim_hours
                    ds["co2_surface"][:, :, snapshot_idx] = Float32.(c_sfc)
                    ds["co2_column_mean"][:, :, snapshot_idx] = Float32.(c_col)
                    ds["mass_total"][snapshot_idx] = Float64(sum(tracers.c))
                    ds["emitted_total_kg"][snapshot_idx] = cumulative_emitted_kg

                    elapsed = round(time() - wall_start, digits=1)
                    rate = met_window > 1 ? round((time() - wall_start) / (met_window - 1), digits=2) : 0.0

                    @info @sprintf(
                        "  Window %d/%d (day %.1f): precomp=%.2fs advect=%.2fs | sfc_max=%.3f mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
                        met_window, total_met_windows, sim_hours/24,
                        t_phase1_done, t_phase2_done,
                        maximum(c_sfc), sum(tracers.c)/length(tracers.c),
                        elapsed, rate)
                end
            end
        end

        wall_total = round(time() - wall_start, digits=1)
        c_final = Array(tracers.c)
        ps_last = load_era5_timestep(files[end], Nt_per_file, FT)[3]
        final_mass = total_mass(c_final, grid, ps_last, A_coeff, B_coeff)
        n_neg = count(x -> x < 0, c_final)

        @info "\n" * "=" ^ 70
        @info "Simulation complete!"
        @info "=" ^ 70
        @info "  Total: $step steps, $met_window met windows, $(step * Float64(DT) / 3600 / 24) days"
        @info "  Wall time: $(wall_total)s ($(round(wall_total/met_window, digits=2))s/window)"
        @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"
        @info ""
        @info "  Tracer: min=$(@sprintf("%.6f", minimum(c_final))), max=$(@sprintf("%.4f", maximum(c_final))), mean=$(@sprintf("%.6f", sum(c_final)/length(c_final))) ppm"
        @info "  Emitted: $(@sprintf("%.2f", cumulative_emitted_kg/1e9)) kt CO2"
        @info "  Negative cells: $n_neg / $(length(c_final))"
        @info "  Output: $outfile ($snapshot_idx snapshots)"
        @info "=" ^ 70
    end
end

run_era5_edgar()
