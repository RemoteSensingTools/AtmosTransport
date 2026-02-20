#!/usr/bin/env julia
# ===========================================================================
# Forward transport reading PRE-COMPUTED mass fluxes from NetCDF
#
# The mass fluxes (am, bm, cm, m) were computed by preprocess_mass_fluxes.jl
# and saved to disk. This script:
#   1. Reads am/bm/cm/m per met window from the preprocessed file
#   2. Copies to GPU (single H2D transfer per window)
#   3. Injects EDGAR emissions once per window (constant field)
#   4. Runs Strang-split advection sub-steps (pure GPU, zero allocation)
#
# This separation means:
#   - No wind staggering at runtime
#   - No compute_mass_fluxes! at runtime
#   - Adjoint runs reuse the same preprocessed file
#   - The GPU only does advection kernels
#
# Usage:
#   USE_GPU=true USE_FLOAT32=true julia --project=. scripts/run_forward_preprocessed.jl
#
# Environment variables:
#   USE_GPU      — "true" for GPU execution (default: false)
#   USE_FLOAT32  — "true" for Float32 precision (default: false)
#   DT           — advection sub-step [s] (read from preprocessed file if not set)
#   MASSFLUX_FILE — path to preprocessed mass-flux NetCDF
#   EDGAR_FILE   — path to EDGAR emission NetCDF
#   OUTDIR       — output directory
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
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

const FT = USE_FLOAT32 ? Float32 : Float64

const MASSFLUX_FILE = expanduser(get(ENV, "MASSFLUX_FILE",
    "~/data/metDrivers/era5/massflux_era5_202406_$(USE_FLOAT32 ? "float32" : "float64").nc"))
const EDGAR_FILE = expanduser(get(ENV, "EDGAR_FILE",
    "~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc"))
const OUTDIR = expanduser(get(ENV, "OUTDIR",
    "~/data/output/era5_edgar_preprocessed_$(USE_FLOAT32 ? "f32" : "f64")"))

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
# Main simulation
# ===========================================================================
function run_forward()
    @info "=" ^ 70
    @info "AtmosTransportModel — Forward Run from Preprocessed Mass Fluxes"
    @info "=" ^ 70

    isfile(MASSFLUX_FILE) || error("Preprocessed file not found: $MASSFLUX_FILE")
    isfile(EDGAR_FILE) || error("EDGAR file not found: $EDGAR_FILE")

    # --- Read metadata from preprocessed file ---
    ds_mf = NCDataset(MASSFLUX_FILE)
    lons = FT.(ds_mf["lon"][:])
    lats = FT.(ds_mf["lat"][:])
    Nx   = length(lons)
    Ny   = length(lats)
    Nz   = ds_mf.dim["lev"]
    Nt   = ds_mf.dim["time"]

    dt_file = FT(ds_mf.attrib["dt_seconds"])
    half_dt_file = FT(ds_mf.attrib["half_dt_seconds"])
    steps_per_met = ds_mf.attrib["steps_per_met_window"]
    level_top = ds_mf.attrib["level_top"]
    level_bot = ds_mf.attrib["level_bot"]

    DT = haskey(ENV, "DT") ? parse(FT, ENV["DT"]) : dt_file
    met_interval = FT(DT * steps_per_met)
    total_steps = Nt * steps_per_met

    @info "  Preprocessed: $MASSFLUX_FILE"
    @info "  Grid: Nx=$Nx, Ny=$Ny, Nz=$Nz, windows=$Nt"
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
    am_dev = AT(zeros(FT, Nx + 1, Ny, Nz))
    bm_dev = AT(zeros(FT, Nx, Ny + 1, Nz))
    cm_dev = AT(zeros(FT, Nx, Ny, Nz + 1))
    ps_dev = AT(zeros(FT, Nx, Ny))
    c      = AT(zeros(FT, Nx, Ny, Nz))
    tracers = (; c)
    ws = allocate_massflux_workspace(m_dev, am_dev, bm_dev, cm_dev)

    @info "  All arrays pre-allocated ($(USE_GPU ? "GPU" : "CPU"))"

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
               attrib = Dict("units" => "hours since 2024-06-01 00:00:00"))
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
        wall_start = time()

        @info "\n--- Starting simulation ---"

        for win in 1:Nt
            t_win = time()

            # Load preprocessed mass fluxes (CPU read → GPU copy)
            m_cpu  = FT.(ds_mf["m"][:, :, :, win])
            am_cpu = FT.(ds_mf["am"][:, :, :, win])
            bm_cpu = FT.(ds_mf["bm"][:, :, :, win])
            cm_cpu = FT.(ds_mf["cm"][:, :, :, win])
            ps_cpu = FT.(ds_mf["ps"][:, :, win])

            copyto!(m_dev, m_cpu)
            copyto!(am_dev, am_cpu)
            copyto!(bm_dev, bm_cpu)
            copyto!(cm_dev, cm_cpu)
            copyto!(ps_dev, ps_cpu)

            t_load = time() - t_win

            # Inject emissions once for this window
            dt_window = FT(DT * steps_per_met)
            apply_emissions_window!(tracers.c, flux_dev, area_j_dev,
                                     ps_dev, A_coeff, B_coeff, Nz,
                                     grid.gravity, dt_window)

            # Advection sub-steps (pure GPU)
            t_adv = time()
            for sub in 1:steps_per_met
                step += 1
                strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                                       grid, true, ws; cfl_limit = FT(0.95))
            end
            t_adv_done = time() - t_adv

            # Periodic snapshots
            if step % snapshot_every < steps_per_met || step <= steps_per_met
                c_host = Array(tracers.c)
                c_sfc = c_host[:, :, Nz]
                c_col = dropdims(sum(c_host, dims=3), dims=3) ./ Nz
                sim_hours = step * Float64(DT) / 3600.0

                snapshot_idx += 1
                ds_out["time"][snapshot_idx] = sim_hours
                ds_out["co2_surface"][:, :, snapshot_idx] = Float32.(c_sfc)
                ds_out["co2_column_mean"][:, :, snapshot_idx] = Float32.(c_col)
                ds_out["mass_total"][snapshot_idx] = Float64(sum(c_host))

                elapsed = round(time() - wall_start, digits=1)
                rate = win > 1 ? round((time() - wall_start) / (win - 1), digits=2) : 0.0

                @info @sprintf(
                    "  Win %d/%d (day %.1f): load=%.2fs adv=%.2fs | sfc_max=%.3f mean=%.5f ppm | wall=%.0fs (%.2fs/win)",
                    win, Nt, sim_hours/24,
                    t_load, t_adv_done,
                    maximum(c_sfc), sum(c_host)/length(c_host),
                    elapsed, rate)
            end
        end

        wall_total = round(time() - wall_start, digits=1)
        c_final = Array(tracers.c)
        n_neg = count(x -> x < 0, c_final)

        @info "\n" * "=" ^ 70
        @info "Simulation complete!"
        @info "=" ^ 70
        @info "  Steps: $step, windows: $Nt, days: $(step * Float64(DT) / 3600 / 24)"
        @info "  Wall time: $(wall_total)s ($(round(wall_total/Nt, digits=2))s/window)"
        @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"
        @info @sprintf("  Tracer: min=%.6f, max=%.4f, mean=%.6f ppm",
                       minimum(c_final), maximum(c_final),
                       sum(c_final)/length(c_final))
        @info "  Negative cells: $n_neg / $(length(c_final))"
        @info "  Output: $outfile ($snapshot_idx snapshots)"
        @info "=" ^ 70
    end

    close(ds_mf)
end

run_forward()
