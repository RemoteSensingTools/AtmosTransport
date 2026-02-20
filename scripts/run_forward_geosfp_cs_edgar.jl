#!/usr/bin/env julia
# ===========================================================================
# Forward transport of EDGAR CO2 emissions on GEOS-FP native cubed-sphere
#
# Architecture:
#   - Reads native GEOS-FP cubed-sphere mass fluxes (MFXC, MFYC) and DELP
#   - No wind-to-flux conversion needed (mass fluxes are pre-computed by GEOS)
#   - Panel-local mass-flux advection with halo exchange
#   - Strang-split X-Y-Z-Z-Y-X per timestep
#   - EDGAR CO2 emissions regridded to cubed-sphere panels
#
# Usage:
#   julia --project=. scripts/run_forward_geosfp_cs_edgar.jl
#
# Environment variables:
#   USE_GPU      — "true" for GPU execution (default: false)
#   USE_FLOAT32  — "true" for Float32 precision (default: false)
#   DT           — advection sub-step [s] (default: 900)
#   GEOSFP_DATA_DIR — directory with downloaded GEOS-FP CS data
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Grids: fill_panel_halos!
using AtmosTransportModel.Advection
using AtmosTransportModel.Parameters
using AtmosTransportModel.Sources: load_edgar_co2, M_AIR, M_CO2
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate,
                              read_geosfp_cs_timestep, to_haloed_panels,
                              GeosFPCubedSphereTimestep
using KernelAbstractions: synchronize, get_backend
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
const DT = parse(FT, get(ENV, "DT", "900"))

const GEOSFP_DIR = get(ENV, "GEOSFP_DATA_DIR",
                       joinpath(homedir(), "data", "geosfp_cs"))
const EDGARFILE  = get(ENV, "EDGAR_FILE",
                       joinpath(homedir(), "data", "emissions", "edgar_v8",
                                "v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc"))
const OUTDIR     = joinpath(homedir(), "data", "output", "geosfp_cs_edgar_june2024")

# ---------------------------------------------------------------------------
# Find GEOS-FP cubed-sphere files for a date range
# ---------------------------------------------------------------------------
function find_geosfp_cs_files(datadir::String, start_date::Date, end_date::Date)
    mst_files = String[]  # mass flux files (MFXC, MFYC)
    asm_files = String[]  # assembly files (DELP)

    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        daydir = joinpath(datadir, datestr)
        isdir(daydir) || continue

        for f in sort(readdir(daydir))
            fp = joinpath(daydir, f)
            if contains(f, "tavg3_3d_mst_Cp")
                push!(mst_files, fp)
            elseif contains(f, "inst3_3d_asm_Cp")
                push!(asm_files, fp)
            end
        end
    end

    return mst_files, asm_files
end

# ---------------------------------------------------------------------------
# Regrid EDGAR lat-lon emissions to cubed-sphere panels (nearest-neighbor)
# ---------------------------------------------------------------------------
function regrid_edgar_to_cs(edgar_flux::Matrix{FT},
                            edgar_lons::Vector{FT},
                            edgar_lats::Vector{FT},
                            grid::CubedSphereGrid{FT}) where FT
    Nc = grid.Nc
    Δlon_edgar = edgar_lons[2] - edgar_lons[1]
    Δlat_edgar = edgar_lats[2] - edgar_lats[1]

    flux_panels = ntuple(6) do p
        panel_flux = zeros(FT, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            lon = grid.λᶜ[p][i, j]
            lat = grid.φᶜ[p][i, j]

            lon = mod(lon + 180, 360) - 180
            ii = clamp(round(Int, (lon - edgar_lons[1]) / Δlon_edgar) + 1,
                       1, length(edgar_lons))
            jj = clamp(round(Int, (lat - edgar_lats[1]) / Δlat_edgar) + 1,
                       1, length(edgar_lats))
            panel_flux[i, j] = edgar_flux[ii, jj]
        end
        panel_flux
    end

    return flux_panels
end

# ---------------------------------------------------------------------------
# Apply emissions to the surface layer of each panel
# ---------------------------------------------------------------------------
function apply_emissions_cs!(c_panels::NTuple{6},
                             flux_panels::NTuple{6},
                             m_panels::NTuple{6},
                             gc::CubedSphereGeometryCache,
                             dt_window::FT) where FT
    mol_ratio = FT(1e6 * M_AIR / M_CO2)
    Hp = gc.Hp
    Nc = gc.Nc
    Nz = gc.Nz

    for p in 1:6
        @inbounds for j in 1:Nc, i in 1:Nc
            f = flux_panels[p][i, j]
            if f != zero(FT)
                area = gc.area[p][i, j]
                m_air = m_panels[p][Hp + i, Hp + j, Nz]
                ΔM = f * dt_window * area
                c_panels[p][Hp + i, Hp + j, Nz] += ΔM * mol_ratio / m_air
            end
        end
    end
end

# ===========================================================================
# Main simulation
# ===========================================================================
function run_geosfp_cs_edgar()
    @info "=" ^ 70
    @info "AtmosTransportModel — GEOS-FP Cubed-Sphere + EDGAR CO2 Forward"
    @info "=" ^ 70

    start_date = Date(2024, 6, 1)
    end_date   = Date(2024, 6, 30)

    mst_files, asm_files = find_geosfp_cs_files(GEOSFP_DIR, start_date, end_date)
    isempty(mst_files) && error("No GEOS-FP CS mst files found in $GEOSFP_DIR")
    isempty(asm_files) && error("No GEOS-FP CS asm files found in $GEOSFP_DIR")
    @info "  MST files: $(length(mst_files)), ASM files: $(length(asm_files))"

    # Inspect first file to get grid dimensions
    ts0 = read_geosfp_cs_timestep(mst_files[1]; asm_file=asm_files[1], FT)
    Nc, Nz = ts0.Nc, ts0.Nz
    @info "  Grid: C$(Nc), Nz=$Nz"

    config = default_met_config("geosfp")
    vc = build_vertical_coordinate(config; FT)
    params = load_parameters(FT)
    pp = params.planet

    Hp = 3
    arch = USE_GPU ? GPU() : CPU()
    @info "  Architecture: $(USE_GPU ? "GPU (CUDA)" : "CPU")"

    grid = CubedSphereGrid(arch;
        FT, Nc, Nz,
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    # Build geometry cache (device-side cell areas, metrics, bt)
    ref_panel = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
    gc = build_geometry_cache(grid, ref_panel)

    # Workspace for advection (reused across panels)
    ws = allocate_cs_massflux_workspace(ref_panel)

    # Allocate tracer and air mass (haloed)
    c_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    m_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)

    # Load EDGAR emissions
    @info "\n--- Loading EDGAR CO2 emissions ---"
    isfile(EDGARFILE) || error("EDGAR file not found: $EDGARFILE")
    ds = NCDataset(EDGARFILE)
    edgar_lons = FT.(ds["lon"][:])
    edgar_lats = FT.(ds["lat"][:])
    edgar_flux_raw = FT.(ds["emi_co2"][:, :])
    close(ds)
    edgar_flux_raw[ismissing.(edgar_flux_raw)] .= zero(FT)

    flux_panels = regrid_edgar_to_cs(edgar_flux_raw, edgar_lons, edgar_lats, grid)
    @info "  EDGAR regridded to C$(Nc) cubed-sphere panels"

    # --- Timing setup ---
    met_interval = FT(10800)   # 3 hours between GEOS-FP snapshots
    steps_per_met = max(1, round(Int, met_interval / DT))
    dt_window = DT * steps_per_met

    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "geosfp_cs_edgar_$(USE_FLOAT32 ? "f32" : "f64").nc")
    rm(outfile; force=true)
    @info "  Output: $outfile"

    # --- Main loop ---
    n_windows = min(length(mst_files), length(asm_files))
    step = 0
    wall_start = time()

    @info "\n--- Starting simulation ($n_windows met windows) ---"

    for w in 1:n_windows
        t_load = time()

        ts = read_geosfp_cs_timestep(mst_files[w]; asm_file=asm_files[w], FT)
        delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)

        # Compute air mass from DELP
        for p in 1:6
            m_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= zero(FT)
            compute_air_mass_panel!(m_panels[p], delp_haloed[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # Compute vertical mass flux (cm) from horizontal convergence
        cm_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
        for p in 1:6
            compute_cm_panel!(cm_panels[p], mfxc[p], mfyc[p], gc.bt, Nc, Nz)
        end

        t_load_done = time() - t_load

        # Inject emissions once per met window
        apply_emissions_cs!(c_panels, flux_panels, m_panels, gc, dt_window)

        # Advection sub-steps
        t_adv = time()
        for sub in 1:steps_per_met
            step += 1
            strang_split_massflux!(c_panels, m_panels,
                                   mfxc, mfyc, cm_panels,
                                   grid, true, ws)
        end
        t_adv_done = time() - t_adv

        # Diagnostics
        if w % 8 == 0 || w == 1 || w == n_windows
            max_c = maximum(maximum(c_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
            elapsed = round(time() - wall_start, digits=1)
            @info @sprintf(
                "  Window %d/%d: load=%.2fs advect=%.2fs | max=%.4f ppm | wall=%.0fs",
                w, n_windows, t_load_done, t_adv_done, max_c, elapsed)
        end
    end

    wall_total = round(time() - wall_start, digits=1)
    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "  Total: $step steps, $n_windows met windows"
    @info "  Wall time: $(wall_total)s ($(round(wall_total/n_windows, digits=2))s/window)"
    @info "=" ^ 70
end

run_geosfp_cs_edgar()
