#!/usr/bin/env julia
# ===========================================================================
# Forward transport of EDGAR CO2 emissions on GEOS-FP native cubed-sphere
#
# Architecture:
#   - Reads native GEOS-FP C720 cubed-sphere mass fluxes (MFXC, MFYC) and DELP
#     from Washington University archive (geoschemdata.wustl.edu)
#   - No wind-to-flux conversion needed (mass fluxes are pre-computed by GEOS)
#   - Panel-local mass-flux advection with halo exchange
#   - Strang-split X-Y-Z-Z-Y-X per timestep
#   - EDGAR CO2 emissions regridded to cubed-sphere panels
#
# Data files: GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HHMM.V01.nc4
#   Contains MFXC, MFYC, DELP, PS, CX, CY in a single file.
#   C-grid convention: MFXC(i,j) = flux through east face of cell (i,j).
#   Units: Pa·m² (pressure-weighted accumulated), converted to kg/s.
#
# Usage:
#   julia --project=. scripts/run_forward_geosfp_cs_edgar.jl
#
# Environment variables:
#   USE_GPU      — "true" for GPU execution (default: false)
#   USE_FLOAT32  — "true" for Float32 precision (default: true)
#   DT           — advection sub-step [s] (default: 900)
#   GEOSFP_DATA_DIR — directory with downloaded GEOS-FP CS data
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Grids: fill_panel_halos!
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.Sources: load_edgar_co2, M_AIR, M_CO2
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              read_geosfp_cs_timestep, to_haloed_panels,
                              cgrid_to_staggered_panels,
                              GeosFPCubedSphereTimestep
using KernelAbstractions: synchronize, get_backend
using NCDatasets
using Dates

const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "false"))
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "true"))
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
# Find GEOS-FP cubed-sphere files (WashU format)
# ---------------------------------------------------------------------------
function find_geosfp_cs_files(datadir::String, start_date::Date, end_date::Date)
    files = String[]
    for date in start_date:Day(1):end_date
        datestr = Dates.format(date, "yyyymmdd")
        daydir = joinpath(datadir, datestr)
        isdir(daydir) || continue
        for f in sort(readdir(daydir))
            if contains(f, "tavg_1hr_ctm_c0720_v72") && endswith(f, ".nc4")
                push!(files, joinpath(daydir, f))
            end
        end
    end
    return files
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
    @info "AtmosTransport — GEOS-FP C720 Cubed-Sphere + EDGAR CO2"
    @info "Data source: geoschemdata.wustl.edu (Washington University)"
    @info "=" ^ 70

    start_date = Date(2024, 6, 1)
    end_date   = Date(2024, 6, 30)

    files = find_geosfp_cs_files(GEOSFP_DIR, start_date, end_date)
    isempty(files) && error("No GEOS-FP CS files found in $GEOSFP_DIR")
    @info "  Found $(length(files)) tavg_1hr_ctm files"

    # Read first file to get grid dimensions
    @info "  Reading first file to get grid dimensions..."
    ts0 = read_geosfp_cs_timestep(files[1]; FT)
    Nc, Nz = ts0.Nc, ts0.Nz
    @info "  Grid: C$(Nc), Nz=$Nz"

    config = default_met_config("geosfp")
    vc = build_vertical_coordinate(config; FT)
    params = load_parameters(FT)
    pp = params.planet

    Hp = 3
    arch = USE_GPU ? GPU() : CPU()
    @info "  Architecture: $(USE_GPU ? "GPU (CUDA)" : "CPU")"
    @info "  Precision: $(USE_FLOAT32 ? "Float32" : "Float64")"
    @info "  Timestep DT=$DT s"

    grid = CubedSphereGrid(arch;
        FT, Nc,
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    ref_panel = zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
    gc = build_geometry_cache(grid, ref_panel)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)

    # Allocate tracer and air mass (haloed)
    c_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    m_panels = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)

    # Load EDGAR emissions
    @info "\n--- Loading EDGAR CO2 emissions ---"
    isfile(EDGARFILE) || error("EDGAR file not found: $EDGARFILE")
    ds = NCDataset(EDGARFILE)
    edgar_lons = FT.(ds["lon"][:])
    edgar_lats = FT.(ds["lat"][:])
    edgar_flux_raw = FT.(replace(ds["emi_co2"][:, :], missing => zero(FT)))
    close(ds)

    flux_panels = regrid_edgar_to_cs(edgar_flux_raw, edgar_lons, edgar_lats, grid)
    @info "  EDGAR regridded to C$(Nc) cubed-sphere panels"

    # Met interval = 1 hour (hourly tavg files)
    met_interval = FT(3600)
    steps_per_met = max(1, round(Int, met_interval / DT))
    dt_window = DT * steps_per_met

    mkpath(OUTDIR)
    @info "  Output directory: $OUTDIR"

    # --- Main loop ---
    n_files = length(files)
    step = 0
    wall_start = time()
    save_every = 24  # save output every 24 files (= every day)

    @info "\n--- Starting simulation ($n_files met windows, $steps_per_met sub-steps each) ---"

    for w in 1:n_files
        t_load = time()

        # Read timestep — single file contains MFXC, MFYC, DELP, PS
        ts = read_geosfp_cs_timestep(files[w]; FT, convert_to_kgs=true)
        delp_haloed, mfxc_cgrid, mfyc_cgrid = to_haloed_panels(ts; Hp)

        # Convert C-grid fluxes to staggered arrays for advection kernels
        am_panels, bm_panels = cgrid_to_staggered_panels(mfxc_cgrid, mfyc_cgrid)

        # Compute air mass from DELP
        for p in 1:6
            compute_air_mass_panel!(m_panels[p], delp_haloed[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end

        # Compute vertical mass flux from horizontal convergence
        cm_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
        for p in 1:6
            compute_cm_panel!(cm_panels[p], am_panels[p], bm_panels[p], gc.bt, Nc, Nz)
        end

        t_load_done = time() - t_load

        # Inject emissions once per met window
        apply_emissions_cs!(c_panels, flux_panels, m_panels, gc, dt_window)

        # Advection sub-steps
        t_adv = time()
        for sub in 1:steps_per_met
            step += 1
            strang_split_massflux!(c_panels, m_panels,
                                   am_panels, bm_panels, cm_panels,
                                   grid, true, ws)
        end
        t_adv_done = time() - t_adv

        # Diagnostics
        if w % 24 == 0 || w == 1 || w == n_files
            max_c = maximum(maximum(c_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
            elapsed = round(time() - wall_start, digits=1)
            day = div(w - 1, 24) + 1
            @info "  Day $day, window $w/$n_files: load=$(round(t_load_done,digits=2))s " *
                  "advect=$(round(t_adv_done,digits=2))s | max=$(round(max_c,digits=4)) ppm | " *
                  "wall=$(elapsed)s"
        end

        # Save daily snapshots
        if w % save_every == 0
            day = div(w, 24)
            save_snapshot(OUTDIR, ts.time, c_panels, m_panels, gc, Nc, Nz, Hp, day)
        end
    end

    # Final save
    save_snapshot(OUTDIR, DateTime(2024, 6, 30, 23, 30),
                  c_panels, m_panels, gc, Nc, Nz, Hp, 30)

    wall_total = round(time() - wall_start, digits=1)
    @info "\n" * "=" ^ 70
    @info "Simulation complete!"
    @info "  Total: $step steps, $n_files met windows"
    @info "  Wall time: $(wall_total)s ($(round(wall_total/60, digits=1)) min)"
    @info "  Per window: $(round(wall_total/n_files, digits=2))s"
    @info "=" ^ 70
end

function save_snapshot(outdir, time, c_panels, m_panels, gc, Nc, Nz, Hp, day)
    fname = joinpath(outdir, "snapshot_day$(lpad(day, 2, '0')).nc")
    ds = NCDataset(fname, "c")
    defDim(ds, "x", Nc)
    defDim(ds, "y", Nc)
    defDim(ds, "z", Nz)
    defDim(ds, "panel", 6)

    v_c = defVar(ds, "co2_ppm", Float32, ("x", "y", "panel", "z"))
    v_m = defVar(ds, "air_mass", Float32, ("x", "y", "panel", "z"))

    for p in 1:6
        v_c[:, :, p, :] = Float32.(c_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])
        v_m[:, :, p, :] = Float32.(m_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])
    end

    ds.attrib["time"] = string(time)
    ds.attrib["grid"] = "CubedSphere C$(Nc)"
    ds.attrib["Nz"] = Nz
    close(ds)
    @info "  Saved snapshot: $fname"
end

run_geosfp_cs_edgar()
