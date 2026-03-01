#!/usr/bin/env julia
#=
Diagnose the correct mass_flux_dt for GEOS-FP C720 data.

Uses the CX (accumulated Courant number) variable available in GEOS-FP files.

Key relationship:
  CFL_model = CX_file × half_dt / mass_flux_dt

where:
  CX_file = accumulated Courant number in the GEOS file
  half_dt = dt_sub / 2 = 200 s (for dt=400s Strang splitting)
  CFL_model = max CFL reported by the run (0.26-0.44)

By comparing the file's max CX with the model's max CFL, we can solve for mass_flux_dt.

Also estimates surface wind speed by accounting for cell face area:
  u ≈ MFXC / (mass_flux_dt × DELP × dy)

Usage:
  julia --project=. scripts/diagnose_massflux_dt.jl [nc_file]
=#

using NCDatasets
using Statistics: mean
using Printf

const GRAV = 9.80665  # m/s²
const R_EARTH = 6.371e6  # m

function main()
    nc_file = if length(ARGS) >= 1
        ARGS[1]
    else
        datadir = expanduser("~/data/geosfp_cs/20240601")
        files = filter(f -> endswith(f, ".nc4") && contains(f, "tavg_1hr_ctm_c0720"),
                       readdir(datadir))
        isempty(files) && error("No GEOS-FP C720 files found in $datadir")
        joinpath(datadir, first(files))
    end
    isfile(nc_file) || error("File not found: $nc_file")
    @info "Reading: $(basename(nc_file))"

    ds = NCDataset(nc_file, "r")

    mfxc_raw = Float64.(ds["MFXC"][:, :, :, :, 1])  # (720,720,6,72)
    delp_raw = Float64.(ds["DELP"][:, :, :, :, 1])
    cx_raw   = Float64.(ds["CX"][:, :, :, :, 1])
    cy_raw   = Float64.(ds["CY"][:, :, :, :, 1])

    # Print MFXC attributes
    @info "MFXC attributes:"
    for (k, v) in ds["MFXC"].attrib
        k in ("long_name", "units") && @info "  $k = $v"
    end
    @info "CX attributes:"
    for (k, v) in ds["CX"].attrib
        k in ("long_name", "units") && @info "  $k = $v"
    end
    close(ds)

    Nc = size(mfxc_raw, 1)
    Nz = size(mfxc_raw, 4)
    @info "Grid: C$Nc, Nz=$Nz"

    # Auto-detect vertical ordering
    mid = div(Nc, 2)
    if delp_raw[mid, mid, 1, 1] > 10 * delp_raw[mid, mid, 1, Nz]
        @info "Detected bottom-to-top ordering — flipping"
        reverse!(mfxc_raw, dims=4)
        reverse!(delp_raw, dims=4)
        reverse!(cx_raw, dims=4)
        reverse!(cy_raw, dims=4)
    end

    # Cell edge length (approximate for equatorial panels)
    dy = 2π * R_EARTH / (4 * Nc)
    @info @sprintf("Approximate cell edge length: %.0f m (%.2f km)", dy, dy / 1000)

    println()
    println("=" ^ 80)
    println("COURANT NUMBER STATISTICS (CX, CY)")
    println("=" ^ 80)

    # Statistics across ALL panels and ALL levels
    global_max_cx = 0.0
    global_max_cy = 0.0
    for p in 1:6
        for k in 1:Nz
            local_max_cx = maximum(abs, cx_raw[:, :, p, k])
            local_max_cy = maximum(abs, cy_raw[:, :, p, k])
            global_max_cx = max(global_max_cx, local_max_cx)
            global_max_cy = max(global_max_cy, local_max_cy)
        end
    end

    # Surface-level statistics (level Nz = surface, TOA-first)
    sfc = Nz
    sfc_max_cx = maximum(abs, cx_raw[:, :, :, sfc])
    sfc_max_cy = maximum(abs, cy_raw[:, :, :, sfc])
    sfc_rms_cx = sqrt(mean(x -> x^2, cx_raw[:, :, :, sfc]))

    @info @sprintf("Global max |CX| (all levels, all panels): %.4f", global_max_cx)
    @info @sprintf("Global max |CY| (all levels, all panels): %.4f", global_max_cy)
    @info @sprintf("Surface max |CX| (all panels):            %.4f", sfc_max_cx)
    @info @sprintf("Surface max |CY| (all panels):            %.4f", sfc_max_cy)
    @info @sprintf("Surface RMS |CX| (all panels):            %.4f", sfc_rms_cx)

    # Find which level has the max CX
    for k in 1:Nz
        maxcx_k = maximum(abs, cx_raw[:, :, :, k])
        if maxcx_k > 0.95 * global_max_cx
            mean_delp_k = mean(delp_raw[:, :, :, k])
            @info @sprintf("  Max CX at level %d (mean DELP=%.1f Pa): |CX|=%.4f", k, mean_delp_k, maxcx_k)
        end
    end

    println()
    println("=" ^ 80)
    println("MASS_FLUX_DT INFERENCE")
    println("=" ^ 80)
    println()
    println("Relationship: CFL_model = CX_file × half_dt / mass_flux_dt")
    println("Model reports: max CFL ≈ 0.26-0.44 (from run log)")
    println()

    # Solve for mass_flux_dt given different observed CFL values
    for cfl_observed in [0.26, 0.35, 0.44]
        dt_inferred = global_max_cx * 200.0 / cfl_observed
        @printf("  If max CFL = %.2f → mass_flux_dt = %.0f s (= %.1f min)\n",
                cfl_observed, dt_inferred, dt_inferred / 60)
    end

    println()
    println("=" ^ 80)
    println("WIND SPEED ESTIMATES (corrected for cell area)")
    println("=" ^ 80)
    println()

    # Surface wind estimate using CX:
    # CX = U × T / dy  →  U = CX × dy / T
    candidates = [112.5, 225.0, 450.0, 900.0, 1800.0, 3600.0]

    println("  mass_flux_dt [s]  |  RMS u_sfc [m/s]  |  Max u_sfc [m/s]  |  Assessment")
    println("-" ^ 80)

    for dt in candidates
        # u = CX × dy / dt (where CX is accumulated Courant number over time dt)
        rms_u = sfc_rms_cx * dy / dt
        max_u = sfc_max_cx * dy / dt

        assessment = if 3.0 < rms_u < 15.0
            "  << PLAUSIBLE"
        elseif rms_u > 30.0
            "  TOO HIGH"
        elseif rms_u < 1.0
            "  TOO LOW"
        else
            ""
        end

        @printf("  %10.1f        |  %10.2f        |  %10.2f        | %s\n",
                dt, rms_u, max_u, assessment)
    end
    println("=" ^ 80)
    println()
    println("Expected: correct mass_flux_dt gives RMS surface wind ~5-10 m/s")
end

main()
