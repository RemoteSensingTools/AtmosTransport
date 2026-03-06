#!/usr/bin/env julia
# =====================================================================
# Validate Julia regridding: per-cell accuracy diagnostics
#
# Tests:
#   --test-uniform          Uniform flux → CS: all cells should be 1.0
#   --test-gradient         Sharp gradient field: nearest-neighbor vs conservative
#   --test-source <file>    Real emission file: per-cell accuracy + percentiles
#   --source + --reference  Compare against xESMF reference
#
# Usage:
#   julia --project=. scripts/validation/validate_regridding_julia.jl --test-uniform
#   julia --project=. scripts/validation/validate_regridding_julia.jl --test-source ~/data/.../edgar.nc
#   julia --project=. scripts/validation/validate_regridding_julia.jl \
#       --source <file> --reference xesmf_reference.nc
# =====================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))

using AtmosTransport
using AtmosTransport.Sources
using AtmosTransport.Sources: _to_m180_180
using AtmosTransport.Grids
using AtmosTransport.IO: read_geosfp_cs_grid_info
using NCDatasets
using Printf
using Statistics: mean, median, quantile

# =====================================================================
# Helpers
# =====================================================================

function build_grid(; Nc=180, FT=Float64, coord_file=nothing)
    vert = AtmosTransport.Grids.HybridSigmaPressure(FT[0, 0], FT[0, 1])
    grid = CubedSphereGrid(AtmosTransport.Architectures.CPU(); FT, Nc, vertical=vert)

    if coord_file !== nothing && isfile(coord_file)
        lons, lats, clons, clats = read_geosfp_cs_grid_info(coord_file)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            grid.λᶜ[p][i, j] = lons[i, j, p]
            grid.φᶜ[p][i, j] = lats[i, j, p]
        end
        if clons !== nothing && clats !== nothing
            R = Float64(grid.radius)
            gmao_areas = Sources.compute_areas_from_corners(
                Float64.(clons), Float64.(clats), R, Nc)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                grid.Aᶜ[p][i, j] = FT(gmao_areas[p][i, j])
            end
            println("  Grid: GMAO coords + corner-based areas")
        else
            println("  Grid: GMAO coords, gnomonic areas")
        end
        Grids.set_coord_status!(grid, :gmao, coord_file)
    else
        println("  Grid: gnomonic (no coord file)")
    end
    return grid
end

function print_percentiles(label, vals; prefix="  ")
    isempty(vals) && return
    sorted = sort(vals)
    n = length(sorted)
    p = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    names = ["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"]
    println("$(prefix)$label (N=$n):")
    for (name, pct) in zip(names, p)
        idx = clamp(round(Int, pct * (n - 1)) + 1, 1, n)
        @printf("%s  %-4s = %+.6f\n", prefix, name, sorted[idx])
    end
end

"""
    conservative_average(flux, lons, lats, cs_lon, cs_lat, half_dx_deg) → Float64

For a single CS cell at (cs_lon, cs_lat), find all source cells whose centers
overlap the CS cell footprint (approximated as ±half_dx_deg in lon/lat), and
return their area-weighted average flux density.
"""
function conservative_average(flux::Matrix{FT}, lons, lats, src_areas,
                               cs_lon, cs_lat, half_dx_deg) where FT
    Nlon = length(lons)
    Nlat = length(lats)
    Δlon_s = abs(lons[2] - lons[1])
    Δlat_s = abs(lats[2] - lats[1])

    # CS cell bounding box
    lon_lo = cs_lon - half_dx_deg
    lon_hi = cs_lon + half_dx_deg
    lat_lo = max(cs_lat - half_dx_deg, -90.0)
    lat_hi = min(cs_lat + half_dx_deg, 90.0)

    # Find source cell index range
    j_lo = max(1, floor(Int, (lat_lo - lats[1] + Δlat_s/2) / Δlat_s) + 1)
    j_hi = min(Nlat, ceil(Int, (lat_hi - lats[1] + Δlat_s/2) / Δlat_s))

    # Handle longitude wrapping
    i_lo = floor(Int, (lon_lo - lons[1] + Δlon_s/2) / Δlon_s) + 1
    i_hi = ceil(Int, (lon_hi - lons[1] + Δlon_s/2) / Δlon_s)

    total_mass = zero(Float64)
    total_area = zero(Float64)

    for j in j_lo:j_hi
        (j < 1 || j > Nlat) && continue
        for i_raw in i_lo:i_hi
            i = mod1(i_raw, Nlon)
            src_lat = lats[j]
            src_lon = lons[i]

            # Compute overlap fraction (simplified: lon fraction × lat fraction)
            src_lon_lo = src_lon - Δlon_s / 2
            src_lon_hi = src_lon + Δlon_s / 2
            src_lat_lo = src_lat - Δlat_s / 2
            src_lat_hi = src_lat + Δlat_s / 2

            ov_lon = max(0.0, min(lon_hi, src_lon_hi) - max(lon_lo, src_lon_lo))
            ov_lat_s = max(src_lat_lo, lat_lo)
            ov_lat_n = min(src_lat_hi, lat_hi)
            ov_lat_n <= ov_lat_s && continue

            frac_lon = ov_lon / Δlon_s
            frac_lat = abs(sind(ov_lat_n) - sind(ov_lat_s)) /
                       abs(sind(src_lat_hi) - sind(src_lat_lo))

            frac = frac_lon * frac_lat
            frac < 1e-10 && continue

            overlap_area = src_areas[j] * frac
            total_mass += Float64(flux[i, j]) * overlap_area
            total_area += overlap_area
        end
    end

    return total_area > 0 ? total_mass / total_area : 0.0
end

# =====================================================================
# Test: Uniform flux
# =====================================================================

function test_uniform(; Nc=90, coord_file=nothing)
    println("\n" * "=" ^ 70)
    println("  TEST: Uniform flux → C$Nc")
    println("=" ^ 70)

    FT = Float64
    grid = build_grid(; Nc, FT, coord_file)

    lons = FT.(collect(0.5:1.0:359.5))
    lats = FT.(collect(-89.5:1.0:89.5))
    flux = ones(FT, length(lons), length(lats))

    cs_map = Sources.build_conservative_cs_map(lons, lats, grid)
    panels = Sources.regrid_latlon_to_cs(flux, lons, lats, grid; cs_map, renormalize=false)

    # Per-cell deviation from 1.0
    all_vals = Float64[]
    for p in 1:6
        append!(all_vals, vec(panels[p]))
    end
    deviations = all_vals .- 1.0

    println("\n  Per-cell value (should be 1.0):")
    @printf("    mean = %.8f\n", mean(all_vals))
    @printf("    min  = %.8f\n", minimum(all_vals))
    @printf("    max  = %.8f\n", maximum(all_vals))
    @printf("    std  = %.2e\n", std(all_vals))

    # Check for zero cells (unmapped)
    n_zero = count(==(0.0), all_vals)
    n_total = length(all_vals)
    @printf("    zero cells: %d / %d (%.2f%%)\n", n_zero, n_total, 100 * n_zero / n_total)

    for p in 1:6
        vals = vec(panels[p])
        @printf("    Panel %d: mean=%.6f, min=%.6f, max=%.6f, zeros=%d\n",
                p, mean(vals), minimum(vals), maximum(vals), count(==(0.0), vals))
    end

    pass = maximum(abs.(deviations)) < 0.01
    println(pass ? "\n  PASS" : "\n  FAIL")
    return pass
end

# =====================================================================
# Test: Source file — per-cell accuracy vs conservative average
# =====================================================================

function test_source(source_path; Nc=180, coord_file=nothing, var_name=nothing, max_cells=50000)
    println("\n" * "=" ^ 70)
    println("  TEST: Per-cell accuracy on $(basename(source_path))")
    println("=" ^ 70)

    FT = Float64
    grid = build_grid(; Nc, FT, coord_file)
    R = FT(grid.radius)

    # Load source
    ds = NCDataset(source_path)
    all_keys = collect(keys(ds))
    lon_key = nothing; lat_key = nothing
    for k in all_keys
        lk = lowercase(k)
        if lon_key === nothing && lk in ("longitude", "lon")
            lon_key = k
        elseif lat_key === nothing && lk in ("latitude", "lat")
            lat_key = k
        end
    end
    lon_key === nothing && error("No longitude coordinate found")
    lat_key === nothing && error("No latitude coordinate found")

    lons = FT.(Array(ds[lon_key]))
    lats = FT.(Array(ds[lat_key]))

    if var_name === nothing
        skip = Set(keys(ds.dim)) ∪ Set(["time", "time_bnds", lon_key, lat_key])
        candidates = [k for k in all_keys if k ∉ skip && ndims(ds[k]) >= 2]
        var_name = if length(candidates) == 1
            candidates[1]
        else
            idx = findfirst(c -> any(contains(lowercase(c), kw) for kw in
                            ["emission", "flux", "fire", "co2", "total"]), candidates)
            idx !== nothing ? candidates[idx] : candidates[1]
        end
    end

    data = Array(ds[var_name])
    raw = ndims(data) == 3 ? FT.(replace(data[:, :, 1], missing => zero(FT))) :
                             FT.(replace(data, missing => zero(FT)))
    close(ds)

    if length(lats) > 1 && lats[1] > lats[end]
        lats = reverse(lats)
        raw = raw[:, end:-1:1]
    end

    Nlon, Nlat = size(raw)
    @printf("  Source: %d×%d, var=%s\n", Nlon, Nlat, var_name)

    # Source field statistics (non-zero cells only)
    nz_src = filter(!=(0.0), vec(raw))
    @printf("  Source stats (non-zero): N=%d, min=%.4e, max=%.4e, median=%.4e\n",
            length(nz_src), minimum(nz_src), maximum(nz_src), median(nz_src))

    # Regrid WITHOUT renormalization
    cs_map = Sources.build_conservative_cs_map(lons, lats, grid)
    panels_norenorm = Sources.regrid_latlon_to_cs(raw, lons, lats, grid; cs_map, renormalize=false)

    # Regrid WITH renormalization
    panels_renorm = Sources.regrid_latlon_to_cs(raw, lons, lats, grid; cs_map, renormalize=true)

    # --- Diagnostic 1: Renormalization factor ---
    src_areas = Sources.latlon_cell_areas(lons, lats, R)
    total_src = sum(Float64(raw[i,j]) * Float64(src_areas[j]) for j in 1:Nlat, i in 1:Nlon)
    total_tgt_norenorm = sum(sum(Float64.(panels_norenorm[p]) .* Float64.(grid.Aᶜ[p])) for p in 1:6)
    total_tgt_renorm = sum(sum(Float64.(panels_renorm[p]) .* Float64.(grid.Aᶜ[p])) for p in 1:6)
    renorm_factor = total_src / total_tgt_norenorm

    println("\n  --- Mass integrals ---")
    @printf("    Source:          %.6e kg/s\n", total_src)
    @printf("    Target (raw):    %.6e kg/s\n", total_tgt_norenorm)
    @printf("    Target (renorm): %.6e kg/s\n", total_tgt_renorm)
    @printf("    Renorm factor:   %.6f  (%.2f%% shift per cell)\n",
            renorm_factor, (renorm_factor - 1) * 100)

    # --- Diagnostic 2: Per-cell value distribution ---
    all_norenorm = Float64[]
    all_renorm = Float64[]
    for p in 1:6
        append!(all_norenorm, vec(Float64.(panels_norenorm[p])))
        append!(all_renorm, vec(Float64.(panels_renorm[p])))
    end

    n_zero_nr = count(==(0.0), all_norenorm)
    n_zero_r = count(==(0.0), all_renorm)
    n_total = length(all_norenorm)
    nz_norenorm = filter(!=(0.0), all_norenorm)
    nz_renorm = filter(!=(0.0), all_renorm)

    println("\n  --- CS grid value distribution ---")
    @printf("    Total cells: %d, non-zero (raw): %d, non-zero (renorm): %d\n",
            n_total, length(nz_norenorm), length(nz_renorm))
    @printf("    Zero cells: %d (%.2f%%)\n", n_zero_nr, 100 * n_zero_nr / n_total)

    if !isempty(nz_renorm)
        @printf("    Renormed: min=%.4e, p5=%.4e, p50=%.4e, p95=%.4e, max=%.4e\n",
                minimum(nz_renorm),
                quantile(nz_renorm, 0.05),
                quantile(nz_renorm, 0.50),
                quantile(nz_renorm, 0.95),
                maximum(nz_renorm))
    end

    # --- Diagnostic 3: Per-cell nearest-neighbor vs conservative average ---
    println("\n  --- Per-cell accuracy: nearest-neighbor vs conservative ---")
    println("  Computing conservative averages (sampling up to $max_cells cells)...")

    half_dx = 90.0 / Nc  # approximate CS cell half-width in degrees
    Δlon_s = abs(lons[2] - lons[1])

    # Sample cells (all non-zero, up to max_cells)
    nz_indices = Tuple{Int, Int, Int}[]
    for p in 1:6, j in 1:Nc, i in 1:Nc
        panels_norenorm[p][i, j] != 0.0 && push!(nz_indices, (p, i, j))
    end

    if length(nz_indices) > max_cells
        # Random subsample preserving spatial distribution
        step = length(nz_indices) ÷ max_cells
        nz_indices = nz_indices[1:step:end]
    end
    @printf("  Evaluating %d non-zero cells...\n", length(nz_indices))

    rel_diffs = Float64[]
    abs_diffs = Float64[]

    # Normalize lons to match source convention
    src_lon_min = minimum(lons)

    for (p, i, j) in nz_indices
        nn_val = Float64(panels_norenorm[p][i, j])  # our nearest-neighbor value
        nn_val == 0.0 && continue

        cs_lon = Float64(grid.λᶜ[p][i, j])
        cs_lat = Float64(grid.φᶜ[p][i, j])

        # Normalize CS lon to source convention
        if src_lon_min >= 0  # source is [0, 360]
            cs_lon = mod(cs_lon, 360.0)
        else  # source is [-180, 180]
            cs_lon = _to_m180_180(cs_lon)
        end

        cons_val = conservative_average(raw, lons, lats, src_areas,
                                         cs_lon, cs_lat, half_dx)

        if abs(cons_val) > 1e-30
            push!(rel_diffs, (nn_val - cons_val) / cons_val)
            push!(abs_diffs, nn_val - cons_val)
        end
    end

    if !isempty(rel_diffs)
        print_percentiles("Relative difference (nn - conservative) / conservative", rel_diffs)

        # Summary statistics
        n_large = count(x -> abs(x) > 0.10, rel_diffs)
        n_vlarge = count(x -> abs(x) > 0.50, rel_diffs)
        n_exact = count(x -> abs(x) < 1e-10, rel_diffs)
        @printf("\n  Summary:\n")
        @printf("    Exact match (<1e-10):  %d / %d (%.1f%%)\n",
                n_exact, length(rel_diffs), 100 * n_exact / length(rel_diffs))
        @printf("    |error| > 10%%:        %d / %d (%.1f%%)\n",
                n_large, length(rel_diffs), 100 * n_large / length(rel_diffs))
        @printf("    |error| > 50%%:        %d / %d (%.1f%%)\n",
                n_vlarge, length(rel_diffs), 100 * n_vlarge / length(rel_diffs))
        @printf("    RMS relative error:    %.4f%%\n", 100 * sqrt(mean(rel_diffs .^ 2)))
        @printf("    Mean relative error:   %+.4f%%\n", 100 * mean(rel_diffs))
        @printf("    Median relative error: %+.4f%%\n", 100 * median(rel_diffs))
    else
        println("  No non-zero cells to compare.")
    end

    # --- Diagnostic 4: Per-panel breakdown ---
    println("\n  --- Per-panel breakdown ---")
    @printf("    %-7s %10s %10s %10s %10s %8s\n",
            "Panel", "mean", "min", "max", "std", "zeros")
    for p in 1:6
        vals = vec(Float64.(panels_renorm[p]))
        nz = filter(!=(0.0), vals)
        @printf("    P%-6d %10.4e %10.4e %10.4e %10.4e %8d\n",
                p,
                isempty(nz) ? 0.0 : mean(nz),
                isempty(nz) ? 0.0 : minimum(nz),
                isempty(nz) ? 0.0 : maximum(nz),
                isempty(nz) ? 0.0 : std(nz),
                count(==(0.0), vals))
    end
end

# =====================================================================
# Test: Synthetic gradient (step function at equator)
# =====================================================================

function test_gradient(; Nc=90, coord_file=nothing)
    println("\n" * "=" ^ 70)
    println("  TEST: Step-function gradient → C$Nc")
    println("=" ^ 70)

    FT = Float64
    grid = build_grid(; Nc, FT, coord_file)
    R = FT(grid.radius)

    # Source: 1° grid with step function at equator (NH=1, SH=0)
    lons = FT.(collect(0.5:1.0:359.5))
    lats = FT.(collect(-89.5:1.0:89.5))
    flux = zeros(FT, length(lons), length(lats))
    for j in eachindex(lats), i in eachindex(lons)
        flux[i, j] = lats[j] >= 0 ? 1.0 : 0.0
    end

    cs_map = Sources.build_conservative_cs_map(lons, lats, grid)
    panels = Sources.regrid_latlon_to_cs(flux, lons, lats, grid; cs_map, renormalize=false)

    # Check cells near equator: should be 0 or 1, not intermediate
    src_areas = Sources.latlon_cell_areas(lons, lats, R)
    half_dx = 90.0 / Nc

    n_boundary = 0
    nn_errors = Float64[]
    for p in 1:6, j in 1:Nc, i in 1:Nc
        cs_lat = Float64(grid.φᶜ[p][i, j])
        cs_lon = Float64(grid.λᶜ[p][i, j])
        nn_val = Float64(panels[p][i, j])

        # Only check cells near the equator (±2°)
        abs(cs_lat) > 2.0 && continue

        cs_lon_use = mod(cs_lon, 360.0)
        cons_val = conservative_average(flux, lons, lats, src_areas,
                                         cs_lon_use, cs_lat, half_dx)
        n_boundary += 1

        # NN gives 0 or 1; conservative gives fractional overlap
        if abs(cons_val) > 1e-10 && abs(cons_val) < 1.0 - 1e-10
            push!(nn_errors, nn_val - cons_val)
        end
    end

    @printf("  Cells within ±2° of equator: %d\n", n_boundary)
    @printf("  Cells where conservative gives fractional value: %d\n", length(nn_errors))
    if !isempty(nn_errors)
        @printf("  NN vs conservative difference at boundary:\n")
        @printf("    min = %+.6f, max = %+.6f, mean = %+.6f\n",
                minimum(nn_errors), maximum(nn_errors), mean(nn_errors))
        @printf("    (NN assigns 0 or 1; conservative gives partial overlap)\n")
    end
end

# =====================================================================
# Compare against xESMF reference
# =====================================================================

function compare_with_xesmf(source_path, reference_path; Nc=180, var_name=nothing, coord_file=nothing)
    println("\n" * "=" ^ 70)
    println("  Compare Julia vs xESMF regridding")
    println("=" ^ 70)

    FT = Float64
    grid = build_grid(; Nc, FT, coord_file)

    ds = NCDataset(source_path)
    all_keys = collect(keys(ds))
    lon_key = nothing; lat_key = nothing
    for k in all_keys
        lk = lowercase(k)
        if lon_key === nothing && lk in ("longitude", "lon"); lon_key = k
        elseif lat_key === nothing && lk in ("latitude", "lat"); lat_key = k; end
    end
    lon_key === nothing && error("No longitude coordinate found")
    lat_key === nothing && error("No latitude coordinate found")

    lons = FT.(Array(ds[lon_key]))
    lats = FT.(Array(ds[lat_key]))

    if var_name === nothing
        skip = Set(keys(ds.dim)) ∪ Set(["time", "time_bnds", lon_key, lat_key])
        candidates = [k for k in all_keys if k ∉ skip && ndims(ds[k]) >= 2]
        var_name = if length(candidates) == 1
            candidates[1]
        else
            idx = findfirst(c -> any(contains(lowercase(c), kw) for kw in
                            ["emission", "flux", "fire", "co2"]), candidates)
            idx !== nothing ? candidates[idx] : candidates[1]
        end
    end

    data = Array(ds[var_name])
    raw = ndims(data) == 3 ? FT.(replace(data[:, :, 1], missing => zero(FT))) :
                             FT.(replace(data, missing => zero(FT)))
    close(ds)

    if length(lats) > 1 && lats[1] > lats[end]
        lats = reverse(lats)
        raw = raw[:, end:-1:1]
    end

    println("  Source: $(length(lons))×$(length(lats)), var=$var_name")

    cs_map = Sources.build_conservative_cs_map(lons, lats, grid)
    julia_panels = Sources.regrid_latlon_to_cs(raw, lons, lats, grid; cs_map)

    ref = NCDataset(reference_path)
    ref_Nc = Int(ref.attrib["Nc"])
    ref_Nc == Nc || @warn "Nc mismatch: Julia=$Nc, reference=$ref_Nc"

    println("\n  --- Per-panel comparison ---")
    @printf("  %-7s %12s %12s %12s %12s\n", "Panel", "RMS", "rel_RMS%", "mass_diff%", "corr")

    all_rel_diffs = Float64[]
    for p in 1:6
        xesmf_panel = FT.(Array(ref["flux_panel$p"]))
        julia_panel = julia_panels[p]

        # Per-cell relative difference (where xesmf is non-zero)
        for j in 1:Nc, i in 1:Nc
            xv = xesmf_panel[i, j]
            jv = julia_panel[i, j]
            if abs(xv) > 1e-30
                push!(all_rel_diffs, (jv - xv) / xv)
            end
        end

        diff = julia_panel .- xesmf_panel
        rms = sqrt(mean(diff .^ 2))
        nz_mean = mean(abs.(filter(!=(0.0), vec(xesmf_panel))))
        rel_rms = nz_mean > 0 ? rms / nz_mean : 0.0

        julia_mass = sum(julia_panel .* grid.Aᶜ[p])
        xesmf_mass = sum(xesmf_panel .* grid.Aᶜ[p])
        mass_diff = abs(xesmf_mass) > 1e-30 ?
            (julia_mass - xesmf_mass) / abs(xesmf_mass) * 100 : 0.0

        # Spatial correlation
        jf = vec(julia_panel); xf = vec(xesmf_panel)
        jm = mean(jf); xm = mean(xf)
        num = sum((jf .- jm) .* (xf .- xm))
        den = sqrt(sum((jf .- jm).^2) * sum((xf .- xm).^2))
        corr = den > 0 ? num / den : 1.0

        @printf("  P%-6d %12.4e %12.4f %+12.4f %12.6f\n",
                p, rms, rel_rms * 100, mass_diff, corr)
    end
    close(ref)

    if !isempty(all_rel_diffs)
        println()
        print_percentiles("Per-cell (Julia - xESMF) / xESMF", all_rel_diffs)
    end
end

using Statistics: std

# =====================================================================
# CLI
# =====================================================================

function main()
    coord_file = let idx = findfirst(==("--coord-file"), ARGS)
        idx !== nothing && idx < length(ARGS) ? ARGS[idx + 1] : nothing
    end

    Nc = let idx = findfirst(==("--Nc"), ARGS)
        idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx + 1]) : 180
    end

    if "--test-uniform" in ARGS
        test_uniform(; Nc, coord_file)
        return
    end

    if "--test-gradient" in ARGS
        test_gradient(; Nc, coord_file)
        return
    end

    test_idx = findfirst(==("--test-source"), ARGS)
    if test_idx !== nothing && test_idx < length(ARGS)
        test_source(ARGS[test_idx + 1]; Nc, coord_file)
        return
    end

    source_idx = findfirst(==("--source"), ARGS)
    ref_idx = findfirst(==("--reference"), ARGS)
    if source_idx !== nothing && ref_idx !== nothing
        compare_with_xesmf(ARGS[source_idx + 1], ARGS[ref_idx + 1]; Nc, coord_file)
        return
    end

    println("Usage:")
    println("  julia --project=. $(@__FILE__) --test-uniform [--Nc 90]")
    println("  julia --project=. $(@__FILE__) --test-gradient [--Nc 90]")
    println("  julia --project=. $(@__FILE__) --test-source <emission.nc> [--Nc 180] [--coord-file <geos.nc>]")
    println("  julia --project=. $(@__FILE__) --source <file> --reference <xesmf_ref.nc> [--Nc 180]")
end

main()
