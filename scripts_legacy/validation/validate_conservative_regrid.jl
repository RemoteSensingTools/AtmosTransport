#!/usr/bin/env julia
#
# Validate conservative LL→CS regridding with exact areas from gridspec file.
#
# Tests:
#   1. Uniform field: 1° constant 1.0 → C180, every CS cell should be ~1.0
#   2. EDGAR SF6: regrid 0.1° source, compare vs GC reference per-cell
#   3. Rn222: regrid 0.5° source, compare vs GC reference per-cell
#
# Usage:
#   julia --project=. scripts/validation/validate_conservative_regrid.jl

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids
using AtmosTransport.Grids: HybridSigmaPressure
using AtmosTransport.Parameters
using AtmosTransport.Sources
using NCDatasets
using Printf

const FT = Float32

# =====================================================================
# Build grid with GMAO coordinates
# =====================================================================

coord_file = expanduser("~/data/geosit_c180_catrine/20211201/GEOSIT.20211201.CTM_A1.C180.nc")
gridspec_file = joinpath(@__DIR__, "..", "..", "data", "grids", "cs_c180_gridspec.nc")

Nc = 180

@info "Building CubedSphereGrid C$Nc..."
params = load_parameters(FT)
pp = params.planet
vc = HybridSigmaPressure(FT[0, 0], FT[0, 1])
grid = CubedSphereGrid(CPU(); FT, Nc, vertical=vc,
                        radius=pp.radius, gravity=pp.gravity,
                        reference_pressure=pp.reference_surface_pressure)

# Load GMAO centers from coord file
ds_coord = NCDataset(coord_file)
lons_gmao = Array(ds_coord["lons"])  # (180, 180, 6)
lats_gmao = Array(ds_coord["lats"])
close(ds_coord)
for p in 1:6, j in 1:Nc, i in 1:Nc
    grid.λᶜ[p][i, j] = lons_gmao[i, j, p]
    grid.φᶜ[p][i, j] = lats_gmao[i, j, p]
end
Grids.set_coord_status!(grid, :gmao, coord_file)

# Load exact areas from gridspec
ds_gs = NCDataset(gridspec_file)
gs_areas = Array(ds_gs["areas"])  # (180, 180, 6)
close(ds_gs)
cs_areas = ntuple(p -> FT.(gs_areas[:, :, p]), 6)

@info "Grid ready. Total exact area: $(sum(sum.(cs_areas))) m²"

# =====================================================================
# Test 1: Uniform field
# =====================================================================

function test_uniform(grid, cs_areas)
    @info "\n=== TEST 1: Uniform field (1° → C180) ==="
    Nlon, Nlat = 360, 180
    lons = FT.(range(0.5, 359.5, length=Nlon))
    lats = FT.(range(-89.5, 89.5, length=Nlat))
    flux = ones(FT, Nlon, Nlat)

    cs_map = Sources.build_conservative_cs_map(lons, lats, grid; cs_areas)
    panels = Sources.regrid_latlon_to_cs(flux, lons, lats, grid; cs_map)

    # Every CS cell should be ~1.0
    all_vals = Float64[]
    for p in 1:6
        append!(all_vals, vec(Float64.(panels[p])))
    end

    min_v, max_v = extrema(all_vals)
    mean_v = sum(all_vals) / length(all_vals)
    nonzero = count(v -> v > 0, all_vals)
    total_cells = 6 * Nc * Nc

    @info "  Results:"
    @info "    Min:  $(round(min_v, digits=8))"
    @info "    Max:  $(round(max_v, digits=8))"
    @info "    Mean: $(round(mean_v, digits=8))"
    @info "    Non-zero: $nonzero / $total_cells"

    # Relative error distribution
    rel_err = abs.(all_vals .- 1.0) ./ 1.0
    p50 = sort(rel_err)[div(length(rel_err), 2)]
    p95 = sort(rel_err)[div(length(rel_err) * 95, 100)]
    p99 = sort(rel_err)[div(length(rel_err) * 99, 100)]

    @info "    Relative error percentiles:"
    @info "      50th: $(round(p50 * 100, digits=4))%"
    @info "      95th: $(round(p95 * 100, digits=4))%"
    @info "      99th: $(round(p99 * 100, digits=4))%"
    @info "      Max:  $(round(maximum(rel_err) * 100, digits=4))%"

    pass = maximum(rel_err) < 0.05  # 5% max error
    @info "  $(pass ? "PASS" : "FAIL") ✓"
    return pass
end

# =====================================================================
# Test 2/3: Compare vs GEOS-Chem emissions
# =====================================================================

function test_vs_geoschem(grid, cs_areas, source_file, gc_var, source_name;
                           conversions=nothing, time_idx=nothing)
    @info "\n=== TEST: $source_name vs GEOS-Chem ==="

    # Load source data
    ds_src = NCDataset(source_file)
    var_names = filter(n -> !(n in ("lon", "lat", "longitude", "latitude",
                                     "time", "x", "y", "crs")), keys(ds_src))
    var_name = length(var_names) == 1 ? first(var_names) : first(var_names)
    raw = Array(ds_src[var_name])
    if ndims(raw) == 3
        ti = time_idx !== nothing ? time_idx : 1
        raw = raw[:, :, ti]
        @info "  Using time index $ti"
    end

    # Get coordinates
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lon_name = first(filter(n -> haskey(ds_src, n), lon_candidates))
    lat_name = first(filter(n -> haskey(ds_src, n), lat_candidates))
    lon_src = Float64.(Array(ds_src[lon_name]))
    lat_src = Float64.(Array(ds_src[lat_name]))
    close(ds_src)

    # Ensure S→N
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        lat_src = reverse(lat_src)
        raw = raw[:, end:-1:1]
    end

    # Replace missing
    flux = FT.(replace(raw, missing => zero(FT)))

    # Apply conversions if needed
    if conversions !== nothing
        conversions(flux, lon_src, lat_src)
    end

    @info "  Source: $(size(flux)) ($(length(lon_src))×$(length(lat_src)))"
    @info "  Source range: [$(minimum(flux)), $(maximum(flux))]"

    # Regrid
    cs_map = Sources.build_conservative_cs_map(FT.(lon_src), FT.(lat_src), grid; cs_areas)
    panels = Sources.regrid_latlon_to_cs(flux, FT.(lon_src), FT.(lat_src), grid; cs_map)

    # Load GC reference
    gc_file = expanduser("~/data/AtmosTransport/catrine/geos-chem/GEOSChem.CATRINE_inst.20211201_0300z.nc4")
    ds_gc = NCDataset(gc_file)
    gc_emis = Array(ds_gc[gc_var])  # (180, 180, 6, 1) or (180, 180, 6)
    if ndims(gc_emis) == 4
        gc_emis = gc_emis[:, :, :, 1]
    end
    close(ds_gc)

    # Per-panel comparison
    @info "  Per-panel comparison:"
    @info "  Panel |  Julia total  |   GC total   | Rel diff  | Correlation"
    @info "  ------|---------------|--------------|-----------|------------"

    total_julia = 0.0
    total_gc = 0.0
    all_julia = Float64[]
    all_gc = Float64[]

    for p in 1:6
        julia_panel = Float64.(panels[p])
        gc_panel = Float64.(gc_emis[:, :, p])

        julia_mass = sum(julia_panel .* Float64.(cs_areas[p]))
        gc_mass = sum(gc_panel .* Float64.(cs_areas[p]))
        total_julia += julia_mass
        total_gc += gc_mass

        append!(all_julia, vec(julia_panel))
        append!(all_gc, vec(gc_panel))

        # Spatial correlation
        jv = vec(julia_panel)
        gv = vec(gc_panel)
        mask = (jv .> 0) .| (gv .> 0)
        corr = if sum(mask) > 10
            jm = jv[mask] .- mean(jv[mask])
            gm = gv[mask] .- mean(gv[mask])
            dot_prod = sum(jm .* gm)
            dot_prod / (sqrt(sum(jm.^2)) * sqrt(sum(gm.^2)) + 1e-30)
        else
            NaN
        end

        rel_diff = abs(julia_mass) > 1e-30 ? (julia_mass - gc_mass) / gc_mass * 100 : NaN
        @info @sprintf("    %d   | %12.6e | %12.6e | %+7.3f%%  | %.6f",
                        p, julia_mass, gc_mass, rel_diff, corr)
    end

    rel_total = abs(total_gc) > 1e-30 ? (total_julia - total_gc) / total_gc * 100 : NaN
    @info @sprintf("  Total | %12.6e | %12.6e | %+7.3f%%", total_julia, total_gc, rel_total)

    # Per-cell error distribution — split by magnitude
    mask = (all_gc .> 0)
    if sum(mask) > 0
        rel_errs = abs.(all_julia[mask] .- all_gc[mask]) ./ all_gc[mask]
        sorted = sort(rel_errs)
        n = length(sorted)
        @info "  Per-cell relative error (ALL non-zero cells: $n):"
        @info "    50th: $(round(sorted[div(n,2)] * 100, digits=2))%"
        @info "    95th: $(round(sorted[div(n*95,100)] * 100, digits=2))%"
        @info "    Max:  $(round(sorted[end] * 100, digits=2))%"

        # Only cells above 1% of max (significant emission cells)
        max_gc = maximum(all_gc)
        sig_mask = all_gc .> 0.01 * max_gc
        if sum(sig_mask) > 0
            sig_errs = abs.(all_julia[sig_mask] .- all_gc[sig_mask]) ./ all_gc[sig_mask]
            sorted_sig = sort(sig_errs)
            ns = length(sorted_sig)
            @info "  Per-cell relative error (>1% of max, $ns cells):"
            @info "    50th: $(round(sorted_sig[div(ns,2)] * 100, digits=2))%"
            @info "    95th: $(round(sorted_sig[div(ns*95,100)] * 100, digits=2))%"
            @info "    Max:  $(round(sorted_sig[end] * 100, digits=2))%"
        end

        # Absolute error: RMS relative to global mean
        abs_errs = abs.(all_julia .- all_gc)
        mean_gc = sum(abs.(all_gc)) / length(all_gc)
        rms = sqrt(sum(abs_errs.^2) / length(abs_errs))
        @info "  RMS error / mean(|GC|): $(round(rms / mean_gc * 100, digits=2))%"
    end

    pass = abs(rel_total) < 2.0  # total mass within 2%
    @info "  $(pass ? "PASS" : "FAIL")"
    return pass
end

using LinearAlgebra: dot
using Statistics: mean

# =====================================================================
# Run tests
# =====================================================================

results = Bool[]

# Test 1: Uniform
push!(results, test_uniform(grid, cs_areas))

# Test 2: EDGAR SF6 (0.1° → C180)
edgar_sf6_file = expanduser("~/data/AtmosTransport/catrine/Emissions/edgar_v8/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc")
if isfile(edgar_sf6_file)
    # EDGAR needs tonnes/cell/year → kg/m²/s conversion
    function edgar_convert!(flux, lon_src, lat_src)
        R = 6.371e6
        areas = Sources.latlon_cell_areas(FT.(lon_src), FT.(lat_src), FT(R))
        sec_per_yr = FT(365.25 * 86400)
        for j in eachindex(lat_src), i in eachindex(lon_src)
            flux[i, j] = flux[i, j] * FT(1000) / (sec_per_yr * areas[j])
        end
    end
    push!(results, test_vs_geoschem(grid, cs_areas, edgar_sf6_file, "EmisSF6", "EDGAR SF6";
                                      conversions=edgar_convert!))
else
    @warn "EDGAR SF6 file not found: $edgar_sf6_file"
end

# Test 3: Zhang Rn222 (0.5° → C180)
rn222_dir = expanduser("~/data/AtmosTransport/catrine/Emissions/ZHANG_Rn222/")
rn222_files = isdir(rn222_dir) ? filter(f -> endswith(f, ".nc"), readdir(rn222_dir, join=true)) : String[]
if !isempty(rn222_files)
    # GC CATRINE file is Dec 2021 → use time_idx=12 (December)
    push!(results, test_vs_geoschem(grid, cs_areas, first(rn222_files), "EmisRn_Soil", "Zhang Rn222";
                                      time_idx=12))
else
    @warn "Rn222 files not found"
end

# Summary
@info "\n=== SUMMARY ==="
n_pass = count(results)
n_total = length(results)
@info "  $n_pass / $n_total tests passed"
all(results) || error("Some tests failed!")
