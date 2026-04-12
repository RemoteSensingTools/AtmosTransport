#!/usr/bin/env julia
# =====================================================================
# Compare CATRINE emission integrals: Julia preprocessed vs GEOS-Chem
#
# Loads preprocessed emission binaries and GEOS-Chem reference output,
# computes global integrals using GEOS-Chem GMAO cell areas, and
# compares per-panel and global flux totals.
#
# Reference: GEOSChem.CATRINE_inst.20211201_0300z.nc4 contains
# emission fields + Met_AREAM2 + corner_lons/corner_lats.
#
# Usage:
#   julia --project=. scripts/validation/compare_emissions_geoschem.jl
# =====================================================================

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))

using AtmosTransport
using AtmosTransport.Sources
using AtmosTransport.Grids
using NCDatasets
using Printf
using Statistics: mean
using Unitful: @u_str, ustrip

# =====================================================================
# Paths
# =====================================================================

const GEOSCHEM_FILE = expanduser(
    "~/data/AtmosTransport/catrine/geos-chem/GEOSChem.CATRINE_inst.20211201_0300z.nc4")
const BIN_DIR = expanduser("~/data/AtmosTransport/catrine/preprocessed_c180")

const FT = Float32
const Nc = 180

# Map: (Julia binary, GEOS-Chem variable, species, display unit, conversion from kg/s)
const SOURCES = [
    (bin  = "edgar_sf6_cs_c180_float32.bin",
     gcvar = "EmisSF6",
     species = :sf6,
     unit = "kt/yr",
     convert = v -> v * Sources.SECONDS_PER_YEAR / 1e6),

    (bin  = "gridfed_fossil_co2_cs_c180_float32.bin",
     gcvar = "Emis_FossilCO2_Total",
     species = :fossil_co2,
     unit = "PgCO2/yr",
     convert = v -> v * Sources.SECONDS_PER_YEAR / 1e12),

    (bin  = "zhang_rn222_cs_c180_float32.bin",
     gcvar = "EmisRn_Soil",
     species = :rn222,
     unit = "kg/yr",
     convert = v -> v * Sources.SECONDS_PER_YEAR),

    (bin  = "lmdz_co2_cs_c180_float32.bin",
     gcvar = "EmisCO2_Total",
     species = :co2,
     unit = "PgCO2/yr",
     convert = v -> v * Sources.SECONDS_PER_YEAR / 1e12),
]

# =====================================================================
# Load GEOS-Chem reference
# =====================================================================

println("=" ^ 70)
println("  Compare Julia preprocessed emissions vs GEOS-Chem reference")
println("=" ^ 70)

isfile(GEOSCHEM_FILE) || error("GEOS-Chem file not found: $GEOSCHEM_FILE")

ds = NCDataset(GEOSCHEM_FILE)

# NCDatasets reads (time,nf,Ydim,Xdim) as Julia (Xdim,Ydim,nf,time)
# lons/lats have no time dim: (nf,Ydim,Xdim) → Julia (Xdim,Ydim,nf)
gc_area = Float64.(ds["Met_AREAM2"][:, :, :, 1])  # (Xdim, Ydim, nf)
gc_lons = Float64.(ds["lons"][:, :, :])            # (Xdim, Ydim, nf)
gc_lats = Float64.(ds["lats"][:, :, :])

total_gc_area = sum(gc_area)
earth_area = 4π * 6.371e6^2
@printf("  GMAO total area: %.6e m² (Earth: %.6e, err: %.4f%%)\n",
        total_gc_area, earth_area, abs(total_gc_area - earth_area) / earth_area * 100)

# Check GEOS-Chem panel ordering (index as [:,:,p])
println("\n  GEOS-Chem panel centers:")
for p in 1:6
    mlon = mean(gc_lons[:, :, p])
    mlat = mean(gc_lats[:, :, p])
    @printf("    Panel %d: mean_lon=%.1f°, mean_lat=%.1f°\n", p, mlon, mlat)
end

# =====================================================================
# Compare each source
# =====================================================================

println("\n" * "-" ^ 70)

function run_comparison(ds, gc_area)
    n_pass = 0
    n_fail = 0
    n_info = 0

for src in SOURCES
    bin_path = joinpath(BIN_DIR, src.bin)
    println("\n  $(uppercase(string(src.species))): $(src.bin)")

    if !isfile(bin_path)
        println("    SKIP — binary not found")
        continue
    end

    # NCDatasets: (Xdim, Ydim, nf, time) — take time=1
    gc_flux = Float64.(ds[src.gcvar][:, :, :, 1])  # (Xdim, Ydim, nf)

    # GEOS-Chem global integral
    gc_total_kgs = sum(gc_flux .* gc_area)
    gc_display = src.convert(gc_total_kgs)

    # Load Julia binary
    panels_vec, time_hours, hdr = Sources.load_cs_emission_binary(bin_path, FT)
    panels = panels_vec[1]  # First timestep

    # Julia global integral using GMAO areas
    # Note: Julia binary panels are (Nc, Nc), GEOS-Chem is (nf, Ydim, Xdim)
    # Need to figure out if panel ordering matches
    jl_total_kgs = zero(Float64)
    for p in 1:6
        # Julia panels[p] is (Nc, Nc) — i=x, j=y in Julia column-major
        # GEOS-Chem gc_area[p, :, :] is (Ydim, Xdim) — row-major in Python, but
        # NCDatasets reads as (nf, Ydim, Xdim) which in Julia is (Xdim, Ydim, nf)
        # so gc_area[:, :, p] is (Xdim, Ydim) = (Nc, Nc) matching panels[p]
        jl_total_kgs += sum(Float64.(panels[p]) .* gc_area[:, :, p])
    end
    jl_display = src.convert(jl_total_kgs)

    # Per-panel comparison
    println("    Panel-by-panel (GMAO areas):")
    max_panel_diff = 0.0
    for p in 1:6
        gc_panel_kgs = sum(gc_flux[:, :, p] .* gc_area[:, :, p])
        jl_panel_kgs = sum(Float64.(panels[p]) .* gc_area[:, :, p])
        gc_p_display = src.convert(gc_panel_kgs)
        jl_p_display = src.convert(jl_panel_kgs)
        rel_diff = abs(gc_panel_kgs) > 1e-30 ?
            abs(gc_panel_kgs - jl_panel_kgs) / abs(gc_panel_kgs) * 100 : 0.0
        max_panel_diff = max(max_panel_diff, rel_diff)
        @printf("      P%d: GC=%.4f, Julia=%.4f %s (diff=%.2f%%)\n",
                p, gc_p_display, jl_p_display, src.unit, rel_diff)
    end

    # Global comparison
    rel_err = abs(gc_total_kgs) > 1e-30 ?
        abs(gc_total_kgs - jl_total_kgs) / abs(gc_total_kgs) * 100 : 0.0

    @printf("    Global: GC=%.4f, Julia=%.4f %s (diff=%.2f%%)\n",
            gc_display, jl_display, src.unit, rel_err)

    # Spatial pattern comparison (correlation + RMS)
    gc_flat = vcat([vec(gc_flux[:, :, p]) for p in 1:6]...)
    jl_flat = vcat([vec(Float64.(panels[p])) for p in 1:6]...)
    # Correlation
    gc_m = mean(gc_flat)
    jl_m = mean(jl_flat)
    cov_gj = mean((gc_flat .- gc_m) .* (jl_flat .- jl_m))
    std_g = sqrt(mean((gc_flat .- gc_m) .^ 2))
    std_j = sqrt(mean((jl_flat .- jl_m) .^ 2))
    corr = std_g > 0 && std_j > 0 ? cov_gj / (std_g * std_j) : NaN
    # RMS difference
    rms = sqrt(mean((gc_flat .- jl_flat) .^ 2))
    rel_rms = mean(abs.(gc_flat)) > 0 ? rms / mean(abs.(gc_flat)) * 100 : 0.0
    @printf("    Pattern: correlation=%.6f, rel_RMS=%.2f%%\n", corr, rel_rms)

    # Pass/fail (tolerances depend on source — CO2 total is net flux so large relative diffs are OK)
    if src.species === :co2
        @printf("    → INFO (CO2 total is net flux, large relative diffs expected)\n")
        n_info += 1
    else
        tol = src.species === :rn222 ? 5.0 : 2.0  # % tolerance
        if rel_err < tol && corr > 0.99
            println("    → PASS")
            n_pass += 1
        else
            println("    → FAIL (expected <$(tol)% global diff and corr>0.99)")
            n_fail += 1
        end
    end
end

    println("\n" * "-" ^ 70)
    @printf("  Results: %d PASS, %d FAIL, %d INFO\n", n_pass, n_fail, n_info)

    if n_fail > 0
        println("\n  FAILURES — preprocessed emissions don't match GEOS-Chem reference")
        return 1
    else
        println("\n  All emission sources consistent with GEOS-Chem reference.")
        return 0
    end
end  # run_comparison

rc = run_comparison(ds, gc_area)
close(ds)

# Unitful verification
println("\n  Unitful sanity checks:")
@assert Sources.SECONDS_PER_YEAR ≈ ustrip(u"s", 1u"yr")
@assert Sources.KGC_TO_KGCO2 ≈ 44.01 / 12.011
println("  Conversion constants verified ✓")

rc != 0 && exit(rc)
