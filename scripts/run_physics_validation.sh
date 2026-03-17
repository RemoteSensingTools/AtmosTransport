#!/bin/bash
# Run 3 flat-IC physics validation tests sequentially, then analyze
set -e
cd /home/cfranken/code/gitHub/AtmosTransportModel

echo "=== Test 1/3: Diffusion Only ==="
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_flat_v4_diff.toml 2>&1 | tail -5

echo ""
echo "=== Test 2/3: Convection Only ==="
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_flat_v4_conv.toml 2>&1 | tail -5

echo ""
echo "=== Test 3/3: Both (Full Physics) ==="
julia --threads=2 --project=. scripts/run.jl config/runs/catrine_geosit_c180_gchp_flat_v4_fullphys.toml 2>&1 | tail -5

echo ""
echo "=== Analysis ==="
julia --project=. -e '
using NCDatasets, Statistics

ref = 0.0004  # 400 ppm
tests = [
    ("advection only (baseline)", "/temp1/catrine/output/catrine_gchp_flat_v4_dry"),
    ("+ diffusion", "/temp1/catrine/output/catrine_gchp_flat_v4_diff"),
    ("+ convection", "/temp1/catrine/output/catrine_gchp_flat_v4_conv"),
    ("+ both", "/temp1/catrine/output/catrine_gchp_flat_v4_fullphys"),
]

println("Flat IC Validation (end of day 2, deviation from 400 ppm):")
println("="^90)
@printf("%-25s %12s %12s %12s %12s\n", "Test", "TOA std", "Mid std", "Sfc std", "Mass Δ%")
println("-"^90)

for (label, prefix) in tests
    f = prefix * "_20211202.nc"
    isfile(f) || (println("  $label: FILE NOT FOUND ($f)"); continue)
    ds = NCDataset(f, "r")
    co2 = Float64.(ds["co2_3d"][:,:,:,:,end])
    Nz = size(co2, 4)

    for (name, k) in [("TOA", 1), ("mid", div(Nz,2)), ("sfc", Nz)]
        lev = co2[:,:,:,k]
        v = filter(x -> x < 1e30 && x > 0, vec(lev))
        avg = mean(v)
        s = std(v)
        if name == "TOA"
            toa_s = s * 1e6
        elseif name == "mid"
            mid_s = s * 1e6
        else
            sfc_s = s * 1e6
        end
    end

    # Mass from column_mass
    cm = Float64.(ds["co2_column_mass"][:,:,:,end])
    cm_v = filter(x -> x < 1e30, vec(cm))
    total_mass = sum(cm_v)

    close(ds)

    # Also check day 1 for mass drift
    f1 = prefix * "_20211201.nc"
    if isfile(f1)
        ds1 = NCDataset(f1, "r")
        cm1 = Float64.(ds1["co2_column_mass"][:,:,:,1])
        cm1_v = filter(x -> x < 1e30, vec(cm1))
        mass0 = sum(cm1_v)
        close(ds1)
        drift = (total_mass - mass0) / mass0 * 100
    else
        drift = NaN
    end

    @printf("%-25s %10.4f ppm %10.4f ppm %10.4f ppm %10.5f%%\n",
            label, toa_s, mid_s, sfc_s, drift)
end
println("="^90)
' 2>&1

echo ""
echo "=== Done ==="
