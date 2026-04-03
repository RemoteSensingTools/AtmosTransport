# Session 1 Summary: Grid Convention + cm Precision Fix

**Commit**: `590aa8a` (2026-04-03)  
**Status**: Validated by `compare_transport_binaries.jl` diagnostic  
**Reviewer**: Codex — please verify against TM5 F90 reference

---

## Problem

The ERA5 lat-lon spectral preprocessor used a grid with cell centers at ±90° (poles), while the runtime `LatitudeLongitudeGrid` places poles at cell faces with centers at ±89.75°. This 0.25° offset caused:

1. `cos(-90°) = 0` → division by zero in am computation → am = ±∞ at poles
2. X-CFL of 1.4 × 10⁹ at pole rows in the preprocessed binary
3. Runtime pole zeroing (am=0 at j=1,Ny) killed transport at real cells
4. cm accumulated Float32 roundoff → 32,800 kg surface residual
5. Z-CFL cascade during Strang cycle → NaN

## Root Cause

`preprocess_spectral_massflux.jl` line 113 (OLD):
```julia
dlat_deg = 180.0 / (Nlat - 1)   # = 0.5° for Nlat=361 (NODE spacing)
lats = [-90.0 + dlat_deg * (j-1) for j in 1:Nlat]  # centers AT poles
```

TM5 convention (`grid_type_ll.F90:247-278`):
```fortran
lat(j) = south_deg + (j-1) * dlat_deg   ! south_deg = -90 + dlat/2
blat(0) = -90.0; blat(jm) = +90.0       ! faces clamped to ±90°
```

## Changes Made

### 1. Spectral preprocessor grid fix (`preprocess_spectral_massflux.jl`)

```julia
# NEW (TM5 convention):
dlat_deg = 180.0 / Nlat              # = 180/361 ≈ 0.4986° (CELL spacing)
south_deg = -90.0 + dlat_deg / 2     # = -89.7507°
lats = [south_deg + dlat_deg * (j-1) for j in 1:Nlat]
```

- Cell centers at ±89.75° (NOT ±90°)
- `cos(-89.75°) = 0.00435` (finite, not zero)
- Cell faces computed via `blat_deg`, clamped to ±90° (matching TM5)
- Area computation uses actual face latitudes

### 2. Spectral cm: B-coefficient + Float64 (`preprocess_spectral_massflux.jl`)

Replaced simple accumulation with TM5's `dynam0` formula:
```julia
# TM5: cm[k+1] = cm[k] - div_h + (B[k+1] - B[k]) × pit
pit = Σ div_h  (column-integrated, Float64)
acc = acc - div_h + dB[k] * pit  (Float64 accumulation)
cm[k+1] = Float32(acc)
```

- `dB[k] = diff(b_ifc)[k]` = `B[k+1] - B[k]` ← same as TM5's `bt(l+1)-bt(l+2)`
- Surface residual: 32,800 kg → 0.0005 kg (7 orders of magnitude)

### 3. v3 daily preprocessor pole handling (`preprocess_era5_daily.jl`)

- **Removed**: am zeroing at j=1,Ny (these are real cells at ±89.75°, not poles)
- **Kept**: bm zeroing at j=1,Ny+1 (pole FACES at ±90° — geometric singularity)
- **Added**: `grid_convention = "TM5"` and `spectral_half_dt_seconds` in binary header

### 4. Runtime cleanup (`physics_phases.jl`)

- **Removed**: `gpu.am[:, 1, :] .= 0` / `gpu.am[:, Ny, :] .= 0` (was zeroing real cells)
- **Removed**: Pole averaging hack (averaging rm/m at j=1,Ny after advection)
- Polar CFL now handled entirely by reduced grid (cluster_size=720 at j=1,Ny)

## Validation Results

Diagnostic: `scripts/diagnostics/compare_transport_binaries.jl --date 2021-12-01`

| Metric | OLD binary | NEW binary | Factor |
|--------|-----------|------------|--------|
| X-CFL (fine grid) | 1.4 × 10⁹ | 89.2 | 10⁷× better |
| X-CFL (TM5 reduced) | — | 5.3 | ✓ (cluster_size=720) |
| Z-CFL | 9.29 | 2.5 | 4× better |
| cm surface residual | 32,800 kg | 0.0005 kg | 10⁷× better |
| Lat convention mismatch | 0.249° | **0.000°** | Fixed |
| cm_reconstruction suspect | confirmed | **rejected** | Fixed |
| latitude_convention suspect | confirmed | **rejected** | Fixed |

## What's NOT fixed (Session 2)

- `/n_sub` division in `advection_phase!` — currently removed but flux scaling convention needs validation
- Z-subcycling (TM5 allows gamma > 1; we still subcycle)
- `compute_ll_dry_mass_evolved!` vs `compute_ll_dry_mass!` for output
- Transport speed verification (effective wind at 50°S)
- 2-day model run validation

## Files Changed

| File | Lines | Change |
|------|-------|--------|
| `scripts/preprocessing/preprocess_spectral_massflux.jl` | 111-141 | Grid convention fix |
| `scripts/preprocessing/preprocess_spectral_massflux.jl` | 553-590 | cm B-coefficient + Float64 |
| `scripts/preprocessing/preprocess_spectral_massflux.jl` | 670 | Info message fix |
| `scripts/preprocessing/preprocess_era5_daily.jl` | 679-691 | Remove am pole zeroing, keep bm |
| `scripts/preprocessing/preprocess_era5_daily.jl` | 619-620 | Header metadata |
| `src/Models/physics_phases.jl` | 331-343 | Remove runtime pole zeroing |
| `src/Models/physics_phases.jl` | 576-588 | Remove pole averaging hack |

## TM5 F90 References

- Grid definition: `deps/tm5/base/src/grid_type_ll.F90:247-278`
- cm formula (dynam0): `deps/tm5/base/src/advect_tools.F90:711-789`
- Pole handling in advecty: `deps/tm5/base/src/advecty.F90:434-442, 520-640`
- Reduced grid: `deps/tm5/base/src/redgridZoom.F90`
