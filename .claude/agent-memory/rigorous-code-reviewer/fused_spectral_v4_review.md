---
name: Fused spectral v4 binary preprocessor review
description: Review of preprocess_spectral_v4_binary.jl - faithful copy from two reference scripts with Float64-only improvement
type: reference
---

## Script: `scripts/preprocessing/preprocess_spectral_v4_binary.jl` (1128 lines)

Combines `preprocess_spectral_massflux.jl` (spectral transforms) and `preprocess_era5_daily.jl`
(level merging + v4 binary writing) into a single GRIB-to-binary pipeline.

### Key facts
- All spectral math (Legendre, vod2uv, SHT, mass fluxes) copied verbatim from reference
- Deliberate improvement: all computation in Float64 (reference could use Float32 when FT_STR="Float32")
- Binary layout: m|am|bm|cm|ps|dam|dbm|dm per window, all optional fields zero
- Uses `recompute_cm_from_divergence!` with B-correction (not `correct_cm_residual!`)
- Thread safety: over-allocated per-tid buffers (nt_max = max(nt, 2*nthreads) + 4)
- Cross-day delta: reads next day's hour 0 spectral data; falls back to zero deltas

### Reviewed 2026-04-03
- Verdict: APPROVE
- All functions line-by-line identical to references (except Float64 hardcoding)
- Binary layout matches `MassFluxBinaryReader` v4 offset computation
- Minor: explicit cm boundary zeroing after B-correction is redundant but harmless
