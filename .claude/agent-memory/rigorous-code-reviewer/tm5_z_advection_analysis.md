---
name: TM5 Z-advection CFL analysis
description: Deep analysis of TM5 dynamw_1d — why TM5 never allows Z-CFL >= 1, prognostic vs diagnostic slopes, and implications for Julia subcycling
type: reference
---

## TM5 Z-CFL Handling

TM5 does NOT subcycle Z-advection. Instead, `dynamwm` (advectm_cfl.F90:2802) sets `cfl_ok = .false.` when any Z gamma >= 1. The caller `Check_CFL` (line 278-292) then halves the ENTIRE timestep ndyn and rescales ALL mass fluxes (am, bm, cm). This means X and Y also use smaller fluxes.

Key code locations:
- `advectz.F90:409-498` — `dynamw_1d`: column Z-advection, NO subcycling loop
- `advectm_cfl.F90:2705-2843` — `dynamwm`: Z mass pilot, REJECTS when gamma >= 1
- `advectm_cfl.F90:154-302` — `Check_CFL`: global timestep reduction loop
- `advectm_cfl.F90:1354-1434` — `advectx_get_nloop`: X DOES subcycle (Z does NOT)

## Flux Formula Math

`f = gamma * (rm + (1 - gamma) * rzm)` where gamma = cm/m_donor.
- gamma < 1: slope correction `(1-gamma)*rzm` is positive coefficient, reduces numerical diffusion (standard van Leer).
- gamma = 1: slope correction vanishes, f = rm (entire cell content evacuated).
- gamma > 1: `(1-gamma) < 0`, slope correction REVERSES sign. Becomes antidiffusive and unstable.
- gamma > 1 with rzm = 0 (uniform): f = gamma*rm, overshoots but rm/m ratio preserved IF exact arithmetic.

## Prognostic vs Diagnostic Slopes

TM5 slopes (rxm, rym, rzm) are PROGNOSTIC — evolved via update formula (advectz.F90:481-485), never recomputed from scratch. They carry subgrid memory across timesteps. All three cross-slopes transported through orthogonal advection passes (rzm through X/Y, etc).

Julia uses DIAGNOSTIC slopes (minmod limiter, recomputed each step). With diagnostic slopes, gamma > 1 produces:
1. Sign-reversed flux for nonzero slopes → instability
2. For uniform fields (rzm=0), ratio preserved in exact arithmetic but intermediate rm can be negative

**Conclusion:** Neither prognostic nor diagnostic slopes make gamma > 1 safe. TM5 avoids it entirely.

## Why TM5 Never Hits Z-CFL >= 1

TM5 standard: 3x2° with 25-34 levels (thick layers, small vertical CFL).
ERA5 L137: layers 130-137 are 2-4 hPa thick → Z-CFL reaches 4-19.
TM5 was never designed for thin vertical levels.

## Subcycling Assessment

Current Julia approach (divide cm by n_sub, keep gamma < 0.95) is correct and necessary.
Overhead: n_sub GPU kernel launches per Z-pass, ~50us each.
Numerical diffusion: bounded, comparable to single step at effective CFL = 0.95.

Alternative: PPM remap (already in vertical_remap.jl) handles CFL > 1 natively by integrating parabola across multiple cells. Could replace subcycled slopes for Z.
