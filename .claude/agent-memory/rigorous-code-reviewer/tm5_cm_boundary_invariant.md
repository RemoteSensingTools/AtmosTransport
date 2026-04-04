---
name: TM5 vertical mass flux boundary invariant
description: TM5 enforces cm=0 at surface and TOA via structural construction in dynam0 + fatal assertions in dynamw/dynamwm
type: reference
---

TM5 guarantees `cm(:,:,0) = 0` (surface) and `cm(:,:,lmr) = 0` (TOA) through two mechanisms:

1. **Structural construction in `dynam0`** (advect_tools.F90):
   - Line 717: `cm(0:im+1,0:jm+1,0:lmr+1) = zero` zeros entire array
   - Lines 783-789: computation loop `l=1,lmr-1` never touches boundaries
   - `sd` array declared as `(im,jm,lm-1)` — physically cannot hold boundary values

2. **Fatal assertions** before Z-advection:
   - `dynamw` (advectz.F90:320-323): `any(cm(:,:,0)/=0) .or. any(cm(:,:,lmr)/=0)` -> `stop`
   - `dynamwm` (advectm_cfl.F90:2751-2760): separate checks, `status=1; return`

**Why:** cm at boundary = vertical mass flux through ground (surface) or space (TOA). Both are physically zero. Non-zero values cause unphysical tracer accumulation (e.g., CO2 419->781 ppm in 6 hours).

**How to apply:** When loading or computing cm in Julia, always zero `cm[:,:,1]` and `cm[:,:,end]` (Julia 1-based). The Julia code must replicate TM5's structural guarantee since the met data source may not provide clean boundaries.
