# Palindrome Ordering Study — Plan 17 Commit 7 Results

**Question:** where should surface emissions `S` sit relative to vertical diffusion `V` inside the Strang palindrome? Plan 17 §4.3 Decision 7 adopts the TM5-inspired `V(dt/2) → S(dt) → V(dt/2)` arrangement; this study measures the layer-surface mass fraction under four candidate arrangements so the recommendation is backed by a concrete measurement.

**Date:** 2026-04-19
**Source of measurements:** [`test/test_ordering_study.jl`](../../../test/test_ordering_study.jl) — the test is committed alongside this document; the numbers below were produced on the wurst host, Julia 1.12.5, Float64 CPU.

---

## 1. Setup

| Parameter          | Value         | Notes                                          |
|--------------------|---------------|------------------------------------------------|
| Grid               | 4 × 3 × 10    | Nx × Ny × Nz (enough columns for a meaningful ratio diagnostic) |
| Layer thickness    | dz = 50 m     | Uniform — deliberately simple so the mixing is driven by Kz alone |
| Vertical diffusivity | Kz = 5 m²/s | `ConstantField{Float64, 3}(5.0)` — daytime-PBL-mean order of magnitude |
| Surface emission   | 1.0 kg/s per cell | Uniform `SurfaceFluxSource` at `k = Nz`         |
| Time step          | dt = 3600 s   | 1 h coarse step — the outer-dt the arrangement question actually cares about |
| Run length         | 24 steps      | 24 h simulated, matching plan doc §4.3 Decision 8 scope |
| Advection          | None (zero fluxes) | Isolates the V / S ordering question            |
| Chemistry          | None          | Ditto                                           |
| Precision          | Float64       | F32 results will differ in the last few bits   |

All four arrangements share the same initial state (zero tracer everywhere) and the same Kz + rate.

**Stability regime.** With Kz = 5, dz = 50, dt = 3600:
- Diffusivity coefficient D = Kz / dz² = 2 × 10⁻³ s⁻¹
- α = dt · D ≈ 7.2

That's well into the overdamped Backward-Euler regime; one V(dt) call pulls mass strongly toward the vertical equilibrium profile. This is the regime CATRINE monthly runs actually operate in (typical 20 min - 1 h steps against PBL Kz), so the measurement is physically relevant.

---

## 2. Arrangements tested

| Label | Inner-step sequence            | Rationale                                             |
|-------|--------------------------------|-------------------------------------------------------|
| **A** | `V(dt/2) → S(dt) → V(dt/2)`    | **Recommended** — plan 17 §4.3 Decision 7, TM5-aligned |
| B     | `S(dt) → V(dt)`                | Emissions then a single full-dt mix                   |
| C     | `V(dt) → S(dt)`                | Single full-dt mix then emissions                     |
| D     | `S(dt)` only                   | Pathological baseline — no mixing                     |

Arrangements B and C share a single V(dt) call; they differ only in whether the emission adds mass before or after the diffusion step. Arrangement A splits V into two half-steps wrapping S.

For each, the arrangement is repeated 24 times on a shared initial condition; the diagnostic is the layer-surface (k = Nz) mass divided by the total column mass, averaged over all (i, j) cells.

---

## 3. Results

| Label | Arrangement                    | Layer-surface mass fraction (24-h, Kz=5) |
|-------|--------------------------------|-----------------------------------------:|
| A     | `V(dt/2) → S(dt) → V(dt/2)`    | **0.1208** (12.1%)                        |
| B     | `S(dt) → V(dt)`                | 0.1165 (11.6%)                            |
| C     | `V(dt) → S(dt)`                | 0.1540 (15.4%)                            |
| D     | `S(dt)` only                   | 1.0000 (100%)                             |

Global mass conservation is exact for all four arrangements (Σ mass equals `rate × dt × n_steps × N_cells`).

### Physical interpretation

- **D = 100% pileup** — with no vertical mixing, every kg of emitted mass is stuck at k = Nz. This is the pathological reference and matches the plan 17 §2.3 expectation.
- **A ≈ B** — both are reasonable operational choices; they keep the layer-surface fraction around 12%, close to the plan's "well-mixed" target of 5–15%.
- **C is worst of the three mixing arrangements** — V(dt) runs BEFORE the step's new emission, so fresh mass sits at k = Nz for the duration of the step and only gets mixed during the *next* step. The 15.4% is a 1-step lag effect; over long runs it remains non-trivially larger than A or B.

### Why A ≈ B with A slightly higher

Backward-Euler V(dt) is not exactly `V(dt/2) ∘ V(dt/2)` — see [plan 16b NOTES.md §"Commit 4 refinement"](../16_VERTICAL_DIFFUSION_PLAN/NOTES.md) for the O((dt·D)²) derivation. Concretely:

- B applies `V(dt)` to the full step's emission already in the surface layer, giving maximum effective mixing per step of the just-added mass.
- A applies `V(dt/2)` to the accumulated profile, then `S(dt)` adds fresh mass at the surface, then `V(dt/2)` mixes that fresh mass once. The fresh mass gets only half-dt mixing within the step it was emitted in.

A's slightly higher surface fraction (0.121 vs 0.116) reflects this under-mixing of fresh emissions within the step. Over 24 steps the effect is cumulative but modest (~4% relative).

### Rankings

```
   D  >  C  >  A  >  B     (surface mass fraction, higher = more pileup)
  100% >15.4% >12.1% >11.6%
```

The ordering `D > C > A ≈ B` confirms plan 17 §2.3's expectation that mixing the surface layer somewhere in the inner step (anywhere but D) gives an order-of-magnitude reduction in layer-surface pileup.

---

## 4. Recommendation

**Keep the plan-17 Commit 5 default: arrangement A (`V(dt/2) → S(dt) → V(dt/2)`).**

Rationale, in priority order:

1. **Physics symmetry.** A is symmetric around the palindrome center, matching the Strang-splitting symmetry principle OPERATOR_COMPOSITION.md §3.2 adopts for the full operator palindrome (`X Y Z V(dt/2) S V(dt/2) Z Y X`). The surrounding X, Y, Z half-steps cancel their splitting error against the reverse-half X, Y, Z — the V half-steps play the analogous role on either side of S.

2. **Second-order accuracy.** Strang splitting is formally 2nd-order in dt when operators are symmetrically composed around a non-symmetric center. B and C both break the symmetry and are formally 1st-order in dt for the V-S composition. A preserves the 2nd-order property.

3. **Measurable difference is small.** The A-vs-B difference (12.1% vs 11.6%) is ~4% relative at dt = 1 h, Kz = 5 m²/s. This is small compared to the A-vs-C (~25% relative) and A-vs-D (order-of-magnitude) differences. Choosing A over B sacrifices 0.5 percentage points of "steady-state mixing efficiency" to gain the formal 2nd-order accuracy — a clear win for any run that reduces dt.

4. **Future convection C.** Plan 18 will add convection at the palindrome center; if we keep A, convection can be wrapped similarly (`V C S C V`) and remain symmetric. Arrangements B/C would not compose cleanly with an additional centered operator.

---

## 5. Limitations

- **No advection in the study.** The ordering study decouples V from X/Y/Z to isolate the question. When horizontal advection is active, the A-vs-B difference could be amplified or damped depending on PBL gradients. A follow-up study with real ERA5 fluxes would pin this down; scope for plan 19+, not needed for plan 17.
- **Single resolution.** The study uses 4×3×10 cells; results are expected to be resolution-insensitive at the per-column level (the diagnostic averages over cells), but verification at C180 × Nz=72 would strengthen the recommendation. The existing benchmark harness (Commit 8) could be extended.
- **Constant Kz.** `DerivedKzField` introduces diurnal Kz variation that interacts with dt; a stable/unstable PBL study would be richer.
- **No Monin-Obukhov similarity.** Real BL mixing near the surface is not a single Kz; the recommendation holds for any diffusion operator whose linearity is preserved.

None of these weaken the core finding (A is operationally OK, D is pathological, symmetry > asymmetry).

---

## 6. How to reproduce

```bash
julia --project=. test/test_ordering_study.jl
```

Prints the four fractions and asserts the pathological-vs-mixed ordering. The test runs in ~2 seconds on CPU.
