# Completed experiments

Scripts from closed-loop investigations — each one corresponds to a
specific plan-era experiment that has since landed, been validated,
or been superseded. Kept for reference value in case a similar
question is asked again; not part of any active workflow.

If you're writing new visualization, start from the canonical scripts
in `scripts/visualization/` (animate_catrine_vs_geoschem.jl,
animate_gchp_v4_fullphys.jl, animate_era5_vs_geoschem.jl, etc.).

## Index

**Hybrid PE → direct cumsum PE (fix landed 2026-03-13):**
- `animate_hybrid_pe_vs_geoschem.jl`

**Mass fixer development (Invariant 11):**
- `animate_fixer_vs_nofixer.jl`

**Plan 14 vertical remap study:**
- `animate_vremap_vs_strang_vs_geoschem.jl`
- `animate_vremap_vs_geoschem_co2.jl`
- `animate_vremap_diff_vs_geoschem.jl`
- `animate_perremap_comparison.jl`

**Plan 13/14 scheme experiments:**
- `animate_linrood_vs_geoschem.jl`
- `animate_advonly_3way.jl`
- `animate_advonly_vs_ord7damp.jl`
- `animate_qspace_vs_geoschem_fast.jl`

**GCHP path validation (Plans 14-17):**
- `animate_gchp_flat_nsub.jl`
- `animate_gchp_vs_geoschem.jl`

**CATRINE variant explorations:**
- `animate_catrine_ord7damp.jl` / `animate_catrine_ord7damp_co2.jl`
- `animate_catrine_comparison.jl` / `_co2.jl` / `_natural_co2.jl`

**Other:**
- `animate_local_vs_nonlocal_pbl.jl` — PBL variant comparison.
- `animate_lmdz_emissions.jl` — LMDZ emissions visualization.
- `animate_era5_merged_vs_geoschem.jl` — ERA5 merged-level validation
  intermediate.
