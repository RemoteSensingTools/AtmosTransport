# GEOS native cubed-sphere

The GEOS path takes **GEOS-IT C180** native NetCDF — the FV3
dynamical core's own cubed-sphere output — and writes a v4 transport
binary on the same cubed-sphere grid (no horizontal regrid). GEOS-FP
native C720 hourly CTM files use the same source contract, with one
file per UTC hour. It
uses the FV3 mass fluxes and pressure-fixer formula directly; this
is the highest-fidelity path for any GEOS-driven simulation.

GEOS-FP C720 support covers the native hourly CTM layout
(`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HH30.V01.nc4`, with
`HH00` accepted for legacy fixtures). Surface/convection physics from
the 0.25° GEOS-FP products can be attached by setting
`[source] physics_dir`, `include_surface = true`, and
`include_convection = true`; the day handle validates and embeds those
payloads into the same transport binary.

## Why no Poisson balance?

A spectral preprocessor needs a Poisson balance step because winds
synthesized from VO + D do **not** satisfy the continuity equation
exactly on the discrete grid; balance closes the residual.

GEOS native MFXC / MFYC are **already discrete-conservative** — the
FV3 dynamical core integrated them under its own discrete continuity.
Running a Poisson balance on top would add a small spurious
correction with no physical justification. The GEOS path therefore
**skips Poisson balance entirely** and uses the FV3 pressure-fixer
formula to derive the vertical mass flux `cm` from the horizontal
flux divergence and the surface-tendency closure.

## Required input per day

| Collection | Cadence | Variables |
|---|---|---|
| `CTM_A1` | hourly (window-averaged) | `MFXC`, `MFYC`, `DELP` |
| `CTM_I1` | hourly (instantaneous) | `PS`, `QV` |
| `A1` | hourly | `PBLH`, `USTAR`, `HFLUX`, `T2M` *(GEOS-IT native or GEOS-FP physics fallback, only if surface is enabled)* |
| `A3mstE` | 3-hourly (window-averaged) | `CMFMC` *(only if convection is enabled)* |
| `A3dyn` | 3-hourly (window-averaged) | `DTRAIN` *(only if convection is enabled)* |

GEOS-IT file-naming convention: `GEOSIT.YYYYMMDD.<COLLECTION>.C180.nc`.
GEOS-FP native CTM file-naming convention:
`GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.YYYYMMDD_HH30.V01.nc4`. The
preprocessor needs the **next day's hour-0** for the last window's
forward-flux endpoint, mirroring the spectral path; the trailing peek
at `<end+1>` is unavoidable.

## GEOS preprocessing TOML

A working GEOS-IT C180 → CS C180 preprocessing config has two TOML
files: a small **preprocessing TOML** (per-run knobs) and a separate
**source descriptor TOML** (shared with the runtime, declares the
source's invariants):

```toml
# config/preprocessing/geosit_c180_to_cs180.toml
[source]
toml     = "config/met_sources/geosit.toml"   # source descriptor (below)
root_dir = "~/data/AtmosTransport/met/geosit_c180/raw_catrine"
include_surface = true
include_convection = true

[output]
directory  = "~/data/AtmosTransport/met/geosit/C180/preprocessed/v4_dec2021"
mass_basis = "dry"                             # binary header

[grid]
type                = "cubed_sphere"
Nc                  = 180
panel_convention    = "geos_native"
regridder_cache_dir = "~/.cache/AtmosTransport/cr_regridding"

[vertical]
coefficients = "config/geos_L72_coefficients.toml"
# 72-level passthrough preserves source resolution; level merging is
# available but is usually deferred for the spectral path, not GEOS.

[numerics]
float_type     = "Float64"
dt_met_seconds = 3600.0           # CTM cadence (hourly for GEOS-IT)
```

The source descriptor (referenced via `[source] toml = …`) declares
collection mappings and FV3-specific invariants. The actual
descriptor TOML is in `config/met_sources/geosit.toml`; the keys the
preprocessor reads sit under `[preprocessing]`:

```toml
# config/met_sources/geosit.toml (extract — full file is longer)
[preprocessing]
mass_flux_dt_seconds  = 450.0     # FV3 dynamics step — see Conventions
level_orientation     = "auto"    # bottom_up | top_down | auto
collections_required  = ["CTM_A1", "CTM_I1"]
collections_optional  = ["A3mstE", "A3dyn"]   # used when convection is on
```

The preprocessing TOML's `[source]` block can override
`include_surface`, `include_convection`, and, for GEOS-FP,
`physics_dir` / `physics_layout`. GEOS-IT reads native A1/A3 files next
to CTM_A1/CTM_I1. GEOS-FP reads native C720 CTM files from `root_dir`
and physics fallback files from `physics_dir`.

## Per-window pipeline

For each of the 24 hourly windows:

1. **Read** (`src/Preprocessing/sources/geos.jl::read_window!`):
    - Open `CTM_A1` for hourly `MFXC`, `MFYC`, `DELP` (window-averaged).
    - Open `CTM_I1` for instantaneous `PS`, `QV` at hour `n` and `n+1`.
    - Apply `mass_flux_dt = 450 s` scaling: `MFXC` is accumulated
      over 450 s of FV3 dynamics, not the 3600 s window. Dividing
      `MFXC` by `mass_flux_dt` gives a per-second mass-flux rate
      (in raw GEOS units of Pa·m²/s); the GEOS-CS orchestrator
      then converts that to the binary's
      `flux_kind = :substep_mass_amount` units (kg per substep)
      before write.
    - Auto-detect level orientation
      (`detect_level_orientation`): GEOS-IT files are **bottom-up**
      (k=1 surface), the runtime expects **top-down** (k=1 TOA).
      The reader flips once at read time so all downstream code sees
      the runtime convention.

2. **Endpoint dry-mass reconstruction**
   (`endpoint_dry_mass!`):
    - From `PS_total` + `QV` + the hybrid-σ-pressure coefficients,
      reconstruct `DELP_dry`, `PS_dry`, ensuring `Σ DELP_dry = PS_dry`
      to machine precision.

3. **Native MFXC/MFYC → v4 face flux layout**
   (`geos_native_to_face_flux!`):
    - GEOS-IT MFXC/MFYC are **already on a dry basis** (per the GMAO
      product manual). No `(1 − qv)` correction is applied.
    - Re-stagger from the FV3 layout (cell-centered values that
      represent the east / north face) into the v4 layout
      (`am[i+1, j, k] = MFXC[i, j, k]`).
    - Sync west / south halos via `_propagate_cs_outflow_to_halo!`.

4. **FV3 pressure-fixer cm**
   (`compute_cs_cm_pressure_fixer!`):
    - Per column, compute `pit = Σ_k (am_inflow + bm_inflow)`.
    - Apply the FV3 formula
      `cm[k+1] = cm[k] + (C_k − ΔB[k] · pit)`
      with `cm[1] = 0` (TOA boundary). The closure
      `cm[Nz+1] = 0` (surface boundary) holds **exactly** by
      construction — no residual to redistribute.

5. **Chained mass evolution**
   (`_evolve_mass_pressure_fixer!`):
    - The pressure-fixer rule implies a per-level mass increment of
      `Δm[k] = +2 · steps · ΔB[k] · pit`. The orchestrator carries the
      evolved `m_next_pf` forward as the **next window's `m_cur`**, so
      every window's mass endpoints close to floating-point.

6. **Write-time replay gate** — same contract as the spectral path
   (`verify_write_replay_cs!`); failures abort.

7. **Cross-day chain** — the day's `final_m` is threaded as `seed_m`
   for the next day's `process_day` invocation. Cross-day continuity
   is bit-identical when both days were produced with the same
   pressure-fixer chain.

## GCHP-style convection wiring (Section C, recently shipped)

When `[source] include_convection = true`:

1. The reader opens `A3mstE` and `A3dyn` (3-hourly NetCDF). If either
   collection is missing, the open fails loudly — there is no silent
   fallback.

2. **3-hourly hold-constant binding** — every hourly window `w`
   reads from A3 record index `(w − 1) ÷ 3 + 1`. Windows 1, 2, 3 see
   the same A3 record; windows 22, 23, 24 see the 8th. The dry-basis
   correction is still per-window (see step 3).

3. **Dry-basis correction**
   (`_moist_to_dry_dtrain!`, `_moist_to_dry_cmfmc!`):
    - GMAO ships `CMFMC` and `DTRAIN` as **moist-air** mass fluxes,
      kg / m² / s. The runtime needs them on the same dry basis as
      `state.air_mass`, so the reader multiplies by `(1 − qv_face)`
      using the window-mean QV (average of `t_n` and `t_{n+1}`
      humidity).
    - `CMFMC` lives at NZ+1 interfaces; the dry factor at face `k`
      is the four-corner mean of QV at the two adjacent centers
      averaged over the two endpoints. `DTRAIN` is at centers; the
      dry factor is the simple two-endpoint mean.

4. **Binary write** — the `:cmfmc` and `:dtrain` payload sections
   land in the binary; `inspect_binary` advertises
   `cmfmc_convection = true` once they're present. The runtime's
   `CMFMCConvection` operator picks them up via the
   `ConvectionForcing` carrier.

If `include_convection = false`, the orchestrator's per-window
NamedTuple omits the `:cmfmc` / `:dtrain` keys and the writer
no-ops them (the binary's `payload_sections` does not list them and
runtime convection is automatically gated off).

## Performance notes

- A 1-day GEOS-IT C180 → CS C180 F32 preprocess is ~1 minute on a
  recent workstation; with `include_convection = true`, add ~10 s per
  day for the A3 reads.
- The pressure-fixer / chained-mass approach has **zero global Poisson
  iteration** — every step is a per-column scan, embarrassingly
  parallel. There's no diminishing return from adding cores.
- Real GEOS-IT data closes write-time replay at machine epsilon
  (`5.94e-16` F64, `3.5e-7` F32) thanks to the chained pressure-fixer
  state — no residual redistribution needed.

## What's next

- [Regridding](@ref) — how the conservative weights are built and
  cached (relevant for cross-topology paths).
- [Conventions cheat sheet](@ref) — `mass_flux_dt = 450 s`, panel
  conventions, units cheat sheet.
- *Tutorials* — once the LL+CS quickstart bundle becomes a Pkg
  LazyArtifact, a Literate tutorial will end-to-end-demo a GEOS-IT
  preprocess on a synthetic C8 fixture.
