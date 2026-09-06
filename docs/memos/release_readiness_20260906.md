# Release preparation and documentation review — 2026-09-06

This review follows source `567bc96b` and prepares version 0.4.0. The version
is unreleased; creating the release tag and registration remain separate steps.

## Apple Metal evidence

The user supplied complete successful six- and 32-tracer smoke-run output from
an Apple M5 Pro with 20 GPU cores. Both cold and warmed samples passed.

- Source: `567bc96b`, distributed in `metal-test-20260906.tar.gz`.
- Environment: macOS 26.5.2, Julia 1.12.6, Metal.jl 1.10.3,
  KernelAbstractions 0.9.42, GPUArrays 11.5.14, GPUCompiler 2.5.3.
- Input: `era5_c90_l66_f32_20181201_2h.bin`, 259,642,112 bytes.
  SHA-256: `74476d4c976db6faa45904a1c64d5953081909183ddc6ea217ef3e22f810bb37`.
- Two hourly windows, ten transport steps per window; six C90 faces and all
  66 levels, Float32, PPM, exact TM5 Dkg diffusion, full-column collaborative
  TM5 convection (`lmax_conv=0`, `n_merge=1`), column means and total-mass output.
- Runner commands: `julia --project=env --threads=4 run.jl metal 6` and
  `julia --project=env --threads=4 run.jl metal 32` in the extracted bundle.
- Input probe: panel mass shape `(96,96,66)`, Float32,
  `mass[4,4,33]=2.829608e12` kg and `ps[1,1]=100869.164` Pa.

| Tracers | Cold elapsed (s) | Warmed elapsed (s) | Maximum relative mass drift | Maximum column relative L2 difference from CUDA |
|---|---|---|---|---|
| 6 | 27.605140334 | 2.899053042 | `5.4686988125563676e-8` | `8.880751930515834e-8` |
| 32 | 32.172329458 | 8.279450167 | `5.629938873080514e-8` | `8.907857326903221e-8` |

The runner disables scalar indexing and checks Float32 `MtlArray` state,
finite output, snapshots at hours 0 and 2, `completed_snapshots=2`, relative
mass drift below `1e-6`, and column relative L2 error below `5e-5` against
bundled L40S CUDA references. Each invocation ended with
`METAL_SMOKE_WORKLOAD_PASSED`. The reported errors were identical in the cold
and warmed samples for each tracer count.

These are single warmed end-to-end samples including setup and output, not
repeated benchmark medians. The separate full-day L40S/V100 benchmark uses a
different duration and timing protocol. The two-hour input retains the original
archive payload, including its three-hourly held TM5 convection. This checks
backend execution and numerical agreement, not independent meteorological
validation. Metal transport uses Float32; output accumulates in Float64 host
slabs. Metal adjoints and broader operator coverage remain unverified.

## Documentation findings addressed

- Replace the blanket README instability warning with version-pinning guidance,
  verified workflow coverage and explicit remaining limitations.
- Add a beginner's adjoint/inversion explanation and an executed footprint
  tutorial, including units, physical indices, fixed meteorology, derivative
  interpretation and a centered finite-difference check.
- Distinguish synthetic inversion CLI configuration from forward-run TOML and
  programmatic real-data inversion assembly. Correct unsupported checkpoint
  promises and the meaning of a cell/layer observation.
- Use an explicit local GPU environment in installation instructions.
- Correct the spectral preprocessing example's vertical keys and document the
  global column-integrated Poisson correction in the native GEOS path.
- Point data acquisition at the maintained downloader and distinguish native
  bundles from the older split spectral/thermodynamic input layout.
- Document partial streamed output, completion markers, spatial precision,
  conservative tracer-storage units and the cubed-sphere `nf` dimension.
- Add `SnapshotFrame` and `SnapshotWriteOptions` to the curated API page and
  distinguish hosted CPU CI from separate GPU verification.

## Additional checks

- Aqua passes all ten package-health assertions. JET reports 154 hot-path
  findings against the existing baseline of 154; this gate does not assert
  that every package function has perfect inference.
- The strict documentation build passes with executed Literate examples and
  export coverage enabled. Local destination checks on the rendered landing,
  installation, adjoint lesson/tutorial and validation pages find no broken
  navigation links.
- Configuration preflight now rejects scalar/array `input.staging` values
  before binary inspection, including configs without a tracer table. The
  focused configuration/CLI tests pass, with 25 additional assertions.
- Explicit CPU launches of the split Dkg kernels pass 798 assertions using
  CUDA-shaped tiles, Float32/Float64, partial tiles, signed tracers and an
  independent mass-space solve. CPU PPM layout coverage adds 480 assertions
  for all sweep axes, partial tiles and constant signed mixing ratios.
- The new footprint example returns `J=3.6099183423320106e-5`; its first-step
  adjoint derivative is `0.001992032000102811`, versus centered finite
  difference `0.0019920320001028074`.
- The existing synthetic inversion lowers cost from `0.03125` to
  `0.0022119987808685566`. The documented spectral TOML block resolves with
  Float32, levels 1–137 and minimum layer thickness 1000 Pa. The ARCO download
  dry run plans ten files for the documented one-day request.
- Two real C90/L66 daily binaries each expose 24 hourly windows. Loading their
  first windows succeeds; the multi-file range guard accepts `(1,nothing)`
  and `(1,24)` and rejects `(1,12)` and `(2,24)`. This is a range/load check,
  not a new 48-hour simulation.
- Nine old 48-hour configs refer to missing historical binaries and obsolete
  runtime sections. They are preserved under `config/runs/likely_legacy/`;
  their original forcing was not available for an acceptance run.

Existing detailed benchmark archives remain as provenance. This review adds
curated results rather than another set of full test logs or home-directory
paths. Hosted checks for the final commit are tracked by
[PR #6](https://github.com/RemoteSensingTools/AtmosTransport.jl/pull/6).
