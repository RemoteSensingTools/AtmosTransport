---
name: CS runner flux scaling bugs
description: Three bugs in run_cs_transport.jl: LinRood 4× undertransport, stale pm_raw, silent surface_flux drop
type: project
---

## LinRood undertransport bug (CRITICAL)

`run_cs_transport.jl:335`: `fs = 1.0 / total_sub` where `total_sub = steps_per_window * n_lr_sub`.
But the outer loop already repeats `steps_per_window` times. The correct scale is `1/n_lr_sub`.

**Why correct is `1/n_lr_sub`:**
- CS binary stores `am = velocity_flux × dt_met / (2 × steps_per_window)` (proven from `transport_binary_v2_cs_conservative.jl:278`)
- `strang_split_linrood_ppm!` calls `fv_tp_2d_cs!` twice per call (no internal scaling)
- So total horizontal flux per `strang_split_linrood_ppm!` call = 2 × pam_p
- For full window: need `steps_per_window × 2 × stored_am` (same as non-LinRood path)
- With current code (fs = 1/total_sub): total = total_sub × 2 × stored_am / total_sub = 2 × stored_am
- Missing factor: steps_per_window (= 4 for catrine C90, dt=900, met_interval=3600)

The fix is `fs = 1.0 / n_lr_sub` — the outer `steps_per_window` loop handles the window subdivision.

## Non-LinRood path is CORRECT

`strang_split_cs!` with `flux_scale=1.0` applies each direction TWICE (forward + reverse Strang).
Each call = one substep = one `dt_met/steps_per_window` timestep.
`steps_per_window` calls = full window. Verified by tracing through `CubedSphereStrang.jl:480-535`.

## pm_raw staleness (MEDIUM)

`run_cs_transport.jl:244`: `pm_raw` set from window 1 only.
`run_cs_transport.jl:324`: CFL pilot uses `pm_raw` as donor mass.
`pm` is updated per window (line 310) but `pm_raw` is never refreshed.
For windows 2+, CFL estimates use wrong (window-1) air masses.
Impact is limited in practice because the CFL pilot already yields n_lr_sub=1 almost always.

## Surface flux silently dropped (HIGH)

`run_cs_transport.jl:209`: only `get(tcfg, "init", ...)` is read per tracer.
The `surface_flux` key is never accessed anywhere in lines 1-424.
`fossil_co2` starts at 0.0 and stays zero. No warning is emitted.
