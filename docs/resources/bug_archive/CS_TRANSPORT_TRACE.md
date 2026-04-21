# GEOS-IT / GEOS-FP Cubed-Sphere Transport: Complete Algorithmic Trace

Reference document for debugging CS transport in AtmosTransport.jl.
Traces every operation from `run!(model)` to output for the cubed-sphere grid path.

**Target configuration:**
- Grid: `CubedSphereGrid{Float32}` (GEOS-IT C180 or GEOS-FP C720)
- Driver: `GEOSFPCubedSphereMetDriver` (binary mode)
- Advection: `PPMAdvection{7}` (three paths: Strang, LinRood, GCHP)
- Convection: `RASConvection` or `TiedtkeConvection`
- Diffusion: `PBLDiffusion` or `NonLocalPBLDiffusion`
- Buffering: `DoubleBuffer`
- Pressure fixer + global mass fixer
- `mass_flux_dt = 450` (GEOS dynamics timestep)

---

## Phase 0: Model Construction (before run!)

### 0.1 Config parsing
- **What:** TOML config parsed into `TransportModel` struct
- **File:** `src/IO/configuration.jl`
- **Key fields:** `grid`, `met_data`, `advection_scheme`, `convection`, `diffusion`, `chemistry`, `sources`, `output_writers`, `buffering`, `tracers`, `metadata`

### 0.2 Grid construction
- **What:** `CubedSphereGrid{FT}(Nc, Nz, Hp=3)` built from config
- **Key data:**
  - `Nc` = cells per panel edge (180 for C180, 720 for C720)
  - `Nz` = vertical levels (72 native, possibly merged)
  - `Hp` = halo width (always 3 for PPM-7)
  - `connectivity` = 6-panel adjacency table (FILE convention)
  - `Ac[p]` = cell areas [m^2], `Dxc[p]`/`Dyc[p]` = cell widths [m]
  - `vertical.A`, `vertical.B` = hybrid sigma-pressure coefficients

### 0.3 Met driver construction
- **What:** `GEOSFPCubedSphereMetDriver{FT}` with mode=`:binary`
- **File:** `src/IO/geosfp_cs_met_driver.jl:137-176`
- **Key fields:**
  - `files` = ordered `.bin` file paths (mmap'd on first access)
  - `mass_flux_dt` = 450 [s] (GEOS dynamics timestep, NOT met interval)
  - `dt` = advection sub-step size [s] (e.g. 450)
  - `steps_per_win` = `met_interval / dt` (e.g. 3600/450 = 8)
  - `merge_map` = vertical level merging index, or `nothing`

### 0.4 Panel map
- **What:** `SingleGPUMap()` or `PanelGPUMap(n_gpus)` for multi-GPU
- **File:** `src/Architectures/multi_gpu.jl`
- **Convention:** `for_panels_nosync(f)` iterates `f(p)` for p=1:6, grouped by GPU

---

## Phase 1: Initialization (inside `_run_loop!`)

**Entry point:** `run!(model)` -> `_run_loop!(model, model.grid, model.buffering)`
**File:** `src/Models/run_loop.jl:27-252`

### 1.1 Compute loop constants
```
run_loop.jl:29-38
```
- `n_win = total_windows(driver)` -- total met windows
- `n_sub = steps_per_window(driver)` -- advection sub-steps per window (e.g. 8)
- `dt_sub = FT(driver.dt)` -- sub-step size [s] (e.g. 450)
- `dt_window = dt_sub * n_sub` -- full window duration [s] (e.g. 3600)
- `half_dt = dt_sub / 2` -- half sub-step for flux scaling [s] (e.g. 225)

### 1.2 Set active panel map
```
run_loop.jl:41-42
```
- `pm = get_panel_map(grid)` -> `set_panel_map!(pm)`
- Global `_ACTIVE_PANEL_MAP[]` used by `for_panels_nosync()`

### 1.3 IOScheduler construction (DoubleBuffer, CS)
```
run_loop.jl:46-47  ->  io_scheduler.jl:91-101
```
- **What:** Allocates TWO pairs of GPU+CPU met buffers (A and B)
- `CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp; use_gchp)` per pair:
  - `delp[p]` = `(Nc+2Hp, Nc+2Hp, Nz)` -- haloed pressure thickness [Pa]
  - `am[p]` = `(Nc+1, Nc, Nz)` -- x-direction mass flux [kg/s]
  - `bm[p]` = `(Nc, Nc+1, Nz)` -- y-direction mass flux [kg/s]
  - `cm[p]` = `(Nc, Nc, Nz+1)` -- z-direction mass flux [kg/s]
  - `cx[p]` = `(Nc+1, Nc, Nz)` -- x Courant number (GCHP only)
  - `cy[p]` = `(Nc, Nc+1, Nz)` -- y Courant number (GCHP only)
  - `xfx[p]` = `(Nc+1, Nc, Nz)` -- x area flux (GCHP only)
  - `yfx[p]` = `(Nc, Nc+1, Nz)` -- y area flux (GCHP only)
- `CubedSphereCPUBuffer(FT, Nc, Nz, Hp; use_gchp)` -- same shapes on CPU
- **File:** `src/IO/met_buffers.jl:156-217`
- **Units:** am/bm in kg/s (from preprocessor); delp in Pa

### 1.4 Physics buffer allocation
```
run_loop.jl:48  ->  simulation_state.jl:86-156
```
- **What:** Per-panel CPU+GPU buffers for convection, diffusion, QV, PBL
- `cmfmc_gpu[p]` = `(Nc+2Hp, Nc+2Hp, Nz+1)` -- updraft mass flux [kg/m^2/s] (MOIST basis)
- `dtrain_gpu[p]` = `(Nc+2Hp, Nc+2Hp, Nz)` -- detraining mass flux [kg/m^2/s] (MOIST)
- `qv_gpu[p]` = `(Nc+2Hp, Nc+2Hp, Nz)` -- specific humidity [kg/kg]
- `qv_next_gpu[p]` = same shape -- next-window QV (for GCHP PE computation)
- `pbl_sfc_gpu` = `(pblh, ustar, hflux, t2m)` each `(Nc+2Hp, Nc+2Hp)` per panel
- `ras_workspace[p]` = `(Nc+2Hp, Nc+2Hp, Nz)` -- scratch for RAS q_cloud
- `ps_cpu[p]` = `(Nc+2Hp, Nc+2Hp)` -- surface pressure [Pa]
- `planet` = `PlanetParameters` (gravity, kappa_vk, cp_dry, etc.)

### 1.5 Tracer allocation
```
run_loop.jl:49  ->  simulation_state.jl:170-182
```
- **What:** 6-panel NTuple per tracer, shape `(Nc+2Hp, Nc+2Hp, Nz)`
- **Units:** tracer mass `rm` [kg] (NOT mixing ratio -- TM5 convention)
- `PendingInitialConditions` applied here: `apply_pending_ic!(cs_tracers, pending_ic, grid)`
- IC values are dry VMR; converted to rm in `finalize_ic_phase!` (Phase 2.6)

### 1.6 Air mass allocation
```
run_loop.jl:50  ->  simulation_state.jl:198-204
```
- `air.m[p]` = working air mass `(Nc+2Hp, Nc+2Hp, Nz)` [kg]
- `air.m_ref[p]` = reference (start-of-window) air mass [kg]
- `air.m_wet[p]` = MOIST air mass [kg] (for convection rm<->q conversion)

### 1.7 Geometry + workspace allocation
```
run_loop.jl:54-63  ->  simulation_state.jl:221-238
```
- `gc = build_geometry_cache(grid, ref_panel)` -> `CubedSphereGeometryCache`
  - `gc.area[p]` = cell area [m^2], shape `(Nc, Nc)` (NO halo)
  - `gc.dx[p]`, `gc.dy[p]` = cell widths [m], shape `(Nc, Nc)`
  - `gc.bt` = vertical B-ratio `dB[k] / sum(dB)`, length `Nz` (for cm closure)
  - `gc.gravity` = g [m/s^2]
  - **File:** `src/Advection/cubed_sphere_mass_flux.jl:38-102`
- `ws = allocate_cs_massflux_workspace(ref_panel, Nc)` -> `CubedSphereMassFluxWorkspace`
  - `ws.rm_buf`, `ws.m_buf` = double-buffer copies for Z-advection
  - `ws.cfl_x`, `ws.cfl_y` = CFL scratch arrays
- `ws_lr = LinRoodWorkspace(grid)` (if linrood or gchp)
  - `ws_lr.q_buf[p]` = mixing ratio buffer for Lin-Rood averaging
  - `ws_lr.fx_in[p]`, `ws_lr.fx_out[p]` = X-direction face fluxes
  - `ws_lr.fy_in[p]`, `ws_lr.fy_out[p]` = Y-direction face fluxes
- `ws_vr = VerticalRemapWorkspace(grid, arch)` (if vertical_remap)
- `geom_gchp = GCHPGridGeometry(grid)` (if gchp)
- `ws_gchp = GCHPTransportWorkspace(grid)` (if gchp)

### 1.8 Emission preparation
```
run_loop.jl:64  ->  simulation_state.jl:249-251  ->  run_helpers.jl:318-332
```
- **What:** Upload CPU emission flux panels to GPU
- Per source: `flux_dev = ntuple(p -> AT(panels[p]), 6)`
- Returns `(emission_data, nothing, nothing, nothing)` tuple

### 1.9 Diffusion workspace
```
run_loop.jl:65  ->  physics_phases.jl:47-52
```
- BLD: `DiffusionWorkspace(grid, Kz_max, H_scale, dt, ref_panel)`
- PBL: handled per-call (no static workspace)

### 1.10 Initial load (synchronous)
```
run_loop.jl:69-70  ->  io_scheduler.jl:146-161
```
- `kw = physics_load_kwargs(phys, grid)` -- builds kwargs with cmfmc_cpu, qv_cpu, etc.
- `initial_load!(sched, driver, grid, 1; kw...)` calls `load_all_window!` synchronously:
  1. `load_met_window!` -- DELP + am + bm from binary (fast mmap)
  2. `load_cmfmc_window!` -- CMFMC from A3dyn binary or NetCDF
  3. `load_dtrain_window!` -- DTRAIN from A3dyn binary
  4. `load_surface_fields_window!` -- PBLH, USTAR, HFLUX, T2M + PS + TROPPT
  5. `load_qv_window!` -- QV (try v4 binary first, then CTM_I1 NetCDF fallback)

---

## Phase 2: Per-Window Main Loop

```
run_loop.jl:98-244  (for w in 1:n_win)
```

### 2.1 Upload met to GPU
```
run_loop.jl:103  ->  io_scheduler.jl:246-249
```
- `upload_met!(sched)` -> `upload!(current_gpu(sched), current_cpu(sched))`
- **File:** `src/IO/met_buffers.jl:224-242`
- Copies per panel: `copyto!(buf.delp[p], cpu.delp[p])`, same for am, bm
- If GCHP: also copies cx, cy, xfx, yfx

### 2.2 Upload physics fields
```
run_loop.jl:104  ->  physics_phases.jl:204-288
```
- `wait_phys!(sched)` -- wait for async physics load task (DoubleBuffer)
- Upload per panel via `for_panels_nosync()`:
  - CMFMC: `copyto!(phys.cmfmc_gpu[p], phys.cmfmc_cpu[p])` -- [kg/m^2/s], MOIST basis
  - DTRAIN: same pattern -- [kg/m^2/s], MOIST basis
  - QV: upload + `fill_panel_halos!(phys.qv_gpu, grid)` -- needed for face-avg in dry corrections
  - QV_next: from v4 binary or CTM_I1 fallback (for GCHP target PE)
  - Surface: pblh [m], ustar [m/s], hflux [W/m^2], t2m [K]
- `invalidate_ras_cfl_cache!()` on fresh CMFMC/DTRAIN data

### 2.3 Begin async load of next window (DoubleBuffer)
```
run_loop.jl:107-109  ->  io_scheduler.jl:183-205
```
- **Split IO:** Two async tasks spawned:
  - `load_task` = met-only (DELP + am + bm + cx/cy/xfx/yfx) -- fast, from mmap binary
  - `phys_task` = physics (CMFMC + DTRAIN + QV + surface) -- slower, from separate files
- These run in background while GPU computes current window

### 2.4 Compute PS from DELP (CPU)
```
run_loop.jl:115  ->  physics_phases.jl:290-311
```
- `compute_ps_phase!(phys, sched, grid)` -> `_compute_ps_from_delp!`
- Sums `delp[Hp+i, Hp+j, k]` over k=1:Nz for each interior cell
- Writes to `phys.ps_cpu[p]` -- used for output metadata
- Skipped if PS already loaded from binary surface fields
- **Runs on CPU, reads from `current_cpu(sched)` -- safe because only met-only task uses next_cpu**

### 2.5 Process met after upload (GPU flux scaling)
```
run_loop.jl:121-122  ->  physics_phases.jl:349-374
```
- **Non-GCHP path:**
  ```julia
  for_panels_nosync() do p
      gpu.am[p] .*= half_dt    # am *= dt_sub/2
      gpu.bm[p] .*= half_dt    # bm *= dt_sub/2
  end
  ```
  - **Units after scaling:** am, bm in [kg] per half-step
  - **Convention:** Each Strang half-sweep applies one `am` worth of flux.
    Two half-sweeps per sub-step * n_sub sub-steps = full window transport.
  - If CX/CY available: `cx[p] *= 0.5`, `cy[p] *= 0.5`
  - If XFX/YFX available: `xfx[p] *= 0.5`, `yfx[p] *= 0.5`

- **GCHP path:** No scaling -- am/bm stay in kg/s, conversion to Pa*m^2 happens in advection kernel

### 2.6 Compute air mass from DELP (GPU)
```
run_loop.jl:124-126  ->  physics_phases.jl:395-416
```
- **Dry air mass** (when QV available + dry_correction=true):
  ```
  compute_air_mass_panel!(air.m[p], gpu.delp[p], phys.qv_gpu[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
  ```
  - **Kernel:** `_dry_air_mass_cs_kernel!` at `cubed_sphere_mass_flux.jl:129-132`
  - `m[Hp+i, Hp+j, k] = delp[Hp+i, Hp+j, k] * (1 - qv[Hp+i, Hp+j, k]) * area[i,j] / g`
  - **Units:** m in [kg] (dry air mass per grid cell)

- **Moist air mass** (always, for convection):
  ```
  compute_air_mass_panel!(air.m_wet[p], gpu.delp[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
  ```
  - `m_wet[Hp+i, Hp+j, k] = delp[Hp+i, Hp+j, k] * area[i,j] / g`
  - **Units:** m_wet in [kg] (total moist air mass per grid cell)

### 2.7 First window: IC finalization + initial output
```
run_loop.jl:132-143  ->  physics_phases.jl:436-451
```
- `finalize_ic_phase!` converts VMR initial conditions to tracer mass:
  - `finalize_ic_vertical_interp!(tracers, air.m, gpu.delp, grid; qv_panels=nothing)`
  - IC values are dry VMR, `air.m` is dry when QV available, so `rm = q * m` is correct
- `save_reference_mass!` copies `air.m[p]` -> `air.m_ref[p]` for all panels
- `record_initial_mass!` sums total tracer mass across 6 panels (GPU->CPU transfer)
- `write_ic_output!` writes t=0 snapshot

### 2.8 CFL diagnostic (periodic)
```
run_loop.jl:145  ->  physics_phases.jl:497-506
```
- Every 24 windows: compute max CFL in X and Y across all 6 panels
- `max_cfl_x_cs(gpu.am[p], air.m_ref[p], ws.cfl_x, Hp)` per panel

### 2.9 Pre-advection mass snapshot
```
run_loop.jl:148  ->  mass_diagnostics.jl:246-248
```
- `snapshot_pre_advection!(diag, tracers, grid)` = `compute_mass_totals(tracers, grid)`
- This is the **target** for the global mass fixer (Phase 7)
- GPU->CPU transfer for all tracers, sum interior cells only (skip halos)

### 2.10 Deferred DELP fetch (DoubleBuffer only)
```
run_loop.jl:152  ->  physics_phases.jl:478-490
```
- `wait_and_upload_next_delp!(sched, grid)`
- **Waits** for met-only load task to complete (fast: mmap binary)
- Uploads next-window DELP to next GPU buffer: `copyto!(ng.delp[p], nc.delp[p])`
- **Purpose:** Next-window DELP needed for:
  - Pressure fixer (cm computation)
  - Vertical remap target PE
  - GCHP dry dp interpolation

### 2.11 Compute vertical mass flux (cm)
```
run_loop.jl:158-162  ->  physics_phases.jl:461-472
```
- **Skipped** for vertical remap path (`ws_vr !== nothing`)
- For Strang/LinRood path:
  ```julia
  for_panels_nosync() do p
      compute_cm_panel!(gpu.cm[p], gpu.am[p], gpu.bm[p], gc.bt, Nc, Nz)
  end
  ```
  - **Kernel:** `_cm_column_cs_kernel!` at `cubed_sphere_mass_flux.jl:321-337`
  - Algorithm per column (i,j):
    1. `pit = sum_k(am[i,j,k] - am[i+1,j,k] + bm[i,j,k] - bm[i,j+1,k])` (total H-divergence)
    2. `cm[i,j,1] = 0` (TOA boundary: no flux)
    3. `cm[i,j,k+1] = cm[i,j,k] + (conv_k - bt[k] * pit)` (continuity + B-correction)
  - **B-correction:** `bt[k] = dB[k] / sum(dB)` distributes the total column H-divergence
    proportionally to the B-coefficient change per level. This ensures cm=0 at surface
    when the pressure tendency is zero (steady state).
  - **Units:** cm in [kg] per half-step (same units as am/bm after scaling)
  - **Convention:** `cm[i,j,k]` = flux at interface between level k-1 (above) and k (below)
    - `cm[i,j,1] = 0` (TOA boundary)
    - `cm[i,j,Nz+1]` = residual (should be ~0 for mass-conserving fluxes)

---

## Phase 3: Advection (three paths)

```
run_loop.jl:166-169  ->  physics_phases.jl:1344-1515
```

The CS `advection_phase!` dispatches on advection scheme configuration:

### Path A: GCHP (use_gchp=true, ws_vr != nothing, cx != nothing)

Dispatches further on `pressure_basis`:
- `"moist"` -> `_gchp_advection_moist!()` (physics_phases.jl:1008-1340)
- `"dry"` -> `_gchp_advection_dry!()` (physics_phases.jl:725-992)

**GCHP Dry Path (summary):**
1. `rm_to_q_panels!(rm_t, air.m, grid)` -- rm -> dry VMR per panel
2. Per substep (`n_sub` times):
   a. Interpolate dry dp: `dp_work = lerp(delp*(1-qv), delp_next*(1-qv_next), frac)`
   b. Scale am/bm to Pa*m^2: `am *= g * mass_flux_dt`
   c. `gchp_tracer_2d!()` -- Lin-Rood horizontal advection on all tracers
   d. Unscale am/bm
   e. Source PE from evolved dpA (cumsum of post-horizontal air mass)
   f. Target PE from prescribed dp (proportional column scaling)
   g. q -> rm, vertical remap (`vertical_remap_cs!`), `fillz_panels!`
   h. Inter-substep: rm -> q using target mass
   i. Final substep: `gchp_calc_scaling_factor` + `apply_scaling_factor!`
3. Update `air.m` from prescribed endpoint DELP

**GCHP Moist Path (additional steps):**
1. `rm_to_q_panels!(rm_t, air.m_wet, grid)` -- divide by MOIST air mass
2. Humidity correction on fluxes: `am *= g*mfx_dt; am = am/(1-qv_face)` (GCHPctmEnv:1029)
3. dp_work = interpolated MOIST DELP (no QV correction)
4. Back-conversion at end: `q_dry = q_wet / (1-QV_next)` using met-data QV (not prognostic)

### Path B: Vertical Remap (ws_vr != nothing, no GCHP)

```
physics_phases.jl:1384-1470
```

1. Save prescribed m: `copyto!(ws_vr.m_save[p], air.m[p])`
2. Per substep (`n_sub` times), per tracer:
   a. Reset `air.m[p] = ws_vr.m_save[p]` (prescribed mass)
   b. `fv_tp_2d_cs!()` -- Lin-Rood horizontal transport (twice: once with damping, once without)
   c. Rescale: `rm_t[p] *= m_save[p] / air.m[p]` (compensate for mass evolution)
3. Convection ONCE per window (after all substeps)
4. Source PE from m_save (direct cumsum)
5. Target PE from next-window dry DELP (direct cumsum, no hybrid)
6. `vertical_remap_cs!()` per tracer
7. `update_air_mass_from_target!()` -- air.m = dp_tgt * area / g
8. Recompute m_wet = m_dry / (1-qv)

### Path C: Strang Split (default, no GCHP, no vertical remap)

```
physics_phases.jl:1471-1515
```

Per substep (`n_sub` times):

#### 3.C.1 Per-tracer advection loop
```julia
for (tname, rm_t) in pairs(tracers)
    # Reset m to prescribed reference for each tracer
    for_panels_nosync() do p
        copyto!(air.m[p], air.m_ref[p])
    end
    _apply_advection_cs!(rm_t, air.m, gpu.am, gpu.bm, gpu.cm,
                          grid, model.advection_scheme, ws; ws_lr, ...)
end
```

#### 3.C.2 Scheme dispatch: PPM with LinRood
```
run_helpers.jl:199-201  ->  cubed_sphere_fvtp2d.jl:776-786
```
```julia
strang_split_linrood_ppm!(rm_panels, m_panels, am, bm, cm, grid, Val(ORD), ws, ws_lr)
```
Decomposition: **H(LR) -> Z -> Z -> H(LR)**
1. `fv_tp_2d_cs!()` -- Lin-Rood horizontal (with damp_coeff)
2. `_sweep_z!()` -- Vertical advection (with limiter)
3. `_sweep_z!()` -- Vertical advection (with limiter)
4. `fv_tp_2d_cs!()` -- Lin-Rood horizontal (no damping)

#### 3.C.3 Scheme dispatch: PPM without LinRood
```
run_helpers.jl:203-204  ->  cubed_sphere_mass_flux_ppm.jl:458-475
```
```julia
strang_split_massflux_ppm!(rm_panels, m_panels, am, bm, cm, grid, Val(ORD), ws)
```
Decomposition: **X -> Y -> Z -> Z -> Y -> X**
1. `_sweep_x_ppm!()` -- X-direction PPM
2. `_sweep_y_ppm!()` -- Y-direction PPM
3. `_sweep_z!()` -- Vertical (upwind)
4. `_sweep_z!()` -- Vertical (upwind)
5. `_sweep_y_ppm!()` -- Y-direction PPM
6. `_sweep_x_ppm!()` -- X-direction PPM

#### 3.C.4 Lin-Rood horizontal: `fv_tp_2d_cs!`
```
cubed_sphere_fvtp2d.jl:562-645
```

Three-phase algorithm (FV3's `fv_tp_2d`):

**Phase 1: Y-PPM + pre-advect q_i**
1. `fill_panel_halos!(rm_panels, grid)` -- exchange edge halos across 6 panels
2. `fill_panel_halos!(m_panels, grid)`
3. `copy_corners!(rm_panels, grid, 2)` -- fill Hp*Hp corner ghosts for Y-direction
4. `copy_corners!(m_panels, grid, 2)` -- (dir=2: perpendicular panel edge rotation)
5. Initialize q_buf: `q_buf[ii,jj,k] = rm[ii,jj,k] / m[ii,jj,k]`
6. Compute Y-face fluxes: `_ppm_y_face_kernel!(fy_in, rm, m, bm, Hp, Nc, Val(ORD))`
7. Pre-advect in Y: `_pre_advect_y_kernel!(q_buf, rm, m, bm, fy_in, Hp)`
   - `q_i = q_buf[ii,jj,k]` = Y-pre-advected mixing ratio

**Phase 2: X-PPM on q_i + inner X-PPM + pre-advect q_j**
1. `copy_corners!(ws_lr.q_buf, grid, 1)` -- X-direction corner fill
2. `copy_corners!(rm_panels, grid, 1)` + `copy_corners!(m_panels, grid, 1)`
3. Outer X-face from q_i: `_ppm_x_face_from_q_kernel!(fx_out, q_buf, am, m, Hp, Nc, Val(ORD))`
4. Inner X-face: `_ppm_x_face_kernel!(fx_in, rm, m, am, Hp, Nc, Val(ORD))`
5. Re-initialize q_buf from original rm/m
6. Pre-advect in X: `_pre_advect_x_kernel!(q_buf, rm, m, am, fx_in, Hp)`
   - `q_j = q_buf[ii,jj,k]` = X-pre-advected mixing ratio

**Phase 3: Outer Y-PPM on q_j + averaged update**
1. `copy_corners!(ws_lr.q_buf, grid, 2)` -- Y-direction corner fill on q_j
2. Outer Y-face from q_j: `_ppm_y_face_from_q_kernel!(fy_out, q_buf, bm, m, Hp, Nc, Val(ORD))`
3. **Combined update:** `_linrood_update_kernel!`
   - `rm_new = rm - 0.5*(fx_in - fx_in_shift + fx_out - fx_out_shift + fy_in - fy_in_shift + fy_out - fy_out_shift)`
   - `m_new = m - 0.5*(|am_x| + |am_x+1| + |bm_y| + |bm_y+1|) + correction`
   - Averages X-first (fx_in + fy_out) and Y-first (fx_out + fy_in) orderings

#### 3.C.5 X-sweep PPM detail: `_sweep_x_ppm!`
```
cubed_sphere_mass_flux_ppm.jl:149-210
```

1. `fill_panel_halos_nosync!(rm_panels, grid)` -- edge halo exchange
2. `fill_panel_halos_nosync!(m_panels, grid)` -- edge halo exchange
3. `sync_all_gpus(pm)` -- barrier before reading halos
4. Compute max CFL: `max_cfl_x_cs(am[p], m[p], ws.cfl_x, Hp)` across all panels
5. Adaptive subcycling: `n_sub = ceil(Int, max_cfl / cfl_limit)` where cfl_limit=0.95
6. If n_sub > 1: divide am by n_sub
7. For each sub-cycle:
   - `_massflux_x_cs_kernel_ppm!` computes PPM reconstruction + flux divergence
   - `_copy_interior_nosync!` writes result back to rm/m panels
8. Restore am if divided

#### 3.C.6 Z-sweep detail: `_sweep_z!`
```
cubed_sphere_mass_flux.jl:1275-1298
```
1. Per panel:
   - `copyto!(ws.rm_buf, rm_panels[p])` -- **DOUBLE BUFFER**: copy rm to read buffer
   - `copyto!(ws.m_buf, m_panels[p])` -- copy m to read buffer
   - `advect_z_cs_panel_column!` -- reads from rm_buf/m_buf, writes to rm/m
2. **Critical invariant:** Without this double buffering, in-place update breaks flux
   telescoping, causing ~10% mass loss per step

#### 3.C.7 Per-cell mass fixer (Strang path only)
```
physics_phases.jl:1495-1500
```
```julia
apply_mass_fixer!(rm_t[p], air.m_ref[p], air.m[p], Nc, Nz, Hp)
```
- **Kernel:** `_mass_fixer_kernel!` at `cubed_sphere_mass_flux.jl:286-298`
- `rm[ii,jj,k] = (rm[ii,jj,k] / m_evolved[ii,jj,k]) * m_ref[ii,jj,k]`
- Converts rm to VMR using evolved mass, then back to rm using reference mass
- This compensates for the fact that Strang splitting evolves m away from m_ref

#### 3.C.8 Convection (once per window, after all substeps)
```
physics_phases.jl:1506-1514
```
```julia
dt_conv = FT(n_sub) * dt_sub  # e.g. 8 * 450 = 3600s
for (_, rm_t) in pairs(tracers)
    convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
              model.convection, grid, dt_conv, phys.planet;
              dtrain_panels=phys.dtrain_gpu, workspace=phys.ras_workspace)
end
```
- **Key:** Uses `air.m_wet` (MOIST mass) because CMFMC/DTRAIN are on MOIST basis
- Operates directly on `rm` (tracer mass) -- convection kernel handles rm<->q internally
- `dt_conv = dt_window` (full window duration)

**RAS convection detail:**
```
ras_convection.jl:297-339
```
1. `_ras_subcycling()` -- Compute adaptive subcycling:
   - CFL = `(cmfmc[k] + dtrain[k]) * dt * g / delp[k]`
   - `n_sub_conv = ceil(Int, max_cfl / 0.9)` -- cached per window
2. Per sub-step: `_ras_column_kernel!` per column (i,j) per panel
   - Computes updraft + downdraft mass transport
   - q_cloud from environment mixing ratio
   - Detrainment from DTRAIN field
3. **Unit convention:** CMFMC, DTRAIN in [kg/m^2/s]; delp in [Pa]

---

## Phase 4: Mass Change Tracking (post-advection)

```
run_loop.jl:172-174
```
- `compute_mass_totals(tracers, grid)` -- GPU->CPU, sum interior cells
- `record_mass_change!(diag.cumulative_transport, before, after)`
- Tracks advection mass drift separately from emissions/physics

---

## Phase 5: Emissions

```
run_loop.jl:177-179  ->  physics_phases.jl:536-547
```

### 5.1 CS emission dispatch
```julia
apply_emissions_phase!(tracers, emi_state, sched, phys, gc, grid, dt_window; sim_hours, arch)
```
- Extracts `emi_data` from `emi_state[1]`
- Calls `_apply_emissions_cs!` with `delp=gpu.delp`, `pblh=phys.pbl_sfc_gpu.pblh`

### 5.2 Per-source emission
```
run_helpers.jl:337-374
```
- For each `(src, flux_ref, idx_ref)` in emission_data:
  1. `TimeVaryingSurfaceFlux`: `update_time_index!(src, sim_hours)` -- advance to correct month
  2. Match species name to tracer
  3. **PBL injection** (when delp+pblh available):
     ```julia
     apply_surface_flux_pbl!(rm_t, flux_dev, area_panels, delp, pblh, dt_window, mol_ratio, Nc, Hp)
     ```
     - **Kernel:** `_emit_cs_pbl_kernel!` at `cubed_sphere_emission.jl`
     - Distributes surface emission across PBL levels proportional to DELP
     - `mol_ratio = M_AIR / molar_mass` (e.g. 28.97/44.01 for CO2)
     - Isothermal hydrostatic approx (T_ref=280K) converts PBLH [m] to pressure
     - **Units:** flux in [kg/m^2/s], area in [m^2], dt in [s] -> delta_rm in [kg]
  4. **Surface-only** (no PBL): `apply_surface_flux!` -- all emission to bottom level

### 5.3 Mass change tracking
```
run_loop.jl:183-184
```
- `record_mass_change!(diag.cumulative_emissions, after_adv, after_emi)`

---

## Phase 6: Post-Advection Physics

```
run_loop.jl:188-189  ->  physics_phases.jl:1552-1578
```

### 6.1 BLD diffusion (static Kz)
```
physics_phases.jl:1563-1564
```
```julia
_apply_bld_cs!(rm_t, air.m, dw, Nc, Nz, Hp)
```
- Uses `air.m` (DRY mass) for rm<->q conversion
- `diffuse_cs_panels!` applies pre-computed tridiagonal factors per panel

### 6.2 PBL diffusion (Kz from surface fluxes)
```
physics_phases.jl:1567-1575
```
```julia
diffuse_pbl!(rm_t, air.m, gpu.delp,
              phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
              phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
              phys.w_scratch, model.diffusion, grid, dt_window, phys.planet)
```
- **File:** `src/Diffusion/pbl_diffusion.jl:335-359`
- Kernel: `_pbl_diffuse_kernel!` per column (i,j) per panel
- **Basis:** Uses DRY air mass (`air.m`) -- matches GeosChem `pbl_mix_mod.F90` AD convention
- Operates on `rm` directly (kernel handles rm<->q conversion internally via `Val(:rm)`)
- Pressure-based tridiagonal coefficients: `D = Kz * (g/R)^2 * (p/T)^2 / (dp_mid * delp)`
- Per-column mass correction after Thomas solve

### 6.3 Chemistry
```
physics_phases.jl:1577
```
```julia
apply_chemistry!(tracers, grid, model.chemistry, dt_window)
```
- **RadioactiveDecay:** `panels[p] .*= exp(-lambda * dt)` (e.g. Rn-222)
- **CompositeChemistry:** iterates over sub-schemes in order
- **Units:** operates on rm [kg] -- decay factor is dimensionless

### 6.4 Mass change tracking
```
run_loop.jl:193-194
```
- `record_mass_change!(diag.cumulative_physics, after_emi, after_phys)`

---

## Phase 7: Global Mass Correction

```
run_loop.jl:198-201  ->  physics_phases.jl:1588-1597
```

### 7.1 Apply global mass fixer (CS only)
```
mass_diagnostics.jl:142-185
```
- **Target:** `diag.pre_adv_mass` = total mass BEFORE advection this window
- For each tracer:
  1. Sum current mass across 6 panels (GPU->CPU)
  2. `scale = target / current`
  3. `rm_t[p] .*= FT(scale)` for all panels
  4. `correction_ppm = (scale - 1) * 1e6` -- logged per window
- **Allowed tracers:** filtered by `mass_fixer_tracers` config
- Cumulative scaling tracked in `fix_interval_scale` and `fix_total_scale`

---

## Phase 8: Output

```
run_loop.jl:212-227
```

### 8.1 Air mass for output
```
physics_phases.jl:1619-1624
```
- `compute_output_mass(sched, air, phys, grid)` returns `air.m` (dry, current state)
- After vertical remap: `air.m` is target mass, `rm` is remapped to match

### 8.2 Build met fields for output
```
physics_phases.jl:1640-1658
```
- `(; ps=phys.ps_cpu, mass_flux_x=gpu.am, mass_flux_y=gpu.bm, mf_scale=half_dt, dt_window, ...)`
- Optionally includes pblh, troph, qv

### 8.3 CS rm -> VMR conversion
```
physics_phases.jl:1617
```
- `rm_to_vmr(tracers, sched, phys, grid::CubedSphereGrid) = tracers`
- **Identity for CS:** tracers stay as rm; output writer handles conversion

### 8.4 Binary output writer
```
binary_output_writer.jl:307-377
```
- Writes Float64 timestamp + Float32 field data
- **CS -> lat-lon regridding:** `_write_binary_field!` calls `regrid_cs_to_latlon()`
- **Native CS output:** concatenates 6 panels as Float32

### 8.5 Mass fixer interval snapshot
```
run_loop.jl:225 -> mass_diagnostics.jl:188-197
```
- Records per-window fixer scale into time series
- Resets interval accumulator to 1.0

---

## Phase 9: Buffer Management

```
run_loop.jl:231-233
```

### 9.1 Wait for async load
```
io_scheduler.jl:215-219
```
- `wait_load!(sched)` -- waits for met-only load_task to complete
- `fetch(sched.load_task)` blocks until background thread finishes

### 9.2 Swap buffers
```
io_scheduler.jl:236-239
```
- `swap!(sched)` -- toggles `sched.current` between `:a` and `:b`
- Next iteration: `current_gpu(sched)` returns the other buffer pair
- The buffer just loaded (in background) is now "current"
- The buffer just used for compute becomes "next" (available for loading)

---

## Complete Window Timeline

```
Window w:
  |-- upload_met!(sched)                     # CPU->GPU: DELP, am, bm
  |-- load_and_upload_physics!(...)          # wait_phys + CPU->GPU: CMFMC, QV, sfc
  |-- begin_load!(sched, w+1)               # ASYNC: start loading next window
  |-- compute_ps_phase!(...)                 # CPU: PS = sum(DELP) per column
  |-- process_met_after_upload!(...)         # GPU: am *= half_dt, bm *= half_dt
  |-- compute_air_mass_phase!(...)           # GPU: m = DELP*(1-QV)*area/g
  |-- (w==1: finalize_ic + save_m_ref)      # GPU: rm = q*m; m_ref = m
  |-- save_reference_mass!(...)             # GPU: m_ref = m
  |-- snapshot_pre_advection!(...)          # GPU->CPU: mass totals
  |-- wait_and_upload_next_delp!(...)       # SYNC: wait for next DELP, upload
  |-- compute_cm_phase!(...)                # GPU: cm from div(am,bm) + bt correction
  |-- advection_phase!(...)                 # GPU: n_sub * (per-tracer Strang/LR/GCHP)
  |     |-- per substep:
  |     |     |-- reset m = m_ref
  |     |     |-- _apply_advection_cs!(rm, m, am, bm, cm, ...)
  |     |     |     |-- (LR) fv_tp_2d_cs! -> Z -> Z -> fv_tp_2d_cs!
  |     |     |     |-- (Strang) X -> Y -> Z -> Z -> Y -> X
  |     |     |     |-- (GCHP) gchp_tracer_2d! + vertical_remap_cs!
  |     |     |-- mass_fixer: rm = (rm/m_evolved) * m_ref
  |     |-- convection (once): convect!(rm, m_wet, cmfmc, delp, ...)
  |-- apply_emissions_phase!(...)           # GPU: PBL injection per tracer
  |-- post_advection_physics!(...)          # GPU: BLD + PBL diffusion + chemistry
  |-- apply_mass_correction!(...)           # GPU: global mass fixer
  |-- write_output!(...)                    # GPU->CPU: regrid + write binary
  |-- wait_load!(sched)                     # SYNC: wait for w+1 met loading
  |-- swap!(sched)                          # toggle A<->B buffers
```

---

## Key Differences from LL Path

| Aspect | LatLon | CubedSphere |
|--------|--------|-------------|
| Data layout | Single 3D array (Nx, Ny, Nz) | 6-panel NTuple, each (Nc+2Hp, Nc+2Hp, Nz) |
| Halo exchange | Periodic wrap in X, pole logic in Y | `fill_panel_halos!` across 6 panels + `copy_corners!` |
| Advection dispatch | All tracers together | Per-tracer with m reset |
| Mass flux cm | Pre-computed from spectral (LL) or continuity equation (CS) | Continuity + B-correction (`_cm_column_cs_kernel!`) |
| CFL subcycling | In X/Y sweep (adaptive per direction) | In X/Y sweep (adaptive per direction, same algorithm) |
| Mass fixer | None (LL has better conservation) | Per-cell fixer + global fixer |
| Pressure fixer | None | `_cm_pressure_fixer_kernel!` (uses next-window DELP) |
| Convection basis | MOIST (rm/m_ref for VMR) | MOIST (rm/m_wet for VMR, since CMFMC is moist) |
| Diffusion basis | MOIST (rm/m_ref) | DRY (rm/air.m, matches GeosChem AD) |
| Emissions | VMR conversion: rm/m_ref -> emit -> rm*m_ref | Direct rm addition (PBL kernel) |
| Output VMR | `rm / m_dry` explicit conversion | Identity (tracers are rm; writer handles) |
| Air mass | Single m_ref on GPU | `air.m` (working), `air.m_ref` (reference), `air.m_wet` (moist) |
| IO scheduling | Met + physics in same load | Split: met-only (fast, mmap) + physics (async) |

---

## Data Flow Summary: Units Through the Pipeline

```
Binary file (Float32):
  DELP[Nc+2Hp, Nc+2Hp, Nz]  = pressure thickness [Pa]
  AM[Nc+1, Nc, Nz]           = x mass flux [kg/s] (DRY, accumulated over mass_flux_dt)
  BM[Nc, Nc+1, Nz]           = y mass flux [kg/s] (DRY, accumulated over mass_flux_dt)

After process_met_after_upload! (non-GCHP):
  am[p] *= half_dt            -> [kg] per half-step (half_dt = dt_sub/2)
  bm[p] *= half_dt            -> [kg] per half-step

After compute_air_mass_phase!:
  m = DELP * (1-QV) * area / g  -> [kg] dry air mass per cell
  m_wet = DELP * area / g       -> [kg] total air mass per cell

After compute_cm_phase!:
  cm = continuity(am, bm, bt)   -> [kg] per half-step

Tracers:
  rm = tracer mass [kg]
  q = rm / m (dry mixing ratio [mol/mol] when using mol-weighted m)

Convection:
  CMFMC [kg/m^2/s] on MOIST basis
  DTRAIN [kg/m^2/s] on MOIST basis
  Operates on rm via m_wet for VMR conversion

Emissions:
  flux [kg/m^2/s]
  delta_rm = flux * area * dt * (M_AIR / M_species) [kg]
  Distributed across PBL levels proportional to DELP

Output:
  rm [kg] written as Float32
  Or regridded to lat-lon via ConservativeCSMap
```

---

## Critical Invariants (CS-specific)

1. **Halo width Hp=3** required for PPM-7 (7-point stencil needs 3 ghost cells)
2. **Halo exchange before every directional sweep** -- `fill_panel_halos!` must precede X/Y advection
3. **Corner fill for Lin-Rood** -- `copy_corners!` with direction-dependent rotation
4. **Double buffering in Z-sweep** -- `ws.rm_buf`/`ws.m_buf` copies prevent in-place corruption
5. **m reset per tracer** -- Strang path resets `air.m = air.m_ref` before each tracer's advection
6. **MOIST convection, DRY advection/diffusion** -- mixing bases must not be confused
7. **mass_flux_dt = 450** -- AM/BM accumulated over dynamics timestep, not met interval
8. **B-correction in cm** -- distributes column H-divergence proportionally to dB/B_total
9. **Pressure fixer** -- incorporates DELP tendency from next window into cm
10. **Global mass fixer target** -- pre-advection mass (includes prior emissions, before current advection)
