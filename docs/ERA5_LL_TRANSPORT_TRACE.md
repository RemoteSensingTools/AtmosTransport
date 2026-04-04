# ERA5 Lat-Lon Transport: Complete Algorithmic Trace

Complete step-by-step execution trace for the ERA5 lat-lon grid path with
`PreprocessedLatLonMetDriver`, `SlopesAdvection`, `DoubleBuffer`, and optional
`TiedtkeConvection`, `PBLDiffusion`, and uniform surface emissions.

Every step lists: Julia file:line, TM5 F90 reference, key variables, units, dimensions,
and discrepancies.

## Summary: Julia vs TM5 F90 Differences

| Aspect | Julia Implementation | TM5 F90 Reference | Status |
|--------|---------------------|-------------------|--------|
| **Met loading** | Pre-computed binary (v3 constant, v4 interpolated dam/dbm/dm) | `TimeInterpolation` (meteodata.F90:680) blends pu/pv per substep → `dynam0` (advect_tools.F90:638) computes am/bm/cm | ⚠️ v4 approximates |
| **CFL control** | Per-direction inner subdivision, n_sub = ceil(CFL/0.95), capped at 50 | Outer loop `Check_CFL` (advectm_cfl.F90:154) halves ndyn globally, scales ALL fluxes, retries | ⚠️ Differs |
| **X subcycling** | Global max CFL → uniform n_sub for all rows | Per-row adaptive nloop in `dynamum` (advectm_cfl.F90:1217-1268) with mass evolution tracking | ⚠️ Differs |
| **Y subcycling** | Same uniform subdivision as X | NO subcycling — relies on outer `Check_CFL` loop (`dynamvm`, advecty.F90:334) | ⚠️ Differs |
| **Z subcycling** | Uniform subdivision: cm/n_sub applied n_sub times | REJECTS CFL >= 1 entirely (`dynamwm` advectm_cfl.F90:2801), halves ndyn | ⚠️ Differs |
| **Slopes** | Diagnostic (minmod, recomputed each step) | Prognostic rxm/rym/rzm (evolve via update formulas, advectx.F90:707, advectz.F90:481) | ⚠️ More diffusive |
| **Cross-slopes** | Not implemented | fy/fz advected alongside rm in X; fx/fz in Y; fx/fy in Z | ❌ Missing |
| **Operator split** | X→Y→Z→Z→Y→X in `strang_split_massflux!` | Same via `splitorderzoom` (advectm_cfl.F90:651) | ✅ Matches |
| **Z mass update** | Double-buffered 3D kernel (rm→rm_buf→copyto!) | Two-pass: all fluxes computed (loop 454-474), then all rm updated (479-491) in `dynamw_1d` | ✅ Equivalent |
| **cm boundaries** | `_enforce_cm_boundaries!` zeros cm[:,:,1] and cm[:,:,end] | `dynam0` zeros all cm first (line 717), fills only interior; `dynamw` asserts zero (line 320) | ✅ Matches |
| **Multi-tracer** | Sequential Strang sweeps per tracer with m_save/restore | Inner tracer loop within single mass evolution (`dynamw_1d` loops over ntr at line 451) | ⚠️ Different order |

### TM5 F90 Source Files Reference

| File | Key Subroutines | Purpose |
|------|----------------|---------|
| `deps/tm5/base/src/advect_tools.F90` | `dynam0` (line 638) | Compute am/bm/cm from pu/pv winds |
| `deps/tm5/base/src/advectm_cfl.F90` | `Check_CFL` (154), `Setup_MassFlow` (313), `dynamum` (1118), `dynamvm` (2083), `dynamwm` (2705), `advectx_get_nloop` (1354) | CFL control, mass-only pilots |
| `deps/tm5/base/src/advectx.F90` | `dynamu` | X tracer advection with prognostic slopes |
| `deps/tm5/base/src/advecty.F90` | `dynamv` | Y tracer advection with prognostic slopes |
| `deps/tm5/base/src/advectz.F90` | `dynamw`, `dynamw_1d` (409) | Z tracer advection with prognostic rzm |
| `deps/tm5/base/src/meteodata.F90` | `TimeInterpolation` (533) | Linear blend of wind fields between time levels |

---

## Phase 0: Configuration & Model Build

### Step 0.1: TOML Parsing

**Julia implementation:**
- File: `src/IO/configuration.jl` (`build_model` entrypoint)
- Input: TOML config Dict from `config/runs/<name>.toml`
- Output: `TransportModel` struct
- Key config params:
  - `[grid]`: Nx, Ny, Nz
  - `[met_data]`: driver type, files, dt
  - `[advection]`: scheme ("slopes" or "ppm"), ppm_order, linrood, cfl_limit
  - `[convection]`: type ("tiedtke", "ras", "tm5_matrix", "none")
  - `[diffusion]`: type ("bld", "pbl", "nonlocal_pbl")
  - `[output]`: interval, format, variables

**TM5 F90 reference:**
- TM5 uses `rcfile` key-value settings, not TOML
- No direct equivalent -- TM5 configuration is compile-time + runtime rc files

### Step 0.2: Grid Construction

**Julia implementation:**
- File: `src/Grids/latitude_longitude_grid.jl`
- Output: `LatitudeLongitudeGrid{FT}` with:
  - `phi_c` (cell centers), `phi_f` (cell faces) in degrees
  - `radius = 6371229.0` m, `gravity = 9.80665` m/s^2
  - `vertical`: A/B hybrid coefficients (Nz+1 interfaces)
  - `reduced_grid`: per-latitude `cluster_sizes` for polar x-advection
- Convention: poles at +/-90, cell centers at +/-89.75 for 0.5-deg grid

**TM5 F90 reference:**
- File: `deps/tm5/base/src/dims.F90`
- Variables: `im`, `jm`, `lm` (grid dimensions)
- `dxyp(j)` = cell area by latitude [m^2]
- `bt(l)` = normalized hybrid B coefficient (B(l)-B(l+1))/(B(1)-B(lm+1))
- `at(l)`, `bt(l)` = hybrid A/B at interfaces

**Status:** Matches -- same hybrid coordinate system, same gravity/radius constants.

### Step 0.3: Met Driver Construction

**Julia implementation:**
- File: `src/IO/preprocessed_latlon_driver.jl:64-143`
- Function: `PreprocessedLatLonMetDriver(; files, dt, merge_map)`
- Opens first `.bin` file via `MassFluxBinaryReader`, reads JSON header
- Reads: Nx, Ny, Nz, dt_seconds, steps_per_met_window, level_top, level_bot
- Computes: `steps_per_win = met_interval / actual_dt`
- Counts windows per file; sums to `n_windows`
- Units: dt in seconds; lons/lats in degrees

**TM5 F90 reference:**
- TM5 reads met data via `MeteoData` module (`TimeInterpolation`)
- Wind fluxes `pu`, `pv` (kg/s) read from met files per window
- `ndyn` = dynamics timestep [s], typically = `dt_sub * 2 * tref`

**Status:** Differs -- Julia uses pre-computed mass fluxes; TM5 reads raw winds and computes fluxes in `dynam0`.

### Step 0.4: Advection Scheme Construction

**Julia implementation:**
- File: `src/IO/configuration.jl`
- `SlopesAdvection()` or `PPMAdvection{ORD}(use_linrood, use_gchp, ...)`
- For slopes: `use_limiter = true` (minmod limiter, matches TM5 `with_limits`)

**TM5 F90 reference:**
- File: `deps/tm5/base/src/dims.F90`
- `adv_scheme = "slopes"` (compile-time `#ifdef slopes`)
- `limits = .true.` when `#ifdef with_limits`

**Status:** Matches -- both use van Leer slopes with minmod limiter.

---

## Phase 1: Run Loop Entry + Pre-Loop Allocation

### Step 1.1: Entry Point

**Julia implementation:**
- File: `src/Models/run_loop.jl:16-18`
- Function: `run!(model)` dispatches to `_run_loop!(model, model.grid, model.buffering)`

### Step 1.2: Timing Constants

**Julia implementation:**
- File: `src/Models/run_loop.jl:29-38`
- Variables:
  - `n_win = total_windows(driver)` -- total met windows across all files
  - `n_sub = steps_per_window(driver)` -- advection substeps per window (typically 4)
  - `dt_sub = FT(driver.dt)` -- substep duration [s] (typically 900s for ERA5)
  - `dt_window = dt_sub * n_sub` -- window duration [s] (typically 3600s)
  - `half_dt = dt_sub / 2` -- half-step [s] (typically 450s)

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advect_tools.F90:711-713`
- `dtu = ndyn / (2 * tref(region))` -- substep duration [s]
- For region=1 with tref=1: `dtu = ndyn / 2`
- `ndyn` is the full dynamics timestep; `dtu` is the half-step applied to fluxes
- `am = dtu * pu`, `bm = dtv * pv` -- mass flux = time * wind flux

**Status:** Matches conceptually. Julia `half_dt` = TM5 `dtu`. The preprocessor stores
`am = half_dt * pu` so runtime Julia doesn't need to multiply.

### Step 1.3: IOScheduler Construction (DoubleBuffer)

**Julia implementation:**
- File: `src/Models/io_scheduler.jl:69-79`
- Allocates two GPU buffer pairs (A and B) and two CPU staging buffers
- Each `LatLonMetBuffer` (`src/IO/met_buffers.jl:72-91`):
  - `m_ref  : GPU (Nx, Ny, Nz)` -- reference air mass [kg]
  - `m_dev  : GPU (Nx, Ny, Nz)` -- working air mass [kg], modified by advection
  - `am     : GPU (Nx+1, Ny, Nz)` -- x mass flux [kg per half_dt]
  - `bm     : GPU (Nx, Ny+1, Nz)` -- y mass flux [kg per half_dt]
  - `cm     : GPU (Nx, Ny, Nz+1)` -- z mass flux [kg per half_dt]
  - `ps     : GPU (Nx, Ny)` -- surface pressure [Pa]
  - `Dp     : GPU (Nx, Ny, Nz)` -- pressure thickness [Pa]
  - `u, v   : GPU (staggered)` -- wind scratch (raw met only, unused for preprocessed)
  - `ws     : MassFluxWorkspace` -- pre-allocated advection buffers
  - `dam, dbm, dm : GPU or nothing` -- v4 flux deltas for TM5-style temporal interpolation
- Each `LatLonCPUBuffer` (`src/IO/met_buffers.jl:100-122`):
  - `m, am, bm, cm, ps` -- CPU staging arrays (same dimensions)
  - `dam, dbm, dm` -- v4 deltas (empty 0x0x0 arrays if v3)

### Step 1.4: Physics Buffers + Workspace Allocation

**Julia implementation:**
- File: `src/Models/run_loop.jl:48-67`
- `phys = allocate_physics_buffers(...)` -- CMFMC, DTRAIN, QV, surface fields, PBL
- `gc_ws = allocate_geometry_and_workspace(...)` -- GridGeometryCache + MassFluxWorkspace
  - `gc.area_j` [m^2], `gc.dy_j` [m], `gc.dx_face` [m], `gc.bt` [dimensionless]
  - `gc.bt[k] = (B[k+1] - B[k]) / (B[Nz+1] - B[1])` -- normalized B thickness
- `emi_state = prepare_emissions(...)` -- GPU flux arrays + area vectors

---

## Phase 2: Initial Load (Window 1)

### Step 2.1: Load Met Window from Binary

**Julia implementation:**
- File: `src/Models/io_scheduler.jl:138-141` (initial_load! for DoubleBuffer)
- Calls: `load_met_window!(current_cpu(sched), driver, grid, 1)`
- File: `src/IO/preprocessed_latlon_driver.jl:210-257`
- Function: `load_met_window!(cpu_buf, driver, grid, win)`
- Maps global window index to (file_idx, local_win) via `window_to_file_local`
- Calls: `load_window!(cpu_buf.m, cpu_buf.am, cpu_buf.bm, cpu_buf.cm, cpu_buf.ps, reader, local_win)`

### Step 2.2: Binary Reader (Mmap Zero-Copy)

**Julia implementation:**
- File: `src/IO/binary_readers.jl:198-208`
- Function: `load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu, reader, win)`
- Layout: `[header | window_1 | window_2 | ...]`
- Per window: `m (Nx*Ny*Nz) | am ((Nx+1)*Ny*Nz) | bm (Nx*(Ny+1)*Nz) | cm (Nx*Ny*(Nz+1)) | ps (Nx*Ny)`
- Optional v2+: `| qv | cmfmc | tm5conv(4) | surface(4) | temperature`
- Optional v4: `| dam | dbm | dm` -- flux deltas for temporal interpolation
- All via `copyto!` from mmap'd `Vector{FT}` -- zero allocation

### Step 2.3: Level Merging (if merge_map provided)

**Julia implementation:**
- File: `src/IO/preprocessed_latlon_driver.jl:265-298`
- Function: `_merge_levels_latlon!(cpu_buf, m_native, am_native, bm_native, cm_native, mm)`
- For m/am/bm: sum native levels within each merged group
- For cm: recompute from horizontal divergence of merged am/bm (continuity equation)
  ```
  cm[i,j,k+1] = cm[i,j,k] - ((am[i+1,j,k] - am[i,j,k]) + (bm[i,j+1,k] - bm[i,j,k]))
  ```
- NOTE: does NOT apply `_correct_cm_residual!` -- recomputed cm is exact by construction

**TM5 F90 reference:**
- TM5 has no equivalent -- it runs at native resolution (no level merging)

### Step 2.4: Enforce cm Boundaries

**Julia implementation:**
- File: `src/IO/preprocessed_latlon_driver.jl:250-254, 303-306`
- Function: `_enforce_cm_boundaries!(cpu_buf.cm)`
- Sets: `cm[:, :, 1] = 0` (TOA) and `cm[:, :, end] = 0` (surface)

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advectz.F90:320`
- `if ( any(cm(:,:,0) /= 0.0) .or. any(cm(:,:,lmr) /= 0.0) ) then` -- fatal assertion
- TM5 enforces cm=0 at boundaries in `dynam0` by construction: `cm(i,j,0) = 0` and `cm(i,j,lmr) = 0`

**Status:** Matches -- both enforce zero cm at TOA and surface.

### Step 2.5: Load v4 Flux Deltas (if v4 binary)

**Julia implementation:**
- File: `src/IO/binary_readers.jl:299-307`
- Function: `load_flux_delta_window!(dam_cpu, dbm_cpu, dm_cpu, reader, win)`
- `dam = am_next - am_curr` ((Nx+1)*Ny*Nz elements)
- `dbm = bm_next - bm_curr` (Nx*(Ny+1)*Nz elements)
- `dm  = m_next  - m_curr`  (Nx*Ny*Nz elements)
- Used for TM5-style per-substep linear interpolation

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advectm_cfl.F90:242-243`
- `call Setup_MassFlow(tr, ndyn, status)` -- called per substep inside Check_CFL
- File: `deps/tm5/base/src/advectm_cfl.F90:313-380`
- `call TimeInterpolation(pu_dat(n), tr, status)` -- interpolates winds to substep time
- Then `call dynam0(n, ndyn, status)` -- converts interpolated winds to am/bm/cm

**Status:** Equivalent purpose. Julia pre-computes deltas in preprocessor; TM5 interpolates winds at runtime.

---

## Phase 3: Main Window Loop (w = 1 to n_win)

### Step 3A: Upload Met to GPU

**Julia implementation:**
- File: `src/Models/run_loop.jl:103`
- Calls: `upload_met!(sched)` which dispatches to `upload!(buf::LatLonMetBuffer, cpu::LatLonCPUBuffer)`
- File: `src/IO/met_buffers.jl:129-143`
- Copies:
  ```
  copyto!(buf.m_ref, cpu.m)    -- m_ref = prescribed air mass [kg]
  copyto!(buf.m_dev, cpu.m)    -- m_dev = working copy (will evolve during advection)
  copyto!(buf.am, cpu.am)      -- x mass flux [kg per half_dt]
  copyto!(buf.bm, cpu.bm)      -- y mass flux [kg per half_dt]
  copyto!(buf.cm, cpu.cm)      -- z mass flux [kg per half_dt]
  copyto!(buf.ps, cpu.ps)      -- surface pressure [Pa]
  if v4: copyto!(buf.dam, cpu.dam), copyto!(buf.dbm, cpu.dbm), copyto!(buf.dm, cpu.dm)
  ```
- **Critical:** Both `m_ref` and `m_dev` start identical each window. `m_dev` evolves during advection; `m_ref` stays constant until post-advection sync.

### Step 3B: Load and Upload Physics Fields

**Julia implementation:**
- File: `src/Models/physics_phases.jl:147-202` (`load_and_upload_physics!` for LL)
- Loads per field from driver (binary or NetCDF):
  1. TM5 convection (entu, detu, entd, detd) -- if TM5MatrixConvection
  2. CMFMC (convective mass flux) [Pa/s at interfaces, Nx x Ny x Nz+1]
  3. DTRAIN (detraining mass flux) -- if RAS convection
  4. Surface fields (pblh, ustar, hflux, t2m) -- if PBL diffusion
  5. QV (specific humidity) [kg/kg, Nx x Ny x Nz] -- for dry-air corrections
- Each: CPU load via `load_*_window!` then `copyto!` to GPU

### Step 3C: Async Begin Load of Next Window

**Julia implementation:**
- File: `src/Models/run_loop.jl:107-109`
- For DoubleBuffer:
  - File: `src/Models/io_scheduler.jl:170-178`
  - `sched.load_task = Threads.@spawn load_met_window!(next_cpu(sched), driver, grid, w+1)`
  - Loads into the "other" CPU buffer asynchronously while GPU computes

### Step 3D: Process Met After Upload

**Julia implementation:**
- File: `src/Models/physics_phases.jl:318-347` (`process_met_after_upload!` for LL)
- For `PreprocessedLatLonMetDriver` (preprocessed binary):
  - NO flux scaling -- fluxes are stored per half_dt by the preprocessor
  - am/bm/cm used directly as-is
- For `AbstractRawMetDriver` (raw winds):
  - Computes: `Dp = m_ref` (pressure thickness from air mass)
  - `compute_air_mass!(m_ref, Dp, grid)` -- m = Dp * area_j / g
  - `compute_mass_fluxes!(am, bm, cm, u, v, grid, Dp, half_dt)` -- converts winds to mass fluxes
- **Pole zeroing** (both paths):
  ```julia
  gpu.am[:, 1, :]  .= zero(FT)    -- zero x-flux at south pole row
  gpu.am[:, Ny, :] .= zero(FT)    -- zero x-flux at north pole row
  ```
  bm at pole FACES (j=1, j=Ny+1) is already zero from preprocessor.

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advect_tools.F90:638-809` (`dynam0`)
- `dtu = ndyn / (2 * tref(region))` -- half-step duration [s]
- Zeros all fluxes: `am = 0; bm = 0; cm = 0`
- Computes horizontal convergence:
  ```fortran
  conv_adv(i,j,l) = pu(i-1,j,l) - pu(i,j,l) + pv(i,j,l) - pv(i,j+1,l)
  ```
- Computes vertically integrated convergence: `pit(i,j) = sum(conv_adv(i,j,:))`
- Computes vertical mass flux via B-ratio distribution:
  ```fortran
  sd(i,j,lmr-1) = conv_adv(i,j,lmr) - (bt(lmr)-bt(lmr+1)) * pit(i,j)
  do l = lmr-2, 1, -1
    sd(i,j,l) = sd(i,j,l+1) + conv_adv(i,j,l+1) - (bt(l+1)-bt(l+2)) * pit(i,j)
  end do
  ```
- Stores mass fluxes:
  ```fortran
  am(i,j,l) = dtu * pu(i,j,l)
  bm(i,j,l) = dtv * pv(i,j,l)
  cm(i,j,l) = -dtw * sd(i,j,l)
  ```
- Periodic BCs: `am(0,:,:) = am(im,:,:); am(im+1,:,:) = am(1,:,:)`

**Status:** Preprocessor equivalent. Julia's preprocessor (`preprocess_spectral_massflux.jl`) performs the same computation offline, storing `am = half_dt * pu` directly.

**Discrepancy:** Julia's `_cm_column_kernel!` accumulates top-down; TM5 accumulates bottom-up via `sd`. Both yield the same cm by construction (using the same B-ratio distribution).

### Step 3E: Compute Air Mass

**Julia implementation:**
- File: `src/Models/physics_phases.jl:393`
- For LL: `compute_air_mass_phase!` is a **no-op** -- m_ref is already loaded from binary
- Air mass was computed by preprocessor: `m = Dp * area_j / g`

### Step 3F: Compute Dry Mass for VMR Conversions

**Julia implementation:**
- File: `src/Models/physics_phases.jl:110-117` (`compute_ll_dry_mass!`)
- If QV loaded and dimensions match:
  ```julia
  phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
  ```
- Otherwise: `copyto!(phys.m_dry, gpu.m_ref)` (moist = dry approximation)
- `m_dry` used ONLY for output (dry VMR = rm / m_dry)

**TM5 F90 reference:**
- TM5 operates entirely on moist basis. No dry-air correction in transport.
- Dry VMR output uses post-processing (not during runtime).

**Status:** Matches for transport (both moist). Differs for output convention.

### Step 3G: IC Finalization (Window 1 Only)

**Julia implementation:**
- File: `src/Models/physics_phases.jl:425-434` (`finalize_ic_phase!` for LL)
- If deferred vertical interpolation pending: `finalize_ic_vertical_interp!(tracers, gpu.m_ref, grid)`
- Convert VMR to tracer mass:
  ```julia
  for (_, c) in pairs(tracers)
    c .*= gpu.m_ref    -- rm = VMR * m_ref [kg, moist basis]
  end
  ```
- After this, ALL tracers are in `rm` form (tracer mass in kg)

**TM5 F90 reference:**
- TM5's prognostic variable is `rm(i,j,l,n)` -- tracer mass [kg]
- ICs loaded as mixing ratios, converted: `rm = c * m`
- Same convention.

**Status:** Matches.

### Step 3H: Save Reference Mass

**Julia implementation:**
- File: `src/Models/physics_phases.jl:445`
- For LL: `save_reference_mass!` is a **no-op** (m_ref is already the reference from binary)

---

## Phase 4: Advection + Convection

### Step 4.1: advection_phase! Entry

**Julia implementation:**
- File: `src/Models/physics_phases.jl:620-698`
- Function: `advection_phase!(tracers, sched, air, phys, model, grid::LatitudeLongitudeGrid, ws, n_sub, dt_sub, step; ...)`

### Step 4.2: Build Advection Workspace

**Julia implementation:**
- File: `src/Models/physics_phases.jl:628`
- `adv_ws = _build_advection_workspace(gpu.ws, scheme, tracers, gpu.m_ref)`
- For `SlopesAdvection`: returns `gpu.ws` (MassFluxWorkspace) unchanged
- `MassFluxWorkspace` (file: `src/Advection/mass_flux_advection.jl:554-562`):
  - `rm     : GPU (Nx, Ny, Nz)` -- per-tracer copy for double-buffered kernels
  - `rm_buf : GPU (Nx, Ny, Nz)` -- output buffer for kernel writes
  - `m_buf  : GPU (Nx, Ny, Nz)` -- output buffer for mass update
  - `cfl_x  : GPU (Nx+1, Ny, Nz)` -- CFL scratch / subdivided am storage
  - `cfl_y  : GPU (Nx, Ny+1, Nz)` -- CFL scratch / subdivided bm storage
  - `cfl_z  : GPU (Nx, Ny, Nz+1)` -- CFL scratch / subdivided cm storage
  - `cluster_sizes : GPU Vector{Int32}(Ny)` -- per-latitude cluster sizes for reduced grid

### Step 4.3: v4 Flux Delta Detection + CM Clamp

**Julia implementation:**
- File: `src/Models/physics_phases.jl:633-635`
- `has_deltas = (gpu.dam !== nothing)`
- If NOT v4: `_clamp_cm_cfl!(gpu.cm, gpu.m_ref, FT(0.95))`
  - File: `src/Models/physics_phases.jl:587-614`
  - Pulls cm, m to CPU
  - For each interior interface k=2..Nz:
    - donor = cm>0 ? m[k] (below) : m[k-1] (above)
    - if |cm| > 0.95 * donor: clamp to sign(cm) * 0.95 * donor
  - Pushes clamped cm back to GPU
  - Purpose: prevent Z-CFL > 0.95 before advection

**TM5 F90 reference:**
- TM5 NEVER clamps cm. Instead, it HALVES ndyn (the global timestep) if CFL > 1.
- File: `deps/tm5/base/src/advectm_cfl.F90:278-296`
- If CFL violation: `ndyn = ndyn / 2`, `n = n * 2`, rescale all fluxes:
  ```fortran
  fraction = real(ndyn) / real(ndyn_old)
  wind_dat(region)%am_t = wind_dat(region)%am_t * fraction
  wind_dat(region)%bm_t = wind_dat(region)%bm_t * fraction
  wind_dat(region)%cm_t = wind_dat(region)%cm_t * fraction
  ```
- This doubles the number of substeps while halving flux per step.

**Status:** Differs -- Julia clamps cm locally; TM5 halves global timestep. Julia also does per-direction CFL subcycling (see Step 4.7).

### Step 4.4: Save Base Fluxes for v4 Interpolation

**Julia implementation:**
- File: `src/Models/physics_phases.jl:641-644`
- If v4: `am0 = copy(gpu.am); bm0 = copy(gpu.bm); cm0 = copy(gpu.cm); m0 = copy(gpu.m_ref)`
- If v3: `am0 = gpu.am` (alias, no copy); `m0 = nothing`

### Step 4.5: Initialize m_dev

**Julia implementation:**
- File: `src/Models/physics_phases.jl:646`
- `copyto!(gpu.m_dev, gpu.m_ref)` -- working mass starts as reference mass

### Step 4.6: Substep Loop

**Julia implementation:**
- File: `src/Models/physics_phases.jl:647-666`
- `for s in 1:n_sub` (typically n_sub=4 for ERA5 with dt_window=3600s, dt_sub=900s)

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advectm_cfl.F90:223-272` (Check_CFL)
- `do i = 1, n` -- loop over ndyn timesteps
- Each substep:
  1. `call Setup_MassFlow(tr, ndyn, status)` -- interpolate winds, compute am/bm/cm
  2. `call determine_cfl_iter(1, cfl_ok, status)` -- xyz half
  3. `call determine_cfl_iter(1, cfl_ok, status)` -- zyx half

**Status:** Both loop over substeps. TM5 recomputes fluxes each substep via `Setup_MassFlow`; Julia either uses constant fluxes (v3) or linearly interpolates (v4).

#### Step 4.6a: v4 Flux Interpolation (per substep)

**Julia implementation:**
- File: `src/Models/physics_phases.jl:650-660`
- Midpoint fraction: `t = (s - 0.5) / n_sub`
  - For n_sub=4: t = 0.125, 0.375, 0.625, 0.875
- Interpolated fluxes:
  ```julia
  gpu.am .= am0 .+ t .* gpu.dam
  gpu.bm .= bm0 .+ t .* gpu.dbm
  gpu.m_dev .= m0 .+ t .* gpu.dm    -- prescribed mass at midpoint
  ```
- Recompute cm from divergence:
  ```julia
  _compute_cm_from_divergence_gpu!(gpu.cm, gpu.am, gpu.bm, gpu.m_dev, grid)
  ```
  - File: `src/Models/physics_phases.jl:560-580`
  - Pull to CPU; for each column: acc=0, cm[1]=0; for k=1:Nz: div_h = (am[i+1]-am[i])+(bm[j+1]-bm[j]); acc -= div_h; cm[k+1] = acc
  - Enforce: cm[:,:,1] = cm[:,:,end] = 0
  - Push back to GPU
- Then clamp: `_clamp_cm_cfl!(gpu.cm, gpu.m_dev, 0.95)`

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advectm_cfl.F90:313-380` (`Setup_MassFlow`)
- `call TimeInterpolation(pu_dat(n), tr, status)` -- linear time interpolation of winds
- `call dynam0(n, ndyn, status)` -- recomputes am/bm/cm from interpolated winds
- TM5 interpolates winds (pu, pv) at the midpoint of each substep interval, then recomputes all fluxes from scratch.

**Status:** Equivalent -- both achieve linear temporal interpolation of mass fluxes. Julia pre-computes deltas; TM5 interpolates winds and recomputes.

#### Step 4.6b: Strang Split Advection

**Julia implementation:**
- File: `src/Models/run_helpers.jl:218-222`
- Dispatches to: `strang_split_massflux!(tracers, gpu.m_dev, gpu.am, gpu.bm, gpu.cm, grid, true, adv_ws; cfl_limit=0.95)`

### Step 4.7: strang_split_massflux! (Workspace Version)

**Julia implementation:**
- File: `src/Advection/mass_flux_advection.jl:1267-1304`
- Strang splitting order: **X -> Y -> Z -> Z -> Y -> X**
- Multi-tracer handling:
  ```julia
  if n_tr > 1: m_save = similar(m); copyto!(m_save, m)
  for each (i, (name, rm_tracer)):
    if i > 1: copyto!(m, m_save)     -- RESTORE m for each tracer
    copyto!(ws.rm, rm_tracer)
    rm_single = NamedTuple{(name,)}((ws.rm,))
    # Strang sweep:
    advect_x_massflux_subcycled!(rm_single, m, am, grid, true, ws)
    advect_y_massflux_subcycled!(rm_single, m, bm, grid, true, ws)
    advect_z_massflux_subcycled!(rm_single, m, cm, true, ws)
    advect_z_massflux_subcycled!(rm_single, m, cm, true, ws)
    advect_y_massflux_subcycled!(rm_single, m, bm, grid, true, ws)
    advect_x_massflux_subcycled!(rm_single, m, am, grid, true, ws)
    copyto!(rm_tracer, ws.rm)
  ```

**TM5 F90 reference:**
- File: `deps/tm5/base/src/advectm_cfl.F90:44-45`
- `splitorder = (/'x','y','z','z','y','x'/)`
- Same X-Y-Z-Z-Y-X Strang splitting.
- TM5 calls `do_steps` which dispatches to `advectmxzoom/advectmyzoom/advectmzzoom` (CFL-only air mass) then `advectxzoom/advectyzoom/advectzzoom` (full tracer advection).

**Status:** Matches. Same Strang splitting order.

**Discrepancy:** Multi-tracer handling. Julia saves/restores `m` so each tracer sees the SAME starting mass. TM5 advects all tracers simultaneously within each direction call (m is updated once, all tracers see the evolving m).

### Step 4.7a: X-Direction Advection (CFL Subcycled)

**Julia implementation:**
- File: `src/Advection/mass_flux_advection.jl:1137-1154`
- Function: `advect_x_massflux_subcycled!(rm, m, am, grid, use_limiter, ws; cfl_limit=0.95)`
- Step 1: CFL check
  ```julia
  cfl = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)
  ```
  - Launches `_cfl_x_kernel!`: for each face, `cfl = |am| / m_donor`
  - Reduced grid: uses cluster-aggregated m for CFL computation
- Step 2: Compute n_sub
  ```julia
  n_sub = min(50, max(1, ceil(Int, cfl / cfl_limit)))
  ```
- Step 3: If n_sub > 1: `ws.cfl_x .= am ./ n_sub` (reuse cfl_x array for subdivided flux)
- Step 4: Loop n_sub times:
  ```julia
  advect_x_massflux!(rm, m, am_eff, grid, true, ws.rm_buf, ws.m_buf, ws.cluster_sizes)
  ```

**Julia kernel:** `_massflux_x_kernel!`
- File: `src/Advection/mass_flux_advection.jl:86-243`
- Per thread (i, j, k):
  - `r = cluster_sizes[j]`
  - **If r == 1 (uniform row):**
    - 5-point stencil: c[imm], c[im], c[i], c[ip], c[ipp] where c = rm/m (VMR)
    - Slopes at im, i, ip: `sc = (c_right - c_left) / 2`
    - Minmod limiter: `sc = minmod(sc, 2*(c_right - c_center), 2*(c_center - c_left))`
    - Mass-weighted slope: `sx = m * sc`
    - Clamp: `sx = max(min(sx, rm), -rm)`
    - Left face flux (am >= 0): `alpha * (rm_im + (1-alpha) * sx_im)` where `alpha = am / m_donor`
    - Left face flux (am < 0): `alpha * (rm_i - (1+alpha) * sx_i)`
    - Similarly for right face
    - `rm_new = rm + flux_left - flux_right`
    - `m_new  = m  + am[i] - am[i+1]`
  - **If r > 1 (reduced row):**
    - Aggregate rm, m over r fine cells via `_cluster_sum`
    - Compute slopes on coarse grid
    - Compute fluxes at cluster boundaries
    - Distribute back: `rm_new[i] = (rm_cluster + delta_rm) * (m[i] / m_cluster)`
  - Periodic BCs: `im = i==1 ? Nx : i-1`, `ip = i==Nx ? 1 : i+1`
- After kernel: `copyto!(rm, rm_buf)` per tracer, `copyto!(m, m_buf)`

**TM5 F90 reference: `dynamu`**
- File: `deps/tm5/base/src/advectx.F90:438-789`
- Per-row CFL loop:
  ```fortran
  call advectx_get_nloop(is, iie, xcyc==1, m(is-1:iie+1,j,l), am(is-1:iie,j,l), nloop, status)
  do iloop = 1, nloop
  ```
- CFL handling: `nloop` determined per row, fluxes divided by `nloop` in `advectx_get_nloop`. After all loops: `am(is-1:ie,j,l) = am(is-1:ie,j,l) * nloop` (restore original)
- Tracer flux formula (slopes with `rxm`):
  ```fortran
  if (am(i,j,l) >= zero) then
    alpha = am(i,j,l) / m(i,j,l)
    f(i)  = alpha * (rm(i,j,l,n) + (one-alpha) * rxm(i,j,l,n))
    pf(i) = am(i,j,l) * (alpha*alpha*rxm(i,j,l,n) - 3.*f(i))
    fy(i) = alpha * rym(i,j,l,n)
    fz(i) = alpha * rzm(i,j,l,n)
  else
    alpha = am(i,j,l) / m(i+1,j,l)
    f(i)  = alpha * (rm(i+1,j,l,n) - (one+alpha) * rxm(i+1,j,l,n))
    pf(i) = am(i,j,l) * (alpha*alpha*rxm(i+1,j,l,n) - 3.*f(i))
    fy(i) = alpha * rym(i+1,j,l,n)
    fz(i) = alpha * rzm(i+1,j,l,n)
  end if
  ```
- Update: `rm(i) = rm(i) + f(i-1) - f(i)`
- **Prognostic slope update:**
  ```fortran
  rxm(i,j,l,n) = rxm(i,j,l,n) + (pf(i-1)-pf(i)
       - (am(i-1)-am(i))*rxm(i,j,l,n)
       + 3.*((am(i-1)+am(i))*rm(i,j,l,n)
       - (f(i-1)+f(i))*m(i,j,l))) / mnew(i)
  ```
  Also updates: `rym(i) = rym(i) + fy(i-1) - fy(i)`, `rzm(i) = rzm(i) + fz(i-1) - fz(i)`
- m update: `m(is:iie,j,l) = mnew(is:iie)`
- Periodic BCs: `rm(0) = rm(iie); rm(iie+1) = rm(1)` (same for m, rxm, rym, rzm)

**Key Differences:**
| Aspect | Julia | TM5 |
|--------|-------|-----|
| Slopes | **Diagnostic** (recomputed via minmod each step) | **Prognostic** (rxm/rym/rzm carried between steps) |
| Flux formula | `f = alpha * (rm + (1-alpha) * sx)` where `sx = m * sc` | `f = alpha * (rm + (1-alpha) * rxm)` |
| Slope update | None -- slopes discarded after each sweep | `rxm += (pf(i-1)-pf(i) - ...) / mnew(i)` |
| Cross-slopes | Not advected | `rym, rzm` advected alongside `rm` |
| CFL handling | Global max CFL -> uniform subdivision | Per-row `nloop` from `advectx_get_nloop` |
| Reduced grid | GPU kernel cluster aggregation | CPU `uni2red` / `red2uni` transforms |
| Periodic BCs | Modular indexing in kernel | Ghost cells: `rm(0)=rm(iie), rm(iie+1)=rm(1)` |

**Status:** Flux formula MATCHES for the base case (diagnostic slopes with minmod = limited `rxm` when `rxm` is recomputed fresh). The prognostic slope update is MISSING in Julia -- this means Julia cannot sustain subgrid gradients across multiple advection steps. For typical ERA5 runs this is acceptable (gradients are recaptured each step) but becomes important for highly structured tracers or when CFL > 1 (prognostic slopes prevent negatives).

### Step 4.7b: Y-Direction Advection (CFL Subcycled)

**Julia implementation:**
- File: `src/Advection/mass_flux_advection.jl:1161-1178`
- Same subcycling structure as X
- Kernel: `_massflux_y_kernel!` (`mass_flux_advection.jl:245-328`)
- Per thread (i, j, k):
  - Slopes at j: 3-point stencil `[j-1, j, j+1]`, minmod limited
  - Slopes zero at j=1 (south pole) and j=Ny (north pole)
  - South face flux (j): `bm >= 0 ? beta*(rm_jm + (1-beta)*sy_jm) : beta*(rm_j - (1+beta)*sy_j)`
  - Special: at pole-adjacent rows (j=1 or j=Ny), flux is purely upwind (no slope correction)
  - North face flux: similarly
  - `rm_new = rm + flux_s - flux_n`
  - `m_new = m + bm[j] - bm[j+1]`
  - **No periodic BCs** in y (finite domain, not cyclic)

**TM5 F90 reference: `dynamv`**
- File: `deps/tm5/base/src/advecty.F90:334-499`
- NO per-row subcycling (unlike dynamu). Single pass over all j.
- Scope: `js=2, je=jmr-1` (excludes pole cells for region 1)
- Flux formula: same structure as dynamu but with `rym` (prognostic y-slope)
  ```fortran
  if (bm(i,j,l) >= zero) then
    beta = bm(i,j,l) / m(i,j-1,l)
    f(i,j-1) = beta * (rm(i,j-1,l,n) + (one-beta) * rym(i,j-1,l,n))
  ```
- Updates: `rm(i,j) = rm(i,j) + f(i,j-1) - f(i,j)`
- Prognostic rym update, rxm/rzm transported as scalars
- m update: `mnew = m + bm(j) - bm(j+1)`

**Key Differences:**
| Aspect | Julia | TM5 |
|--------|-------|-----|
| Subcycling | CFL-adaptive subdivision | NO subcycling |
| Pole handling | Slopes zero at j=1 and j=Ny; flux upwind-only at poles | js=2, je=jmr-1 (pole cells EXCLUDED from advection) |
| Slopes | Diagnostic (minmod) | Prognostic (rym carried) |

**Status:** Matches in flux formula. Differs in pole treatment and subcycling.
**Discrepancy:** Julia advects at j=1 and j=Ny with upwind flux; TM5 excludes these cells entirely.

### Step 4.7c: Z-Direction Advection (CFL Subcycled)

**Julia implementation:**
- File: `src/Advection/mass_flux_advection.jl:1189-1213`
- CFL: `cfl = max_cfl_massflux_z(cm, m, ws.cfl_z)`; `n_sub = min(50, ceil(cfl/0.95))`
- If n_sub > 1: `ws.cfl_z .= cm ./ n_sub`
- Loop n_sub times:
  - For each tracer: launch `_massflux_z_kernel!`, synchronize, `copyto!(rm, rm_buf)`
  - Then: `copyto!(m, m_buf)`

**Julia kernel:** `_massflux_z_kernel!`
- File: `src/Advection/mass_flux_advection.jl:330-413`
- Per thread (i, j, k):
  - Slopes at k: 3-point stencil `[k-1, k, k+1]`, minmod limited
  - Slopes zero at k=1 (TOA) and k=Nz (surface)
  - Top face flux (interface k):
    - `cm > 0`: `gamma * (rm[k-1] + (1-gamma) * sz_km)` where `gamma = cm/m[k-1]`
    - `cm < 0`: `gamma * (rm[k] - (1+gamma) * sz_k)` where `gamma = cm/m[k]`
    - `cm == 0`: zero
  - Bottom face flux (interface k+1): similarly
  - `rm_new = rm + flux_top - flux_bot`
  - `m_new  = m + cm[k] - cm[k+1]`
  - **Double buffering**: kernel READS from `rm` (const), WRITES to `rm_buf`. Then `copyto!(rm, rm_buf)`.

**TM5 F90 reference: `dynamw_1d`**
- File: `deps/tm5/base/src/advectz.F90:409-498`
- Column-at-a-time subroutine:
  ```fortran
  ! Step 1: Compute ALL new masses first
  do l = 1, lmr
    mnew(l) = m(l) + cm(l-1) - cm(l)
  end do
  ```
- Limit slopes: `rzm = max(min(rzm, rm), -rm)` (prognostic, pre-limited)
- Step 2: Compute ALL fluxes at ALL interfaces FIRST:
  ```fortran
  do l = 0, lmr
    if (cm(l) == 0.0) then
      f(l,n) = 0.0; pf(l) = 0.0; fx(l) = 0.0; fy(l) = 0.0
    else if (cm(l) > zero) then
      gamma = cm(l) / m(l)
      f(l,n) = gamma * (rm(l,n) + (one-gamma) * rzm(l,n))
      pf(l) = cm(l) * (gamma*gamma*rzm(l,n) - 3.*f(l,n))
      fx(l) = gamma * rxm(l,n)
      fy(l) = gamma * rym(l,n)
    else
      gamma = cm(l) / m(l+1)
      f(l,n) = gamma * (rm(l+1,n) - (one+gamma) * rzm(l+1,n))
      pf(l) = cm(l) * (gamma*gamma*rzm(l+1,n) - 3.*f(l,n))
      fx(l) = gamma * rxm(l+1,n)
      fy(l) = gamma * rym(l+1,n)
    end if
  end do
  ```
- Step 3: Update ALL levels:
  ```fortran
  do l = 1, lmr
    rm(l,n) = rm(l,n) + f(l-1,n) - f(l,n)
    rzm(l,n) = rzm(l,n) + (pf(l-1)-pf(l) - (cm(l-1)-cm(l))*rzm(l,n)
                + 3.*((cm(l-1)+cm(l))*rm(l,n) - (f(l-1,n)+f(l,n))*m(l))) / mnew(l)
    rxm(l,n) = rxm(l,n) + (fx(l-1)-fx(l))
    rym(l,n) = rym(l,n) + (fy(l-1)-fy(l))
  end do
  ```
- Step 4: `m = mnew` (update mass AFTER all tracers processed)
- **Critical:** TM5 computes ALL fluxes using the ORIGINAL m and rm values (lines 454-474), THEN updates rm and rzm (lines 479-491). This two-phase approach is mathematically equivalent to Julia's double buffering.

**Key Differences:**
| Aspect | Julia | TM5 |
|--------|-------|-----|
| Subcycling | CFL-adaptive (n_sub = ceil(max_CFL / 0.95), cap 50) | NONE. TM5 halves global ndyn instead. |
| Slopes | Diagnostic (minmod each step) | Prognostic (rzm carried) |
| Slope sign at CFL>1 | Cannot prevent negatives (diagnostic slopes insufficient) | Prognostic rzm carries history, better at CFL>1 |
| Compute order | Per-cell: compute flux_top + flux_bot + update (one pass) | Two-phase: ALL fluxes first, THEN all updates |
| Double buffer | Explicit: read rm, write rm_buf, then copyto | Implicit: new f array computed from original m/rm |
| Cross-slopes | Not transported | rxm, rym transported alongside rm |
| CFL>1 behavior | Subcycles to keep CFL<0.95 | Allows CFL>1 (rzm prevents negatives) |
| cm boundary check | Enforced before advection (cm[:,1]=cm[:,end]=0) | Fatal assertion in dynamw (line 320) |
| m update timing | After each tracer (per n_sub iteration) | After ALL tracers in column (line 496: m=mnew) |

**Status:** Flux formula MATCHES (base Russell-Lerner). Double buffering is semantically equivalent. Differs in subcycling (Julia) vs global ndyn halving (TM5), and prognostic vs diagnostic slopes.

### Step 4.7d-f: Second Z, Reverse Y, Reverse X

**Julia implementation:**
- File: `src/Advection/mass_flux_advection.jl:1297-1299`
- Same kernels called in reverse order: Z -> Y -> X
- Uses the SAME am/bm/cm (not negated) -- Strang splitting symmetry
- After all 6 sweeps: `copyto!(rm_tracer, ws.rm)` writes workspace back to original tracer array

**TM5 F90 reference:**
- Splitorder: `(/'x','y','z','z','y','x'/)` (same)
- Each direction call is a separate subroutine (advectxzoom, advectyzoom, advectzzoom)
- The `do_steps` subroutine dispatches xyz, then zyx

**Status:** Matches.

### Step 4.8: Post-Substep Mass Sync

**Julia implementation:**
- File: `src/Models/physics_phases.jl:668-672`
- If v4: `gpu.m_ref .= m0 .+ gpu.dm` (prescribe end-of-window mass)
- Then: `copyto!(gpu.m_ref, gpu.m_dev)` (sync m_ref to evolved m_dev)
- **NOTE:** After v4 line, m_ref = prescribed end. After `copyto!`, m_ref = m_dev = evolved mass. For v3, only `copyto!` runs.
- This means m_ref always tracks the EVOLVED mass after advection.

**TM5 F90 reference:**
- In TM5, `m` is updated in-place during each sweep (dynamu line 734: `m(is:iie,j,l) = mnew(is:iie)`)
- After advection, `m` contains the evolved mass. No separate m_ref/m_dev distinction.

**Status:** Equivalent. Both have evolved mass available after advection.

### Step 4.9: Convection (Once per Window)

**Julia implementation:**
- File: `src/Models/physics_phases.jl:674-697`
- Condition: `phys.cmfmc_loaded[] || phys.tm5conv_loaded[]`
- Step 1: `rm -> c` (VMR): `rm ./= gpu.m_ref`
  - Units: rm [kg] / m_ref [kg] = c [mol/mol] (moist VMR)
- Step 2: `convect!(tracers, phys.cmfmc_gpu, gpu.Dp, model.convection, grid, dt_conv, phys.planet)`
  - `dt_conv = n_sub * dt_sub = dt_window` (one convection step per window)
  - Operates on VMR directly; vertical redistribution conserving column total
- Step 3: `c -> rm`: `c .*= gpu.m_ref`
  - Same m_ref used for both directions -> exact roundtrip (no mass leak)

**TM5 F90 reference:**
- TM5 applies convection as a separate operator in the splitting sequence
- File: `deps/tm5/base/src/advectm_cfl.F90:44`: splitorder includes 'c' for convection
- Convection operates on `rm` directly (not VMR), using separate convection modules

**Status:** Differs in operator ordering. Julia: advection -> convection (once per window). TM5: convection interleaved in splitting sequence. Both conserve total mass.

---

## Phase 5: Emissions

### Step 5.1: apply_emissions_phase! (LL)

**Julia implementation:**
- File: `src/Models/physics_phases.jl:515-534`
- Step 1: `rm -> c` (VMR): `rm ./= gpu.m_ref`
- Step 2: Apply emissions:
  - File: `src/Models/run_helpers.jl:274-312` (`_apply_emissions_latlon!`)
  - For each source:
    - Update time index for time-varying fluxes
    - Find tracer by species name
    - If PBL injection available: `apply_emissions_window_pbl!`
    - If A/B coefficients available: `apply_emissions_window!` (uses ps and hybrid coords for bottom-level Dp)
    - Fallback: `c[:,:,Nz] += flux * dt_window * (M_AIR/M_species) * g / Dp_approx`
      - `flux`: kg/m^2/s
      - `dt_window`: s
      - `M_AIR/M_species`: molar mass ratio (converts mass to moles)
      - `g / Dp_approx`: converts surface emission to VMR increment
- Step 3: `c -> rm`: `c .*= gpu.m_ref`

**TM5 F90 reference:**
- TM5 applies emissions in dedicated emission routines, converting between kg and mixing ratio
- Not directly comparable (different emission infrastructure)

**Status:** Both apply emissions to VMR (mixing ratio) and convert back.

---

## Phase 6: Post-Advection Physics (Diffusion + Chemistry)

### Step 6.1: post_advection_physics! (LL)

**Julia implementation:**
- File: `src/Models/physics_phases.jl:1525-1550`
- Step 1: `rm -> c` (moist VMR): `rm ./= gpu.m_ref`
- Step 2: BLD diffusion: `_apply_bld!(tracers, dw)`
  - Static Kz profile diffusion if `BoundaryLayerDiffusion` configured
- Step 3: PBL diffusion: `diffuse_pbl!(tracers, gpu.Dp, pblh, ustar, hflux, t2m, ...)`
  - Pressure-based tridiagonal diffusion (GeosChem vdiff_mod.F90 style)
  - `D = Kz * (g/R)^2 * (p/T)^2 / (dp_mid * delp)`
  - Thomas algorithm (tridiagonal solve) per column
- Step 4: Chemistry: `apply_chemistry!(tracers, grid, model.chemistry, dt_window)`
  - RadioactiveDecay, CompositeChemistry, or NoChemistry dispatch
- Step 5: `c -> rm`: `c .*= gpu.m_ref`

---

## Phase 7: Mass Diagnostics

**Julia implementation:**
- File: `src/Models/run_loop.jl:172-206`
- `compute_mass_totals(tracers, grid)` after advection, emissions, physics
- `record_mass_change!(diag.cumulative_transport, ...)` -- tracks transport mass change
- `apply_mass_correction!(tracers, grid, diag)` -- NO-OP for LL (CS only)
- `update_mass_diagnostics!(diag, tracers, grid)` -- per-window running totals

---

## Phase 8: Output

### Step 8.1: Prepare Output Mass

**Julia implementation:**
- File: `src/Models/physics_phases.jl:1604-1611`
- `compute_output_mass(sched, air, phys, grid::LatitudeLongitudeGrid)`
  - If QV loaded: `phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)` -> returns m_dry
  - Otherwise: returns gpu.m_ref (moist mass)

### Step 8.2: Recompute Dry Mass

**Julia implementation:**
- File: `src/Models/run_loop.jl:217`
- `compute_ll_dry_mass!(phys, sched, grid)` -- recomputes m_dry from current m_ref
  - `phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)`

### Step 8.3: Convert rm to Dry VMR

**Julia implementation:**
- File: `src/Models/physics_phases.jl:1615-1616`
- `rm_to_vmr(tracers, sched, phys, grid::LatitudeLongitudeGrid)`
  - Returns: `map(rm -> rm ./ ll_dry_mass(phys), tracers)`
  - `c_dry = rm / m_dry = rm / (m_moist * (1 - QV))` [mol/mol, dry air]

### Step 8.4: Write Output

**Julia implementation:**
- File: `src/Models/run_loop.jl:220-224`
- For each writer: `write_output!(writer, model, sim_time; air_mass, tracers=c_tracers, met_fields, rm_tracers)`
- Output contains: dry VMR (from c_tracers) and optionally rm (from rm_tracers)
- Surface slice: `c[i, j, Nz]`
- Column mean: mass-weighted `sum(c * m) / sum(m)`

---

## Phase 9: Buffer Management

### Step 9.1: Wait for Async Load

**Julia implementation:**
- File: `src/Models/run_loop.jl:231-232`
- `wait_load!(sched)` -- blocks until the async load of next window completes
  - For DoubleBuffer: `fetch(sched.load_task); sched.load_task = nothing`
  - For SingleBuffer: no-op (load was synchronous)

### Step 9.2: Swap Buffers

**Julia implementation:**
- File: `src/Models/run_loop.jl:233`
- `swap!(sched)` -- toggles current buffer: `:a <-> :b`
  - `sched.current = _other(sched.current)`
  - Next window's data (in "next" buffer) becomes current

---

## Key Data Flow Summary

```
Binary file on disk (v3/v4)
    |
    v  load_met_window!  (mmap zero-copy)
CPU staging buffer: m, am, bm, cm, ps [, dam, dbm, dm]
    |
    v  _enforce_cm_boundaries!
cm[:,:,1] = 0 (TOA), cm[:,:,end] = 0 (surface)
    |
    v  upload!(gpu_buf, cpu_buf)
GPU: m_ref = m_dev = m (from binary)
GPU: am, bm, cm (from binary)
GPU: dam, dbm, dm (v4 only)
    |
    v  process_met_after_upload!
am[:, 1, :] = 0  (south pole zeroing)
am[:, Ny, :] = 0 (north pole zeroing)
    |
    v  compute_ll_dry_mass!
phys.m_dry = m_ref * (1-QV) [for output only]
    |
    v  finalize_ic_phase! (window 1 only)
tracers: VMR -> rm = VMR * m_ref  [kg, moist basis]
    |
    v  advection_phase!
    |  For each substep s in 1:n_sub:
    |    [v4: interpolate am/bm/cm to midpoint t=(s-0.5)/n_sub]
    |    [v4: prescribe m_dev = m0 + t*dm]
    |    [v4: recompute cm from divergence + clamp]
    |    strang_split_massflux!: X-Y-Z-Z-Y-X
    |      Each direction: CFL check -> n_sub -> uniform subdivision -> kernel
    |      Kernel: van Leer slopes (diagnostic minmod), upwind flux, double-buffer
    |      Multi-tracer: m_save/restore per tracer
    |  Post-loop:
    |    [v4: m_ref = m0 + dm]
    |    copyto!(m_ref, m_dev)  -- sync reference to evolved
    |  Convection (once per window):
    |    rm/m_ref -> VMR -> convect! -> VMR*m_ref -> rm
    |
    v  apply_emissions_phase!
    |  rm/m_ref -> VMR -> apply flux -> VMR*m_ref -> rm
    |
    v  post_advection_physics!
    |  rm/m_ref -> VMR -> BLD -> PBL diffusion -> chemistry -> VMR*m_ref -> rm
    |
    v  output
       compute_ll_dry_mass! (recompute from current m_ref)
       rm / (m_ref * (1-QV)) = c_dry [mol/mol]  -- dry VMR for output
```

---

## TM5 F90 Reference Comparison Summary

| Julia Step | TM5 Equivalent | File:Line | Status |
|-----------|----------------|-----------|--------|
| Binary load | `Setup_MassFlow` | `advectm_cfl.F90:313-380` | Preprocessor = dynam0 |
| cm computation | `dynam0` sd computation | `advect_tools.F90:749-753` | Matches (different accumulation direction) |
| cm boundary | `dynamw` assertion | `advectz.F90:320` | Matches |
| Strang order | `splitorder` | `advectm_cfl.F90:44-45` | Matches: X-Y-Z-Z-Y-X |
| X-advection | `dynamu` | `advectx.F90:438-789` | Flux formula matches; diagnostic vs prognostic slopes |
| X-CFL subcycling | `advectx_get_nloop` per row | `advectx.F90:617-624` | Julia: global; TM5: per-row |
| Y-advection | `dynamv` | `advecty.F90:334-499` | Flux matches; Julia subcycles, TM5 does not |
| Y pole handling | Slopes zero at j=1,Ny | js=2, je=jmr-1 | Julia advects poles (upwind); TM5 excludes |
| Z-advection | `dynamw_1d` | `advectz.F90:409-498` | Flux matches; Julia subcycles, TM5 does not |
| Z flux order | Per-cell simultaneous | All-fluxes-then-update | Both equivalent (double buffer vs two-phase) |
| Reduced grid | GPU cluster kernel | CPU uni2red/red2uni | Different implementation, same concept |
| CFL handling | Per-direction n_sub | Global ndyn halving | Julia local; TM5 global |
| Multi-tracer | m save/restore per tracer | All tracers simultaneous | Julia per-tracer; TM5 batch |
| v4 interpolation | Linear am0+t*dam | TimeInterpolation + dynam0 | Equivalent purpose |
| Convection timing | Once per window (after all substeps) | Interleaved in split sequence | Differs |
| m_ref sync | copyto!(m_ref, m_dev) after advection | m = mnew in-place | Equivalent |

---

## Critical Invariants for the LL Path

1. **No runtime flux scaling**: Preprocessed binary am/bm/cm are stored per half_dt.
   The preprocessor handles the scaling. n_sub Strang cycles accumulate correctly.

2. **Moist transport basis**: rm = c * m_moist. Dry correction ONLY at output time.

3. **cm boundaries**: cm[:,:,1]=0 (TOA), cm[:,:,Nz+1]=0 (surface). Enforced after load.

4. **Pole zeroing**: am[:, 1, :] = am[:, Ny, :] = 0. Prevents zonal transport at poles.

5. **Z-advection double buffer**: Kernel reads from rm, writes to rm_buf. Then `copyto!(rm, rm_buf)`.
   Never writes in-place. Violation causes ~10% mass loss per step.

6. **Multi-tracer m restore**: Mass m is saved before first tracer, restored before each
   subsequent tracer. All tracers see the same starting mass field for their Strang sweep.

7. **m_ref sync after advection**: `copyto!(m_ref, m_dev)` after all substeps. This ensures
   convection and emission rm<->c conversions use the EVOLVED mass, not prescribed.

8. **CFL subcycling cap**: n_sub capped at 50 per direction to prevent infinite loops.

9. **Convection timing**: One convection call per window (dt_conv = dt_window),
   NOT per substep. Applied after all substeps complete.

10. **Dry VMR output only**: Transport and all physics operate on moist basis.
    `c_dry = rm / (m_ref * (1 - QV))` computed at output time only.

11. **v4 cm recomputation**: When using v4 flux deltas, cm is recomputed from interpolated
    am/bm divergence each substep (not interpolated directly). This ensures continuity.
