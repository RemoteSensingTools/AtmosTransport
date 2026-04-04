# ERA5 Lat-Lon Transport: Complete Algorithmic Trace

Complete step-by-step execution trace for the ERA5 lat-lon grid path with
`PreprocessedLatLonMetDriver`, `SlopesAdvection`, `DoubleBuffer`, and optional
`TiedtkeConvection`, `PBLDiffusion`, and uniform surface emissions.

Every step lists: what happens, exact file:line, key variables, units, and dependencies.

---

## Phase 0: Model Construction

```
[0.1] Parse TOML config
  File: src/IO/configuration.jl (build_model entrypoint)
  Input: TOML config Dict
  Output: TransportModel struct
  Notes: All parameters from config/runs/<name>.toml
  |
  v
[0.2] Construct LatitudeLongitudeGrid
  File: src/Grids/latitude_longitude_grid.jl
  Input: Nx, Ny, Nz, vertical coordinate (A/B coefficients)
  Output: LatitudeLongitudeGrid{FT} with phi_c, phi_f, radius, gravity, vertical, reduced_grid
  Units: degrees for lat/lon, m for radius, m/s^2 for gravity
  Notes: Poles at +/-90, cell centers at +/-89.75 for 0.5 deg grid
         reduced_grid: per-latitude cluster_sizes for polar x-advection
  |
  v
[0.3] Construct PreprocessedLatLonMetDriver
  File: src/IO/preprocessed_latlon_driver.jl:64-143
  Input: files (Vector{String} of .bin paths), dt (override or from header)
  Output: PreprocessedLatLonMetDriver{FT} with n_windows, dt, steps_per_win, lons, lats, merge_map
  Details:
    - Opens first .bin file, reads JSON header (MassFluxBinaryReader)
    - Reads: Nx, Ny, Nz, dt_seconds, steps_per_met_window, level_top, level_bot
    - Computes steps_per_win = met_interval / actual_dt
    - Counts windows per file (sum = n_windows)
    - Auto-detects start_date from file
  Units: dt in seconds; lons/lats in degrees
  |
  v
[0.4] Construct tracer arrays (GPU)
  File: src/IO/configuration.jl (tracer allocation section)
  Input: tracer config (names, initial conditions)
  Output: NamedTuple{(:co2, :sf6, ...)} of GPU arrays, each (Nx, Ny, Nz)
  Units: Initially VMR (mol/mol) from IC; converted to rm (kg) in Phase 1
  |
  v
[0.5] Construct sources, writers, physics schemes
  File: src/IO/configuration.jl
  Output: Vector{AbstractSurfaceFlux}, Vector{AbstractOutputWriter},
          SlopesAdvection(), TiedtkeConvection()/nothing, PBLDiffusion()/nothing
  |
  v
[0.6] Construct TransportModel
  File: src/Models/Models.jl:62-92
  Output: TransportModel{GPU, LatitudeLongitudeGrid, ...} with all components
```

---

## Phase 1: Run Loop Entry + Pre-Loop Allocation

```
[1.1] run!(model) dispatches to _run_loop!
  File: src/Models/run_loop.jl:16-18
  Input: TransportModel
  Output: calls _run_loop!(model, model.grid, model.buffering)
  |
  v
[1.2] Extract timing constants
  File: src/Models/run_loop.jl:29-38
  Computed:
    n_win    = total_windows(driver)     -- total met windows
    n_sub    = steps_per_window(driver)  -- advection substeps per window (typically 4)
    dt_sub   = FT(driver.dt)             -- substep duration [s] (typically 900s for ERA5)
    dt_window = dt_sub * n_sub           -- window duration [s] (typically 3600s)
    half_dt  = dt_sub / 2                -- half-step [s] (typically 450s)
  |
  v
[1.3] Build IOScheduler (DoubleBuffer, LL)
  File: src/Models/io_scheduler.jl:69-79
  Allocates:
    gpu_a = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu)
    gpu_b = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu)
    cpu_a = LatLonCPUBuffer(FT, Nx, Ny, Nz)
    cpu_b = LatLonCPUBuffer(FT, Nx, Ny, Nz)
  Each LatLonMetBuffer (src/IO/met_buffers.jl:72-91) contains:
    m_ref  : GPU (Nx, Ny, Nz)    -- reference air mass [kg]
    m_dev  : GPU (Nx, Ny, Nz)    -- working air mass [kg]
    am     : GPU (Nx+1, Ny, Nz)  -- x mass flux [kg per half_dt]
    bm     : GPU (Nx, Ny+1, Nz)  -- y mass flux [kg per half_dt]
    cm     : GPU (Nx, Ny, Nz+1)  -- z mass flux [kg per half_dt]
    ps     : GPU (Nx, Ny)         -- surface pressure [Pa]
    Dp     : GPU (Nx, Ny, Nz)    -- pressure thickness [Pa]
    u, v   : GPU (staggered)     -- wind scratch [m/s] (raw met only)
    ws     : MassFluxWorkspace    -- pre-allocated advection buffers
    dam, dbm, dm : GPU or nothing -- v4 flux deltas
  Each LatLonCPUBuffer (src/IO/met_buffers.jl:100-122) contains:
    m   : CPU (Nx, Ny, Nz)
    am  : CPU (Nx+1, Ny, Nz)
    bm  : CPU (Nx, Ny+1, Nz)
    cm  : CPU (Nx, Ny, Nz+1)
    ps  : CPU (Nx, Ny)
    dam, dbm, dm : CPU (v4) or empty
  |
  v
[1.4] Allocate physics buffers (LL)
  File: src/Models/simulation_state.jl:18-84
  Allocates (conditional on has_conv, has_pbl):
    cmfmc_cpu/gpu : (Nx, Ny, Nz+1)  -- convective mass flux [Pa/s] (interface)
    dtrain_cpu/gpu: (Nx, Ny, Nz)    -- detraining mass flux (RAS only)
    pbl_sfc_cpu/gpu: NamedTuple with pblh, ustar, hflux, t2m (each Nx x Ny)
    w_scratch     : GPU (Nx, Ny, Nz) -- tridiagonal solver workspace
    delp_cpu      : CPU (Nx, Ny, Nz) -- pressure thickness for convection
    area_j        : CPU Vector{FT}(Ny) -- cell area by latitude [m^2]
    qv_cpu/gpu    : (Nx, Ny, Nz)    -- specific humidity [kg/kg]
    m_dry         : GPU (Nx, Ny, Nz) -- dry air mass [kg]
    planet        : PlanetParameters  -- g, R, cp, etc.
  |
  v
[1.5] Allocate tracers (LL passthrough)
  File: src/Models/simulation_state.jl:168
  Returns: model.tracers as-is (already GPU arrays from config)
  |
  v
[1.6] Allocate air mass (LL = nothing)
  File: src/Models/simulation_state.jl:194-196
  Returns: nothing
  NOTE: LL air mass lives INSIDE LatLonMetBuffer (m_ref, m_dev). No separate allocation.
  |
  v
[1.7] Allocate geometry + workspace (LL = all nothing)
  File: src/Models/simulation_state.jl:211-218
  Returns: gc=nothing, ws=nothing, ws_lr=nothing, ws_vr=nothing
  NOTE: LL advection workspace is in LatLonMetBuffer.ws (MassFluxWorkspace)
  |
  v
[1.8] Prepare emissions
  File: src/Models/simulation_state.jl:245-247 -> run_helpers.jl:246-268
  For each SurfaceFlux/TimeVaryingSurfaceFlux:
    Upload flux_dev = AT(src.flux)  -- emission flux [kg/m^2/s] to GPU
    Compute area_j_dev = AT([cell_area(1,j,grid) for j in 1:Ny])
    Extract A_coeff, B_coeff from driver (vertical coordinate coefficients)
  Returns: (emission_data, area_j_dev, A_coeff, B_coeff)
  |
  v
[1.9] Setup diffusion workspace
  File: src/Models/physics_phases.jl:43-44
  For BoundaryLayerDiffusion: builds DiffusionWorkspace from m_ref template
  For PBLDiffusion/NonLocalPBL: returns nothing (PBL diffusion uses its own tridiagonal solver)
```

---

## Phase 2: Initial Load (Window 1)

```
[2.1] initial_load! (DoubleBuffer, LL)
  File: src/Models/io_scheduler.jl:138-142
  Calls: load_met_window!(current_cpu(sched), driver, grid, 1)
  Loads into: cpu_a (the "current" buffer)
  |
  v
[2.2] load_met_window! for PreprocessedLatLonMetDriver
  File: src/IO/preprocessed_latlon_driver.jl:210-257
  Steps:
    a) window_to_file_local(driver, 1) -> (file_idx, local_win)
    b) Open MassFluxBinaryReader for the file
    c) load_window!(cpu.m, cpu.am, cpu.bm, cpu.cm, cpu.ps, reader, local_win)
       File: src/IO/binary_readers.jl:198-208
       - Zero-copy from mmap via sequential copyto!
       - Binary layout per window: [m | am | bm | cm | ps]
       - m:  Nx*Ny*Nz floats (air mass) [kg]
       - am: (Nx+1)*Ny*Nz floats (x mass flux) [kg per half_dt]
       - bm: Nx*(Ny+1)*Nz floats (y mass flux) [kg per half_dt]
       - cm: Nx*Ny*(Nz+1) floats (z mass flux) [kg per half_dt]
       - ps: Nx*Ny floats (surface pressure) [Pa]
    d) _enforce_cm_boundaries!(cpu.cm)
       File: src/IO/preprocessed_latlon_driver.jl:303-306
       - Sets cm[:,:,1] = 0 (TOA) and cm[:,:,end] = 0 (surface)
       - TM5 invariant: no flux through top or bottom of atmosphere
  |
  UNITS at this point (CPU buffer):
    m  : kg (air mass per grid cell)
    am : kg per half_dt (x mass flux = integral of rho*u*dA over half_dt)
    bm : kg per half_dt (y mass flux)
    cm : kg per half_dt (z mass flux, from preprocessor continuity equation)
    ps : Pa (surface pressure)
```

---

## Phase 3: Main Window Loop (w = 1 to n_win)

### 3A: Upload Met to GPU

```
[3A.1] upload_met!(sched)
  File: src/Models/io_scheduler.jl:246-248
  Calls: upload!(current_gpu(sched), current_cpu(sched))
  |
  v
[3A.2] upload! (LatLonMetBuffer, LatLonCPUBuffer)
  File: src/IO/met_buffers.jl:129-143
  Performs:
    copyto!(buf.m_ref, cpu.m)   -- air mass -> GPU m_ref
    copyto!(buf.m_dev, cpu.m)   -- air mass -> GPU m_dev (SAME data initially)
    copyto!(buf.am, cpu.am)     -- x flux -> GPU
    copyto!(buf.bm, cpu.bm)     -- y flux -> GPU
    copyto!(buf.cm, cpu.cm)     -- z flux -> GPU
    copyto!(buf.ps, cpu.ps)     -- surface pressure -> GPU
    If v4: copyto! dam, dbm, dm -- flux deltas -> GPU
  ALL units preserved from binary: kg, kg/half_dt, Pa
  CRITICAL: m_ref AND m_dev both get the SAME m values here.
            m_ref is the prescribed reference; m_dev evolves during advection.
```

### 3B: Load Physics Fields

```
[3B.1] load_and_upload_physics! (LL)
  File: src/Models/physics_phases.jl:148-202
  Calls in sequence:
  |
  v
[3B.2] Load CMFMC (if has_conv && !has_tm5conv)
  File: src/IO/preprocessed_latlon_driver.jl:358-391
  Dispatches to MassFluxBinaryReader: binary_readers.jl:230-235
  Binary layout: QV block at offset = core_offset, CMFMC at core_offset + n_qv
  Shape: (Nx, Ny, Nz+1) -- convective mass flux at interfaces
  Units: Pa/s (ECMWF convention) or kg/m^2/s depending on source
  copyto!(phys.cmfmc_gpu, phys.cmfmc_cpu)
  |
  v
[3B.3] Load DTRAIN (if RAS convection)
  File: src/IO/preprocessed_latlon_driver.jl (via binary reader)
  Shape: (Nx, Ny, Nz)
  |
  v
[3B.4] Load surface fields (if has_pbl)
  File: src/IO/preprocessed_latlon_driver.jl:438-465
  Binary: pblh, t2m, ustar, hflux -- each (Nx, Ny)
  Units: pblh [m], ustar [m/s], hflux [W/m^2], t2m [K]
  copyto! for each field: phys.pbl_sfc_gpu.{pblh,ustar,hflux,t2m}
  |
  v
[3B.5] Load QV (specific humidity)
  File: src/IO/preprocessed_latlon_driver.jl:474-506
  Binary reader: binary_readers.jl:218-223
  Shape: (Nx, Ny, Nz)
  Units: kg/kg (dimensionless mass fraction)
  copyto!(phys.qv_gpu, phys.qv_cpu)
```

### 3C: Async Begin Load of Next Window

```
[3C.1] begin_load! (DoubleBuffer, LL)
  File: src/Models/io_scheduler.jl:170-178
  Spawns: Threads.@spawn load_met_window!(next_cpu(sched), driver, grid, w+1)
  Loads into: cpu_b (the "next" buffer, while GPU works on gpu_a data)
  This runs in parallel with all GPU computation below.
```

### 3D: Compute PS from DELP

```
[3D.1] compute_ps_phase! (LL)
  File: src/Models/physics_phases.jl:291
  Returns: nothing (no-op for LL; PS already in binary)
```

### 3E: Process Met After Upload

```
[3E.1] process_met_after_upload! (LL, preprocessed)
  File: src/Models/physics_phases.jl:318-347
  For PreprocessedLatLonMetDriver (AbstractMassFluxMetDriver):
    - NOT a raw met driver, so skips wind->flux computation
    - If needs_delp (has_conv or has_pbl):
      [3E.2] _compute_delp_ll!
        File: src/Models/physics_phases.jl:377-386
        For each (i,j,k): delp_cpu[i,j,k] = cpu.m[i,j,k] * g / area_j[j]
        Then: copyto!(gpu.Dp, delp_cpu)
        Units: Pa (pressure thickness per level)
    - am/bm/cm are NOT scaled at runtime for preprocessed binary
      Comment at line 334-338: "NO runtime scaling -- the preprocessor stores
      fluxes per half-step (450s), and n_sub Strang cycles accumulate the correct
      total: n_sub * 2 * am = steps_per_met * 2 * half_dt = window_dt"
    - Pole zeroing (line 345-346):
        gpu.am[:, 1, :]  .= 0  -- zero x-flux at south pole row
        gpu.am[:, Ny, :] .= 0  -- zero x-flux at north pole row
        (bm at pole FACES j=1, j=Ny+1 already zero from preprocessor)
```

### 3F: Compute Air Mass

```
[3F.1] compute_air_mass_phase! (LL)
  File: src/Models/physics_phases.jl:393
  Returns: nothing (no-op for LL; air mass already in m_ref from upload)
  NOTE: LL air mass = m_ref = binary m field (MOIST air mass)
```

### 3G: Compute Dry Mass for VMR Conversions

```
[3G.1] compute_ll_dry_mass!
  File: src/Models/physics_phases.jl:111-118
  If QV loaded and shapes match:
    phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
    Units: kg (dry air mass per cell)
  Else:
    copyto!(phys.m_dry, gpu.m_ref)  -- fallback: treat as dry
  Used by: output VMR conversion (Phase 7), NOT by transport
```

### 3H: First Window Only -- IC Finalization

```
[3H.1] finalize_ic_phase! (LL)
  File: src/Models/physics_phases.jl:425-434
  Step a: finalize_ic_vertical_interp! if deferred
    - Vertical interpolation of IC VMR to model levels
  Step b: Convert VMR -> tracer mass
    for (_, c) in pairs(tracers):
      c .*= gpu.m_ref
    Input:  c = VMR (mol/mol) from IC
    Output: c = rm = c * m_ref (kg tracer mass, MOIST basis)
    Units:  rm in kg (tracer mass = VMR * moist_air_mass)
    CRITICAL: This is the TM5 convention. Transport uses rm on MOIST basis.
              Dry correction is ONLY applied at output time.
  |
  v
[3H.2] save_reference_mass! (LL)
  File: src/Models/physics_phases.jl:445
  Returns: nothing (no-op for LL; m_ref is already the reference)
  |
  v
[3H.3] record_initial_mass!
  File: src/Models/mass_diagnostics.jl
  Computes: sum(rm) for each tracer across entire 3D grid
  Stores: diag.initial_mass[tname] = total [kg]
  |
  v
[3H.4] write_ic_output! (LL)
  File: src/Models/physics_phases.jl:1665-1666
  Returns: nothing (no-op for LL)
```

### 3I: CFL Diagnostic + Pre-Advection Snapshot

```
[3I.1] update_cfl_diagnostic! (LL)
  File: src/Models/physics_phases.jl:497
  Returns: nothing (no-op for LL; CFL diagnostic is CS-only)
  |
  v
[3I.2] snapshot_pre_advection!
  File: src/Models/mass_diagnostics.jl
  Computes: sum(rm) for each tracer -> diag.pre_adv_mass
  Purpose: target mass for global mass fixer (CS only, but computed for LL tracking)
```

### 3J: Wait for Next DELP + Compute CM

```
[3J.1] wait_and_upload_next_delp! (LL)
  File: src/Models/physics_phases.jl:476
  Returns: nothing (no-op for LL)
  |
  v
[3J.2] compute_cm_phase! (LL)
  File: src/Models/physics_phases.jl:458-459
  Returns: nothing (no-op for LL; cm is PRE-COMPUTED in binary)
  NOTE: For LL, cm comes from the spectral preprocessor which solves the continuity
        equation from spectral wind divergence. This is more accurate than runtime
        computation because it uses the full spectral resolution.
```

---

## Phase 4: Advection + Convection

```
[4.1] advection_phase! (LL)
  File: src/Models/physics_phases.jl:620-698
  |
  v
[4.2] Build advection workspace
  File: src/Models/physics_phases.jl:628
  adv_ws = _build_advection_workspace(gpu.ws, scheme, tracers, gpu.m_ref)
  For SlopesAdvection: returns gpu.ws (MassFluxWorkspace) unchanged
  MassFluxWorkspace contains (src/Advection/mass_flux_advection.jl:554-562):
    rm     : GPU (Nx, Ny, Nz)    -- per-tracer copy for double-buffer advection
    rm_buf : GPU (Nx, Ny, Nz)    -- output buffer for kernel writes
    m_buf  : GPU (Nx, Ny, Nz)    -- output buffer for mass update
    cfl_x  : GPU (Nx+1, Ny, Nz)  -- CFL scratch / subcycled flux storage
    cfl_y  : GPU (Nx, Ny+1, Nz)  -- CFL scratch / subcycled flux storage
    cfl_z  : GPU (Nx, Ny, Nz+1)  -- CFL scratch / subcycled flux storage
    cluster_sizes: GPU Vector{Int32}(Ny) -- per-latitude cluster sizes
  |
  v
[4.3] Check for v4 flux deltas
  File: src/Models/physics_phases.jl:633-635
  has_deltas = (gpu.dam !== nothing)
  If NOT v4 (no deltas):
    _clamp_cm_cfl!(gpu.cm, gpu.m_ref, 0.95)
      File: src/Models/physics_phases.jl:587-614
      Pulls cm, m to CPU
      For each interior interface k=2..Nz:
        donor = cm>0 ? m[k] : m[k-1]
        if |cm| > 0.95 * donor: clamp to sign(cm) * 0.95 * donor
      Pushes clamped cm back to GPU
      Purpose: prevent CFL > 0.95 in Z-direction before advection
  |
  v
[4.4] Save base fluxes for interpolation
  File: src/Models/physics_phases.jl:641-644
  If has_deltas (v4):
    am0 = copy(gpu.am)   -- save original x-flux
    bm0 = copy(gpu.bm)   -- save original y-flux
    cm0 = copy(gpu.cm)   -- save original z-flux
    m0  = copy(gpu.m_ref) -- save original air mass
  If no deltas (v3):
    am0 = gpu.am (alias, no copy)
    bm0 = gpu.bm
    cm0 = gpu.cm
    m0  = nothing
  |
  v
[4.5] Initialize m_dev from m_ref
  File: src/Models/physics_phases.jl:646
  copyto!(gpu.m_dev, gpu.m_ref)
  |
  v
[4.6] === SUBSTEP LOOP: for s in 1:n_sub ===
  (typically n_sub=4 for ERA5 spectral with dt_window=3600s, dt_sub=900s)
  |
  v
[4.6a] (v4 only) Interpolate fluxes to substep midpoint
  File: src/Models/physics_phases.jl:650-660
  t = (s - 0.5) / n_sub  (midpoint fraction, e.g. 0.125, 0.375, 0.625, 0.875)
  gpu.am .= am0 .+ t .* gpu.dam    -- interpolated x-flux
  gpu.bm .= bm0 .+ t .* gpu.dbm    -- interpolated y-flux
  gpu.m_dev .= m0 .+ t .* gpu.dm   -- prescribed m at substep midpoint
  Recompute cm from divergence:
    _compute_cm_from_divergence_gpu!(gpu.cm, gpu.am, gpu.bm, gpu.m_dev, grid)
      File: src/Models/physics_phases.jl:560-580
      Pull to CPU, for each column:
        acc = 0; cm[:,j,1] = 0
        for k=1:Nz: div_h = (am[i+1]-am[i]) + (bm[j+1]-bm[j])
                    acc -= div_h
                    cm[i,j,k+1] = acc
      cm[:,:,1] = cm[:,:,end] = 0  (enforce boundaries)
      Push back to GPU
    Then clamp: _clamp_cm_cfl!(gpu.cm, gpu.m_dev, 0.95)
  |
  v
[4.6b] Call _apply_advection_latlon! for SlopesAdvection
  File: src/Models/run_helpers.jl:218-222
  Dispatches to: strang_split_massflux!(tracers, gpu.m_dev, gpu.am, gpu.bm, gpu.cm,
                                         grid, true, adv_ws; cfl_limit=0.95)
  |
  v
[4.7] strang_split_massflux! (workspace version)
  File: src/Advection/mass_flux_advection.jl:1267-1304
  
  Multi-tracer handling:
    n_tr = length(tracers)
    if n_tr > 1: m_save = similar(m); copyto!(m_save, m)
    For each tracer (name, rm_tracer):
      if i > 1: copyto!(m, m_save)  -- RESTORE m to pre-first-tracer state
      copyto!(ws.rm, rm_tracer)      -- copy tracer to workspace buffer
      rm_single = NamedTuple{(name,)}((ws.rm,))  -- single-tracer NamedTuple
      
      STRANG SWEEP (X -> Y -> Z -> Z -> Y -> X):
      |
      v
[4.7a] advect_x_massflux_subcycled!(rm_single, m, am, grid, true, ws; cfl_limit=0.95)
  File: src/Advection/mass_flux_advection.jl:1137-1154
  Step 1: Compute max CFL
    cfl = max_cfl_massflux_x(am, m, ws.cfl_x, ws.cluster_sizes)
      File: mass_flux_advection.jl:1053-1062
      Launches _cfl_x_kernel!(backend, 256)(cfl_arr, am, m, Nx, cluster_sizes)
        For each face: cfl = |am| / m_donor (donor = upwind cell)
        Reduced grid: uses cluster-aggregated m for CFL
      Returns: maximum(cfl_arr) -- global max CFL
  Step 2: Compute n_sub
    n_sub = min(50, max(1, ceil(Int, cfl / 0.95)))  -- cap at 50
  Step 3: If n_sub > 1: ws.cfl_x .= am ./ n_sub  (use cfl_x as subdivided flux storage)
  Step 4: Loop n_sub times:
    advect_x_massflux!(rm_single, m, am_eff, grid, true, ws.rm_buf, ws.m_buf, ws.cluster_sizes)
      File: mass_flux_advection.jl:749-767
      Launches: _massflux_x_kernel!(backend, 256)(rm_buf, rm, m_buf, m, am, Nx, cluster_sizes, true)
        ndrange = size(m) = (Nx, Ny, Nz)
        For each (i,j,k):
          r = cluster_sizes[j]
          if r == 1:  -- UNIFORM ROW (standard fine-grid)
            Compute 5-point stencil: c[imm], c[im], c[i], c[ip], c[ipp]  (c = rm/m = VMR)
            Compute slopes: sc_im, sc_i, sc_ip via minmod limiter
            Scale slopes: sx = m * sc (convert VMR slope to mass slope)
            Clamp: sx in [-rm, +rm]
            Left face flux:  am>=0 ? alpha*(rm_im + (1-alpha)*sx_im) : alpha*(rm_i - (1+alpha)*sx_i)
              where alpha = am / m_donor
            Right face flux: similarly
            rm_new = rm + flux_left - flux_right
            m_new  = m  + am_left   - am_right    (mass continuity)
          else:  -- REDUCED ROW (cluster size r > 1)
            Aggregate rm, m over r fine cells via _cluster_sum
            Compute slopes on coarse grid
            Compute fluxes at cluster boundaries
            Distribute back: rm_new[i] = (rm_cluster + delta_rm) * (m[i]/m_cluster)
      synchronize(backend)
      For each tracer: copyto!(rm, rm_buf)   -- write back
      copyto!(m, m_buf)                       -- update mass
      |
      v
[4.7b] advect_y_massflux_subcycled!(rm_single, m, bm, grid, true, ws; cfl_limit=0.95)
  Same structure as X but with _massflux_y_kernel!
  File: mass_flux_advection.jl:245-328
  Key differences:
    - No periodic wrapping (j goes 1..Ny, not cyclic)
    - Slopes at j=1 and j=Ny are zero (boundary condition)
    - flux_s at j=1: zero; flux_n at j=Ny: zero
    - At pole-adjacent rows (j=1, j=Ny): upwind-only (no slope correction)
  m_new = m + bm[j] - bm[j+1]
      |
      v
[4.7c] advect_z_massflux_subcycled!(rm_single, m, cm, true, ws; cfl_limit=0.95)
  File: mass_flux_advection.jl:1189-1213
  CFL check: cfl = max_cfl_massflux_z(cm, m, ws.cfl_z) -> n_sub
  If n_sub > 1: ws.cfl_z .= cm ./ n_sub
  Loop n_sub times:
    _massflux_z_kernel!(backend, 256)(rm_buf, rm, m_buf, m, cm_eff, Nz, true)
    File: mass_flux_advection.jl:330-413
    For each (i,j,k):
      Slopes at k: 3-point stencil [k-1, k, k+1], minmod limited
      Slopes zero at k=1 (TOA) and k=Nz (surface)
      Top face flux (k): cm>0 ? gamma*(rm[k-1]+(1-gamma)*sz_km) : gamma*(rm[k]-(1+gamma)*sz_k)
        gamma = cm / m_donor
      Bottom face flux (k+1): similar
      flux_top at k=1: zero (cm[1]=0 enforced)
      flux_bot at k=Nz: zero (cm[Nz+1]=0 enforced)
      rm_new = rm + flux_top - flux_bot
      m_new  = m  + cm[k]   - cm[k+1]     (mass continuity)
    synchronize; copyto!(rm, rm_buf) for each tracer; copyto!(m, m_buf)
  NOTE: Z advection handles DOUBLE BUFFERING correctly:
    kernel READS from rm (input) and WRITES to rm_buf (output).
    Then rm_buf is copied to rm. This prevents the stale-value bug.
      |
      v
[4.7d] advect_z_massflux_subcycled! (SECOND Z call)
  File: mass_flux_advection.jl:1297
  Same kernel, same cm. This is the SYMMETRIC pair in Strang splitting.
  Z is applied TWICE to maintain second-order accuracy.
      |
      v
[4.7e] advect_y_massflux_subcycled! (reverse Y)
  File: mass_flux_advection.jl:1298
  Same as [4.7b] -- reverse half of Strang split
      |
      v
[4.7f] advect_x_massflux_subcycled! (reverse X)
  File: mass_flux_advection.jl:1299
  Same as [4.7a] -- reverse half of Strang split
      
      After all 6 sweeps:
        copyto!(rm_tracer, ws.rm)  -- write back to original tracer array
      END of per-tracer loop
    END of strang_split_massflux!
  END of substep s
  |
  v
[4.8] Post-substep mass sync
  File: src/Models/physics_phases.jl:668-672
  If v4: gpu.m_ref .= m0 .+ gpu.dm  -- prescribe end-of-window mass
  Then:  copyto!(gpu.m_ref, gpu.m_dev)  -- sync m_ref to evolved m_dev
  NOTE: After v4 line, m_ref = prescribed end. After copyto!, m_ref = m_dev.
        For v3 (no deltas), only the copyto! runs, so m_ref = m_dev = evolved mass.
        This means m_ref tracks the EVOLVED mass, not the prescribed mass.
  |
  v
[4.9] Convection (once per window, AFTER all substeps)
  File: src/Models/physics_phases.jl:674-697
  Condition: phys.cmfmc_loaded[] (or tm5conv_loaded)
  
  [4.9a] rm -> c (VMR) conversion
    for (_, rm) in pairs(tracers): rm ./= gpu.m_ref
    Units: rm [kg] / m_ref [kg] = c [mol/mol] (moist VMR)
    
  [4.9b] convect!
    File: src/Convection/ (TiedtkeConvection or RASConvection dispatch)
    Input: c (VMR), CMFMC (Pa/s at interfaces), Dp (pressure thickness)
    convect!(tracers, phys.cmfmc_gpu, gpu.Dp, model.convection, grid, dt_conv, phys.planet)
    dt_conv = n_sub * dt_sub = dt_window
    Operates on VMR directly; vertical redistribution conserving column total
    
  [4.9c] c -> rm conversion
    for (_, c) in pairs(tracers): c .*= gpu.m_ref
    Units: c [mol/mol] * m_ref [kg] = rm [kg]
    NOTE: Same m_ref used for both directions => exact roundtrip
```

---

## Phase 5: Emissions

```
[5.1] apply_emissions_phase! (LL)
  File: src/Models/physics_phases.jl:515-534
  
  [5.1a] rm -> c conversion
    for (_, rm) in pairs(tracers): rm ./= gpu.m_ref
    Units: rm -> VMR (moist basis)
    
  [5.1b] _apply_emissions_latlon!
    File: src/Models/run_helpers.jl:274-312
    For each emission source:
      If TimeVaryingSurfaceFlux: update_time_index!(src, sim_hours)
        - Checks if monthly index needs advancing
        - If changed: re-upload flux data to GPU
      
      name = src.species (e.g. :co2)
      If name in tracers:
        If PBL injection available (delp + pblh):
          apply_emissions_window_pbl!(c, flux_dev, delp, pblh, g, dt_window; molar_mass=mm)
          File: src/Sources/surface_flux.jl
          - Distributes emission vertically across PBL levels proportional to DELP
          
        Elif A_coeff, B_coeff available:
          apply_emissions_window!(c, flux_dev, area_j_dev, ps_dev, A_coeff, B_coeff, Nz, g, dt_window)
          File: src/Sources/surface_flux.jl
          - Computes bottom-level Dp from A/B coefficients and PS
          - dc = flux * dt * (M_AIR/M_species) * g / Dp_bottom
          - Units: flux [kg/m^2/s] * dt [s] * mol_ratio * g [m/s^2] / Dp [Pa]
                   = [kg/m^2] * [mol_air/mol_species] * [m/s^2] / [Pa]
                   = [kg/m^2] * [mol/mol per (kg/m^2)] = [mol/mol]  (VMR increment)
          
        Else (fallback):
          Dp_approx = reference_pressure / Nz
          mol = M_AIR / mm
          c[:,:,Nz] += flux_dev * dt_window * mol * g / Dp_approx
          Units: same as above but with approximate Dp
  
  [5.1c] c -> rm conversion
    for (_, c) in pairs(tracers): c .*= gpu.m_ref
    Units: VMR -> kg (tracer mass)
```

---

## Phase 6: Post-Advection Physics (Diffusion + Chemistry)

```
[6.1] post_advection_physics! (LL)
  File: src/Models/physics_phases.jl:1525-1550
  
  [6.1a] rm -> c conversion
    for (_, rm) in pairs(tracers): rm ./= gpu.m_ref
    Units: rm -> VMR (moist basis)
    
  [6.1b] BLD diffusion (if BoundaryLayerDiffusion configured)
    _apply_bld!(tracers, dw)
    File: src/Models/run_helpers.jl:105
    Calls: diffuse_gpu!(tracers, dw::DiffusionWorkspace)
    - Pre-computed tridiagonal coefficients from Kz_max, H_scale
    - Thomas algorithm per column
    - Operates on VMR directly
    
  [6.1c] PBL diffusion (if PBLDiffusion configured + surface fields loaded)
    diffuse_pbl!(tracers, gpu.Dp, phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                  phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                  phys.w_scratch, model.diffusion, grid, dt_window, phys.planet)
    File: src/Diffusion/pbl_diffusion.jl
    - GeosChem vdiff_mod.F90 style pressure-based tridiagonal
    - D = Kz * (g/R)^2 * (p/T)^2 / (dp_mid * delp)
    - Includes counter-gradient term for unstable BL
    - Per-column mass correction after Thomas solve
    
  [6.1d] Chemistry
    apply_chemistry!(tracers, grid, model.chemistry, dt_window)
    File: src/Chemistry/chemistry.jl
    - RadioactiveDecay: c *= exp(-lambda * dt)
    - NoChemistry: no-op
    
  [6.1e] c -> rm conversion
    for (_, c) in pairs(tracers): c .*= gpu.m_ref
    Units: VMR -> kg (tracer mass)
```

---

## Phase 7: Mass Diagnostics

```
[7.1] Mass tracking after each operator
  File: src/Models/run_loop.jl:172-194
  After advection:  _mass_after_adv  = compute_mass_totals(tracers, grid)
  After emissions:  _mass_after_emi  = compute_mass_totals(tracers, grid)
  After physics:    _mass_after_phys = compute_mass_totals(tracers, grid)
  record_mass_change! accumulates diffs into diag.cumulative_{transport,emissions,physics}
  |
  v
[7.2] apply_mass_correction! (LL)
  File: src/Models/physics_phases.jl:1585-1586
  Returns: nothing (no-op for LL; global mass fixer is CS-only)
  |
  v
[7.3] update_mass_diagnostics!
  File: src/Models/mass_diagnostics.jl
  Computes current total mass, formats diag.showvalue string for progress bar
```

---

## Phase 8: Output

```
[8.1] Compute output air mass (dry)
  File: src/Models/physics_phases.jl:1604-1611
  compute_output_mass(sched, air, phys, grid::LL):
    If QV loaded: phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
    Returns: phys.m_dry [kg, dry air mass]
    Else: returns gpu.m_ref [kg, moist air mass]
  |
  v
[8.2] Build met_fields for writer
  File: src/Models/physics_phases.jl:1631-1638
  base = (; ps=Array(gpu.ps))
  If PBL loaded: merge with pblh
  |
  v
[8.3] Recompute dry mass for output VMR
  File: src/Models/run_loop.jl:217
  compute_ll_dry_mass!(phys, sched, grid)
  phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)  [if QV loaded]
  |
  v
[8.4] Convert rm -> dry VMR for output
  File: src/Models/physics_phases.jl:1615-1616
  rm_to_vmr(tracers, sched, phys, grid::LL):
    Returns: map(rm -> rm ./ ll_dry_mass(phys), tracers)
    c_dry = rm / m_dry = rm / (m_ref * (1-QV))
    Units: mol/mol (dry air VMR)
    NOTE: This creates NEW GPU arrays (one per tracer per output). Acceptable
          because output is infrequent (once per window).
  |
  v
[8.5] write_output! for each writer
  File: src/IO/ (BinaryOutputWriter or NetCDFOutputWriter dispatch)
  Input: sim_time, air_mass=m_dry, tracers=c_dry, met_fields, rm_tracers=rm
  Writers can extract:
    - surface_slice: c_dry[:,:,Nz]
    - column_mean: sum(c_dry .* m_dry, dims=3) / sum(m_dry, dims=3)
    - full_3d: c_dry[:,:,:]
```

---

## Phase 9: Buffer Management

```
[9.1] wait_load! (DoubleBuffer)
  File: src/Models/io_scheduler.jl:215-219
  fetch(sched.load_task)  -- blocks until Threads.@spawn from [3C.1] completes
  sched.load_task = nothing
  |
  v
[9.2] swap! (DoubleBuffer)
  File: src/Models/io_scheduler.jl:236-239
  sched.current = _other(sched.current)  -- :a <-> :b
  Effect: next iteration's current_gpu/current_cpu points to the buffer
          that was being loaded asynchronously.
  
  After swap:
    - What was gpu_b (pre-loaded) is now current_gpu
    - What was gpu_a (just computed) is now next_gpu (will be overwritten next iteration)
```

---

## Key Data Flow Summary

```
Binary file (mmap)
    |
    v  load_window! (zero-copy)
CPU buffer (m, am, bm, cm, ps)
    |
    v  upload! (copyto!)
GPU buffer:
  m_ref = m_dev = m  (initially same)
  am, bm, cm (fluxes per half_dt)
    |
    v  (no runtime scaling for preprocessed binary!)
    |
    v  finalize_ic_phase! (window 1 only)
tracers: VMR -> rm = VMR * m_ref  [kg, moist basis]
    |
    v  advection_phase!
    |  For each substep:
    |    [v4: interpolate am/bm/cm to substep midpoint]
    |    strang_split_massflux!: X-Y-Z-Z-Y-X
    |      Each direction: CFL check -> n_sub -> uniform subdivision -> kernel launch
    |      Kernel: van Leer slopes (minmod), upwind flux, double-buffer
    |      Multi-tracer: m_save/restore per tracer
    |  copyto!(m_ref, m_dev)  -- sync reference to evolved
    |  convect! (once per window): rm/m_ref -> VMR -> convect -> VMR*m_ref -> rm
    |
    v  apply_emissions_phase!
    |  rm -> VMR -> apply flux -> VMR -> rm  (using m_ref for roundtrip)
    |
    v  post_advection_physics!
    |  rm -> VMR -> BLD -> PBL diffusion -> chemistry -> VMR -> rm
    |
    v  output
       rm / (m_ref * (1-QV)) = c_dry [mol/mol]  -- dry VMR for output
```

---

## Critical Invariants for the LL Path

1. **No runtime flux scaling**: Preprocessed binary am/bm/cm are stored per half_dt.
   The preprocessor handles the scaling. n_sub Strang cycles accumulate correctly.

2. **Moist transport basis**: rm = c * m_moist. Dry correction ONLY at output time.

3. **cm boundaries**: cm[:,:,1]=0 (TOA), cm[:,:,Nz+1]=0 (surface). Enforced after load.

4. **Pole zeroing**: am[:, 1, :] = am[:, Ny, :] = 0. Prevents zonal transport at poles.

5. **Z-advection double buffer**: Kernel reads from rm, writes to rm_buf. Then copyto!(rm, rm_buf).
   Never writes in-place. Violation causes ~10% mass loss per step.

6. **Multi-tracer m restore**: Mass m is saved before first tracer, restored before each
   subsequent tracer. All tracers see the same starting mass field for their Strang sweep.

7. **m_ref sync after advection**: copyto!(m_ref, m_dev) after all substeps. This ensures
   convection and emission rm<->c conversions use the EVOLVED mass, not prescribed.

8. **CFL subcycling cap**: n_sub capped at 50 per direction to prevent infinite loops.

9. **Convection timing**: One convection call per window (dt_conv = dt_window),
   NOT per substep. Applied after all substeps complete.
