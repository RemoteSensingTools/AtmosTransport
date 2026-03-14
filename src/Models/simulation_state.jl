# ---------------------------------------------------------------------------
# Simulation state allocation — dispatched on grid type
#
# Provides factory functions that allocate all pre-loop buffers (physics,
# emissions, air mass, geometry, workspace) for both LL and CS grids.
# ---------------------------------------------------------------------------

# =====================================================================
# Physics buffers — allocate convection, diffusion, QV, PBL arrays
# =====================================================================

"""
    allocate_physics_buffers(grid, arch, model) → NamedTuple

Allocate all physics auxiliary arrays (CMFMC, DTRAIN, QV, PBL surface fields,
scratch arrays) on CPU and GPU. Dispatches on grid type.
"""
function allocate_physics_buffers(grid::LatitudeLongitudeGrid{FT}, arch, model) where FT
    AT = array_type(arch)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    has_conv = _needs_convection(model.convection)
    has_dtrain = _needs_dtrain(model.convection)
    has_pbl = _needs_pbl(model.diffusion)
    needs_delp = has_conv || has_pbl

    cmfmc_cpu = has_conv ? Array{FT}(undef, Nx, Ny, Nz + 1) : nothing
    cmfmc_gpu = has_conv ? AT(zeros(FT, Nx, Ny, Nz + 1)) : nothing

    dtrain_cpu = has_dtrain ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    dtrain_gpu = has_dtrain ? AT(zeros(FT, Nx, Ny, Nz)) : nothing
    ras_workspace = model.convection isa RASConvection ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    pbl_sfc_cpu = has_pbl ? (
        pblh  = Array{FT}(undef, Nx, Ny),
        ustar = Array{FT}(undef, Nx, Ny),
        hflux = Array{FT}(undef, Nx, Ny),
        t2m   = Array{FT}(undef, Nx, Ny)) : nothing
    pbl_sfc_gpu = has_pbl ? (
        pblh  = AT(zeros(FT, Nx, Ny)),
        ustar = AT(zeros(FT, Nx, Ny)),
        hflux = AT(zeros(FT, Nx, Ny)),
        t2m   = AT(zeros(FT, Nx, Ny))) : nothing
    w_scratch = has_pbl ? AT(zeros(FT, Nx, Ny, Nz)) : nothing

    delp_cpu = needs_delp ? Array{FT}(undef, Nx, Ny, Nz) : nothing
    area_j = needs_delp ? FT[cell_area(1, j, grid) for j in 1:Ny] : nothing

    qv_cpu = Array{FT}(undef, Nx, Ny, Nz)
    qv_gpu = AT(zeros(FT, Nx, Ny, Nz))
    m_dry = AT(zeros(FT, Nx, Ny, Nz))

    planet = (has_conv || has_pbl) ? load_parameters(FT).planet : nothing

    return (;
        has_conv, has_dtrain, has_pbl, needs_delp,
        cmfmc_cpu, cmfmc_gpu,
        dtrain_cpu, dtrain_gpu, ras_workspace,
        pbl_sfc_cpu, pbl_sfc_gpu, w_scratch,
        delp_cpu, area_j,
        qv_cpu, qv_gpu, qv_loaded=Ref(false), m_dry,
        planet,
        cmfmc_loaded=Ref(false), dtrain_loaded=Ref(false), sfc_loaded=Ref(false),
        troph_loaded=Ref(false)
    )
end

function allocate_physics_buffers(grid::CubedSphereGrid{FT}, arch, model;
                                   Hp::Int=3) where FT
    AT = array_type(arch)
    Nc, Nz = grid.Nc, grid.Nz

    has_conv = _needs_convection(model.convection)
    has_dtrain = _needs_dtrain(model.convection)
    has_pbl = _needs_pbl(model.diffusion)

    cmfmc_gpu = has_conv ?
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1)), 6) : nothing
    cmfmc_cpu = has_conv ?
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz + 1), 6) : nothing

    dtrain_gpu = has_dtrain ?
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6) : nothing
    dtrain_cpu = has_dtrain ?
        ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6) : nothing

    qv_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    qv_gpu = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)

    ras_workspace = model.convection isa RASConvection ?
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6) : nothing

    pbl_sfc_cpu = has_pbl ? (
        pblh  = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
        ustar = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
        hflux = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6),
        t2m   = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)) : nothing
    pbl_sfc_gpu = has_pbl ? (
        pblh  = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
        ustar = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
        hflux = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6),
        t2m   = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp)), 6)) : nothing
    w_scratch = has_pbl ?
        ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6) : nothing

    troph_cpu = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)
    ps_cpu    = ntuple(_ -> zeros(FT, Nc + 2Hp, Nc + 2Hp), 6)

    planet = load_parameters(FT).planet

    return (;
        has_conv, has_dtrain, has_pbl,
        cmfmc_cpu, cmfmc_gpu,
        dtrain_cpu, dtrain_gpu, ras_workspace,
        pbl_sfc_cpu, pbl_sfc_gpu, w_scratch,
        qv_cpu, qv_gpu, qv_loaded=Ref(false),
        troph_cpu, ps_cpu,
        planet,
        cmfmc_loaded=Ref(false), dtrain_loaded=Ref(false),
        sfc_loaded=Ref(false), troph_loaded=Ref(false)
    )
end

# =====================================================================
# Tracer allocation — CS needs per-panel arrays + deferred IC
# =====================================================================

"""
    allocate_tracers(model, grid) → tracers NamedTuple

For LL grids, returns model.tracers as-is (already GPU arrays).
For CS grids, allocates 6-panel NTuple per tracer and applies pending ICs.
"""
allocate_tracers(model, grid::LatitudeLongitudeGrid) = model.tracers

function allocate_tracers(model, grid::CubedSphereGrid)
    Nz = grid.Nz
    tracer_names = keys(model.tracers)
    n_tracers = length(tracer_names)
    cs_tracers = NamedTuple{tracer_names}(
        ntuple(_ -> allocate_cubed_sphere_field(grid, Nz), n_tracers)
    )
    pending_ic = get_pending_ic()
    if !isempty(pending_ic.entries)
        apply_pending_ic!(cs_tracers, pending_ic, grid)
    end
    return cs_tracers
end

# =====================================================================
# Air mass allocation
# =====================================================================

"""
    allocate_air_mass(grid, arch) → NamedTuple

Allocate air mass arrays. Returns NamedTuple with m (working) and m_ref (reference).
For CS, also includes dm_per_sub for pressure fixer.
"""
function allocate_air_mass(grid::LatitudeLongitudeGrid{FT}, arch) where FT
    # LL uses m_ref and m_dev from the met buffer — no separate allocation needed
    return nothing
end

function allocate_air_mass(grid::CubedSphereGrid{FT}, arch) where FT
    Nz = grid.Nz
    m       = allocate_cubed_sphere_field(grid, Nz)
    m_ref   = allocate_cubed_sphere_field(grid, Nz)
    m_wet   = allocate_cubed_sphere_field(grid, Nz)  # MOIST air mass for convection
    return (; m, m_ref, m_wet)
end

# =====================================================================
# Geometry + workspace allocation
# =====================================================================

"""Allocate geometry cache and advection workspace, dispatched on grid."""
function allocate_geometry_and_workspace(grid::LatitudeLongitudeGrid, arch;
                                          use_linrood::Bool=false,
                                          use_vertical_remap::Bool=false)
    # LL uses workspace from met buffer (gpu_buf.ws)
    return (; gc=nothing, ws=nothing, ws_lr=nothing, ws_vr=nothing)
end

function allocate_geometry_and_workspace(grid::CubedSphereGrid{FT}, arch;
                                          Hp::Int=3, use_linrood::Bool=false,
                                          use_vertical_remap::Bool=false) where FT
    AT = array_type(arch)
    Nc, Nz = grid.Nc, grid.Nz
    ref_panel = AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz))
    gc = build_geometry_cache(grid, ref_panel)
    ws = allocate_cs_massflux_workspace(ref_panel, Nc)
    ws_lr = use_linrood ? LinRoodWorkspace(grid) : nothing
    ws_vr = use_vertical_remap ? VerticalRemapWorkspace(grid, arch) : nothing
    return (; gc, ws, ws_lr, ws_vr)
end

# =====================================================================
# Emission preparation — dispatched on grid type
# =====================================================================

"""Prepare emission data on device, dispatched on grid type."""
function prepare_emissions(sources, grid::LatitudeLongitudeGrid, driver, arch)
    _prepare_latlon_emissions(sources, grid, driver, arch)
end

function prepare_emissions(sources, grid::CubedSphereGrid, driver, arch)
    _prepare_cs_emissions(sources, grid, arch), nothing, nothing, nothing
end
