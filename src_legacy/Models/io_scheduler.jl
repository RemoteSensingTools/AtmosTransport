# ---------------------------------------------------------------------------
# IOScheduler — abstracts SingleBuffer vs DoubleBuffer met-data IO scheduling
#
# SingleBuffer: load met → upload → compute → output (sequential)
# DoubleBuffer: pre-load win 1, then overlap CPU I/O with GPU compute
#
# Parametric on buffering strategy B and buffer types GM/CM.
# No `Any` fields — all concretely typed via parametric types.
# ---------------------------------------------------------------------------

# =====================================================================
# IOScheduler type
# =====================================================================

"""
    IOScheduler{B, GM, CM}

Abstracts met-data I/O scheduling for the unified run loop.

- `B <: AbstractBufferingStrategy` — SingleBuffer or DoubleBuffer
- `GM` — GPU met buffer type (LatLonMetBuffer or CubedSphereMetBuffer)
- `CM` — CPU staging buffer type (LatLonCPUBuffer or CubedSphereCPUBuffer)

For SingleBuffer: only `gpu_a`/`cpu_a` are used; `gpu_b`/`cpu_b` are identical refs.
For DoubleBuffer: two separate buffer pairs alternate via `current` toggle.
"""
mutable struct IOScheduler{B <: AbstractBufferingStrategy, GM, CM}
    const strategy :: B
    const gpu_a    :: GM
    const gpu_b    :: GM
    const cpu_a    :: CM
    const cpu_b    :: CM
    current        :: Symbol        # :a or :b
    load_task      :: Union{Task, Nothing}       # met-only (DELP, am, bm, cx, cy, xfx, yfx)
    phys_task      :: Union{Task, Nothing}       # physics (CMFMC, DTRAIN, QV, surface)
    io_result      :: Union{NamedTuple, Nothing}  # return value from load_all_window!
end

# =====================================================================
# Buffer accessors
# =====================================================================

@inline _other(s::Symbol) = ifelse(s === :a, :b, :a)

@inline current_gpu(s::IOScheduler) = ifelse(s.current === :a, s.gpu_a, s.gpu_b)
@inline current_cpu(s::IOScheduler) = ifelse(s.current === :a, s.cpu_a, s.cpu_b)
@inline next_gpu(s::IOScheduler)    = ifelse(s.current === :a, s.gpu_b, s.gpu_a)
@inline next_cpu(s::IOScheduler)    = ifelse(s.current === :a, s.cpu_b, s.cpu_a)

# =====================================================================
# Construction — dispatched on grid type and buffering strategy
# =====================================================================

"""
    build_io_scheduler(grid, arch, buffering) → IOScheduler

Allocate met buffers and construct the IOScheduler.
"""
function build_io_scheduler(grid::LatitudeLongitudeGrid{FT}, arch,
                             ::SingleBuffer; use_gchp::Bool=false,
                             panel_map::AbstractPanelMap=SingleGPUMap(),
                             flux_delta::Bool=false) where FT
    cs_cpu = _get_cluster_sizes_cpu(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    gpu = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu, flux_delta)
    cpu = LatLonCPUBuffer(FT, Nx, Ny, Nz; flux_delta)
    IOScheduler(SingleBuffer(), gpu, gpu, cpu, cpu, :a, nothing, nothing, nothing)
end

function build_io_scheduler(grid::LatitudeLongitudeGrid{FT}, arch,
                             ::DoubleBuffer; use_gchp::Bool=false,
                             panel_map::AbstractPanelMap=SingleGPUMap(),
                             flux_delta::Bool=false) where FT
    cs_cpu = _get_cluster_sizes_cpu(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    gpu_a = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu, flux_delta)
    gpu_b = LatLonMetBuffer(arch, FT, Nx, Ny, Nz; cluster_sizes_cpu=cs_cpu, flux_delta)
    cpu_a = LatLonCPUBuffer(FT, Nx, Ny, Nz; flux_delta)
    cpu_b = LatLonCPUBuffer(FT, Nx, Ny, Nz; flux_delta)
    IOScheduler(DoubleBuffer(), gpu_a, gpu_b, cpu_a, cpu_b, :a, nothing, nothing, nothing)
end

function build_io_scheduler(grid::CubedSphereGrid{FT}, arch,
                             ::SingleBuffer;
                             Hp::Int=3, use_gchp::Bool=false,
                             panel_map::AbstractPanelMap=SingleGPUMap()) where FT
    Nc, Nz = grid.Nc, grid.Nz
    gpu = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp; use_gchp, panel_map)
    cpu = CubedSphereCPUBuffer(FT, Nc, Nz, Hp; use_gchp)
    IOScheduler(SingleBuffer(), gpu, gpu, cpu, cpu, :a, nothing, nothing, nothing)
end

function build_io_scheduler(grid::CubedSphereGrid{FT}, arch,
                             ::DoubleBuffer;
                             Hp::Int=3, use_gchp::Bool=false,
                             panel_map::AbstractPanelMap=SingleGPUMap()) where FT
    Nc, Nz = grid.Nc, grid.Nz
    gpu_a = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp; use_gchp, panel_map)
    gpu_b = CubedSphereMetBuffer(arch, FT, Nc, Nz, Hp; use_gchp, panel_map)
    cpu_a = CubedSphereCPUBuffer(FT, Nc, Nz, Hp; use_gchp)
    cpu_b = CubedSphereCPUBuffer(FT, Nc, Nz, Hp; use_gchp)
    IOScheduler(DoubleBuffer(), gpu_a, gpu_b, cpu_a, cpu_b, :a, nothing, nothing, nothing)
end

# =====================================================================
# Load operations — dispatch on buffering strategy
# =====================================================================

# --- SingleBuffer: synchronous load into the single buffer pair ---

"""Load met data synchronously for a lat-lon SingleBuffer scheduler."""
function begin_load!(sched::IOScheduler{SingleBuffer}, driver,
                      grid::LatitudeLongitudeGrid, w::Int; kwargs...)
    load_met_window!(sched.cpu_a, driver, grid, w)
    sched.io_result = nothing
    return nothing
end

"""Load all data synchronously for a CS SingleBuffer scheduler."""
function begin_load!(sched::IOScheduler{SingleBuffer}, driver,
                      grid::CubedSphereGrid, w::Int;
                      cmfmc_cpu=nothing, dtrain_cpu=nothing,
                      sfc_cpu=nothing, troph_cpu=nothing,
                      needs_cmfmc::Bool=false, needs_dtrain::Bool=false,
                      needs_sfc::Bool=false, needs_qv::Bool=false,
                      qv_cpu=nothing, ps_panels=nothing,
                      qv_next_cpu=nothing, ps_next_panels=nothing)
    result = load_all_window!(sched.cpu_a, cmfmc_cpu, dtrain_cpu, sfc_cpu, troph_cpu,
                               driver, grid, w;
                               needs_cmfmc, needs_dtrain, needs_sfc,
                               needs_qv, qv_cpu, ps_panels,
                               qv_next_cpu, ps_next_panels)
    sched.io_result = result
    return nothing
end

# --- DoubleBuffer: async load into the "next" buffer pair ---

"""Load met data synchronously for initial window (DoubleBuffer LL)."""
function initial_load!(sched::IOScheduler{DoubleBuffer}, driver,
                        grid::LatitudeLongitudeGrid, w::Int; kwargs...)
    load_met_window!(current_cpu(sched), driver, grid, w)
    sched.io_result = nothing
    return nothing
end

"""Load all data synchronously for initial window (DoubleBuffer CS)."""
function initial_load!(sched::IOScheduler{DoubleBuffer}, driver,
                        grid::CubedSphereGrid, w::Int;
                        cmfmc_cpu=nothing, dtrain_cpu=nothing,
                        sfc_cpu=nothing, troph_cpu=nothing,
                        needs_cmfmc::Bool=false, needs_dtrain::Bool=false,
                        needs_sfc::Bool=false, needs_qv::Bool=false,
                        qv_cpu=nothing, ps_panels=nothing,
                        qv_next_cpu=nothing, ps_next_panels=nothing)
    result = load_all_window!(current_cpu(sched), cmfmc_cpu, dtrain_cpu,
                               sfc_cpu, troph_cpu, driver, grid, w;
                               needs_cmfmc, needs_dtrain, needs_sfc,
                               needs_qv, qv_cpu, ps_panels,
                               qv_next_cpu, ps_next_panels)
    sched.io_result = result
    return nothing
end

"""SingleBuffer initial_load! (just delegates to begin_load!)."""
function initial_load!(sched::IOScheduler{SingleBuffer}, driver, grid, w::Int;
                        kwargs...)
    begin_load!(sched, driver, grid, w; kwargs...)
end

"""Spawn async load for next window (DoubleBuffer LL)."""
function begin_load!(sched::IOScheduler{DoubleBuffer}, driver,
                      grid::LatitudeLongitudeGrid, w::Int; kwargs...)
    nc = next_cpu(sched)
    sched.load_task = Threads.@spawn begin
        load_met_window!(nc, driver, grid, w)
        nothing
    end
    return nothing
end

"""Spawn split async loads for next window (DoubleBuffer CS).
Task 1 (load_task): met-only (DELP, am, bm, cx, cy, xfx, yfx) — fast, needed for advection.
Task 2 (phys_task): physics (CMFMC, DTRAIN, QV, surface) — slower, needed at next window start."""
function begin_load!(sched::IOScheduler{DoubleBuffer}, driver,
                      grid::CubedSphereGrid, w::Int;
                      cmfmc_cpu=nothing, dtrain_cpu=nothing,
                      sfc_cpu=nothing, troph_cpu=nothing,
                      needs_cmfmc::Bool=false, needs_dtrain::Bool=false,
                      needs_sfc::Bool=false, needs_qv::Bool=false,
                      qv_cpu=nothing, ps_panels=nothing,
                      qv_next_cpu=nothing, ps_next_panels=nothing)
    nc = next_cpu(sched)
    # Task 1: met-only (fast path — DELP + fluxes from mmap binary)
    sched.load_task = Threads.@spawn begin
        load_met_window!(nc, driver, grid, w)
        nothing
    end
    # Task 2: physics fields (CMFMC, DTRAIN, QV, surface — can overlap with GPU compute)
    sched.phys_task = Threads.@spawn load_physics_window!(
        cmfmc_cpu, dtrain_cpu, sfc_cpu, troph_cpu,
        driver, grid, w;
        needs_cmfmc, needs_dtrain, needs_sfc,
        needs_qv, qv_cpu, ps_panels,
        qv_next_cpu, ps_next_panels)
    return nothing
end

# =====================================================================
# Wait and swap operations
# =====================================================================

"""No-op wait for SingleBuffer."""
wait_load!(::IOScheduler{SingleBuffer}) = nothing

"""Wait for async met load to complete (DoubleBuffer)."""
function wait_load!(sched::IOScheduler{DoubleBuffer})
    sched.load_task === nothing && return nothing
    fetch(sched.load_task)
    sched.load_task = nothing
    return nothing
end

"""Wait for async physics load to complete (DoubleBuffer). Stores result."""
function wait_phys!(sched::IOScheduler{DoubleBuffer})
    sched.phys_task === nothing && return nothing
    result = fetch(sched.phys_task)
    sched.io_result = result isa NamedTuple ? result : nothing
    sched.phys_task = nothing
    return nothing
end
wait_phys!(::IOScheduler{SingleBuffer}) = nothing

"""No-op swap for SingleBuffer."""
swap!(::IOScheduler{SingleBuffer}) = nothing

"""Toggle current buffer for DoubleBuffer."""
function swap!(sched::IOScheduler{DoubleBuffer})
    sched.current = _other(sched.current)
    return nothing
end

# =====================================================================
# Upload: transfer current CPU buffer to current GPU buffer
# =====================================================================

"""Upload current CPU met buffer to GPU."""
function upload_met!(sched::IOScheduler)
    upload!(current_gpu(sched), current_cpu(sched))
end
