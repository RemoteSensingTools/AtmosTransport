# ---------------------------------------------------------------------------
# Multi-GPU panel-split types and abstractions (CUDA-free core)
#
# Concrete CUDA implementations (CUDA.device!, CUDA.synchronize) live in
# ext/AtmosTransportCUDAExt.jl to maintain the weak-dependency pattern.
#
# Key insight: CUDA.jl + KernelAbstractions auto-route kernels to the GPU
# that owns the output array. Most `for p in 1:6` loops work unchanged.
# Only allocation, workspace duplication, and sync points need explicit handling.
# ---------------------------------------------------------------------------

# =====================================================================
# Abstract type
# =====================================================================

"""Supertype for panel-to-GPU mapping strategies."""
abstract type AbstractPanelMap end

# =====================================================================
# SingleGPUMap — zero-overhead sentinel for n_gpus=1
# =====================================================================

"""
    SingleGPUMap <: AbstractPanelMap

Sentinel for single-GPU execution. All multi-GPU operations are no-ops.
Compiler constant-folds `workspace_for(pgw, ::SingleGPUMap, p)` to a direct access.
"""
struct SingleGPUMap <: AbstractPanelMap end

@inline gpu_for_panel(::SingleGPUMap, ::Int) = 0
@inline is_cross_gpu(::SingleGPUMap, ::Int, ::Int) = false
@inline is_multi_gpu(::SingleGPUMap) = false
@inline n_gpus(::SingleGPUMap) = 1

# No-op sync, allocation, and P2P for single GPU
sync_all_gpus(::SingleGPUMap) = nothing
allocate_ntuple_panels(::SingleGPUMap, f::Function) = ntuple(f, 6)
enable_p2p!(::SingleGPUMap) = nothing

# =====================================================================
# PanelGPUMap — multi-GPU panel assignment (struct only, no CUDA calls)
# =====================================================================

"""
    PanelGPUMap <: AbstractPanelMap

Maps each of the 6 cubed-sphere panels to a GPU device (0-based).
Default split: panels 1,2,3 → GPU 0, panels 4,5,6 → GPU 1.

CUDA-specific methods (`sync_all_gpus`, `allocate_ntuple_panels`) are in the
CUDA extension. This struct is CUDA-free.
"""
struct PanelGPUMap <: AbstractPanelMap
    panel_to_gpu  :: NTuple{6, Int}   # panel → GPU id (0-based)
    n_gpus        :: Int
    groups        :: Vector{Any}  # Pre-computed panel tuples per GPU (cached, NTuple{K,Int})
end

function PanelGPUMap(panel_to_gpu::NTuple{6, Int}, n_gpus::Int)
    groups = [tuple((p for p in 1:6 if panel_to_gpu[p] == g)...)
              for g in 0:(n_gpus - 1)]
    PanelGPUMap(panel_to_gpu, n_gpus, groups)
end

@inline gpu_for_panel(m::PanelGPUMap, p::Int) = m.panel_to_gpu[p]
@inline is_cross_gpu(m::PanelGPUMap, p::Int, q::Int) = m.panel_to_gpu[p] != m.panel_to_gpu[q]
@inline is_multi_gpu(::PanelGPUMap) = true
@inline n_gpus(m::PanelGPUMap) = m.n_gpus

"""Return list of panel indices assigned to GPU `gpu_id` (0-based)."""
@inline function panels_on_gpu(m::PanelGPUMap, gpu_id::Int)
    [p for p in 1:6 if m.panel_to_gpu[p] == gpu_id]
end

# =====================================================================
# Task-local panel map — set once in run loop, used everywhere
# Avoids threading panel_map through 100+ function signatures.
# =====================================================================

const _ACTIVE_PANEL_MAP = Ref{AbstractPanelMap}(SingleGPUMap())

"""Set the active panel map for the current simulation. Enables P2P for multi-GPU."""
function set_panel_map!(pm::AbstractPanelMap)
    _ACTIVE_PANEL_MAP[] = pm
    enable_p2p!(pm)
    nothing
end

"""Get the active panel map. Returns SingleGPUMap if not set."""
active_panel_map() = _ACTIVE_PANEL_MAP[]

"""
    for_panels(f)

Execute `f(p)` for panels 1-6, then sync all GPUs.

For SingleGPUMap: sequential launch + 1 sync (zero overhead).
For PanelGPUMap (Strategy E): sequential device switch per GPU, async kernel queuing.
GPU 0's kernels continue executing while GPU 1's kernels are launched. Final sync
waits for all GPUs. No threading needed — 2× speedup via CUDA async dispatch.
"""
function for_panels(f::F) where F
    pm = _ACTIVE_PANEL_MAP[]
    foreach_gpu_batch(pm) do _, panels
        for p in panels
            f(p)
        end
    end
    nothing
end

"""
    for_panels_nosync(f)

Execute `f(p)` for panels 1-6 WITHOUT a sync barrier at the end.
Use for independent per-panel operations (broadcasts, copyto!, per-panel kernels)
where no cross-panel data dependency exists. Caller must add an explicit
`sync_all_gpus(active_panel_map())` before any operation that reads across panels.

For SingleGPUMap: identical to `for_panels()` (both are zero-overhead).
For PanelGPUMap: device switches + kernel launches, but no CUDA.synchronize().
"""
function for_panels_nosync(f::F) where F
    pm = _ACTIVE_PANEL_MAP[]
    foreach_gpu_batch_nosync(pm) do _, panels
        for p in panels
            f(p)
        end
    end
    nothing
end

# Fallback: same as foreach_gpu_batch for non-CUDA backends
@inline _foreach_gpu_nosync(f::F, ::SingleGPUMap) where F = (f(0, ntuple(identity, 6)); nothing)
@inline _foreach_gpu_nosync(f::F, m::AbstractPanelMap) where F = foreach_gpu_batch(f, m)

"""
    foreach_gpu_batch_nosync(f, panel_map)

Execute `f(gpu_id, panel_ids)` for each GPU group without a terminal barrier.
Caller is responsible for invoking `sync_all_gpus(panel_map)` before any cross-panel
read or host-side reduction that depends on queued work.
"""
function foreach_gpu_batch_nosync(f::F, panel_map::AbstractPanelMap) where F
    _foreach_gpu_nosync(f, panel_map)
end


# Generic fallback for any AbstractPanelMap — no-op sync and simple allocation.
# The CUDA extension adds CUDA-specific methods for PanelGPUMap that call
# CUDA.device!() and CUDA.synchronize(). These don't conflict because they
# import and extend (not redefine) the functions.
sync_all_gpus(::AbstractPanelMap) = nothing
allocate_ntuple_panels(::AbstractPanelMap, f::Function) = ntuple(f, 6)
enable_p2p!(::AbstractPanelMap) = nothing

"""
    foreach_gpu_batch(f, panel_map)

Execute `f(gpu_id, panel_ids)` for each GPU's panel group.
- `SingleGPUMap`: calls `f(0, 1:6)` — zero overhead, no device switching.
- `PanelGPUMap`: groups panels by GPU, switches device once per group.

The function `f` receives 0-based `gpu_id` and a vector of panel indices.
Kernels launched inside `f` should NOT synchronize between panels —
the caller handles sync after all panels on each GPU are launched.
"""
@inline function foreach_gpu_batch(f::F, ::SingleGPUMap) where F
    f(0, ntuple(identity, 6))
    nothing
end

# PanelGPUMap version is in ext/AtmosTransportCUDAExt.jl (needs CUDA.device!)

# =====================================================================
# PerGPUWorkspace — one workspace instance per GPU
# =====================================================================

"""
    PerGPUWorkspace{W, M <: AbstractPanelMap}

Holds one workspace instance per GPU. `workspace_for(pgw, p)` returns
the workspace on panel `p`'s GPU.

For `SingleGPUMap`, stores a single workspace — `workspace_for` compiles
to direct field access (zero overhead).
"""
struct PerGPUWorkspace{W, M <: AbstractPanelMap}
    workspaces :: Vector{W}  # indexed by gpu_id + 1
    panel_map  :: M
end

"""Get the workspace for panel p's GPU."""
@inline function workspace_for(pgw::PerGPUWorkspace{W, SingleGPUMap}, p::Int) where W
    @inbounds pgw.workspaces[1]
end
@inline function workspace_for(pgw::PerGPUWorkspace{W, PanelGPUMap}, p::Int) where W
    gpu_id = pgw.panel_map.panel_to_gpu[p]
    @inbounds pgw.workspaces[gpu_id + 1]
end

"""Construct PerGPUWorkspace for single GPU."""
function PerGPUWorkspace(::SingleGPUMap, f::Function)
    PerGPUWorkspace([f(0)], SingleGPUMap())
end

# Multi-GPU PerGPUWorkspace(::PanelGPUMap, f) is in ext/AtmosTransportCUDAExt.jl

# =====================================================================
# Build panel map from config
# =====================================================================

"""Mapping table for N GPUs."""
function _panel_mapping(ng::Int)
    ng == 1 && return (0, 0, 0, 0, 0, 0)
    ng == 2 && return (0, 0, 0, 1, 1, 1)
    ng == 3 && return (0, 0, 1, 1, 2, 2)
    ng >= 6 && return ntuple(i -> i - 1, 6)
    return ntuple(i -> (i - 1) % ng, 6)
end

"""
    build_panel_map(config) → AbstractPanelMap

Construct a panel map from the model configuration Dict.
Returns `SingleGPUMap()` if `n_gpus ≤ 1` or not specified.
"""
function build_panel_map(config::Dict)
    arch_cfg = get(config, "architecture", Dict())
    ng = get(arch_cfg, "n_gpus", 1)
    ng <= 1 ? SingleGPUMap() : PanelGPUMap(_panel_mapping(ng), ng)
end
build_panel_map(::Nothing) = SingleGPUMap()
