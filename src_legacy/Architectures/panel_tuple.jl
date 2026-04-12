# ---------------------------------------------------------------------------
# PanelTuple — GPU-aware panel container for cubed-sphere grids
#
# Wraps NTuple{6,T} with an AbstractPanelMap so that map/reduce operations
# automatically dispatch to the correct GPU. Drop-in replacement for
# NTuple{6} — supports indexing, iteration, and destructuring.
#
# SingleGPUMap methods compile to bare loops (zero overhead).
# PanelGPUMap methods (in ext/AtmosTransportCUDAExt.jl) use Threads.@spawn
# for concurrent multi-GPU execution.
# ---------------------------------------------------------------------------

using KernelAbstractions: get_backend, synchronize

"""
    PanelTuple{N, T, M <: AbstractPanelMap}

GPU-aware container for cubed-sphere panel data.
Carries its `panel_map` so that `map_panels!` and `reduce_panels`
automatically group panels by GPU.

Supports `pt[p]` indexing, `for p in eachindex(pt)`, and `length(pt)`.
"""
struct PanelTuple{N, T, M <: AbstractPanelMap}
    data :: NTuple{N, T}
    panel_map :: M
end

# --- Construction helpers ---

"""Wrap an existing NTuple{6} with a panel map."""
PanelTuple(data::NTuple{6}, pm::AbstractPanelMap) = PanelTuple{6, eltype(data), typeof(pm)}(data, pm)
PanelTuple(data::NTuple{6}) = PanelTuple(data, SingleGPUMap())

"""Allocate a PanelTuple via a factory function (uses allocate_ntuple_panels)."""
function PanelTuple(pm::AbstractPanelMap, f::Function)
    data = allocate_ntuple_panels(pm, f)
    PanelTuple(data, pm)
end

# --- AbstractArray-like interface ---

Base.getindex(pt::PanelTuple, i::Int) = pt.data[i]
Base.length(::PanelTuple{N}) where N = N
Base.firstindex(::PanelTuple) = 1
Base.lastindex(pt::PanelTuple{N}) where N = N
Base.eachindex(pt::PanelTuple{N}) where N = Base.OneTo(N)
Base.iterate(pt::PanelTuple, state=1) = state > length(pt) ? nothing : (pt[state], state + 1)
Base.eltype(::Type{PanelTuple{N,T,M}}) where {N,T,M} = T
Base.keys(pt::PanelTuple{N}) where N = Base.OneTo(N)

# Allow NTuple-like destructuring and conversion
Base.Tuple(pt::PanelTuple) = pt.data
Base.convert(::Type{NTuple{N,T}}, pt::PanelTuple{N,T}) where {N,T} = pt.data

# --- Core operations ---

"""
    map_panels!(f, panels::PanelTuple)
    map_panels!(f, p1::PanelTuple, p2, p3, ...)

Apply `f(p, panel_data_p, ...)` for each panel index `p`, grouped by GPU.
Synchronizes each GPU after its panels complete.

For `SingleGPUMap`: bare sequential loop + single sync (zero overhead).
For `PanelGPUMap`: concurrent execution via `Threads.@spawn` (in CUDA ext).

`f` receives the panel INDEX as first arg, then the panel data from each PanelTuple.
Extra arguments that are not PanelTuple are passed through unchanged.
"""
function map_panels!(f::F, pt::PanelTuple{N,T,SingleGPUMap}) where {F,N,T}
    for p in 1:N
        f(p, pt[p])
    end
    synchronize(get_backend(pt[1]))
    nothing
end

"""Multi-arg map_panels! — indexes into all PanelTuple args."""
function map_panels!(f::F, pt1::PanelTuple{N,T,SingleGPUMap}, pt2::PanelTuple) where {F,N,T}
    for p in 1:N
        f(p, pt1[p], pt2[p])
    end
    synchronize(get_backend(pt1[1]))
    nothing
end

function map_panels!(f::F, pt1::PanelTuple{N,T,SingleGPUMap},
                     pt2::PanelTuple, pt3::PanelTuple) where {F,N,T}
    for p in 1:N
        f(p, pt1[p], pt2[p], pt3[p])
    end
    synchronize(get_backend(pt1[1]))
    nothing
end

# Variadic fallback for 4+ PanelTuples (slightly less efficient due to tuple indexing)
function map_panels!(f::F, pt1::PanelTuple{N,T,SingleGPUMap},
                     rest::PanelTuple...) where {F,N,T}
    for p in 1:N
        args = (pt[p] for pt in (pt1, rest...))
        f(p, args...)
    end
    synchronize(get_backend(pt1[1]))
    nothing
end

# PanelGPUMap versions are defined in ext/AtmosTransportCUDAExt.jl

"""
    map_panels_nosync!(f, panels::PanelTuple)

Like `map_panels!` but does NOT synchronize. Use when caller manages sync
(e.g., between phases of fv_tp_2d where halo exchange provides the barrier).
"""
function map_panels_nosync!(f::F, pt::PanelTuple{N}) where {F,N}
    for p in 1:N
        f(p, pt[p])
    end
    nothing
end

"""
    reduce_panels(f, op, panels::PanelTuple; init)

Apply `f(panel_data_p)` per panel, reduce with `op`. Returns scalar on CPU.
Synchronizes all GPUs before reduction (GPU→CPU transfer in `f` implies sync).

Example: `max_cfl = reduce_panels(p -> max_cfl_x(panels[p], ...), max, panels; init=0f0)`
"""
function reduce_panels(f::F, op::Op, pt::PanelTuple{N}; init) where {F,Op,N}
    acc = init
    for p in 1:N
        acc = op(acc, f(p))
    end
    return acc
end

"""
    copyto_panels!(dst::PanelTuple, src::PanelTuple)
    copyto_panels!(dst::PanelTuple, src)  # src can be NTuple

Panel-wise `copyto!`. No sync (copyto! on CuArray is synchronous).
"""
function copyto_panels!(dst::PanelTuple{N}, src::PanelTuple{N}) where N
    for p in 1:N
        copyto!(dst[p], src[p])
    end
    nothing
end

function copyto_panels!(dst::PanelTuple{N}, src::NTuple{N}) where N
    for p in 1:N
        copyto!(dst[p], src[p])
    end
    nothing
end

function copyto_panels!(dst::NTuple{N}, src::PanelTuple{N}) where N
    for p in 1:N
        copyto!(dst[p], src[p])
    end
    nothing
end

"""
    broadcast_panels!(op, panels::PanelTuple, scalar)

Apply `panels[p] .op= scalar` for each panel.
"""
function broadcast_panels!(::typeof(*), pt::PanelTuple{N}, scalar) where N
    for p in 1:N
        pt[p] .*= scalar
    end
    nothing
end

"""
    sync_panels!(panels::PanelTuple)

Synchronize all GPUs that own panels in this tuple.
"""
function sync_panels!(pt::PanelTuple{N,T,SingleGPUMap}) where {N,T}
    synchronize(get_backend(pt[1]))
    nothing
end

# PanelGPUMap version in CUDA ext

"""
    fill_panels!(panels::PanelTuple, val)

Fill all panels with `val`.
"""
function fill_panels!(pt::PanelTuple{N}, val) where N
    for p in 1:N
        fill!(pt[p], val)
    end
    nothing
end

"""Extract the panel map from a PanelTuple."""
panel_map(pt::PanelTuple) = pt.panel_map
