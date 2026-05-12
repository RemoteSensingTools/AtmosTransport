# ---------------------------------------------------------------------------
# Memory-mapped, on-disk cubed-sphere tape storage policy.
#
# Plan 26 Phase A.1 — `MmapCSTapeStorage` is a sibling of
# `DeviceCSTapeStorage` and `PinnedHostCSTapeStorage` that evicts staged
# panel snapshots from working memory onto a raw appended binary file
# (`records.bin`). A separate human-readable manifest (`manifest.toml`)
# is written at finalisation time and used during resume / inspection.
#
# Design doc: `docs/plans/26_TM5_STYLE_INVERSION/MMAP_TAPE_LAYOUT.md`.
#
# Layout summary (v1):
#
#   tape/<run-id>/
#   ├── manifest.toml      # written on finalize_tape!
#   └── records.bin        # grown via truncate as slots are allocated
#
# Each slot reserves a contiguous span of `records.bin` and holds six
# `(record_id, panel_offset, panel_shape)` triplets. `stage_panels!`
# copies the source panels into `Mmap.mmap` views over the reserved
# region; `_tape_panels` returns mmap views (CPU runs) or copies them
# into a shared device-side cache (GPU runs, see CUDA extension).
# ---------------------------------------------------------------------------

using Mmap: Mmap
using TOML: TOML
using Dates: now

const _MMAP_TAPE_FORMAT_VERSION = "v1"

# One forward record's locator into records.bin. Kept inside the storage
# so the manifest emitter has a single source of truth at finalise time.
struct MmapTapeRecordEntry
    record_id::Int
    eltype::DataType
    offsets::NTuple{6, Int64}
    nbytes::NTuple{6, Int64}
    shapes::NTuple{6, NTuple{3, Int}}
end

"""
    MmapCSTapeStorage(; dir = mktempdir(prefix="atmostransport-cstape-"),
                       cleanup_on_finalize = (dir == ""))

Tape storage policy that streams staged adjoint mass states onto a
single appended raw binary file (`records.bin`) inside `dir` and emits a
TOML manifest (`manifest.toml`) at finalisation. `dir` is created if it
does not exist; if omitted, a unique temporary directory is created and
deleted when the storage object is garbage-collected.

The policy is backend-agnostic: CPU source panels yield in-place
`Mmap.mmap` views at read time; GPU source panels go through a shared
device-side read cache (initialised by the CUDA extension).

See `docs/plans/26_TM5_STYLE_INVERSION/MMAP_TAPE_LAYOUT.md` for the
on-disk format.
"""
mutable struct MmapCSTapeStorage <: AbstractCSTapeStorage
    bin_path::String
    manifest_path::String
    dir::String
    bin_io::IOStream
    cursor::Int64
    records::Vector{MmapTapeRecordEntry}
    device_cache::Any
    synchronize::Any
    finalised::Bool
    closed::Bool
    cleanup_on_finalize::Bool

    function MmapCSTapeStorage(; dir::AbstractString = "",
                               cleanup_on_finalize::Union{Nothing,Bool} = nothing)
        if isempty(dir)
            dir = mktempdir(; prefix = "atmostransport-cstape-")
            cleanup = cleanup_on_finalize === nothing ? true : cleanup_on_finalize
        else
            isdir(dir) || mkpath(dir)
            cleanup = cleanup_on_finalize === nothing ? false : cleanup_on_finalize
        end
        bin_path = joinpath(dir, "records.bin")
        manifest_path = joinpath(dir, "manifest.toml")
        bin_io = open(bin_path, "w+")
        local storage
        try
            storage = new(bin_path, manifest_path, String(dir), bin_io, Int64(0),
                          MmapTapeRecordEntry[], nothing, nothing,
                          false, false, cleanup)
            finalizer(_mmap_storage_gc_close!, storage)
        catch
            close(bin_io)
            rethrow()
        end
        return storage
    end
end

_tape_storage(::Val{:mmap}) = MmapCSTapeStorage()

# ---------------------------------------------------------------------------
# Slot type — minimal descriptor; all data lives on disk.
# ---------------------------------------------------------------------------

struct MmapCSTapeSlot{S <: MmapCSTapeStorage}
    storage::S
    record_id::Int
    eltype::DataType
    offsets::NTuple{6, Int64}
    shapes::NTuple{6, NTuple{3, Int}}
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_mmap_file_size(io::IOStream) = stat(io).size

"""
    _bump_cursor!(storage, nbytes)

Reserve `nbytes` of contiguous space at the end of `records.bin`, extending
the file via `truncate` (sparse on filesystems that support it) so that
subsequent `Mmap.mmap` calls find the region backed.
Returns the base offset of the reserved region.
"""
# NOTE: this assumes POSIX semantics — growing the file via
# `Base.truncate` while earlier mmap views are still alive is
# well-defined on Linux/macOS (existing pages stay valid; the kernel
# refcounts the file). Windows is not supported.
function _bump_cursor!(storage::MmapCSTapeStorage, nbytes::Int64)
    base = storage.cursor
    storage.cursor += nbytes
    if storage.cursor > _mmap_file_size(storage.bin_io)
        Base.truncate(storage.bin_io, storage.cursor)
    end
    return base
end

# Build per-panel offsets from the record base.
function _panel_offsets(base::Int64, nbytes_per::NTuple{6, Int64})
    o1 = base
    o2 = o1 + nbytes_per[1]
    o3 = o2 + nbytes_per[2]
    o4 = o3 + nbytes_per[3]
    o5 = o4 + nbytes_per[4]
    o6 = o5 + nbytes_per[5]
    return (o1, o2, o3, o4, o5, o6)
end

_mmap_view(io::IOStream, ::Type{T}, shape::NTuple{3, Int}, offset::Int64) where {T} =
    Mmap.mmap(io, Array{T, 3}, shape, offset; grow = false, shared = true)

# Register the slot in the manifest queue. Called from the type-specialised
# `_allocate_tape_slot` methods so the queue is consistent across backends.
function _push_record!(storage::MmapCSTapeStorage, ::Type{T},
                       offsets, nbytes_per, shapes) where {T}
    record_id = length(storage.records) + 1
    push!(storage.records,
          MmapTapeRecordEntry(record_id, T, offsets, nbytes_per, shapes))
    return record_id
end

# ---------------------------------------------------------------------------
# Slot allocation + staging — CPU path.
# ---------------------------------------------------------------------------

"""
    _mmap_prepare_for_panels!(storage, panels)

Hook called once per slot allocation, **before** the disk reservation,
that lets the storage install per-policy state (e.g. a shared
device-side read cache, a backend `synchronize` callback).

The **default** method assumes the source panels are non-`Array`
device-resident buffers (CuArray, MtlArray, …) and pre-allocates a
backend-matched read cache via `_ensure_tape_read_cache!`. The
specialised method for `NTuple{6, <:Array}` is a no-op so the CPU read
path can return mmap views directly without an intermediate copy.

The CUDA extension layers a `synchronize = CUDA.synchronize` hook on top
of the default to amortise asynchronous device transfers; correctness
of mmap → device traffic does **not** depend on that extension because
`copyto!(::CuArray, ::Array)` is synchronous on CUDA.jl.
"""
function _mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                                   panels::NTuple{6})
    _ensure_tape_read_cache!(storage, panels)
    return storage
end

_mmap_prepare_for_panels!(storage::MmapCSTapeStorage,
                          _panels::NTuple{6, <:Array}) = storage

"""
    _allocate_tape_slot(storage::MmapCSTapeStorage, panels)

Reserve a span of `records.bin` for one tape record (six panel arrays)
and return a slot that can be staged into and later read from. The disk
bookkeeping is backend-agnostic; backend-specific setup (read-cache
allocation, synchronisation hooks) happens in
`_mmap_prepare_for_panels!`, which the CUDA extension specialises for
`NTuple{6, <:CuArray}`.
"""
function _allocate_tape_slot(storage::MmapCSTapeStorage,
                             panels::NTuple{6})
    storage.finalised && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is finalised; cannot allocate new slot"))
    storage.closed && throw(ArgumentError(
        "MmapCSTapeStorage at $(storage.dir) is closed; cannot allocate new slot"))
    _mmap_prepare_for_panels!(storage, panels)
    T = eltype(panels[1])
    shapes = ntuple(p -> size(panels[p]), 6)
    nbytes_per = ntuple(p -> Int64(sizeof(T)) * Int64(prod(shapes[p])), 6)
    base = _bump_cursor!(storage, sum(nbytes_per))
    offsets = _panel_offsets(base, nbytes_per)
    record_id = _push_record!(storage, T, offsets, nbytes_per, shapes)
    return MmapCSTapeSlot(storage, record_id, T, offsets, shapes)
end

"""
    stage_panels!(slot::MmapCSTapeSlot, src)

Write the six source panels into the slot's reserved region of
`records.bin` via `Mmap.mmap` views. Works uniformly for CPU `Array` and
GPU `CuArray` sources because `copyto!(::Array, ::CuArray)` performs a
device→host transfer.
"""
function stage_panels!(slot::MmapCSTapeSlot, src::NTuple{6})
    storage = slot.storage
    @inbounds for p in 1:6
        view = _mmap_view(storage.bin_io, slot.eltype, slot.shapes[p],
                          slot.offsets[p])
        copyto!(view, src[p])
    end
    _after_tape_stage!(storage)
    return slot
end

# ---------------------------------------------------------------------------
# Slot read — CPU returns mmap views; GPU is overloaded in the CUDA ext.
# ---------------------------------------------------------------------------

# Single-reader during reverse pass: the `_collect_surface_footprints`
# loop in Footprint/ReverseLoop.jl consumes slots sequentially. The
# `device_cache` is mutated in-place across reads, so this method is
# not safe under concurrent multi-threaded reverse walks against the
# same storage. Document via this comment until/unless that scenario
# materialises.
#
# Cache-matching contract (relied on for GPU correctness): on the GPU
# path the cache stored in `storage.device_cache` is a tuple of
# `CuArray`s, while `views` here are `Array` mmap views. The check in
# `_cache_matches` (TapeStorage.jl) ignores array type and looks only
# at `eltype` and `size` — so an existing `CuArray` cache is kept and
# `copyto!(CuArray, Array)` runs the host→device transfer below. If
# `_cache_matches` is ever tightened to also compare `typeof`, this
# path breaks and must be updated.
function _tape_panels(slot::MmapCSTapeSlot)
    storage = slot.storage
    views = ntuple(p -> _mmap_view(storage.bin_io, slot.eltype,
                                   slot.shapes[p], slot.offsets[p]), 6)
    if storage.device_cache === nothing
        return views
    end
    cache = _ensure_tape_read_cache!(storage, views)
    @inbounds for p in 1:6
        copyto!(cache[p], views[p])
    end
    _after_tape_read!(storage)
    return cache
end

# ---------------------------------------------------------------------------
# Stage / read hooks — sync the device cache for GPU runs.
# ---------------------------------------------------------------------------

function _sync_mmap_tape_storage!(storage::MmapCSTapeStorage)
    storage.synchronize === nothing || storage.synchronize()
    return nothing
end

_after_tape_stage!(storage::MmapCSTapeStorage) =
    _sync_mmap_tape_storage!(storage)
_after_tape_read!(storage::MmapCSTapeStorage) =
    _sync_mmap_tape_storage!(storage)

# `_ensure_tape_read_cache!(storage::MmapCSTapeStorage, panels)` reuses the
# shared `device_cache`-shape check defined for `PinnedHostCSTapeStorage`
# (in `TapeStorage.jl`). For the CPU path the cache is left as `nothing`
# and `_tape_panels` returns mmap views directly; for the GPU path the
# CUDA extension installs a CuArray cache via `_ensure_tape_read_cache!`.
function _ensure_tape_read_cache!(storage::MmapCSTapeStorage,
                                  panels::NTuple{6})
    if !_cache_matches(storage.device_cache, panels)
        storage.device_cache = ntuple(p -> similar(panels[p]), 6)
    end
    return storage.device_cache
end

# ---------------------------------------------------------------------------
# Finalisation + cleanup
# ---------------------------------------------------------------------------

"""
    finalize_tape!(storage::MmapCSTapeStorage)

Sync any pending mmap state, write the TOML manifest, and close the
underlying `records.bin` handle. Safe to call more than once; subsequent
calls are no-ops. If the storage owns its temp directory and
`cleanup_on_finalize` is true, the directory is also removed after
manifest emission.
"""
function finalize_tape!(storage::MmapCSTapeStorage; quiet::Bool = false)
    storage.closed && return storage
    if !storage.finalised && isdir(storage.dir)
        try
            _write_manifest(storage)
            storage.finalised = true
        catch err
            quiet ||
                @warn "MmapCSTapeStorage: manifest emission failed" exception = err
        end
    end
    try
        close(storage.bin_io)
    catch err
        quiet || @warn "MmapCSTapeStorage: bin close failed" exception = err
    end
    storage.closed = true
    if storage.cleanup_on_finalize && isdir(storage.dir)
        try
            rm(storage.dir; recursive = true, force = true)
        catch err
            quiet ||
                @warn "MmapCSTapeStorage: temp dir cleanup failed" exception = err
        end
    end
    return storage
end

# GC-driven close. Closes the IOStream and optionally removes the temp
# directory, but does **not** emit the TOML manifest — emitting that
# from the GC thread risks running TOML.print across a task switch
# inside a finalizer. Manifest emission requires an explicit
# `finalize_tape!(storage)` call by the user.
function _mmap_storage_gc_close!(storage::MmapCSTapeStorage)
    storage.closed && return nothing
    try
        close(storage.bin_io)
    catch
        # Finaliser must never throw.
    end
    storage.closed = true
    if storage.cleanup_on_finalize && isdir(storage.dir)
        try
            rm(storage.dir; recursive = true, force = true)
        catch
            # Finaliser must never throw.
        end
    end
    return nothing
end

function _write_manifest(storage::MmapCSTapeStorage)
    meta = Dict{String, Any}(
        "version"             => _MMAP_TAPE_FORMAT_VERSION,
        "created_at"          => string(now()),
        "endianness"          => _machine_endianness(),
        "julia_version"       => string(VERSION),
        "total_bytes"         => storage.cursor,
        "record_count"        => length(storage.records),
        "finalised"           => true,
    )
    records = [
        Dict{String, Any}(
            "id"      => r.record_id,
            "eltype"  => string(r.eltype),
            "offsets" => collect(r.offsets),
            "nbytes"  => collect(r.nbytes),
            "shapes"  => [collect(s) for s in r.shapes],
        ) for r in storage.records
    ]
    open(storage.manifest_path, "w") do io
        TOML.print(io, Dict{String, Any}("meta" => meta, "record" => records))
    end
    return storage.manifest_path
end

_machine_endianness() =
    ENDIAN_BOM == 0x04030201 ? "little" :
    ENDIAN_BOM == 0x01020304 ? "big" : "unknown"

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, storage::MmapCSTapeStorage)
    print(io, "MmapCSTapeStorage(dir=", repr(storage.dir),
          ", records=", length(storage.records),
          ", bytes=", storage.cursor,
          storage.closed ? ", closed" : "",
          ")")
end
