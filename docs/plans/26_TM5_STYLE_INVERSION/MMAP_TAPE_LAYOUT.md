# Phase A — `MmapCSTapeStorage` on-disk layout

_Status: working design — written 2026-05-11, to be revised when Phase A.1 lands._
_Decision context: Plan 26 NOTES `Phase A` storage-format decision (raw binary + mmap default;
NetCDF/Zarr deferred to A4 fallback). This document fills in the concrete data layout._

## Top-level filesystem layout

```
tape/<run-id>/
├── manifest.toml          # human-readable index, finalised at run end
├── records.bin            # raw appended panel data, sparse-OK preallocation
└── records.bin.partial    # only present if a run was interrupted mid-write
```

### Two-file split: why metadata is separated from data

- Manifest is small (~10-20 MB at C180/14 d for ~20 k records) and read once at the
  start of the reverse pass.
- Records is large (~600 GB at C180/14 d, Float32) and read once in LIFO order during
  the reverse pass.
- TOML is human-readable: `cat`, `less`, `grep`, version-control diffs are useful for
  debugging tape generation and resume.
- Records.bin stays pure data: zero serialization, zero per-record magic bytes; a single
  `Mmap.mmap` view at the right offset is the kernel-callable Julia array.

## Manifest schema (TOML)

```toml
[meta]
version            = "v1"
created_at         = "2026-05-11T20:14:33Z"
arch               = "x86_64-linux-gnu"
endianness         = "little"
julia_version      = "1.11.3"
atmostransport_sha = "629bcd8"
mesh.Nc            = 180
mesh.Hp            = 3
mesh.Nz            = 72
dtype              = "Float32"
total_bytes_preallocated = 644245094400
finalised          = true   # set on close; if false, run was interrupted

[[record]]
id        = 1
op        = "sweep_x"      # sweep_x | sweep_y | sweep_z | halo | midpoint
                           # | diffusion | convection | linrood_horiz
step      = 1
offset    = 0
nbytes    = 28311552       # 6 panels * 180 * 180 * 72 * sizeof(Float32)
panel_shapes = [[180,180,72], [180,180,72], ..., [180,180,72]]
scheme    = "PPMScheme{MonotoneLimiter}"   # reverse-loop dispatch
substep   = 1
extras    = { flux_scale = 0.125 }         # op-specific scalar metadata

[[record]]
id        = 2
op        = "halo"
step      = 1
offset    = 28311552
nbytes    = 0              # halo records are scalar metadata; no panel data
dir       = 0
```

## records.bin layout

Concatenated, panel-aligned, with no padding between records:
- Each record reserves `nbytes = sum(prod(panel_p_shape) * sizeof(dtype) for p in 1:6)`.
- Panel data inside one record is **panel-contiguous, column-major Julia layout** —
  bit-for-bit `Mmap.mmap(io, Array{Float32, 3}, (Nx, Ny, Nz), offset)`. Zero
  serialization.
- File is preallocated at run start via `fallocate(POSIX_FALLOC_FL_KEEP_SIZE)` using
  the estimate from `_tape_byte_estimate(...; storage=:mmap)`. Underallocated tapes grow
  via `truncate + remap`; over-allocation leaves a sparse hole on the filesystem.

## Why these specific choices

| Design choice | Reason |
|---|---|
| Per-panel contiguous, not `(panel, i, j)` interleaved | Panel-level reads are the natural unit (each adjoint kernel takes one panel at a time); avoids stride waste. |
| Append-only with explicit offset table in manifest | Decouples write order from read order; revolve-checkpointing can overwrite a slot's data without manifest changes. |
| `op` field as enumerated string in manifest, **not** a magic byte in `records.bin` | Records.bin stays pure data; debug-readable manifest; no binary parser. |
| Float32 default, Float64 opt-in via `meta.dtype` | C180 Float64 tape is ~1.2 TB; F32 halves it with negligible adjoint accuracy loss. Matches transport-binary defaults. |
| Manifest `finalised` flag | Resume detection: if `finalised = false`, the last manifest entry might be torn — skip it on replay. |
| Endianness + arch in `meta` | Refuse to load a tape created on a different machine. |
| `atmostransport_sha` in `meta` | Refuse to replay if kernel ABI changed between runs. |
| No checksums in v1 | mmap does not lend itself to per-record checksums; rely on filesystem checksums (ZFS/btrfs) or skip. |

## Open questions to settle before implementation

1. **Manifest write cadence.** One TOML entry per record means ~20 k file ops. Options:
   buffer in memory and flush every N records (~64 ≈ 4 MB manifest delta), or use an
   append-only binary index alongside `records.bin` and only emit TOML at close.
   Buffered TOML probably right for ~28 k records.

2. **Record alignment.** Today `nbytes` is exact (no padding). For mmap views per
   record, offsets need to be page-aligned (4 KB on x86). Trade-off: page-aligning
   wastes ~2 KB per record ≈ ~40 MB total at C180. Probably worth it for clean
   mmap semantics. (Otherwise reads cross page boundaries and we lose mmap's
   page-cache amortisation.)

3. **Variable-size records.** Halo / midpoint records carry no panel data (just
   scalar metadata). Either reserve zero bytes (records.bin offset does not
   advance) or omit from `records.bin` entirely (manifest still tracks them).
   The latter is cleaner.

4. **Revolve checkpointing slot reuse.** If we overwrite slot N's data
   mid-run (Revolve schedule), the manifest entry for N stays the same —
   only the bytes change. Safe under mmap because we hold no stale view.
   Should the manifest record a generation counter per slot to detect
   mid-replay-write-races? Probably overkill for single-process runs.

5. **What goes in `extras`.** Some op records need a few scalar parameters
   (`flux_scale`, `dt`, scheme params). Inline TOML table per record is
   flexible; alternative is a fixed per-op struct. Inline table preferred
   for v1; promote to fixed struct only if performance dictates.

6. **Op enumeration vs scheme polymorphism.** Today's `_CSSweepRecord` carries
   a `scheme::S` field with the actual scheme type. On disk we can't
   preserve a Julia type; we'd serialise as a string
   (`"PPMScheme{MonotoneLimiter}"`) and reconstruct via a small dispatcher
   in the loader. Acceptable but is a fragile-shaped surface.

7. **NetCDF interop later.** If/when we want a `NetCDFCSTapeStorage` for
   archiving (A4 fallback), the natural shape is: `manifest.toml` stays
   the same human-readable index, `records.bin` becomes one big NetCDF
   dataset chunked along `record_id`. The on-the-wire layout is
   irrelevant once the manifest carries enough metadata to find any
   record by id.

## How this fits the existing API

The P0.1 `AbstractCSTapeStorage` abstraction means `MmapCSTapeStorage` plugs in
as a sibling of `DeviceCSTapeStorage` / `PinnedHostCSTapeStorage`:

- `_allocate_tape_slot(::MmapCSTapeStorage, panels)` reserves a manifest
  entry, bumps the records cursor, and returns a slot descriptor (record
  id + offset).
- `stage_panels!(slot, src)` `pwrite`s 6 panels into `records.bin` at the
  slot's offset; appends the record entry to the buffered manifest queue.
- `_tape_panels(slot)` `Mmap.mmap`s a view at the slot's offset, copies
  into a device-side LRU cache (reused from the
  `PinnedHostCSTapeStorage.device_cache` field shape), returns the cache
  tuple.
- `_after_tape_stage!` / `_after_tape_read!` are no-ops for the mmap
  policy (no GPU device synchronisation needed when bytes live on disk).

## References

- Plan 26 NOTES `Phase A` decision matrix (NetCDF vs Zarr vs mmap) — the
  earlier-stage analysis that produced "use mmap as default."
- `docs/reference/BINARY_FORMAT_V5.md` — the existing transport-binary
  format whose offset + manifest pattern this design mirrors.
- `src/Tape/TapeStorage.jl` — the `AbstractCSTapeStorage` interface this
  new storage policy must satisfy.
