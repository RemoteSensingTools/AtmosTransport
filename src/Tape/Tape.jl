"""
    Tape

Tape storage policies, tape record types, and (in Phase A) on-disk
checkpointing for the AtmosTransport.jl cubed-sphere adjoint pipeline.

Plan 26 Commit P0.1 extracts these utilities from the previously-monolithic
`src/Adjoints/Adjoints.jl` into a focused sibling module so subsequent
Plan 26 phases (NetCDF tape, sliding-window replay) land here rather than
growing `Adjoints.jl` further.

Module dependency order (Plan 26 P0.0 NOTES):
    Adjoints  →  Tape  →  Footprint  →  Inversion
                  ↑
                  this module

Phase P0.1 is **pure code motion** — no semantic change. The reverse-loop
driver that dispatches on tape records currently lives in `Adjoints.jl`;
it moves to `Footprint/` in P0.3.
"""
module Tape

export AbstractCSTapeStorage,
       DeviceCSTapeStorage,
       PinnedHostCSTapeStorage,
       CSTapeSlot,
       PinnedHostCSTapeSlot,
       _tape_storage, _tape_panels,
       _allocate_tape_slot, stage_panels!, _stage_panels,
       _after_tape_stage!, _after_tape_read!,
       _sync_pinned_tape_storage!,
       _ensure_tape_read_cache!,
       _bytes_per_panel_tuple,
       _CSSweepRecord, _CSHaloRecord, _CSMidpointRecord,
       _CSDiffusionRecord, _CSConvectionRecord,
       _CSTapeOp

include("TapeStorage.jl")
include("TapeRecords.jl")

end # module Tape
