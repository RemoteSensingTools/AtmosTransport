# Staging integration on current main

The staging implementation and its pre-port tests were identical on current
main and the old review branch, so the validated ownership/source-identity
change could be applied directly.

A directory has one active staging owner. Concurrent runs or unavailable
staging directories fall back to direct-source reads. Reuse requires source
path, size, inode, and modification/change timestamps to match a TOML sidecar;
size alone no longer authorizes reuse. These are filesystem identity checks,
not content checksums, and inputs must remain immutable during a run.

Repeated input paths share an in-flight copy, eviction respects outstanding
references, and cleanup removes only tracked data/metadata and this copy's
partial files. It releases directory ownership after draining tasks.

All 77 staging checks pass on Julia 1.12.6 with four threads and GPUs hidden.
Aqua passes 10 checks. The core tests cover stale equal-sized sources, invalid
metadata, retained-copy reuse, directory ownership/fallback, repeated sources,
and cleanup. No GPU or transport throughput claim is attached to this change.

The remaining optional preprocessing Git-provenance probes are also quiet
outside a checkout, preserving their existing empty/unknown fallback values.
The corresponding output probes were already ported in `36c23f8a`.
