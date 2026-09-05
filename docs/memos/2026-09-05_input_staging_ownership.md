# Input staging: source identity and directory ownership

The previous reuse check compared file sizes only. An equal-sized replacement
of a forcing binary could therefore silently reuse old forcing. Two runs using
one staging directory also had independent eviction policies and could remove
each other's inputs or overwrite the same `.part` file.

Staging now records the source's absolute path, size, modification and change
times, and inode in a TOML sidecar. Reuse requires all metadata to agree. Copies
check source metadata before and after the copy; a changed source or failed
copy falls back to direct-source reads. Old cache files without metadata are
recopied. This uses filesystem identity, not a content checksum: forcing must
remain immutable during a run, and manually preserving all metadata is outside
this check's guarantee.

A `FileWatching.Pidfile` lock gives one run ownership of its staging directory.
A second run, or a run unable to create/lock the directory, continues with
source paths. Separate directories allow concurrent staging. Cleanup waits for
all copy tasks, releases ownership even on failure, and removes only files
tracked by that run. It no longer sweeps other producers' `.part` files.
Repeated source paths in an explicit replay list share in-flight copy tasks;
eviction keeps a file while another current/future entry still refers to it.

Validation:

- Julia 1.12.6 and 1.10.12: 77 staging assertions pass, including equal-sized
  source replacement, corrupt metadata, cross-run reuse, concurrent ownership,
  unavailable directory fallback, repeated paths, eviction, and cleanup.
- A clean index export passes Aqua's 10 package-health checks and JET's unchanged
  threshold (180 reports against the allowed 181).
- No transport arithmetic changes. FileWatching is a Julia standard library;
  both root and direct-test environments declare it explicitly.
