# Cached package loading in maintained command-line tools

The transport preprocessor, LL-to-CS transport-binary regridder, CS inversion
driver, and ATMSNAP-to-NetCDF converter now import `AtmosTransport` through the
active Julia project. They no longer include `src/AtmosTransport.jl` into a
fresh module on every launch. The transport runner already used package loading.
Launch commands retain `--project=.` from the checkout, as documented in the
installation guide. Older diagnostic scripts have not all been migrated.

On the local CPU, Julia 1.12.6 with four Julia threads, one OpenBLAS thread,
GPUs hidden, and `--startup-file=no`, two fresh-process launches of
`scripts/preprocessing/regrid_ll_transport_binary_to_cs.jl --help` took
19.16 / 19.11 seconds before and 7.60 / 7.48 seconds after. Dependencies were
already installed and precompiled. This measures command startup, not regridding
throughput; first-time package precompilation remains a separate cost.

Existing public configuration/CLI, LL-to-CS regridding, and inversion-driver
suites pass. These exercise the synthetic quickstart, parser failures,
regridding conservation and source metadata, actual regrid CLI execution, and
forward/adjoint inversion smoke cases. An additional temporary end-to-end
converter smoke writes an ATMSNAP frame containing `1e8, 1, -1e8`, launches the
converter in a new Julia process, and verifies finite signed fields and the
preserved Float64 total of 1 (four checks pass).

Codex diff review checked import scope, existing aliases, main guards, documented
project selection, and reuse of the same package types by script callers. The
inversion driver uses `import AtmosTransport` to retain its qualified namespace;
the other entry points retain their existing imported exports.
