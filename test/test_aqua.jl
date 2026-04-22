#!/usr/bin/env julia
"""
Aqua.jl package health gates (plan 21 Phase 6A).

Hard CI gate: ambiguities, undefined exports, unbound type parameters,
stale deps, compat bounds, type piracy.

`persistent_tasks` is disabled because the GPU extensions
(`AtmosTransportCUDAExt`, `AtmosTransportMetalExt`) launch async tasks
during precompile that trip false positives.

If Aqua surfaces a real bug, fix it in a follow-up commit. If a check
has a documented false-positive reason (e.g. method extension for an
external type), silence the specific case with a rationale rather
than globally disabling the check.
"""

using Test
using Aqua
using AtmosTransport

# Invocation:
#   julia --project=test test/test_aqua.jl       (targeted)
#   julia --project=. -e 'using Pkg; Pkg.test()' (full suite)
#
# `using AtmosTransport` here works under both invocations because the
# root project's [targets.test] and test/Project.toml both carry the
# package. Avoids the legacy `include("src/AtmosTransport.jl")` pattern
# which requires every transitive dep to be loadable from Main.

@testset "Aqua: package health" begin
    Aqua.test_all(
        AtmosTransport;
        ambiguities       = true,
        unbound_args      = true,
        undefined_exports = true,
        project_extras    = true,
        stale_deps        = true,
        deps_compat       = true,
        piracies          = true,
        persistent_tasks  = false,
    )
end
