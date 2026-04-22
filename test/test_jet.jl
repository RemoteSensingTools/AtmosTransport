#!/usr/bin/env julia
"""
JET.jl static analysis (plan 21 Phase 6B/6C).

This test runs JET.report_package on AtmosTransport and records the
number of inference errors filtered to hot-path modules. It's
structured as a **snapshot gate**: the test passes if the count is
at or below a documented baseline, and fails loudly if new reports
appear. That catches regressions without making CI red from day
one (plan 21 budget for JET iteration was 1 day).

## Known-tolerated noise sources

Two patterns dominate the current report count and are NOT bugs:

1. **KernelAbstractions `kwcall` dispatch** — JET can't prove that
   `kernel(args...; ndrange=(...))` dispatches to the CPU or GPU
   backend because `KA.Kernel` is generic over its Backend type
   parameter. Every KA-using package hits this. Documented at
   https://github.com/aviatesk/JET.jl/issues/?q=KernelAbstractions.

2. **Parametric `@kwdef` zero-arg constructors** — `Base.@kwdef`
   auto-generates a zero-arg constructor for
   `struct Foo{FT}; x::FT = FT(1.0); end`, but `FT` is unbound at
   that call site. Users must call `Foo{Float64}()` explicitly.
   Documented in `CLAUDE.md` under "Julia / language gotchas".

## Escape hatch

Set `ATMOSTRANSPORT_JET_ADVISORY=1` to demote any count increase to
a warning instead of a test failure. Useful during local dev when
intermediate refactors temporarily surface new reports.

## Expected behavior

- Baseline count stable: test passes silently.
- Count drops: test passes and prints a "baseline can tighten" hint.
- Count rises: test fails (unless advisory env var is set), printing
  the new reports so the author can triage.
"""

using Test
using JET
using AtmosTransport

# Invocation:
#   julia --project=test test/test_jet.jl        (targeted)
#   julia --project=. -e 'using Pkg; Pkg.test()' (full suite)

# Hot-path modules that plan 21 cares about. IO-heavy modules
# (MetDrivers, Preprocessing) are excluded because they carry
# ~120 JSON3 header-parsing union-split reports that are known-
# tolerated (header field types are guaranteed by the preprocessor
# but JET can't prove it).
const HOT_PATH_MODULES = (
    AtmosTransport.Operators,
    AtmosTransport.State,
    AtmosTransport.Models,
    AtmosTransport.Grids,
)

# Snapshot baseline captured 2026-04-21 during plan 21 Phase 6.
# Dominant sources (see the docstring above):
#   - KernelAbstractions.Kernel kwcall dispatch: ~118 reports
#   - @kwdef zero-arg PBLPhysicsParameters: 1 report
# Bumped 117 → 119 on 2026-04-21 when CS chemistry shipped — the
# per-panel ExponentialDecay launch added two new KA kwcall reports
# at the same dispatch site as CellState. See
# artifacts/plan21/jet_baseline.txt for the full captured output.
const JET_HOT_PATH_BASELINE = 119

const ADVISORY_ONLY = get(ENV, "ATMOSTRANSPORT_JET_ADVISORY", "0") == "1"

@testset "JET: hot-path inference snapshot" begin
    result = JET.report_package(AtmosTransport;
                                target_modules = HOT_PATH_MODULES,
                                toplevel_logger = nothing)
    reports = JET.get_reports(result)
    n = length(reports)

    @info "JET hot-path modules: $n reports (baseline $JET_HOT_PATH_BASELINE)"

    if n > JET_HOT_PATH_BASELINE
        # Print new reports so the author can triage
        println()
        println("⚠  JET report count rose above the baseline.")
        println("   Baseline: $JET_HOT_PATH_BASELINE")
        println("   Current:  $n")
        println("   Delta:    +$(n - JET_HOT_PATH_BASELINE)")
        println()
        println("   First 20 reports:")
        for (i, rep) in enumerate(first(reports, 20))
            println("   [$i] ", rep)
        end
        println()
        println("   If the new reports are genuine bugs: fix them.")
        println("   If they're known-tolerated patterns (see docstring):")
        println("     1. Update JET_HOT_PATH_BASELINE in this file.")
        println("     2. Document the pattern in the 'Known-tolerated")
        println("        noise sources' section of the docstring.")
        println()

        if ADVISORY_ONLY
            @warn "ATMOSTRANSPORT_JET_ADVISORY=1 set — demoting to warning."
            @test true
        else
            @test n <= JET_HOT_PATH_BASELINE
        end
    elseif n < JET_HOT_PATH_BASELINE
        @info "JET baseline can be tightened: $n < $JET_HOT_PATH_BASELINE. " *
              "Lower JET_HOT_PATH_BASELINE in test_jet.jl when this is stable."
        @test true
    else
        @test true
    end
end
