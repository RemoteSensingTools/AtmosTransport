# Invalidate convection caches when installing a new driver window

`DrivenSimulation` invalidated convection caches when advancing within a
meteorological driver, but not when constructing another simulation around a
reused model. CMFMC's cached subcycle count and CMFMC matrix's derived rates could
therefore describe the previous driver's final window. This affects callers
that retain the model/workspace across driver handoffs; the previous CS runner
avoided this particular problem by rebuilding its workspace each day.

Installing the new first-window forcing now invalidates both existing
convection cache hooks. It retains all allocated workspace arrays. The normal
within-driver invalidation is unchanged.

A regression compares a two-window continuous run with two separate one-window
drivers, using identical initial profiles and a substantial forcing change.
Before the fix, five of seven handoff checks failed: both tracer results
differed, CMFMC retained one subcycle instead of twelve, and matrix derived rates
remained stale. After the fix, all seven pass, and the existing 42 convection
runtime checks also pass on Julia 1.12.6 and 1.10.12. The test verifies that the
workspace object itself is still reused.
