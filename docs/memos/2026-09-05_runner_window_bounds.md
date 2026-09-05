# Reject ignored or discontinuous driven window ranges

The CS runner accepted the generic `start_window` config key but always started
from window 1. It now rejects nondefault starts explicitly, before constructing
the model. Later starts remain supported by the LL/RG runner for a single file.

A partial LL/RG window range repeated across multiple files could also carry
state across omitted forcing. Such ranges now require a single input file.
Validation checks the first driver before model allocation and each subsequent
driver before stepping, covering files with different window counts. CS retains
its existing first-file partial-stop rejection and now checks subsequent files
as well. Valid full-file runs and single-file partial debugging remain available.

Validation:

- The input-lifetime suite passes 30 assertions on Julia 1.12.6 and 1.10.12,
  including both rejected multi-file ranges and an actual successful single-file
  partial run with its final tracer/air-mass values checked.
- The CS multifile suite passes 56 assertions on Julia 1.12.6, including explicit
  nondefault-start rejection and its existing physical handoff checks.
- Six two-file CPU pipeline cases pass across LL/RG/CS and full/column-only
  output, checking every output time and tracer through the public reader.
- A final clean export passes Aqua's ten checks and JET's unchanged threshold
  (180 reports against 181). Documentation builds with deployment disabled.
