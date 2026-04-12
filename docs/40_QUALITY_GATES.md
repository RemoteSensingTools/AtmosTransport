# Quality Gates

## Why these gates exist

The goal is to move slowly and correctly, with code that remains readable,
future-proof, and credible to outside users.

Documentation and tests are part of the implementation phase, not cleanup work.

## Code quality gates

New `src` transport paths should aim for:

- concrete, type-stable model structs
- allocation-free stepping after workspace setup
- explicit failure for unsupported modes
- no hidden fallback into legacy `src`

## Binary/driver quality gates

Before a new met path is considered stable, it should have:

- explicit mass basis
- explicit timing/value semantics
- clear payload manifest
- readable `show` output
- roundtrip tests
- driver validation of supported semantics

## Numerical quality gates

Minimum expected tests for a new path:

- zero-flux invariance
- uniform-field invariance
- air-mass conservation
- tracer-mass conservation
- localized pulse transport

## Documentation gates

Before starting the next major grid/runtime path, there should be at least:

- one short reference note
- one binary semantics note
- one end-to-end smoke test path

The point is not maximal documentation early. The point is avoiding ambiguity
and design drift.
