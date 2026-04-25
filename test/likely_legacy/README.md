# Likely Legacy Tests

These tests import deleted preprocessing wrappers such as
`preprocess_spectral_v4_binary.jl`, `preprocess_era5_latlon_transport_binary_v2.jl`,
and `preprocess_era5_cs_conservative_v2.jl`.

They are preserved as references for expected behavior, but are intentionally
outside `test/runtests.jl` until they are ported to the unified
`src/Preprocessing` APIs.
