# ERA5 met-driver utilities

This submodule contains ERA5-specific geometry and continuity helpers used by
the canonical transport-binary stack.

- [`NativeGRIBGeometry.jl`](NativeGRIBGeometry.jl) reconstructs an ERA5 native
  reduced-Gaussian mesh from GRIB metadata.
- [`VerticalClosure.jl`](VerticalClosure.jl) diagnoses structured vertical
  interface mass flux from horizontal convergence and the hybrid-coordinate
  B increments.

Transport binaries are read by the topology-generic readers in
[`../TransportBinary.jl`](../TransportBinary.jl) and
[`../CubedSphereBinaryReader.jl`](../CubedSphereBinaryReader.jl). ERA-specific
runtime binary layouts and runtime moist-to-dry conversion are intentionally
unsupported: preprocessing writes the current v4 dry-basis contract.

See [`../../../docs/reference/BINARY_FORMAT.md`](../../../docs/reference/BINARY_FORMAT.md)
for the canonical on-disk schema.
