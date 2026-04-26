# API reference

This section is the **auto-generated API reference** for the public
surface of every AtmosTransport.jl submodule. It is built directly
from the source docstrings; if a function is exported but the
docstring is sparse, that's both a quick-fix opportunity and a
signal that the function is internal-leaning.

The submodules are listed in load order — earlier modules don't
depend on later ones:

| Module | What's in it |
|---|---|
| [Architectures](@ref Architectures-API) | `CPU`, `GPU`, backend selection. |
| [Parameters](@ref Parameters-API) | `PlanetParameters`, `earth_parameters`. |
| [Grids](@ref Grids-API) | `LatLonMesh`, `CubedSphereMesh`, `ReducedGaussianMesh`, `AtmosGrid`. |
| [State](@ref State-API) | `CellState`, `CubedSphereState`, basis tags, tracer accessors. |
| [MetDrivers](@ref MetDrivers-API) | `TransportBinaryDriver`, `inspect_binary`, the v4 binary I/O. |
| [Operators](@ref Operators-API) | Advection / Convection / Diffusion / SurfaceFlux. |
| [Models](@ref Models-API) | `TransportModel`, `DrivenSimulation`, the runtime stepper. |
| [Preprocessing](@ref Preprocessing-API) | `process_day` and the source/target dispatch. |

### A note on the public surface

The API reference shows **public** symbols (`Private = false`).
Internals — kernel implementations, workspace structs, dispatch
helpers — live in the source under `src/` and are not auto-rendered
here. If you find yourself reaching for a non-exported symbol, that
either means (a) you're doing something the public API supports
through a higher-level entry, or (b) the public API has a gap; in
the latter case, an issue / PR is welcome.

The API reference does NOT yet have curated narrative for every
symbol — that's the natural follow-on to a full docstring audit
across the codebase. The pages below are the canonical entry
points; the [Concepts](@ref) and [Theory & Verification](@ref)
chapters are where you'll find narrative descriptions of what each
type does and how it fits.
