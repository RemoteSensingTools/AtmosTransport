# ---------------------------------------------------------------------------
# Dry flux builder interface — LEGACY for moist-basis binaries
#
# With Invariant 14 (dry-basis default), transport binaries are preprocessed
# on dry basis. The build_dry_fluxes! interface is retained for backward
# compatibility with old moist-basis binaries, but new production runs should
# use dry-basis binaries where the transport core reads dry fields directly.
#
# This is the key boundary between meteorology and transport:
#
#   meteorology → build_dry_fluxes! → (AbstractFaceFluxState, cell dry mass)
#
# Each met driver implements build_dry_fluxes! to convert its native
# meteorological fields into the flux representation that matches the
# mesh's flux_topology.  The transport core never sees raw met fields.
# ---------------------------------------------------------------------------

"""
    build_dry_fluxes!(fluxes::AbstractFaceFluxState, cell_mass,
                      met::MetState, grid::AtmosGrid,
                      driver::AbstractMetDriver,
                      closure::AbstractMassClosure)

Build dry face mass fluxes from meteorological fields.

This is the single most important interface function in the architecture.
After this call, `fluxes` contains horizontal and vertical dry mass fluxes
whose storage layout matches the mesh's `flux_topology`:

- Structured meshes (`StructuredFluxTopology`) → directional arrays
- Unstructured meshes (`FaceIndexedFluxTopology`) → face-indexed array

`cell_mass` is filled with dry air mass per cell [kg].

Each met driver provides a specialized method dispatching on the concrete
flux state type and grid type appropriate for its target mesh.
"""
function build_dry_fluxes! end

"""
    build_air_mass!(cell_mass, met::MetState, grid::AtmosGrid,
                    driver::AbstractMetDriver)

Compute dry air mass from surface pressure and humidity.
`m_dry[i,j,k] = Δp[k](ps) × area[i,j] / g × (1 - q[i,j,k])`
"""
function build_air_mass! end

function build_dry_fluxes!(fluxes, cell_mass, met, grid, driver,
                           ::PressureTendencyClosure)
    throw(ArgumentError("PressureTendencyClosure is Phase 3+: no src implementation exists yet"))
end

function build_dry_fluxes!(fluxes, cell_mass, met, grid, driver,
                           ::NativeVerticalFluxClosure)
    throw(ArgumentError("NativeVerticalFluxClosure is Phase 3+: no src implementation exists yet"))
end

export build_dry_fluxes!, build_air_mass!
