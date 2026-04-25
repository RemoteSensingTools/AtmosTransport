# ===========================================================================
# Identity-passthrough "regridder" for source-mesh == target-mesh paths.
#
# When the orchestrator constructs source and target meshes that are
# structurally equivalent (e.g. GEOS-IT C180 native source and CS C180
# preprocessing target), the conservative regridder is a costly identity
# matrix. `IdentityRegrid` short-circuits that with a `copyto!` — type-stable,
# zero allocation, no `if` branch in the hot path.
# ===========================================================================

"""
    IdentityRegrid{M}

Passthrough sentinel returned by `build_regridder` when source and target
meshes are structurally equivalent (per `meshes_equivalent`). Carries the
shared mesh so callers can introspect it like any `Regridder`.
"""
struct IdentityRegrid{M}
    mesh::M
end

# ---------------------------------------------------------------------------
# Structural equivalence: same shape ⇒ identity passthrough is safe.
# Default fallback returns false — different mesh types are never equivalent.
# Concrete subtype methods compare the geometry-defining fields, mirroring
# what `_hash_mesh!` already serializes for the cache key.
# ---------------------------------------------------------------------------

"""
    meshes_equivalent(src, dst) -> Bool

True when the two meshes describe the same horizontal grid up to
floating-point equality of the geometry-defining fields. Used by
`build_regridder` to skip building a conservative regridder when the
source and target are the same grid.
"""
meshes_equivalent(::Any, ::Any) = false

meshes_equivalent(a::LatLonMesh, b::LatLonMesh) =
    a.Nx == b.Nx && a.Ny == b.Ny &&
    a.λᶠ == b.λᶠ && a.φᶠ == b.φᶠ &&
    a.radius == b.radius

meshes_equivalent(a::CubedSphereMesh, b::CubedSphereMesh) =
    a.Nc == b.Nc &&
    typeof(a.convention) === typeof(b.convention) &&
    a.radius == b.radius

meshes_equivalent(a::ReducedGaussianMesh, b::ReducedGaussianMesh) =
    a.nlon_per_ring == b.nlon_per_ring &&
    a.latitudes     == b.latitudes &&
    a.lat_faces     == b.lat_faces &&
    a.radius        == b.radius

"""
    identity_regrid_or_nothing(src, dst) -> Union{IdentityRegrid, Nothing}

Helper used inside `build_regridder` to early-out when meshes are
equivalent. Returns `nothing` when a real regridder must be built.
"""
identity_regrid_or_nothing(src, dst) =
    meshes_equivalent(src, dst) ? IdentityRegrid(src) : nothing

# ---------------------------------------------------------------------------
# apply_regridder! extension — passthrough is `copyto!`.
# ---------------------------------------------------------------------------

apply_regridder!(dst::AbstractArray, ::IdentityRegrid, src::AbstractArray) =
    copyto!(dst, src)
