"""
    PrecomputedCSKzField(host_cache)

Cubed-sphere layer-centre Kz field whose six panel caches are filled directly
from a binary `:kz` payload — the eddy diffusivity the preprocessor computed
once per met window with the TM5 boundary-layer scheme (`tm5_bldiff`).

Unlike [`LocalHoltslagBovilleKzField`], which *derives* Kz at runtime from the
GEOS VDIFF profiles, this field simply *copies* the precomputed values into its
panel caches on each met-window advance via
[`refresh_precomputed_cs_kz_cache!`](@ref). It is the runtime half of the
`[diffusion] kind = "precomputed_kz"` path: the heavy non-local PBL physics runs
offline in the preprocessor, and the runtime does one `copyto!` per panel per
window — host↔host on CPU, and device↔device on GPU when the field is adapted
alongside the model (the same per-kernel-launch transfer cost as the other CS Kz
fields).

Each panel wraps a [`PreComputedKzField`] over `host_cache[p]` (shape
`(Nc, Nc, Nz)`, the unhaloed panel), so the diffusion kernel reads
`field_value(panel_field(f, p), (i, j, k))` exactly as for the other CS Kz
fields. `host_cache` is parametric so the field is GPU-portable through
`Adapt`.
"""
struct PrecomputedCSKzField{FT, F <: PreComputedKzField{FT, 3}, H} <: AbstractCubedSphereField{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
end

function PrecomputedCSKzField(host_cache::NTuple{6, Array{FT, 3}}) where FT
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    return PrecomputedCSKzField{FT, typeof(panels[1]), typeof(host_cache)}(panels, host_cache)
end

@inline panel_field(f::PrecomputedCSKzField, p::Integer) = f.panels[Int(p)]

# Refresh cadence is per met-window: the copy happens in
# `refresh_precomputed_cs_kz_cache!`, driven by the runner on window advance, so
# the time-keyed `update_field!` is a no-op (the panels already hold the active
# window's Kz).
update_field!(f::PrecomputedCSKzField, ::Real) = f

function Adapt.adapt_structure(to, f::PrecomputedCSKzField)
    panels = Adapt.adapt(to, f.panels)
    return PrecomputedCSKzField{_precomputed_cs_kz_eltype(f), typeof(panels[1]),
                                typeof(f.host_cache)}(panels, f.host_cache)
end
@inline _precomputed_cs_kz_eltype(::PrecomputedCSKzField{FT}) where {FT} = FT

"""
    refresh_precomputed_cs_kz_cache!(field, kz_panels) -> field

Copy the active window's precomputed Kz panels into the field's panel caches.
`kz_panels` is the `kz` field of the loaded transport window (an
`NTuple{6, AbstractArray}` shaped `(Nc, Nc, Nz)`, or `nothing` when the binary
carries no `:kz` section). `copyto!` runs on whichever backend the field and
window share: host↔host on CPU, device↔device on GPU when both have been adapted
with the model.
"""
function refresh_precomputed_cs_kz_cache!(field::PrecomputedCSKzField, kz_panels)
    kz_panels === nothing && throw(ArgumentError(
        "[diffusion] kind = \"precomputed_kz\" requires a `:kz` payload in the \
         cubed-sphere transport window; regenerate the binary with \
         include_tm5_diffusion=true."))
    @inbounds for p in 1:6
        copyto!(field.panels[p].data, kz_panels[p])
    end
    return field
end
