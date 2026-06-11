# ---------------------------------------------------------------------------
# TracerReferences — per-tracer reference-state (anomaly transport) metadata
#
# Large-background tracers (e.g. CO2 at ~412 ppm) lose Float32 precision in
# the flux-form update `rm_new = rm + flux_div` because the increment is tiny
# relative to the background-dominated tracer mass. The reference-state scheme
# stores q = q_ref + q_anom: a single F64 reference VMR per tracer is carried
# analytically (a uniform VMR is an exact eigenstate of conservative mass-flux
# transport, `q const ⇒ rm_new = q·m_new`), and only the small signed anomaly
# mass `q_anom·m` lives in `state.tracers_raw`. See
# docs/plans/45_ANOMALY_REFERENCE_TRANSPORT/PLAN.md.
#
# Contract
# --------
# - Path selection is the explicit `kind` flag, NEVER a `q_ref == 0` value
#   test: a referenced tracer may legitimately have `q_ref = 0` (IC=0) and
#   must still execute the referenced code path.
# - `kind` uses compact `UInt8` codes so device-side consumers never depend
#   on host `Symbol` behavior. Symbols appear only at the config boundary.
# - The carrier is HOST-resident metadata (like `tracer_names`): kernels
#   consume the reference as an `FT` scalar argument computed at launch.
#   `Adapt.adapt_structure` passes it through unchanged, so mutation
#   (IC seeding, re-referencing) through an adapted GPU state stays visible.
# - Vectors are mutated in place; the struct itself is immutable.
# ---------------------------------------------------------------------------

"""
Reference-kind codes (`UInt8`, device-safe):

- `REF_NONE` — tracer stores full mass `q_full·m`; every code path is the
  raw path, bit-identical to a build without reference support.
- `REF_GLOBAL_MEAN` — tracer stores anomaly mass `q_anom·m` against a single
  mass-weighted global-mean reference VMR held in `q_ref`.
"""
const REF_NONE        = 0x00
const REF_GLOBAL_MEAN = 0x01

"""
    TracerReferences

Per-tracer reference-state metadata for anomaly transport (host-resident).

# Fields
- `kind :: Vector{UInt8}` — per-tracer reference kind (`REF_NONE`,
  `REF_GLOBAL_MEAN`), indexed in `tracer_names` storage order.
- `q_ref :: Vector{Float64}` — per-tracer reference dry VMR. Always `Float64`
  regardless of the run `FT`: the reference participates in exact burden
  bookkeeping (`total_mass_full`) and re-reference shifts, which must not
  round. Converted to the run `FT` only at kernel-argument boundaries.
"""
struct TracerReferences
    kind  :: Vector{UInt8}
    q_ref :: Vector{Float64}

    function TracerReferences(kind::Vector{UInt8}, q_ref::Vector{Float64})
        length(kind) == length(q_ref) || throw(DimensionMismatch(
            "TracerReferences kind has length $(length(kind)) but q_ref has " *
            "length $(length(q_ref))"))
        return new(kind, q_ref)
    end
end

"""
    TracerReferences(Nt::Integer)

All-`REF_NONE` carrier for `Nt` tracers — the default: every tracer stores
full mass and every code path is the raw path.
"""
TracerReferences(Nt::Integer) = TracerReferences(fill(REF_NONE, Int(Nt)),
                                                 zeros(Float64, Int(Nt)))

ntracers(refs::TracerReferences) = length(refs.kind)

"""
    tracer_reference_kind(refs::TracerReferences, idx::Integer) -> UInt8

Reference-kind code for the tracer at storage index `idx`.
"""
@inline tracer_reference_kind(refs::TracerReferences, idx::Integer) =
    refs.kind[Int(idx)]

"""
    tracer_reference_value(refs::TracerReferences, idx::Integer)
        -> Union{Nothing, Float64}

Reference VMR for the tracer at storage index `idx`, or `nothing` when the
tracer is unreferenced (`REF_NONE`). Returning `nothing` rather than `0.0`
lets downstream kernels dispatch the raw path away at compile time, and keeps
the `kind`-flag contract: a referenced tracer with `q_ref == 0.0` still
returns `0.0` (not `nothing`) and runs the referenced path.
"""
@inline function tracer_reference_value(refs::TracerReferences, idx::Integer)
    i = Int(idx)
    return refs.kind[i] == REF_NONE ? nothing : refs.q_ref[i]
end

"""
    set_tracer_reference!(refs::TracerReferences, idx::Integer,
                          kind::UInt8, q_ref::Real)

Install reference metadata for the tracer at storage index `idx`. Used by IC
seeding (`REF_GLOBAL_MEAN`) and re-referencing; mutates the carrier in place
so adapted (GPU) states sharing the carrier observe the update.
"""
function set_tracer_reference!(refs::TracerReferences, idx::Integer,
                               kind::UInt8, q_ref::Real)
    kind in (REF_NONE, REF_GLOBAL_MEAN) || throw(ArgumentError(
        "unknown tracer reference kind code $(kind)"))
    i = Int(idx)
    refs.kind[i] = kind
    refs.q_ref[i] = Float64(q_ref)
    return refs
end

"""
    any_tracer_referenced(refs::TracerReferences) -> Bool

`true` when at least one tracer carries a non-`REF_NONE` reference. Cheap
host-side guard so the default path skips reference bookkeeping entirely.
"""
any_tracer_referenced(refs::TracerReferences) = any(!=(REF_NONE), refs.kind)

export TracerReferences, REF_NONE, REF_GLOBAL_MEAN
export tracer_reference_kind, tracer_reference_value, set_tracer_reference!
export any_tracer_referenced
