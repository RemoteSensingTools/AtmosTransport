# Typed runtime-physics configuration specs.
#
# Oceananigans-style: parse loose TOML *once* into typed objects, validate at the
# boundary, then dispatch on concrete types — instead of `Val(Symbol(get(...)))`
# over a raw `Dict{String,Any}` at the point of use. Mirrors the existing
# `RuntimeOutputSpec` in `src/Output/runtime_output.jl`.
#
#   TOML section Dict  --<family>_spec-->  <Spec>  --materialize(spec, style[, …])-->  <operator>
#
# The only place a string becomes a type is the `_parse_*_kind` parser, where all
# parse-time validation (including the lmax_conv/n_merge footgun) lives. The
# materialized operator is identical to what the old `build_runtime_*` produced, so
# this is behavior-preserving EXCEPT for the one intentional tightening noted on
# `convection_spec`.
#
# POLICY: every `materialize(spec, style[, …])` method for EVERY spec family
# (convection now; advection/diffusion/chemistry to follow) lives in THIS file. Do
# not define `materialize` elsewhere in the Models module — keeping them together is
# what makes the dispatch discoverable.
#
# This file currently holds the convection family.

# --- Typed TOML-value accessors (clean errors at the boundary) -------------
# All three give the user the section + key + offending value, instead of a bare
# MethodError/InexactError from a `Float64("1.5")`/`Int(2.0)` deep in the parser.
# `label` is the TOML table name (e.g. "[convection]") for the error message.

function _spec_bool(section, key::AbstractString, default::Bool, label::AbstractString)
    v = get(section, key, default)
    v isa Bool || throw(ArgumentError("$(label).$(key) must be true or false; got $(repr(v))"))
    return v
end
function _spec_int(section, key::AbstractString, default::Int, label::AbstractString)
    v = get(section, key, default)
    v isa Integer || throw(ArgumentError("$(label).$(key) must be an integer; got $(repr(v))"))
    return Int(v)
end
function _spec_float64(section, key::AbstractString, default::Float64, label::AbstractString)
    v = get(section, key, default)
    v isa Real || throw(ArgumentError("$(label).$(key) must be a number; got $(repr(v))"))
    return Float64(v)
end

# =========================================================================
# Convection
# =========================================================================

abstract type AbstractConvectionSpec end

# The two collaborative-LU kinds (`tm5`, `cmfmc_matrix`) share the same knobs and
# the same LU machinery — they differ only in which operator `materialize` builds.
abstract type AbstractCollabLUConvectionSpec <: AbstractConvectionSpec end

struct NoConvectionSpec    <: AbstractConvectionSpec end
struct CMFMCConvectionSpec <: AbstractConvectionSpec end

# Knobs stored as raw/friendly scalars — never the run `FT` (TM5Convection's own FT
# is `typeof(tile_workspace_gib)` = Float64, independent of the run precision).
#
# Knob reference:
#   tile_workspace_gib — per-topology column-tile budget (binary GiB); default 1.0
#     fits C720/L137 with slack on H100; lower it on memory-tight GPUs (e.g. L40S
#     48 GiB), raise it to amortize launch overhead.
#   use_collab_lu      — opt into the workgroup-collaborative kernel (~10× faster,
#     bit-exact within Float32 rounding on CUDA). Default off so existing runs stay
#     bit-identical. REQUIRED for lmax_conv/n_merge to do anything (see convection_spec).
#   lmax_conv          — cap the convection matrix below the full Nz (TM5 tropoX*).
#     0 = no truncation. Pick a safe ceiling with
#     `scripts/diagnostics/per_column_depth_histogram.jl`; e.g. on the ERA5/GEOS-native
#     C180/L85 binary `lmax_conv = 75` is bit-exact for every observed column.
#   n_merge            — aggregate n adjacent fine layers per convection super-layer
#     (LU is O(L_super³): n_merge=3 ≈ 27× cheaper). 1 = no aggregation; 2 is rejected.
struct TM5ConvectionSpec <: AbstractCollabLUConvectionSpec
    tile_workspace_gib :: Float64
    use_collab_lu      :: Bool
    lmax_conv          :: Int
    n_merge            :: Int
end

struct CMFMCMatrixConvectionSpec <: AbstractCollabLUConvectionSpec
    tile_workspace_gib :: Float64
    use_collab_lu      :: Bool
    lmax_conv          :: Int
    n_merge            :: Int
end

function _parse_convection_kind(section)
    raw = lowercase(String(get(section, "kind", "none")))
    raw == "none"         && return :none
    raw == "tm5"          && return :tm5
    raw == "cmfmc"        && return :cmfmc
    raw == "cmfmc_matrix" && return :cmfmc_matrix
    throw(ArgumentError(
        "Unknown [convection] kind: $(repr(raw)). Supported: none | tm5 | cmfmc | cmfmc_matrix"))
end

# Shared knob extraction + validation for the collaborative-LU kinds.
function _collab_lu_knobs(section)
    budget     = _spec_float64(section, "tile_workspace_gib", 1.0, "[convection]")
    use_collab = _spec_bool(section, "use_collab_lu", false, "[convection]")
    lmax_conv  = _spec_int(section, "lmax_conv", 0, "[convection]")
    n_merge    = _spec_int(section, "n_merge", 1, "[convection]")

    if (lmax_conv != 0 || n_merge != 1) && !use_collab
        throw(ArgumentError(
            "[convection] lmax_conv/n_merge only take effect with use_collab_lu = true " *
            "— they steer the collaborative-LU kernel, and the per-thread fallback ignores " *
            "them. Set use_collab_lu = true (requires Float32 + GPU + lmax_conv ≤ 85), or " *
            "remove lmax_conv/n_merge. Got lmax_conv=$(lmax_conv), n_merge=$(n_merge), " *
            "use_collab_lu=false."))
    end
    n_merge == 2 && throw(ArgumentError(
        "[convection] n_merge = 2 is rejected; use n_merge ∈ {1, 3, 4, 5}."))
    return (budget, use_collab, lmax_conv, n_merge)
end

"""
    convection_spec(section) -> AbstractConvectionSpec

Parse a `[convection]` TOML section into a typed spec, validating at the boundary.

INTENTIONAL behavior change vs the old builder: the collaborative-LU knobs
`lmax_conv`/`n_merge` only take effect when `use_collab_lu = true` (they steer the
workgroup-collaborative kernel; the legacy per-thread path ignores them). Setting
them without `use_collab_lu` used to be a silent no-op; it is now a hard error.
"""
function convection_spec(section)
    kind = _parse_convection_kind(section)
    kind === :none  && return NoConvectionSpec()
    kind === :cmfmc && return CMFMCConvectionSpec()
    knobs = _collab_lu_knobs(section)
    kind === :tm5 && return TM5ConvectionSpec(knobs...)
    return CMFMCMatrixConvectionSpec(knobs...)  # :cmfmc_matrix
end

# materialize — turn the spec into the runtime operator. Convection needs neither
# the run `FT` nor the driver context, so the clean form is `materialize(spec, style)`
# (no unused ceremony args). The two collaborative-LU specs share knobs but build
# DIFFERENT operators, so they dispatch on their concrete types.
materialize(::NoConvectionSpec, ::AbstractRuntimeRecipeStyle)    = NoConvection()
materialize(::CMFMCConvectionSpec, ::AbstractRuntimeRecipeStyle) = CMFMCConvection()
materialize(s::TM5ConvectionSpec, ::AbstractRuntimeRecipeStyle) =
    TM5Convection(; tile_workspace_gib = s.tile_workspace_gib, use_collab_lu = s.use_collab_lu,
                    lmax_conv = s.lmax_conv, n_merge = s.n_merge)
materialize(s::CMFMCMatrixConvectionSpec, ::AbstractRuntimeRecipeStyle) =
    CMFMCMatrixConvection(; tile_workspace_gib = s.tile_workspace_gib,
                            use_collab_lu = s.use_collab_lu, lmax_conv = s.lmax_conv,
                            n_merge = s.n_merge)

# The singleton specs print fine via the default `Base.summary`; only the
# knob-carrying specs need a custom one (shared across both collab-LU variants).
Base.summary(s::AbstractCollabLUConvectionSpec) =
    "$(nameof(typeof(s)))(use_collab_lu=$(s.use_collab_lu), lmax_conv=$(s.lmax_conv), n_merge=$(s.n_merge))"

# =========================================================================
# Advection
# =========================================================================

abstract type AbstractAdvectionSpec end

struct UpwindAdvectionSpec <: AbstractAdvectionSpec end
struct SlopesAdvectionSpec <: AbstractAdvectionSpec end
struct PPMAdvectionSpec    <: AbstractAdvectionSpec end
struct NoAdvectionSpec     <: AbstractAdvectionSpec end

# LinRood is cubed-sphere only and carries the reconstruction order.
struct LinRoodAdvectionSpec <: AbstractAdvectionSpec
    order :: Int
end

function _parse_advection_scheme(section)
    raw = lowercase(String(get(section, "scheme", "upwind")))
    raw == "upwind" && return :upwind
    raw == "slopes" && return :slopes
    raw == "ppm"    && return :ppm
    raw == "none"   && return :none
    (raw == "linrood" || raw == "linrood_ppm") && return :linrood   # legacy alias
    throw(ArgumentError(
        "Unknown [advection] scheme: $(repr(raw)). Supported: upwind | slopes | ppm | linrood | none"))
end

"""
    advection_spec(section) -> AbstractAdvectionSpec

Parse an `[advection]` section into a typed spec. `ppm_order` is only meaningful
for `scheme = "linrood"`; pairing it with `scheme = "ppm"` is rejected (the split
PPM path takes no order knob), matching the old builder.
"""
function advection_spec(section)
    kind = _parse_advection_scheme(section)
    kind === :upwind && return UpwindAdvectionSpec()
    kind === :slopes && return SlopesAdvectionSpec()
    kind === :none   && return NoAdvectionSpec()
    if kind === :ppm
        haskey(section, "ppm_order") && throw(ArgumentError(
            "[advection] `ppm_order` is only valid with `scheme = \"linrood\"`; " *
            "`scheme = \"ppm\"` selects the standard split `PPMScheme()` path."))
        return PPMAdvectionSpec()
    end
    return LinRoodAdvectionSpec(_spec_int(section, "ppm_order", 5, "[advection]"))  # :linrood
end

# materialize — upwind/slopes/ppm/none are topology-independent; LinRood is
# cubed-sphere only (the structured method throws, matching the old builder). This
# collapses the old ~14 `Val`-dispatch methods to these 6.
materialize(::UpwindAdvectionSpec, ::AbstractRuntimeRecipeStyle) = UpwindScheme()
materialize(::SlopesAdvectionSpec, ::AbstractRuntimeRecipeStyle) = SlopesScheme()
materialize(::PPMAdvectionSpec,    ::AbstractRuntimeRecipeStyle) = PPMScheme()
materialize(::NoAdvectionSpec,     ::AbstractRuntimeRecipeStyle) = NoAdvection()
materialize(s::LinRoodAdvectionSpec, ::CubedSphereRuntimeRecipeStyle) = LinRoodPPMScheme(s.order)
materialize(::LinRoodAdvectionSpec, ::AbstractStructuredRuntimeRecipeStyle) = throw(ArgumentError(
    "[advection] `scheme = \"linrood\"` is only available on cubed-sphere runs."))

Base.summary(s::LinRoodAdvectionSpec) = "LinRoodAdvectionSpec(order=$(s.order))"

export AbstractConvectionSpec, AbstractCollabLUConvectionSpec, NoConvectionSpec,
       TM5ConvectionSpec, CMFMCConvectionSpec, CMFMCMatrixConvectionSpec
export AbstractAdvectionSpec, UpwindAdvectionSpec, SlopesAdvectionSpec,
       PPMAdvectionSpec, NoAdvectionSpec, LinRoodAdvectionSpec
export convection_spec, advection_spec, materialize
