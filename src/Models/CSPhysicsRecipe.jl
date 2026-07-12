"""
    RuntimePhysicsRecipe

Validated operator composition for runtime-driven transport runners.

The recipe layer separates:

- component selection from TOML (`build_runtime_advection`,
  `build_runtime_diffusion`, `build_runtime_convection`)
- topology-specific construction rules (lat-lon, reduced Gaussian,
  cubed sphere) via dispatch on a lightweight runtime-style trait
- capability checks against readers / drivers
  (`validate_runtime_physics_recipe`)

This keeps the CLI scripts thin and prevents topology-specific
`if/elseif` trees from growing in parallel.
"""

# The runtime-style traits (`AbstractRuntimeRecipeStyle` + the LatLon/RG/CS
# variants) now live in `RuntimeRecipeStyles.jl`, included before this file (the
# `RuntimePhysicsSpecs.jl` `materialize` methods dispatch on them). The
# `_runtime_recipe_style(grid/driver/reader)` resolvers stay below.

struct RuntimePhysicsRecipe{AdvT, DiffT, ConvT, ChemT}
    advection  :: AdvT
    diffusion  :: DiffT
    convection :: ConvT
    chemistry  :: ChemT
end

# Backward-compat: 3-arg constructor defaults chemistry to `NoChemistry`.
RuntimePhysicsRecipe(adv, diff, conv) = RuntimePhysicsRecipe(adv, diff, conv, NoChemistry())

const CSPhysicsRecipe = RuntimePhysicsRecipe

# The flat-411 `catrine_co2` stub is gone. CS tracers
# now flow through the same `build_initial_mixing_ratio` +
# `pack_initial_tracer_mass` pipeline as LL/RG; `kind = "catrine_co2"`
# loads the Catrine NetCDF and regrids + remaps it conservatively onto
# the CS grid. Historical flat-411 behaviour is now expressed as
# `kind = "uniform" background = 4.11e-4`.

function _advection_section(cfg)
    run = get(cfg, "run", Dict{String,Any}())
    if haskey(cfg, "advection")
        for key in ("scheme", "ppm_order")
            haskey(run, key) && throw(ArgumentError(
                "Advection option `[run].$(key)` is ambiguous because `[advection]` " *
                "is present. Move `$(key)` into `[advection]`; legacy `[run].$(key)` " *
                "is only accepted when `[advection]` is absent."))
        end
        return cfg["advection"]
    end
    return run
end
@inline _diffusion_section(cfg) = get(cfg, "diffusion", Dict{String,Any}())
@inline _convection_section(cfg) = get(cfg, "convection", Dict{String,Any}())
@inline _chemistry_section(cfg) = get(cfg, "chemistry", Dict{String,Any}())

@inline _runtime_recipe_style(style::AbstractRuntimeRecipeStyle) = style
@inline _runtime_recipe_style(::AtmosGrid{<:LatLonMesh}) = LatLonRuntimeRecipeStyle()
@inline _runtime_recipe_style(::AtmosGrid{<:ReducedGaussianMesh}) = ReducedGaussianRuntimeRecipeStyle()
@inline _runtime_recipe_style(::AtmosGrid{<:CubedSphereMesh}) = CubedSphereRuntimeRecipeStyle()
@inline _runtime_recipe_style(driver::AbstractMetDriver) = _runtime_recipe_style(driver_grid(driver))
@inline _runtime_recipe_style(::CubedSphereBinaryReader) = CubedSphereRuntimeRecipeStyle()

@inline function _runtime_recipe_style(reader::TransportBinaryReader)
    if grid_type(reader) === :latlon && horizontal_topology(reader) === :structureddirectional
        return LatLonRuntimeRecipeStyle()
    elseif grid_type(reader) === :reduced_gaussian && horizontal_topology(reader) === :faceindexed
        return ReducedGaussianRuntimeRecipeStyle()
    end
    throw(ArgumentError(
        "No runtime recipe style is defined for transport binary grid/topology " *
        "$(grid_type(reader)) / $(horizontal_topology(reader))."))
end

function _runtime_recipe_style(context)
    throw(ArgumentError(
        "No runtime recipe style is defined for context $(typeof(context))."))
end

function build_runtime_advection(cfg, context)
    return build_runtime_advection(cfg, _runtime_recipe_style(context))
end

# Thin wrapper: parse the `[advection]` section into a typed `AbstractAdvectionSpec`
# once (validated), then materialize the scheme. The structured-vs-CS split lives in
# `materialize` (LinRood is CS-only); spec types + parser live in `RuntimePhysicsSpecs.jl`.
build_runtime_advection(cfg, style::AbstractRuntimeRecipeStyle) =
    materialize(advection_spec(_advection_section(cfg)), style)

function build_runtime_diffusion(cfg, context, ::Type{FT}) where FT
    return build_runtime_diffusion(cfg, _runtime_recipe_style(context), FT, context)
end

# Thin wrapper: parse the `[diffusion]` section into a typed `AbstractDiffusionSpec`
# once (validating the legacy `type=`/missing-`kind`/unknown-kind cases), then
# materialize. Diffusion is the one family whose `materialize` needs `style` (Kz-field
# rank / CS-only gating), `FT` (precision), AND the runtime `context` (driver/reader,
# for the Kz-cache shape + binary-capability gate). The spec stays context-free; the
# context work happens in `materialize`, which calls the helpers below. Spec types +
# parser + `materialize` live in `RuntimePhysicsSpecs.jl`.
build_runtime_diffusion(cfg, style::AbstractRuntimeRecipeStyle, ::Type{FT},
                        context = nothing) where FT =
    materialize(diffusion_spec(_diffusion_section(cfg)), style, FT, context)

# --- Context helpers the diffusion `materialize` methods call ----------------
# Kept here (not in RuntimePhysicsSpecs.jl) because they resolve concrete
# grid/reader/driver types; tests stub them via `AtmosTransport.Models._runtime_has_*`
# and `AtmosTransport.Models._pbl_cache_shape`.

@inline _constant_runtime_kz_field(::LatLonRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 3}(value)
@inline _constant_runtime_kz_field(::ReducedGaussianRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 2}(value)
@inline _constant_runtime_kz_field(::CubedSphereRuntimeRecipeStyle, value::FT) where FT =
    CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(value), 6))

function _pbl_cache_shape(context)
    throw(ArgumentError(
        "[diffusion] kind = \"pbl\" requires a cubed-sphere reader or driver " *
        "context so the Kz cache can be sized."))
end
_pbl_cache_shape(reader::CubedSphereBinaryReader) =
    (reader.header.Nc, reader.header.Nc, reader.header.nlevel)
_pbl_cache_shape(driver::CubedSphereTransportDriver) =
    (driver.reader.header.Nc, driver.reader.header.Nc, driver.reader.header.nlevel)

@inline _runtime_has_surface(_context) = false
_runtime_has_surface(reader::TransportBinaryReader) = MetDrivers.has_surface(reader)
_runtime_has_surface(reader::CubedSphereBinaryReader) = MetDrivers.has_surface(reader)
_runtime_has_surface(driver::TransportBinaryDriver) = MetDrivers.has_surface(driver.reader)
_runtime_has_surface(driver::CubedSphereTransportDriver) = MetDrivers.has_surface(driver.reader)

@inline _runtime_has_gchp_vdiff(_context) = false
_runtime_has_gchp_vdiff(reader::CubedSphereBinaryReader) =
    MetDrivers.has_surface(reader) && MetDrivers.has_vdiff_fields(reader)
_runtime_has_gchp_vdiff(driver::CubedSphereTransportDriver) =
    _runtime_has_gchp_vdiff(driver.reader)

_runtime_has_precomputed_kz(_context) = false
_runtime_has_precomputed_kz(reader::CubedSphereBinaryReader) =
    any(s in reader.header.payload_sections for s in (:dkg, :kz))
_runtime_has_precomputed_kz(driver::CubedSphereTransportDriver) =
    _runtime_has_precomputed_kz(driver.reader)

_runtime_has_precomputed_dkg(_context) = false
_runtime_has_precomputed_dkg(reader::CubedSphereBinaryReader) =
    :dkg in reader.header.payload_sections
_runtime_has_precomputed_dkg(driver::CubedSphereTransportDriver) =
    _runtime_has_precomputed_dkg(driver.reader)

# Concrete-operator validators must require the matching section on every
# daily binary. The broad capability above is only for first-context
# materialization and backward-compatible test contexts.
_runtime_has_legacy_precomputed_kz(context) = _runtime_has_precomputed_kz(context)
_runtime_has_legacy_precomputed_kz(reader::CubedSphereBinaryReader) =
    :kz in reader.header.payload_sections
_runtime_has_legacy_precomputed_kz(driver::CubedSphereTransportDriver) =
    _runtime_has_legacy_precomputed_kz(driver.reader)

function build_runtime_convection(cfg, context)
    return build_runtime_convection(cfg, _runtime_recipe_style(context))
end

# Thin wrapper: parse the `[convection]` section into a typed `AbstractConvectionSpec`
# (validated, incl. the lmax_conv/n_merge-needs-use_collab_lu guard) once, then
# materialize the operator. Spec types + parser + `materialize` live in
# `RuntimePhysicsSpecs.jl`. `style` is threaded for API uniformity with
# build_runtime_advection/diffusion; convection materialization is
# topology-independent, so the `materialize` methods ignore it.
build_runtime_convection(cfg, style::AbstractRuntimeRecipeStyle) =
    materialize(convection_spec(_convection_section(cfg)), style)

@inline validate_runtime_advection(::AbstractRuntimeRecipeStyle,
                                   ::AbstractAdvectionScheme,
                                   _context) = nothing
@inline validate_runtime_diffusion(::AbstractRuntimeRecipeStyle,
                                   ::AbstractDiffusion,
                                   _context) = nothing
@inline validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                    ::NoConvection,
                                    _context) = nothing

function validate_runtime_advection(::AbstractStructuredRuntimeRecipeStyle,
                                    ::LinRoodPPMScheme,
                                    _context)
    throw(ArgumentError(
        "LinRoodPPMScheme is only supported on cubed-sphere runtimes."))
end

@inline _runtime_has_tm5conv(_context) = false
@inline _runtime_has_cmfmc(_context) = false
@inline _runtime_has_tm5conv(reader::TransportBinaryReader) = MetDrivers.has_tm5conv(reader)
@inline _runtime_has_tm5conv(reader::CubedSphereBinaryReader) = MetDrivers.has_tm5conv(reader)
@inline _runtime_has_tm5conv(driver::TransportBinaryDriver) = MetDrivers.has_tm5conv(driver.reader)
@inline _runtime_has_tm5conv(driver::CubedSphereTransportDriver) = MetDrivers.has_tm5conv(driver.reader)
@inline _runtime_has_cmfmc(reader::TransportBinaryReader) = MetDrivers.has_cmfmc(reader)
@inline _runtime_has_cmfmc(reader::CubedSphereBinaryReader) = MetDrivers.has_cmfmc(reader)
@inline _runtime_has_cmfmc(driver::TransportBinaryDriver) = MetDrivers.has_cmfmc(driver.reader)
@inline _runtime_has_cmfmc(driver::CubedSphereTransportDriver) = MetDrivers.has_cmfmc(driver.reader)

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::TM5Convection,
                                     context)
    _runtime_has_tm5conv(context) ||
        throw(ArgumentError(
            "[convection] kind = \"tm5\" requires TM5 convection sections " *
            "(`entu`, `detu`, `entd`, `detd`) in the runtime forcing source."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    ::ImplicitVerticalDiffusion{FT, <:WindowPBLKzField},
                                    context) where FT
    _runtime_has_surface(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"pbl\" requires pblh/ustar/pbl_hflux/t2m sections " *
            "in every cubed-sphere transport binary."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    op::ImplicitVerticalDiffusion{FT, <:LocalHoltslagBovilleKzField},
                                    context) where FT
    _runtime_has_gchp_vdiff(context) ||
        throw(ArgumentError(
            "[diffusion] kind = \"geoschem_holtslag_boville_vdiff\" requires " *
            "pblh/ustar/pbl_hflux/t2m and vdiff_u/vdiff_v/vdiff_t/vdiff_qv " *
            "sections in every cubed-sphere transport binary."))
    # GCHP parity requires emissions to be applied as a boundary condition
    # inside the same diffusive solve (see vdiff_mod.F90:679, gchp_chunk_mod.F90:1296).
    # Our default `SplitSurfaceFluxCoupling` does V(dt/2) → S(dt) → V(dt/2)
    # Strang, which is a valid integration but does NOT match GCHP. Warn at
    # config-load time so users picking this Kz field for GCHP parity know
    # to flip `surface_flux_boundary = true` (or equivalent recipe knob).
    if !(op.surface_flux_coupling isa DiffusiveSurfaceFluxBoundary)
        @warn """
        [diffusion] kind = "geoschem_holtslag_boville_vdiff" was selected but
        the surface-flux coupling is $(typeof(op.surface_flux_coupling)).
        For GCHP VDIFF parity, surface emissions must be applied as a boundary
        condition inside the diffusion solve (reference: vdiff_mod.F90:679,
        gchp_chunk_mod.F90:1296). Switch to `DiffusiveSurfaceFluxBoundary`
        (set `surface_flux_boundary = true` in the recipe) for GCHP-equivalent
        behavior. See memory/diffusion_full_pipeline_audit_2026_05_25.md (D3).
        """
    end
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    ::ImplicitVerticalDiffusion{FT, <:PrecomputedCSKzField},
                                    context) where FT
    _runtime_has_legacy_precomputed_kz(context) ||
        throw(ArgumentError(
            "The active diffusion operator was materialized from a legacy `:kz` " *
            "binary, so every cubed-sphere binary in this run must also carry `:kz`. " *
            "Do not mix legacy `:kz` and exact `:dkg` daily binaries; regenerate " *
            "the complete interval with include_tm5_diffusion=true."))
    return nothing
end

function validate_runtime_diffusion(::CubedSphereRuntimeRecipeStyle,
                                    ::ImplicitVerticalDiffusion{FT, <:PrecomputedCSDkgField},
                                    context) where FT
    _runtime_has_precomputed_dkg(context) || throw(ArgumentError(
        "TM5 precomputed diffusion requires a `:dkg` section in every cubed-sphere transport binary."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::CMFMCConvection,
                                     context)
    _runtime_has_cmfmc(context) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc\" requires CMFMC convection forcing " *
            "in the runtime forcing source."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::CMFMCMatrixConvection,
                                     context)
    # The matrix variant requires both cmfmc AND dtrain — see the
    # capability check in `DrivenRunner._validate_capability_match` for
    # the detailed reason. Recipe-level validators don't have access to
    # `caps.payload_sections` so we can only check the cmfmc capability
    # here; DrivenRunner enforces the dtrain requirement directly.
    _runtime_has_cmfmc(context) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc_matrix\" requires CMFMC convection " *
            "forcing (cmfmc + dtrain) in the runtime forcing source. The " *
            "matrix variant reads the same binary sections as kind=\"cmfmc\" " *
            "and derives entu/detu at runtime — no Tiedtke fallback."))
    return nothing
end

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     op::AbstractConvection,
                                     _context)
    throw(ArgumentError(
        "Runtime recipe validation does not support convection operator $(typeof(op)) yet."))
end

function validate_runtime_halo_width(scheme::AbstractAdvectionScheme, halo_width::Integer)
    min_hp = required_halo_width(scheme)
    halo_width >= min_hp ||
        throw(ArgumentError(
            "[run] halo padding Hp=$(halo_width) is too small for $(typeof(scheme)); " *
            "need Hp >= $(min_hp)."))
    return nothing
end

@inline validate_runtime_combination(::AbstractRuntimeRecipeStyle,
                                     ::AbstractAdvectionScheme,
                                     ::AbstractDiffusion,
                                     ::AbstractConvection,
                                     _context) = nothing

function validate_runtime_physics_recipe(recipe::RuntimePhysicsRecipe,
                                         context;
                                         halo_width::Union{Nothing, Integer} = nothing)
    style = _runtime_recipe_style(context)
    validate_runtime_advection(style, recipe.advection, context)
    validate_runtime_diffusion(style, recipe.diffusion, context)
    validate_runtime_convection(style, recipe.convection, context)
    validate_runtime_combination(style,
                                 recipe.advection,
                                 recipe.diffusion,
                                 recipe.convection,
                                 context)
    halo_width === nothing || validate_runtime_halo_width(recipe.advection, halo_width)
    return recipe
end

function build_runtime_physics_recipe(cfg,
                                      context,
                                      ::Type{FT};
                                      halo_width::Union{Nothing, Integer} = nothing) where FT
    recipe = RuntimePhysicsRecipe(
        build_runtime_advection(cfg, context),
        build_runtime_diffusion(cfg, context, FT),
        build_runtime_convection(cfg, context),
        build_runtime_chemistry(cfg, FT),
    )
    return validate_runtime_physics_recipe(recipe, context; halo_width = halo_width)
end

"""
    build_runtime_chemistry(cfg, ::Type{FT}) -> AbstractChemistryOperator

Read the optional `[chemistry]` TOML section and produce the
corresponding chemistry operator.

Supported `kind` values:

- `"none"` (default) — `NoChemistry()`.
- `"decay"` — `ExponentialDecay(FT; ...)`. Half-lives are read from
  the `half_lives_seconds` table:

      [chemistry]
      kind = "decay"
      [chemistry.half_lives_seconds]
      rn222 = 330350.4   # 3.8235 days

  The keyword name must match the corresponding `[tracers.<name>]`
  symbol that the run is carrying (case-insensitive — the builder
  symbolizes the key as-is and `ExponentialDecay.apply!` resolves
  it against `state.tracer_names` at call time).

Thin wrapper: parse the `[chemistry]` section into a typed
`AbstractChemistrySpec` once (validated), then materialize at run precision
`FT`. Spec types + parser live in `RuntimePhysicsSpecs.jl`.
"""
build_runtime_chemistry(cfg, ::Type{FT}) where FT =
    materialize(chemistry_spec(_chemistry_section(cfg)), FT)

function configured_halo_width(cfg, scheme::AbstractAdvectionScheme)
    run_cfg = get(cfg, "run", Dict{String,Any}())
    default_hp = required_halo_width(scheme)

    if haskey(run_cfg, "Hp") && haskey(run_cfg, "halo_padding")
        hp = Int(run_cfg["Hp"])
        halo_padding = Int(run_cfg["halo_padding"])
        hp == halo_padding || throw(ArgumentError(
            "[run] `Hp` ($(hp)) and `halo_padding` ($(halo_padding)) disagree; use one value."))
    end

    return haskey(run_cfg, "Hp") ? Int(run_cfg["Hp"]) :
           Int(get(run_cfg, "halo_padding", default_hp))
end

build_cs_advection(cfg) = build_runtime_advection(cfg, CubedSphereRuntimeRecipeStyle())
build_cs_diffusion(cfg, ::Type{FT}) where FT =
    build_runtime_diffusion(cfg, CubedSphereRuntimeRecipeStyle(), FT)
build_cs_convection(cfg) = build_runtime_convection(cfg, CubedSphereRuntimeRecipeStyle())
validate_cs_physics_recipe(recipe::RuntimePhysicsRecipe, context; halo_width::Union{Nothing, Integer} = nothing) =
    validate_runtime_physics_recipe(recipe, context; halo_width = halo_width)
build_cs_physics_recipe(cfg, context, ::Type{FT}; halo_width::Union{Nothing, Integer} = nothing) where FT =
    build_runtime_physics_recipe(cfg, context, FT; halo_width = halo_width)
configured_cs_halo_width(cfg, scheme::AbstractAdvectionScheme) = configured_halo_width(cfg, scheme)

# CS tracers flow through the unified pipeline:
#
#     vmr = build_initial_mixing_ratio(air_mass, grid, init_cfg)
#     rm  = pack_initial_tracer_mass(grid, air_mass, vmr;
#                                    mass_basis = DryBasis())
#
# See `src/Models/InitialConditionIO.jl`.

export RuntimePhysicsRecipe, CSPhysicsRecipe
export build_runtime_advection, build_runtime_diffusion, build_runtime_convection
export build_runtime_chemistry
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export configured_halo_width
export build_cs_advection, build_cs_diffusion, build_cs_convection
export build_cs_physics_recipe, validate_cs_physics_recipe
export configured_cs_halo_width
