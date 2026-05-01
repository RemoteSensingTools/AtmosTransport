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

abstract type AbstractRuntimeRecipeStyle end
abstract type AbstractStructuredRuntimeRecipeStyle <: AbstractRuntimeRecipeStyle end

struct LatLonRuntimeRecipeStyle <: AbstractStructuredRuntimeRecipeStyle end
struct ReducedGaussianRuntimeRecipeStyle <: AbstractStructuredRuntimeRecipeStyle end
struct CubedSphereRuntimeRecipeStyle <: AbstractRuntimeRecipeStyle end

struct RuntimePhysicsRecipe{AdvT, DiffT, ConvT}
    advection  :: AdvT
    diffusion  :: DiffT
    convection :: ConvT
end

const CSPhysicsRecipe = RuntimePhysicsRecipe

# Plan 40 Commit 2: the flat-411 `catrine_co2` stub is gone. CS tracers
# now flow through the same `build_initial_mixing_ratio` +
# `pack_initial_tracer_mass` pipeline as LL/RG; `kind = "catrine_co2"`
# loads the Catrine NetCDF and regrids + remaps it conservatively onto
# the CS grid. Historical flat-411 behaviour is now expressed as
# `kind = "uniform" background = 4.11e-4`.

@inline _config_symbol(section, key::AbstractString, default::AbstractString) =
    Symbol(lowercase(String(get(section, key, default))))

@inline _advection_section(cfg) = get(cfg, "advection", get(cfg, "run", Dict{String,Any}()))
@inline _diffusion_section(cfg) = get(cfg, "diffusion", Dict{String,Any}())
@inline _convection_section(cfg) = get(cfg, "convection", Dict{String,Any}())

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

function build_runtime_advection(cfg, style::AbstractRuntimeRecipeStyle)
    section = _advection_section(cfg)
    return build_runtime_advection(style,
                                   Val(_config_symbol(section, "scheme", "upwind")),
                                   section)
end

build_runtime_advection(::AbstractStructuredRuntimeRecipeStyle, ::Val{:upwind}, _section) = UpwindScheme()
build_runtime_advection(::AbstractStructuredRuntimeRecipeStyle, ::Val{:slopes}, _section) = SlopesScheme()

function build_runtime_advection(::AbstractStructuredRuntimeRecipeStyle, ::Val{:ppm}, section)
    haskey(section, "ppm_order") &&
        throw(ArgumentError(
            "[advection] `ppm_order` is only valid with `scheme = \"linrood\"`; " *
            "structured runtime `scheme = \"ppm\"` selects `PPMScheme()`."))
    return PPMScheme()
end

function build_runtime_advection(::AbstractStructuredRuntimeRecipeStyle, ::Val{:linrood}, _section)
    throw(ArgumentError(
        "[advection] `scheme = \"linrood\"` is only available on cubed-sphere runs."))
end

build_runtime_advection(style::AbstractStructuredRuntimeRecipeStyle, ::Val{:linrood_ppm}, section) =
    build_runtime_advection(style, Val(:linrood), section)

function build_runtime_advection(::CubedSphereRuntimeRecipeStyle, ::Val{:upwind}, _section)
    return UpwindScheme()
end

function build_runtime_advection(::CubedSphereRuntimeRecipeStyle, ::Val{:slopes}, _section)
    return SlopesScheme()
end

function build_runtime_advection(::CubedSphereRuntimeRecipeStyle, ::Val{:ppm}, section)
    haskey(section, "ppm_order") &&
        throw(ArgumentError(
            "[advection] `ppm_order` is only valid with `scheme = \"linrood\"`; " *
            "`scheme = \"ppm\"` selects the standard split `PPMScheme()` path."))
    return PPMScheme()
end

function build_runtime_advection(::CubedSphereRuntimeRecipeStyle, ::Val{:linrood}, section)
    return LinRoodPPMScheme(Int(get(section, "ppm_order", 5)))
end

build_runtime_advection(style::CubedSphereRuntimeRecipeStyle, ::Val{:linrood_ppm}, section) =
    build_runtime_advection(style, Val(:linrood), section)

function build_runtime_advection(::AbstractRuntimeRecipeStyle, ::Val{name}, _section) where name
    throw(ArgumentError(
        "Unknown [advection] scheme: $(name). Supported: upwind | slopes | ppm | linrood"))
end

function build_runtime_diffusion(cfg, context, ::Type{FT}) where FT
    return build_runtime_diffusion(cfg, _runtime_recipe_style(context), FT)
end

function build_runtime_diffusion(cfg, style::AbstractRuntimeRecipeStyle, ::Type{FT}) where FT
    section = _diffusion_section(cfg)
    # Empty / absent section is explicit "no diffusion".
    isempty(section) && return NoDiffusion()
    # Reject the legacy `type = "..."` schema rather than silently mapping
    # it to NoDiffusion. Configs that said `type = "pbl"` / `"nonlocal_pbl"`
    # etc. expected diffusion to run; the silent fall-through hid that for
    # months. Migrate to `kind` — supported kinds today are "none" and
    # "constant". (Codex Section B P0 fix.)
    haskey(section, "type") && !haskey(section, "kind") &&
        throw(ArgumentError(
            "[diffusion] uses legacy `type = \"$(section["type"])\"`; rename to " *
            "`kind = \"...\"`. Supported kinds: \"none\", \"constant\". Until a " *
            "PBL diffusion operator is wired through the runtime contract, set " *
            "`kind = \"none\"` for no diffusion, or `kind = \"constant\"` with " *
            "a `value = <Kz>` for a uniform Kz."))
    haskey(section, "kind") ||
        throw(ArgumentError(
            "[diffusion] section is present but has no `kind` key. " *
            "Set `kind = \"none\"` for no diffusion, or `kind = \"constant\"` " *
            "with `value = <Kz>` for uniform-Kz diffusion."))
    return build_runtime_diffusion(style,
                                   Val(_config_symbol(section, "kind", "none")),
                                   section,
                                   FT)
end

build_runtime_diffusion(::AbstractRuntimeRecipeStyle, ::Val{:none}, _section, ::Type{FT}) where FT =
    NoDiffusion()

@inline _constant_runtime_kz_field(::LatLonRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 3}(value)
@inline _constant_runtime_kz_field(::ReducedGaussianRuntimeRecipeStyle, value::FT) where FT =
    ConstantField{FT, 2}(value)
@inline _constant_runtime_kz_field(::CubedSphereRuntimeRecipeStyle, value::FT) where FT =
    CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(value), 6))

function build_runtime_diffusion(style::AbstractRuntimeRecipeStyle,
                                 ::Val{:constant},
                                 section,
                                 ::Type{FT}) where FT
    value = FT(get(section, "value", 1.0))
    return ImplicitVerticalDiffusion(; kz_field = _constant_runtime_kz_field(style, value))
end

function build_runtime_diffusion(::AbstractRuntimeRecipeStyle,
                                 ::Val{name},
                                 _section,
                                 ::Type{FT}) where {name, FT}
    throw(ArgumentError(
        "Unknown [diffusion] kind: $(name). Supported: none | constant"))
end

function build_runtime_convection(cfg, context)
    return build_runtime_convection(cfg, _runtime_recipe_style(context))
end

function build_runtime_convection(cfg, style::AbstractRuntimeRecipeStyle)
    section = _convection_section(cfg)
    return build_runtime_convection(style,
                                    Val(_config_symbol(section, "kind", "none")),
                                    section)
end

build_runtime_convection(::AbstractRuntimeRecipeStyle, ::Val{:none}, _section) = NoConvection()
function build_runtime_convection(::AbstractRuntimeRecipeStyle, ::Val{:tm5}, section)
    # `tile_workspace_gib` is the per-topology TM5 column-tile budget
    # in binary GiB. Default 1.0 — fits all production resolutions
    # through C720/L137 with slack on H100. Set lower on memory-tight
    # GPUs (e.g. L40S 48 GiB) or higher to amortize launch overhead.
    budget = Float64(get(section, "tile_workspace_gib", 1.0))
    return TM5Convection(; tile_workspace_gib = budget)
end
build_runtime_convection(::AbstractRuntimeRecipeStyle, ::Val{:cmfmc}, _section) = CMFMCConvection()

function build_runtime_convection(::AbstractRuntimeRecipeStyle, ::Val{name}, _section) where name
    throw(ArgumentError(
        "Unknown [convection] kind: $(name). Supported: none | tm5 | cmfmc"))
end

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
@inline _runtime_has_tm5conv(reader::TransportBinaryReader) = has_tm5conv(reader)
@inline _runtime_has_tm5conv(reader::CubedSphereBinaryReader) = has_tm5conv(reader)
@inline _runtime_has_tm5conv(driver::TransportBinaryDriver) = has_tm5conv(driver.reader)
@inline _runtime_has_tm5conv(driver::CubedSphereTransportDriver) = has_tm5conv(driver.reader)
@inline _runtime_has_cmfmc(reader::TransportBinaryReader) = has_cmfmc(reader)
@inline _runtime_has_cmfmc(reader::CubedSphereBinaryReader) = has_cmfmc(reader)
@inline _runtime_has_cmfmc(driver::TransportBinaryDriver) = has_cmfmc(driver.reader)
@inline _runtime_has_cmfmc(driver::CubedSphereTransportDriver) = has_cmfmc(driver.reader)

function validate_runtime_convection(::AbstractRuntimeRecipeStyle,
                                     ::TM5Convection,
                                     context)
    _runtime_has_tm5conv(context) ||
        throw(ArgumentError(
            "[convection] kind = \"tm5\" requires TM5 convection sections " *
            "(`entu`, `detu`, `entd`, `detd`) in the runtime forcing source."))
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
    )
    return validate_runtime_physics_recipe(recipe, context; halo_width = halo_width)
end

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

# `build_cs_tracer_panels` was a flat-411 stub (plan 23 era). Plan 40
# Commit 2 removed it in favour of the unified pipeline:
#
#     vmr = build_initial_mixing_ratio(air_mass, grid, init_cfg)
#     rm  = pack_initial_tracer_mass(grid, air_mass, vmr;
#                                    mass_basis = DryBasis())
#
# See `src/Models/InitialConditionIO.jl`.

export RuntimePhysicsRecipe, CSPhysicsRecipe
export build_runtime_advection, build_runtime_diffusion, build_runtime_convection
export build_runtime_physics_recipe, validate_runtime_physics_recipe
export configured_halo_width
export build_cs_advection, build_cs_diffusion, build_cs_convection
export build_cs_physics_recipe, validate_cs_physics_recipe
export configured_cs_halo_width
