function _synchronize_backend!(cfg)
    synchronize_backend!(_cfg_runtime_backend(cfg))
    return nothing
end

"""
    _assert_gpu_residency!(state, cfg)

See `feedback_verify_gpu_runs_on_gpu`. When a GPU backend is
selected, assert that `state.air_mass` lives on that backend. A silent CPU
fallback aborts with a precise error. Called once after model construction,
before the run loop.
"""
function _assert_gpu_residency!(state, cfg)
    backend = _cfg_runtime_backend(cfg)
    is_gpu_backend(backend) || return nothing
    backing = assert_backend_residency!(state.air_mass, backend; label = "state.air_mass")
    wrapper = Base.typename(typeof(backing)).wrapper
    @info @sprintf("[gpu verified] backend=%s backing=%s device=%s",
                   String(backend_name(backend)),
                   String(nameof(wrapper)),
                   backend_device_name(backend))
    return nothing
end

# ===========================================================================
# Model construction (hoisted from run_transport_binary.jl:153-188)
#
# Uses `pack_initial_tracer_mass` (C1b) rather than raw `.* air_mass`:
# bit-exact on DryBasis, errors loudly on MoistBasis without qv
# (correctness rule feedback_vmr_to_mass_basis_aware). No LL/RG config
# in-tree uses MoistBasis, so no behaviour change for shipped configs.
# ===========================================================================

function _allocate_structured_runner_fluxes(mesh::LatLonMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_structured_runner_fluxes(mesh::ReducedGaussianMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_structured_runner_fluxes(mesh, _Nz::Int, _FT, _basis)
    throw(ArgumentError(
        "TransportBinaryDriver model construction requires a lat-lon or " *
        "reduced-Gaussian grid; got $(typeof(mesh))"))
end

function _allocate_cs_runner_fluxes(mesh::CubedSphereMesh, Nz::Int, FT, basis)
    return allocate_face_fluxes(mesh, Nz; FT = FT, basis = basis)
end

function _allocate_cs_runner_fluxes(mesh, _Nz::Int, _FT, _basis)
    throw(ArgumentError(
        "CubedSphereTransportDriver returned incompatible horizontal grid " *
        "$(typeof(mesh)); expected CubedSphereMesh"))
end

function _make_structured_model(driver::TransportBinaryDriver;
                                FT::Type{<:AbstractFloat},
                                recipe,
                                tracer_specs,
                                cfg)
    grid = driver_grid(driver)
    mesh = grid.horizontal
    window = load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)

    tracer_specs_tuple = Tuple(tracer_specs)
    isempty(tracer_specs_tuple) && throw(ArgumentError("at least one tracer must be configured"))

    basis_type = air_mass_basis(driver) == :dry ? DryBasis : MoistBasis
    tracer_names_tup = Tuple(spec.name for spec in tracer_specs_tuple)
    rm_arrays = map(tracer_specs_tuple) do spec
        vmr = build_initial_mixing_ratio(air_mass, grid, spec.init_cfg;
                                         surface_pressure = window.surface_pressure)
        # MoistBasis LL/RG runs would need qv threaded from window.qv —
        # none in-tree today; the packer errors with a precise message.
        return pack_initial_tracer_mass(grid, air_mass, vmr;
                                        mass_basis = basis_type())
    end

    tracer_tuple = NamedTuple{tracer_names_tup}(Tuple(rm_arrays))
    state = CellState(basis_type, air_mass; tracer_tuple...)
    fluxes = _allocate_structured_runner_fluxes(
        mesh, nlevels(grid), FT, basis_type)
    model = TransportModel(state, fluxes, grid, recipe.advection;
                           diffusion = recipe.diffusion,
                           convection = recipe.convection,
                           chemistry = recipe.chemistry)
    adaptor = _backend_array_adapter(cfg)
    return adaptor === Array ? model : Base.invokelatest(Adapt.adapt, adaptor, model)
end

# Snapshot capture and NetCDF writing live in `AtmosTransport.Output`. The
# runner only decides when to sample; the output module owns topology-specific
# diagnostics and file layout.

# ===========================================================================
# Capability validation
#
# Validate TOML physics against binary capabilities BEFORE constructing the
# model, so users get a precise error up front instead of silently failing
# partway through. Runs after `build_runtime_physics_recipe` (which already
# validates kind strings against recipe types) but before model construction
# (which discovers problems at the first load).
# ===========================================================================

function _validate_capability_match(driver, recipe)
    _validate_convection_capability(recipe.convection,
                                     binary_capabilities(driver.reader))
    return nothing
end

# Dispatch on the concrete convection-operator type so a new operator is a
# new method (compile-time coverage), not a new branch in an if-chain. The
# raw `cfg` no longer participates — the recipe has already been built and
# its convection field is authoritative.
_validate_convection_capability(::NoConvection, _caps) = nothing

function _validate_convection_capability(::TM5Convection, caps)
    caps.tm5_convection || throw(ArgumentError(
        "[convection] kind = \"tm5\" requires the binary to carry " *
        "entu, detu, entd, detd; this binary's payload_sections are " *
        "$(caps.payload_sections). Regenerate with a TM5-enabled " *
        "preprocessor or set convection.kind = \"none\"."))
    return nothing
end

function _validate_convection_capability(::CMFMCConvection, caps)
    caps.cmfmc_convection || throw(ArgumentError(
        "[convection] kind = \"cmfmc\" requires the binary to carry " *
        "the cmfmc section; this binary's payload_sections are " *
        "$(caps.payload_sections)."))
    return nothing
end

# The matrix variant has NO Tiedtke fallback — `dtrain` is the explicit
# detrainment rate that closes the continuity derivation
# `entu - detu = cmfmc[k] - cmfmc[k+1]`. A binary with cmfmc but no dtrain is
# hard-rejected up front so the failure mode is actionable at recipe-validation
# time (not at the first window load several seconds later).
function _validate_convection_capability(::CMFMCMatrixConvection, caps)
    (caps.cmfmc_convection && :dtrain in caps.payload_sections) ||
        throw(ArgumentError(
            "[convection] kind = \"cmfmc_matrix\" requires the binary " *
            "to carry both cmfmc AND dtrain payloads (no Tiedtke fallback); " *
            "this binary's payload_sections are $(caps.payload_sections). " *
            "Regenerate the binary with a preprocessor that emits :dtrain, " *
            "or fall back to kind=\"cmfmc\" which has a Tiedtke path."))
    return nothing
end

# Catch-all for any future convection operator. Forces a method to be added
# here when a new operator type appears, which is the whole point of the
# dispatch refactor.
function _validate_convection_capability(op::AbstractConvection, _caps)
    throw(ArgumentError(
        "no _validate_convection_capability method for $(typeof(op)); " *
        "add a dispatch in DrivenRunner.jl when introducing a new convection " *
        "operator type."))
end

# ===========================================================================
# run_driven_simulation — top-level entry
# ===========================================================================

function _validate_input_binary_expectations(caps, input_cfg::AbstractDict,
                                             path::AbstractString)
    if haskey(input_cfg, "expected_nlevel")
        expected = Int(input_cfg["expected_nlevel"])
        caps.nlevel == expected || throw(ArgumentError(
            "[input].expected_nlevel=$expected but $(basename(path)) has " *
            "nlevel=$(caps.nlevel). This usually means the run config is " *
            "pointing at an older preprocessing product."))
    end
    if haskey(input_cfg, "required_preprocessor_contract")
        required = String(input_cfg["required_preprocessor_contract"])
        actual = caps.preprocessor_contract
        actual == required || throw(ArgumentError(
            "[input].required_preprocessor_contract=$(repr(required)) but " *
            "$(basename(path)) declares $(repr(actual))."))
    end
    if _config_bool(input_cfg, "require_adaptive_substeps", false,
                    "[input].require_adaptive_substeps")
        caps.adaptive_substeps === true || throw(ArgumentError(
            "[input].require_adaptive_substeps=true but $(basename(path)) " *
            "does not declare adaptive_substeps=true."))
    end
    return nothing
end

function _log_binary_summary(path::AbstractString, caps)
    schedule = caps.variable_step_schedule ?
        "adaptive" : string(caps.steps_per_window)
    fields = "[" * join(String.(sort(collect(caps.payload_sections))), ",") * "]"
    @info "[binary] $(path) grid=$(caps.grid_type) levels=$(caps.nlevel) " *
          "basis=$(caps.mass_basis) steps/window=$(schedule) fields=$(fields)"
    return nothing
end
