"""
    DrivenRunner

Library-level entry point for the driven transport runtime.

Plan 40 Commit 6a hoists the LL/RG runner here from
`scripts/run_transport_binary.jl` so both the old script (now a thin
shim) and the forthcoming unified `scripts/run_transport.jl` share
one implementation. Commit 6b will fold the CS-specific flow from
`scripts/run_cs_driven.jl` into the same top-level
`run_driven_simulation(cfg)` with dispatch driven by the first
binary's header (`inspect_binary(first_path).grid_type`).

## Ownership boundary

- **Binary header** (`grid_type`, `mass_basis`, `payload_sections`,
  panel convention) — authoritative for topology and capability.
  Accessed here via `binary_capabilities(driver.reader)` and
  `air_mass_basis(driver)`.
- **TOML `[input]`** — either an explicit `binary_paths = [...]`
  list (Shape A) or a `folder + start_date + end_date
  (+ file_pattern)` block (Shape B). Both are resolved to a sorted
  `Vector{String}` by `expand_binary_paths`.
- **TOML physics** (`[advection]` / `[diffusion]` / `[convection]`)
  — validated against binary capabilities by
  `validate_runtime_physics_recipe` / `build_runtime_physics_recipe`
  before the loop.
- **TOML `[tracers.*]`** — tracer specs consumed via
  `build_initial_mixing_ratio` + `pack_initial_tracer_mass`
  (basis-aware per `feedback_vmr_to_mass_basis_aware`) and
  `build_surface_flux_sources`.

## GPU residency (feedback_verify_gpu_runs_on_gpu)

When `[architecture].use_gpu = true`, the runner asserts that
`parent(state.air_mass) isa CuArray` after model construction and
prints a `[gpu verified] …` line. A silent CPU fallback aborts the
run with a precise error message.
"""
module DrivenRunner

using Adapt
using Printf: @sprintf, @printf
using NCDatasets
using Logging

using ..State: AbstractMassBasis, DryBasis, MoistBasis, CellState,
                CubedSphereState, total_air_mass, total_mass, tracer_names,
                tracer_index, get_tracer
using ..Grids: AtmosGrid, LatLonMesh, ReducedGaussianMesh, CubedSphereMesh,
                nx, ny, nrings, ring_longitudes, cell_index, ncells,
                nlevels
using ..Architectures: CPU, GPU
using ..MetDrivers: TransportBinaryDriver, CubedSphereTransportDriver,
                     load_transport_window, driver_grid, air_mass_basis,
                     total_windows, window_dt, binary_capabilities,
                     inspect_binary
using ..InitialConditionIO: build_initial_mixing_ratio,
                             pack_initial_tracer_mass,
                             build_surface_flux_sources
using ..BinaryPathExpander: expand_binary_paths
# TransportModel + DrivenSimulation live alongside us in the Models module;
# reach up to the parent and pull them in.
using ..Models: TransportModel
import ..Models: DrivenSimulation, run_window!, run!, step!, allocate_face_fluxes
# Physics-recipe helpers: `build_runtime_physics_recipe` /
# `validate_runtime_physics_recipe` are defined in `CSPhysicsRecipe.jl`
# (loaded before us in Models). Pull them in so we don't have to stutter
# through `Main.AtmosTransport.*`.
using ..Models: build_runtime_physics_recipe, validate_runtime_physics_recipe,
                 configured_halo_width, build_cs_advection

export run_driven_simulation, TransportTracerSpec

# ===========================================================================
# TOML parsing — tracer specs (hoisted from run_transport_binary.jl:57-100)
# ===========================================================================

struct TransportTracerSpec
    name             :: Symbol
    init_cfg         :: Dict{String, Any}
    surface_flux_cfg :: Dict{String, Any}
end

_copy_cfg_dict(cfg) = Dict{String, Any}(String(k) => v for (k, v) in pairs(cfg))

function _tracer_init_cfg(tracer_cfg)
    if haskey(tracer_cfg, "init")
        return _copy_cfg_dict(tracer_cfg["init"])
    end
    cfg = Dict{String, Any}()
    for key in ("kind", "background", "lon0_deg", "lat0_deg", "sigma_lon_deg",
                "sigma_lat_deg", "amplitude", "file", "variable", "time_index")
        haskey(tracer_cfg, key) && (cfg[key] = tracer_cfg[key])
    end
    isempty(cfg) && return Dict{String, Any}("kind" => "uniform", "background" => 0.0)
    return cfg
end

function _tracer_surface_flux_cfg(tracer_cfg)
    if haskey(tracer_cfg, "surface_flux")
        return _copy_cfg_dict(tracer_cfg["surface_flux"])
    end
    cfg = Dict{String, Any}()
    for (src_key, dst_key) in (("surface_flux_kind", "kind"),
                               ("surface_flux_file", "file"),
                               ("surface_flux_variable", "variable"),
                               ("surface_flux_time_index", "time_index"),
                               ("surface_flux_month", "month"),
                               ("surface_flux_scale", "scale"))
        haskey(tracer_cfg, src_key) && (cfg[dst_key] = tracer_cfg[src_key])
    end
    return cfg
end

function _parse_tracer_specs(cfg)
    tracers_cfg = get(cfg, "tracers", nothing)
    tracers_cfg isa AbstractDict || return nothing
    names = sort!(collect(keys(tracers_cfg)))
    isempty(names) && throw(ArgumentError("config has [tracers] but no tracer sections"))
    return Tuple(TransportTracerSpec(Symbol(name),
                                     _tracer_init_cfg(tracers_cfg[name]),
                                     _tracer_surface_flux_cfg(tracers_cfg[name])) for name in names)
end

# ===========================================================================
# GPU runtime helpers (hoisted from run_transport_binary.jl:101-138)
# ===========================================================================

@inline _cfg_use_gpu(cfg) = Bool(get(get(cfg, "architecture", Dict{String, Any}()), "use_gpu", false))

function _ensure_gpu_runtime!(cfg)
    _cfg_use_gpu(cfg) || return false
    Sys.isapple() && throw(ArgumentError("driven runner GPU path is only wired for CUDA hosts"))
    if !isdefined(Main, :CUDA)
        Core.eval(Main, :(using CUDA))
    end
    CUDA = Core.eval(Main, :CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        throw(ArgumentError("CUDA runtime is not functional on this host"))
    Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

function _backend_array_adapter(cfg)
    if _cfg_use_gpu(cfg)
        _ensure_gpu_runtime!(cfg)
        return getproperty(Core.eval(Main, :CUDA), :CuArray)
    end
    return Array
end

function _backend_label(cfg)
    if _cfg_use_gpu(cfg)
        _ensure_gpu_runtime!(cfg)
        CUDA = Core.eval(Main, :CUDA)
        device_name = Base.invokelatest(getproperty(CUDA, :name),
                                        Base.invokelatest(getproperty(CUDA, :device)))
        return "GPU (CUDA, $(device_name))"
    end
    return "CPU"
end

function _synchronize_backend!(cfg)
    if _cfg_use_gpu(cfg)
        _ensure_gpu_runtime!(cfg)
        Base.invokelatest(getproperty(Core.eval(Main, :CUDA), :synchronize))
    end
    return nothing
end

"""
    _assert_gpu_residency!(state, cfg)

Plan 40 Commit 6a / `feedback_verify_gpu_runs_on_gpu`. When
`use_gpu = true`, assert that `state.air_mass` lives on the GPU. A
silent CPU fallback aborts with a precise error. Called once after
model construction, before the run loop.
"""
function _assert_gpu_residency!(state, cfg)
    _cfg_use_gpu(cfg) || return nothing
    backing = state.air_mass isa Tuple ? parent(state.air_mass[1]) : parent(state.air_mass)
    CUDA = Core.eval(Main, :CUDA)
    CuArrayType = getproperty(CUDA, :CuArray)
    backing isa CuArrayType || throw(ErrorException(
        "[gpu residency check] use_gpu = true but state.air_mass is $(typeof(backing)); " *
        "CPU fallback aborted. Verify CUDA extension loaded and `adapt(CuArray, model)` " *
        "completed without error."))
    device_name = Base.invokelatest(getproperty(CUDA, :name),
                                    Base.invokelatest(getproperty(CUDA, :device)))
    @info @sprintf("[gpu verified] backing=%s device=%s", nameof(typeof(backing).name.wrapper), device_name)
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

function _make_structured_model(driver::TransportBinaryDriver;
                                FT::Type{<:AbstractFloat},
                                recipe,
                                tracer_specs,
                                cfg)
    grid = driver_grid(driver)
    window = load_transport_window(driver, 1)
    air_mass = copy(window.air_mass)

    tracer_specs_tuple = Tuple(tracer_specs)
    isempty(tracer_specs_tuple) && throw(ArgumentError("at least one tracer must be configured"))

    basis_type = air_mass_basis(driver) == :dry ? DryBasis : MoistBasis
    tracer_names_tup = Tuple(spec.name for spec in tracer_specs_tuple)
    rm_arrays = map(tracer_specs_tuple) do spec
        vmr = build_initial_mixing_ratio(air_mass, grid, spec.init_cfg)
        # MoistBasis LL/RG runs would need qv threaded from window.qv —
        # none in-tree today; the packer errors with a precise message.
        return pack_initial_tracer_mass(grid, air_mass, vmr;
                                        mass_basis = basis_type())
    end

    tracer_tuple = NamedTuple{tracer_names_tup}(Tuple(rm_arrays))
    state = CellState(basis_type, air_mass; tracer_tuple...)
    fluxes = allocate_face_fluxes(grid.horizontal, nlevels(grid);
                                  FT = FT, basis = basis_type)
    model = TransportModel(state, fluxes, grid, recipe.advection;
                           diffusion = recipe.diffusion,
                           convection = recipe.convection)
    adaptor = _backend_array_adapter(cfg)
    return adaptor === Array ? model : Adapt.adapt(adaptor, model)
end

# ===========================================================================
# Snapshot capture (hoisted from run_transport_binary.jl:195-232)
# ===========================================================================

"""
    _capture_snapshot(model) -> Dict{Symbol, Vector{Float64}}

Column-mean mixing ratio per tracer. Handles structured grids
(`(Nx, Ny, Nz)` 3D arrays → flattened `(Nx * Ny,)`) and face-indexed
grids like reduced Gaussian (`(ncells, Nz)` 2D → `(ncells,)`).
"""
function _capture_snapshot(model)
    names = tracer_names(model.state)
    m = Array(model.state.air_mass)
    result = Dict{Symbol, Vector{Float64}}()
    for name in names
        rm = Array(getfield(model.state.tracers, name))
        if ndims(rm) == 3
            Nx, Ny, Nz = size(rm)
            col_mean = zeros(Float64, Nx * Ny)
            for j in 1:Ny, i in 1:Nx
                sum_rm = 0.0; sum_m = 0.0
                for k in 1:Nz
                    sum_rm += Float64(rm[i, j, k])
                    sum_m  += Float64(m[i, j, k])
                end
                col_mean[(j - 1) * Nx + i] = sum_m > 0 ? sum_rm / sum_m : 0.0
            end
        elseif ndims(rm) == 2
            Nc, Nz = size(rm)
            col_mean = zeros(Float64, Nc)
            for c in 1:Nc
                sum_rm = 0.0; sum_m = 0.0
                for k in 1:Nz
                    sum_rm += Float64(rm[c, k])
                    sum_m  += Float64(m[c, k])
                end
                col_mean[c] = sum_m > 0 ? sum_rm / sum_m : 0.0
            end
        else
            error("_capture_snapshot: unsupported tracer array ndims=$(ndims(rm))")
        end
        result[name] = col_mean
    end
    return result
end

# ===========================================================================
# NetCDF snapshot output (hoisted from run_transport_binary.jl:243-363)
# ===========================================================================

function _write_snapshot_netcdf(path, snapshots, snapshot_hours, grid)
    isempty(snapshots) && return
    mkpath(dirname(path))
    mesh = grid.horizontal
    ntime = length(snapshots)

    NCDataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["grid"] = summary(mesh)
        if mesh isa LatLonMesh
            _write_snapshot_ll!(ds, snapshots, snapshot_hours, mesh, ntime)
        elseif mesh isa ReducedGaussianMesh
            _write_snapshot_rg!(ds, snapshots, snapshot_hours, mesh, ntime)
        else
            error("Unsupported mesh type for snapshot output: $(typeof(mesh))")
        end
    end
    @info "Saved snapshots: $path ($(length(snapshots)) times)"
end

function _write_snapshot_ll!(ds, snapshots, snapshot_hours, mesh, ntime)
    Nx_, Ny_ = nx(mesh), ny(mesh)
    defDim(ds, "lon", Nx_); defDim(ds, "lat", Ny_); defDim(ds, "time", ntime)

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib = Dict("units" => "degrees_east", "long_name" => "longitude",
                                 "standard_name" => "longitude"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib = Dict("units" => "degrees_north", "long_name" => "latitude",
                                 "standard_name" => "latitude"))
    v_time = defVar(ds, "time", Float64, ("time",),
                    attrib = Dict("units" => "hours since start", "long_name" => "time"))

    v_lon[:] = Float64.(mesh.λᶜ)
    v_lat[:] = Float64.(mesh.φᶜ)
    v_time[:] = snapshot_hours[1:ntime]

    for name in keys(first(snapshots))
        v = defVar(ds, "$(name)_column_mean", Float64, ("lon", "lat", "time"),
                   attrib = Dict("units" => "mol mol-1",
                                 "long_name" => "Column-mean $name VMR"))
        for t in 1:ntime
            v[:, :, t] = reshape(snapshots[t][name], Nx_, Ny_)
        end
    end
end

function _write_snapshot_rg!(ds, snapshots, snapshot_hours, mesh, ntime)
    nr = nrings(mesh)
    Nlon = maximum(mesh.nlon_per_ring)
    Nlat = nr
    dlon = 360.0 / Nlon
    out_lons = [(i - 0.5) * dlon for i in 1:Nlon]
    out_lats = Float64.(mesh.latitudes)

    nn_map = zeros(Int, Nlon, Nlat)
    for j in 1:Nlat
        nlon_ring = mesh.nlon_per_ring[j]
        dlon_ring = 360.0 / nlon_ring
        for i in 1:Nlon
            i_ring = clamp(round(Int, out_lons[i] / dlon_ring + 0.5), 1, nlon_ring)
            nn_map[i, j] = cell_index(mesh, i_ring, j)
        end
    end

    defDim(ds, "lon", Nlon); defDim(ds, "lat", Nlat); defDim(ds, "time", ntime)
    ds.attrib["grid_type"] = "reduced_gaussian_regridded"
    ds.attrib["nrings"] = nr
    ds.attrib["regridding"] = "nearest-neighbor from reduced Gaussian"

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib = Dict("units" => "degrees_east", "long_name" => "longitude",
                                 "standard_name" => "longitude"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib = Dict("units" => "degrees_north", "long_name" => "latitude",
                                 "standard_name" => "latitude"))
    v_time = defVar(ds, "time", Float64, ("time",),
                    attrib = Dict("units" => "hours since start", "long_name" => "time"))

    v_lon[:] = out_lons
    v_lat[:] = out_lats
    v_time[:] = snapshot_hours[1:ntime]

    for name in keys(first(snapshots))
        v = defVar(ds, "$(name)_column_mean", Float64, ("lon", "lat", "time"),
                   attrib = Dict("units" => "mol mol-1",
                                 "long_name" => "Column-mean $name VMR"))
        for t in 1:ntime
            flat = snapshots[t][name]
            regridded = zeros(Float64, Nlon, Nlat)
            for j in 1:Nlat, i in 1:Nlon
                regridded[i, j] = flat[nn_map[i, j]]
            end
            v[:, :, t] = regridded
        end
    end
end

# ===========================================================================
# Capability validation (plan 40 Commit 6a)
#
# Validate TOML physics against binary capabilities BEFORE constructing the
# model, so users get a precise error up front instead of silently failing
# partway through. Runs after `build_runtime_physics_recipe` (which already
# validates kind strings against recipe types) but before model construction
# (which discovers problems at the first load).
# ===========================================================================

function _validate_capability_match(driver, recipe, cfg)
    caps = binary_capabilities(driver.reader)

    # Convection kind vs binary sections
    conv_kind = Symbol(lowercase(String(get(get(cfg, "convection", Dict()), "kind", "none"))))
    if conv_kind === :tm5 && !caps.tm5_convection
        throw(ArgumentError(
            "[convection] kind = \"tm5\" requires the binary to carry " *
            "entu, detu, entd, detd; this binary's payload_sections are " *
            "$(caps.payload_sections). Regenerate with a TM5-enabled " *
            "preprocessor or set convection.kind = \"none\"."))
    end
    if conv_kind === :cmfmc && !caps.cmfmc_convection
        throw(ArgumentError(
            "[convection] kind = \"cmfmc\" requires the binary to carry " *
            "the cmfmc section; this binary's payload_sections are " *
            "$(caps.payload_sections)."))
    end
    return nothing
end

# ===========================================================================
# run_driven_simulation — top-level entry
# ===========================================================================

"""
    run_driven_simulation(cfg::AbstractDict) -> TransportModel

Run a driven transport simulation from a TOML config. Resolves
`[input]` to a sorted binary list via `expand_binary_paths`, picks
the right driver based on the binary's `grid_type` header field,
validates physics-vs-capability, verifies GPU residency when
requested, runs the loop, optionally captures column-mean snapshots
to NetCDF, and returns the terminal `TransportModel`.

Plan 40 Commit 6a supports LL/RG only (structured and
reduced-Gaussian). CS dispatch is added in Commit 6b.
"""
function run_driven_simulation(cfg::AbstractDict)
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = expand_binary_paths(input_cfg)
    isempty(binary_paths) &&
        throw(ArgumentError("[input] resolved to an empty binary list"))
    # Plan 40 Commit 6b: dispatch on the first binary's grid_type —
    # the ownership boundary (binary header owns topology, TOML owns
    # physics kinds). The capability probe also runs the load-time
    # gates (stale-binary, cm-continuity) as a side effect of opening
    # the reader in `inspect_binary`.
    caps = inspect_binary(first(binary_paths); io = devnull)
    if caps.grid_type === :cubed_sphere
        return _run_driven_simulation_cs(binary_paths, cfg)
    else
        return _run_driven_simulation_structured(binary_paths, cfg)
    end
end

function _run_driven_simulation_structured(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ?
         Float32 : Float64
    run_cfg = get(cfg, "run", Dict{String, Any}())
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)
    haskey(run_cfg, "reset_air_mass_each_window") &&
        @debug "run.reset_air_mass_each_window is ignored (plan 39 Commit G removed the flag)"

    init_cfg = get(cfg, "init", Dict{String, Any}())
    tracer_specs = something(_parse_tracer_specs(cfg),
                             (TransportTracerSpec(Symbol(get(run_cfg, "tracer_name", "CO2")),
                                                  _copy_cfg_dict(init_cfg),
                                                  Dict{String, Any}()),))

    output_cfg = get(cfg, "output", Dict{String, Any}())
    snapshot_hours = Float64.(get(output_cfg, "snapshot_hours", Float64[]))
    snapshot_file = expanduser(String(get(output_cfg, "snapshot_file", "")))
    do_snapshots = !isempty(snapshot_hours) && !isempty(snapshot_file)

    _ensure_gpu_runtime!(cfg)

    # Open first driver, build recipe, validate capability, build model
    first_driver = TransportBinaryDriver(first(binary_paths);
                                          FT = FT,
                                          arch = CPU())
    recipe = build_runtime_physics_recipe(cfg, first_driver, FT)
    _validate_capability_match(first_driver, recipe, cfg)

    model = _make_structured_model(first_driver;
                                    FT = FT, recipe = recipe,
                                    tracer_specs = tracer_specs, cfg = cfg)
    _assert_gpu_residency!(model.state, cfg)

    grid_of_first = driver_grid(first_driver)
    surface_sources = build_surface_flux_sources(grid_of_first, tracer_specs, FT)
    m0 = total_air_mass(model.state)
    tracer_masses0 = Dict(name => total_mass(model.state, name)
                          for name in tracer_names(model.state))
    source_tracers = Set(source.tracer_name for source in surface_sources)

    @info "Backend: $(_backend_label(cfg))"
    @info "Physics: advection=$(nameof(typeof(recipe.advection))) " *
          "diffusion=$(nameof(typeof(recipe.diffusion))) " *
          "convection=$(nameof(typeof(recipe.convection)))"
    for source in surface_sources
        @info @sprintf("Surface source %s total mass rate: %.12e kg/s",
                       String(source.tracer_name),
                       Float64(sum(source.cell_mass_rate)))
    end

    snapshots = Dict{Symbol, Vector{Float64}}[]
    snap_idx = 1
    total_elapsed_hours = 0.0

    if do_snapshots && snap_idx <= length(snapshot_hours) &&
       abs(snapshot_hours[snap_idx]) < 0.5
        push!(snapshots, _capture_snapshot(model))
        @info @sprintf("Snapshot %d at t=%.0fh", snap_idx, 0.0)
        snap_idx += 1
    end

    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver :
                 TransportBinaryDriver(path; FT = FT, arch = CPU())
        validate_runtime_physics_recipe(recipe, driver)
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) : Int(stop_window_override)
        initialize_air_mass = idx == 1
        sim = DrivenSimulation(model, driver;
                               start_window = start_window,
                               stop_window = stop_window,
                               initialize_air_mass = initialize_air_mass,
                               surface_sources = surface_sources)
        model = sim.model
        if !initialize_air_mass
            boundary_rel = maximum(abs.(model.state.air_mass .- sim.window.air_mass)) /
                           max(maximum(abs.(sim.window.air_mass)), eps(FT))
            @info @sprintf("Boundary air-mass mismatch before %s: %.3e",
                           basename(path), boundary_rel)
        end
        window_hours = Float64(window_dt(driver)) / 3600.0
        n_windows = stop_window - start_window + 1
        @info @sprintf("Running %s with %s on %s (%d windows)",
                       basename(path),
                       nameof(typeof(recipe.advection)),
                       summary(driver_grid(driver).horizontal),
                       n_windows)
        _synchronize_backend!(cfg)
        t0 = time()

        if do_snapshots
            for _ in 1:n_windows
                run_window!(sim)
                total_elapsed_hours += window_hours
                while snap_idx <= length(snapshot_hours) &&
                      abs(total_elapsed_hours - snapshot_hours[snap_idx]) < 0.5
                    push!(snapshots, _capture_snapshot(model))
                    @info @sprintf("Snapshot %d at t=%.0fh",
                                   snap_idx, total_elapsed_hours)
                    snap_idx += 1
                end
            end
        else
            run!(sim)
            total_elapsed_hours += n_windows * window_hours
        end

        _synchronize_backend!(cfg)
        @info @sprintf("Finished %s in %.2f s", basename(path), time() - t0)
        close(driver)
    end

    if do_snapshots && !isempty(snapshots)
        _write_snapshot_netcdf(snapshot_file, snapshots, snapshot_hours,
                               driver_grid(first_driver))
    end

    m1 = total_air_mass(model.state)
    @info @sprintf("Final air-mass change vs initial state:  %.3e", (m1 - m0) / m0)
    for name in tracer_names(model.state)
        rm0 = Float64(tracer_masses0[name])
        rm1 = Float64(total_mass(model.state, name))
        if name in source_tracers
            @info @sprintf("Final tracer mass for %s (with source): %.12e kg",
                           String(name), rm1)
        elseif abs(rm0) > eps(Float64)
            @info @sprintf("Final tracer-mass drift for %s:         %.3e",
                           String(name), (rm1 - rm0) / rm0)
        else
            @info @sprintf("Final tracer mass for %s:               %.12e kg",
                           String(name), rm1)
        end
    end
    return model
end

# ===========================================================================
# CS runner (plan 40 Commit 6b, hoisted from scripts/run_cs_driven.jl)
# ===========================================================================

"""Strip halo padding and return the interior `(Nc, Nc, Nz)` view as a CPU Array."""
_cs_interior(panel, Hp, Nc) = Array(panel[Hp + 1 : Hp + Nc, Hp + 1 : Hp + Nc, :])

_cfg_float_type(cfg) = let s = get(get(cfg, "numerics", Dict()), "float_type", "Float64")
    s == "Float32" ? Float32 : Float64
end

function _cfg_architecture(cfg)
    if _cfg_use_gpu(cfg)
        _ensure_gpu_runtime!(cfg)
        return GPU()
    end
    return CPU()
end

function _write_snapshot_cs!(nc_path, snap_data, snap_m, snapshot_hours, mesh, Nz)
    Nc = mesh.Nc
    ntime = length(snap_data)
    isfile(nc_path) && rm(nc_path)
    mkpath(dirname(nc_path))
    NCDataset(nc_path, "c") do ds
        defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6)
        defDim(ds, "lev", Nz); defDim(ds, "time", ntime)
        ds.attrib["Conventions"] = "CF"
        ds.attrib["Source"] = "AtmosTransport.jl run_driven_simulation (CS)"
        ds.attrib["Nc"] = Nc

        defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since t0"))[:] = Float64.(snapshot_hours[1:ntime])
        defVar(ds, "nf", Int32, ("nf",))[:] = Int32.(1:6)

        m_v = defVar(ds, "air_mass", Float64, ("Xdim", "Ydim", "nf", "lev", "time"))
        for t in 1:ntime, p in 1:6
            m_v[:, :, p, :, t] = snap_m[t][p]
        end
        for name in keys(snap_data[1])
            v = defVar(ds, String(name), Float64,
                       ("Xdim", "Ydim", "nf", "lev", "time"))
            for t in 1:ntime, p in 1:6
                rm_p = snap_data[t][name][p]
                m_p  = snap_m[t][p]
                # Column-diagnostic VMR = rm / m; clamp tiny denominators
                v[:, :, p, :, t] = rm_p ./ max.(m_p, eps(Float64))
            end
        end
    end
    @info "Saved snapshots: $nc_path (C$Nc × 6 panels × $Nz levels × $ntime times)"
end

function _run_driven_simulation_cs(binary_paths::Vector{String}, cfg)
    FT   = _cfg_float_type(cfg)
    arch = _cfg_architecture(cfg)

    run_cfg = get(cfg, "run", Dict{String, Any}())
    advection = build_cs_advection(cfg)
    Hp = configured_halo_width(cfg, advection)
    stop_window_override = get(run_cfg, "stop_window", nothing)

    output_cfg = get(cfg, "output", Dict{String, Any}())
    snapshot_hours = Float64.(get(output_cfg, "snapshot_hours", Float64[0, 24, 48]))
    snapshot_file  = expanduser(String(get(output_cfg, "snapshot_file",
                                           "cs_driven_snapshot.nc")))

    tracers_cfg = get(cfg, "tracers", Dict{String, Any}())
    isempty(tracers_cfg) && error("[tracers] must define at least one tracer")
    tracer_init = Dict(Symbol(n) => get(c, "init",
                                        Dict("kind" => "uniform", "background" => 0.0))
                       for (n, c) in tracers_cfg)

    # First driver + model (reuses air_mass from window 1)
    driver1 = CubedSphereTransportDriver(first(binary_paths);
                                          FT = FT, arch = arch, Hp = Hp)
    recipe  = build_runtime_physics_recipe(cfg, driver1, FT; halo_width = Hp)
    _validate_capability_match(driver1, recipe, cfg)

    grid    = driver_grid(driver1)
    mesh    = grid.horizontal
    window1 = load_transport_window(driver1, 1)
    air_mass = window1.air_mass
    Nz = size(air_mass[1], 3)

    # Honor the binary's mass_basis — `DrivenSimulation._check_basis_compatibility`
    # compares `mass_basis(model.state/fluxes)` against `air_mass_basis(driver)`
    # and throws if they diverge. Hardcoding `DryBasis` would trip a
    # runtime ArgumentError on any moist-basis CS binary.
    basis_sym = air_mass_basis(driver1)
    BasisT    = basis_sym === :dry   ? DryBasis   :
                basis_sym === :moist ? MoistBasis :
                error("CS binary has unsupported mass_basis $(basis_sym); expected :dry or :moist")

    # Plan 40 Commit 2 + 1c: CS tracers flow through the unified IC
    # pipeline. DryBasis is the default per invariant 14; MoistBasis
    # requires qv from window1 (feedback_vmr_to_mass_basis_aware), which
    # CS windows do not carry today — so moist binaries error explicitly
    # here rather than producing silently wrong tracer mass.
    basis_sym === :moist &&
        error("CS driven runner does not yet support moist-basis binaries: " *
              "`pack_initial_tracer_mass` needs qv, which `CubedSphereTransportWindow` " *
              "does not expose. Regenerate the binary on dry basis " *
              "(`regrid_ll_transport_binary_to_cs.jl --mass-basis dry`), " *
              "or extend the CS window + this runner to thread qv.")

    tracer_kwargs = Dict{Symbol, NTuple{6, typeof(air_mass[1])}}()
    for (name, init_cfg) in tracer_init
        vmr = build_initial_mixing_ratio(air_mass, grid, init_cfg)
        tracer_kwargs[name] = pack_initial_tracer_mass(grid, air_mass, vmr;
                                                       mass_basis = BasisT())
    end

    state  = CubedSphereState(BasisT, mesh, air_mass; tracer_kwargs...)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = BasisT)

    model = TransportModel(state, fluxes, grid, recipe.advection;
                            diffusion  = recipe.diffusion,
                            convection = recipe.convection)
    _assert_gpu_residency!(model.state, cfg)

    @info @sprintf("CS driven runner  C%d × %d levels  Hp=%d  %s  FT=%s  %s",
                   mesh.Nc, Nz, Hp, typeof(recipe.advection).name.name, FT,
                   _cfg_use_gpu(cfg) ? "GPU" : "CPU")
    @info @sprintf("Physics: advection=%s  diffusion=%s  convection=%s",
                   typeof(recipe.advection).name.name,
                   typeof(recipe.diffusion).name.name,
                   typeof(recipe.convection).name.name)
    @info @sprintf("Tracers: %s   Binaries: %d → %s",
                   join(String.(keys(tracer_init)), ", "),
                   length(binary_paths), snapshot_file)

    # Snapshot storage
    snap_data = Dict{Symbol, NTuple{6, Array{FT, 3}}}[]
    snap_m    = NTuple{6, Array{FT, 3}}[]
    snap_taken = Float64[]

    function capture_cs!(hour_total)
        cur_m = ntuple(p -> _cs_interior(state.air_mass[p], Hp, mesh.Nc), 6)
        cur_tracers = Dict(n => ntuple(p -> _cs_interior(get_tracer(state, n)[p],
                                                          Hp, mesh.Nc), 6)
                           for n in keys(tracer_init))
        push!(snap_m, cur_m)
        push!(snap_data, cur_tracers)
        push!(snap_taken, hour_total)
    end

    snap_idx = 1
    total_hour = 0.0
    if snap_idx <= length(snapshot_hours) && abs(snapshot_hours[snap_idx]) < 0.5
        capture_cs!(0.0)
        @info @sprintf("  Snapshot %d at t=%.1fh", snap_idx, 0.0)
        snap_idx += 1
    end

    t0 = time()
    drivers = [driver1;
               [CubedSphereTransportDriver(expanduser(p); FT = FT,
                                           arch = arch, Hp = Hp)
                for p in binary_paths[2:end]]]

    for driver in drivers
        validate_runtime_physics_recipe(recipe, driver; halo_width = Hp)
        stop_window = stop_window_override === nothing ?
                      total_windows(driver) :
                      min(Int(stop_window_override), total_windows(driver))
        window_hours = window_dt(driver) / 3600.0

        # Plan-39 Commit G removed the window-boundary air_mass reset, so
        # the cross-day handoff is continuity-consistent. We rebuild the
        # sim around each day's driver; state + physics carry over.
        if driver !== driver1
            fluxes_d = allocate_face_fluxes(mesh, Nz; FT = FT, basis = BasisT)
            model = TransportModel(state, fluxes_d, grid, recipe.advection;
                                    diffusion  = recipe.diffusion,
                                    convection = recipe.convection)
        end
        sim = DrivenSimulation(model, driver;
                               start_window = 1, stop_window = stop_window)

        while sim.iteration < sim.final_iteration
            step!(sim)
            if sim.iteration % sim.steps_per_window == 0
                total_hour += window_hours
                while snap_idx <= length(snapshot_hours) &&
                      abs(total_hour - snapshot_hours[snap_idx]) < 0.5
                    capture_cs!(total_hour)
                    @info @sprintf("  Snapshot %d at t=%.1fh",
                                   snap_idx, total_hour)
                    snap_idx += 1
                end
            end
        end
        close(driver)
    end

    @info @sprintf("Done: %.1fs  (%d snapshots, final t=%.1fh)",
                   time() - t0, length(snap_data), total_hour)

    for name in keys(tracer_init)
        @info @sprintf("  %s total mass: %.6e kg", name, total_mass(state, name))
    end

    _write_snapshot_cs!(snapshot_file, snap_data, snap_m, snap_taken, mesh, Nz)
    return model
end

end # module DrivenRunner
