#!/usr/bin/env julia

using Logging
using TOML
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
include(joinpath(@__DIR__, "..", "run_transport_binary_v2.jl"))

@inline _basis_symbol(::AtmosTransportV2.DryBasis) = "dry"
@inline _basis_symbol(::AtmosTransportV2.MoistBasis) = "moist"

function _parse_snapshot_hours(cfg)
    out_cfg = get(cfg, "output", Dict{String, Any}())
    raw = get(out_cfg, "snapshot_hours", Any[0, 6, 12, 18, 24])
    hours = Int[]
    for value in raw
        push!(hours, Int(value))
    end
    sort!(unique!(hours))
    return hours
end

_column_mean(rm, m) = sum(rm; dims=ndims(rm)) ./ sum(m; dims=ndims(m))

function _capture_snapshot(mesh::AtmosTransportV2.LatLonMesh{FT}, state, tracer_name::Symbol) where FT
    χ = AtmosTransportV2.mixing_ratio(state, tracer_name)
    column = _column_mean(getproperty(state.tracers, tracer_name), state.air_mass)
    surface = permutedims(@view(χ[:, :, size(χ, 3)]), (2, 1))
    column2d = permutedims(dropdims(column; dims=3), (2, 1))
    return (
        lon = collect(mesh.λᶜ),
        lat = collect(mesh.φᶜ),
        surface = Array(surface),
        column_mean = Array(column2d),
    )
end

function _reduced_cell_centers(mesh::AtmosTransportV2.ReducedGaussianMesh{FT}) where FT
    lon = Vector{FT}(undef, AtmosTransportV2.ncells(mesh))
    lat = Vector{FT}(undef, AtmosTransportV2.ncells(mesh))
    for j in 1:AtmosTransportV2.nrings(mesh)
        lons = AtmosTransportV2.ring_longitudes(mesh, j)
        φ = mesh.latitudes[j]
        for i in eachindex(lons)
            c = AtmosTransportV2.cell_index(mesh, i, j)
            lon[c] = lons[i]
            lat[c] = φ
        end
    end
    return lon, lat
end

function _capture_snapshot(mesh::AtmosTransportV2.ReducedGaussianMesh{FT}, state, tracer_name::Symbol) where FT
    χ = AtmosTransportV2.mixing_ratio(state, tracer_name)
    column = _column_mean(getproperty(state.tracers, tracer_name), state.air_mass)
    lon, lat = _reduced_cell_centers(mesh)
    return (
        lon = lon,
        lat = lat,
        surface = Array(@view(χ[:, size(χ, 2)])),
        column_mean = Array(dropdims(column; dims=2)),
    )
end

function _write_snapshot_netcdf(path::AbstractString,
                                mesh::AtmosTransportV2.LatLonMesh{FT},
                                snapshots,
                                metadata) where FT
    mkpath(dirname(path))
    nt = length(snapshots)
    ny = length(snapshots[1].lat)
    nx = length(snapshots[1].lon)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "time", nt)
        defDim(ds, "lat", ny)
        defDim(ds, "lon", nx)

        vtime = defVar(ds, "time_hours", Float64, ("time",))
        vlat = defVar(ds, "lat", Float64, ("lat",))
        vlon = defVar(ds, "lon", Float64, ("lon",))
        vsfc = defVar(ds, "co2_surface", Float64, ("time", "lat", "lon"))
        vcol = defVar(ds, "co2_column_mean", Float64, ("time", "lat", "lon"))
        vair = defVar(ds, "air_mass_total", Float64, ("time",))
        vrm = defVar(ds, "tracer_mass_total", Float64, ("time",))

        ds.attrib["grid_type"] = "latlon"
        ds.attrib["scheme"] = metadata.scheme
        ds.attrib["tracer_name"] = String(metadata.tracer_name)
        ds.attrib["mass_basis"] = metadata.mass_basis
        ds.attrib["source_binary"] = metadata.binary_label

        vlat[:] = snapshots[1].lat
        vlon[:] = snapshots[1].lon
        for (i, snap) in enumerate(snapshots)
            vtime[i] = snap.time_hours
            vsfc[i, :, :] = snap.surface
            vcol[i, :, :] = snap.column_mean
            vair[i] = snap.air_mass_total
            vrm[i] = snap.tracer_mass_total
        end
    finally
        close(ds)
    end
    return path
end

function _write_snapshot_netcdf(path::AbstractString,
                                mesh::AtmosTransportV2.ReducedGaussianMesh{FT},
                                snapshots,
                                metadata) where FT
    mkpath(dirname(path))
    nt = length(snapshots)
    nc = length(snapshots[1].lon)
    ds = NCDataset(path, "c")
    try
        defDim(ds, "time", nt)
        defDim(ds, "cell", nc)

        vtime = defVar(ds, "time_hours", Float64, ("time",))
        vlon = defVar(ds, "lon_cell", Float64, ("cell",))
        vlat = defVar(ds, "lat_cell", Float64, ("cell",))
        vsfc = defVar(ds, "co2_surface", Float64, ("time", "cell"))
        vcol = defVar(ds, "co2_column_mean", Float64, ("time", "cell"))
        vair = defVar(ds, "air_mass_total", Float64, ("time",))
        vrm = defVar(ds, "tracer_mass_total", Float64, ("time",))

        ds.attrib["grid_type"] = "reduced_gaussian"
        ds.attrib["scheme"] = metadata.scheme
        ds.attrib["tracer_name"] = String(metadata.tracer_name)
        ds.attrib["mass_basis"] = metadata.mass_basis
        ds.attrib["source_binary"] = metadata.binary_label

        vlon[:] = snapshots[1].lon
        vlat[:] = snapshots[1].lat
        for (i, snap) in enumerate(snapshots)
            vtime[i] = snap.time_hours
            vsfc[i, :] = snap.surface
            vcol[i, :] = snap.column_mean
            vair[i] = snap.air_mass_total
            vrm[i] = snap.tracer_mass_total
        end
    finally
        close(ds)
    end
    return path
end

function _capture_named_snapshot!(snapshots, model, tracer_name::Symbol, time_hours::Real)
    snap = _capture_snapshot(model.grid.horizontal, model.state, tracer_name)
    push!(snapshots, merge(snap, (
        time_hours = Float64(time_hours),
        air_mass_total = Float64(AtmosTransportV2.total_air_mass(model.state)),
        tracer_mass_total = Float64(AtmosTransportV2.total_mass(model.state, tracer_name)),
    )))
    return nothing
end

_open_driver(path::AbstractString, ::Type{FT}) where FT = AtmosTransportV2.TransportBinaryDriver(path; FT=FT, arch=AtmosTransportV2.CPU())

function export_snapshots(binary_paths::Vector{String}, cfg)
    FT = Symbol(get(get(cfg, "numerics", Dict{String, Any}()), "float_type", "Float64")) == :Float32 ? Float32 : Float64
    run_cfg = get(cfg, "run", Dict{String, Any}())
    scheme_name = Symbol(lowercase(String(get(run_cfg, "scheme", "upwind"))))
    tracer_name = Symbol(get(run_cfg, "tracer_name", "CO2"))
    reset_air_mass_each_window = Bool(get(run_cfg, "reset_air_mass_each_window", false))
    init_cfg = get(cfg, "init", Dict{String, Any}())
    out_cfg = get(cfg, "output", Dict{String, Any}())
    out_path = expanduser(String(get(out_cfg, "snapshot_file",
        joinpath(homedir(), "data", "AtmosTransport", "output", "src_v2_snapshots", "transport_v2_snapshots.nc"))))

    isempty(binary_paths) && throw(ArgumentError("no binary_paths configured"))
    ensure_gpu_runtime!(cfg)

    first_driver = _open_driver(first(binary_paths), FT)
    model = make_model(first_driver; FT=FT, scheme_name=scheme_name, tracer_name=tracer_name, init_cfg=init_cfg, cfg=cfg)
    @info "Backend: $(backend_label(cfg))"

    stop_window_cfg = get(run_cfg, "stop_window", nothing)
    local_stop = stop_window_cfg === nothing ? AtmosTransportV2.total_windows(first_driver) : Int(stop_window_cfg)

    target_hours = _parse_snapshot_hours(cfg)
    target_windows = sort!(unique!(vcat(0, [clamp(round(Int, hour * 3600 / AtmosTransportV2.window_dt(first_driver)), 0, local_stop) for hour in target_hours])))

    snapshots = NamedTuple[]
    captured = Set{Int}()
    if 0 in target_windows
        _capture_named_snapshot!(snapshots, model, tracer_name, 0.0)
        push!(captured, 0)
    end

    global_window = 0
    synchronize_backend!(cfg)
    t0 = time()
    for (idx, path) in enumerate(binary_paths)
        driver = idx == 1 ? first_driver : _open_driver(path, FT)
        # Per-driver stop_window is the minimum of this binary's window count
        # and the remaining target windows (`local_stop - global_window`).
        # Previously the first driver used `local_stop` directly, which
        # failed when `stop_window` in the config exceeded Day 1's window
        # count (e.g. 48h run spanning two daily binaries).
        stop_window = min(AtmosTransportV2.total_windows(driver),
                          max(0, local_stop - global_window))
        stop_window <= 0 && break
        sim = AtmosTransportV2.DrivenSimulation(model, driver;
                                                start_window=1,
                                                stop_window=stop_window,
                                                initialize_air_mass=(idx == 1),
                                                reset_air_mass_each_window=reset_air_mass_each_window)
        for _ in 1:stop_window
            AtmosTransportV2.run_window!(sim)
            global_window += 1
            if global_window in target_windows && !(global_window in captured)
                _capture_named_snapshot!(snapshots, model, tracer_name, global_window * AtmosTransportV2.window_dt(driver) / 3600)
                push!(captured, global_window)
            end
        end
        close(driver)
    end
    synchronize_backend!(cfg)
    @info @sprintf("Captured %d snapshots in %.2f s", length(snapshots), time() - t0)

    metadata = (
        scheme = String(scheme_name),
        tracer_name = tracer_name,
        mass_basis = _basis_symbol(AtmosTransportV2.mass_basis(model.state)),
        binary_label = join(basename.(binary_paths), ", "),
    )
    return _write_snapshot_netcdf(out_path, model.grid.horizontal, snapshots, metadata)
end

function main()
    isempty(ARGS) && error("Usage: julia --project=. scripts/diagnostics/export_transport_v2_snapshots.jl config.toml")
    cfg = TOML.parsefile(expanduser(ARGS[1]))
    binary_paths = [expanduser(String(p)) for p in get(get(cfg, "input", Dict{String, Any}()), "binary_paths", String[])]
    out_path = export_snapshots(binary_paths, cfg)
    @info "Wrote snapshot NetCDF: $out_path"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
