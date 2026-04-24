function _var_attrib(; units::AbstractString,
                     long_name::AbstractString,
                     coordinates::Union{Nothing, AbstractString}=nothing,
                     grid_mapping::Union{Nothing, AbstractString}=nothing,
                     cell_methods::Union{Nothing, AbstractString}=nothing,
                     extra=Dict{String, Any}())
    attrs = Dict{String, Any}("units" => units, "long_name" => long_name)
    coordinates === nothing || (attrs["coordinates"] = coordinates)
    grid_mapping === nothing || (attrs["grid_mapping"] = grid_mapping)
    cell_methods === nothing || (attrs["cell_methods"] = cell_methods)
    for (k, v) in extra
        attrs[k] = v
    end
    return attrs
end

_tracer_units(mass_basis_sym::Symbol) =
    mass_basis_sym === :dry ? "mol mol-1 dry" : "mol mol-1"

function _payload_deflate_level(options::SnapshotWriteOptions)
    return options.deflate_level == 0 ? nothing : options.deflate_level
end

function _payload_shuffle(options::SnapshotWriteOptions)
    return options.deflate_level > 0 && options.shuffle
end

function _def_payload_var(ds, name::AbstractString, T::DataType, dims;
                          attrib, options::SnapshotWriteOptions)
    return defVar(ds, name, T, dims;
                  attrib = attrib,
                  deflatelevel = _payload_deflate_level(options),
                  shuffle = _payload_shuffle(options))
end

function _cs_stack3(panels::NTuple{6, <:AbstractArray})
    Nc1, Nc2, Nz = size(panels[1])
    out = Array{Float64}(undef, Nc1, Nc2, 6, Nz)
    @inbounds for p in 1:6
        out[:, :, p, :] = Float64.(panels[p])
    end
    return out
end

function _cs_stack2(panels::NTuple{6, <:AbstractArray})
    Nc1, Nc2 = size(panels[1])
    out = Array{Float64}(undef, Nc1, Nc2, 6)
    @inbounds for p in 1:6
        out[:, :, p] = Float64.(panels[p])
    end
    return out
end

function _rg_rasterize(native::AbstractVector, nn_map::AbstractMatrix{Int})
    out = Array{Float64}(undef, size(nn_map))
    @inbounds for idx in eachindex(nn_map)
        out[idx] = Float64(native[nn_map[idx]])
    end
    return out
end

function _validate_snapshot_inputs(frames, mesh, mass_basis_sym::Symbol)
    isempty(frames) && throw(ArgumentError("write_snapshot_netcdf requires at least one SnapshotFrame"))
    _check_same_keys(frames)
    _check_mass_basis(frames, mass_basis_sym)
    _check_frame_shapes(frames, mesh)
    return nothing
end

"""
    write_snapshot_netcdf(path, frames, grid; mass_basis=:dry, options=SnapshotWriteOptions())

Write topology-aware runtime snapshots to NetCDF.

The output contains full per-level VMR fields, stored air mass, layer
mass-per-area diagnostics, column air mass per area, and tracer column means.
Reduced-Gaussian files also carry a legacy lon/lat raster view for quick plots.
Cubed-sphere files carry panel lon/lat coordinates and a `cubed_sphere`
grid-mapping variable modeled after GEOS-Chem diagnostics.
"""
function write_snapshot_netcdf(path::AbstractString,
                               frames::AbstractVector{<:SnapshotFrame},
                               grid::AtmosGrid;
                               mass_basis::Symbol=:dry,
                               options::SnapshotWriteOptions=SnapshotWriteOptions())
    expanded = expanduser(String(path))
    _ensure_parent_dir(expanded)
    isfile(expanded) && rm(expanded)
    mesh = grid.horizontal
    _validate_snapshot_inputs(frames, mesh, mass_basis)
    Nz = _nlevel(first(frames), mesh)
    times = [frame.time_hours for frame in frames]
    tracer_keys = _check_same_keys(frames)

    NCDataset(expanded, "c") do ds
        _define_common_attributes!(ds, mesh, frames, mass_basis)
        geometry = _define_geometry!(ds, mesh, Nz, times)
        _write_snapshot_payload!(ds, mesh, frames, tracer_keys, geometry,
                                 mass_basis, options)
    end
    @info @sprintf("Saved snapshots: %s (%d frame(s), %s, mass_basis=%s)",
                   expanded, length(frames), summary(mesh), mass_basis)
    return expanded
end

function _write_snapshot_payload!(ds, mesh::LatLonMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions)
    T = options.float_type
    air = _def_payload_var(ds, "air_mass", T, ("lon", "lat", "lev", "time"),
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = "lon lat"),
                           options = options)
    air_area = _def_payload_var(ds, "air_mass_per_area", T, ("lon", "lat", "lev", "time"),
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = "lon lat"),
                                options = options)
    col_air = _def_payload_var(ds, "column_air_mass_per_area", T, ("lon", "lat", "time"),
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = "lon lat",
                                                    cell_methods = "lev: sum"),
                               options = options)

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    for name in tracer_keys
        s = String(name)
        tracer_vars[name] = _def_payload_var(ds, s, T, ("lon", "lat", "lev", "time"),
                                             attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                  long_name = "per-layer $(s) mixing ratio",
                                                                  coordinates = "lon lat"),
                                             options = options)
        tracer_cm_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, ("lon", "lat", "time"),
                                                attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                     long_name = "air-mass-weighted column-mean $(s) mixing ratio",
                                                                     coordinates = "lon lat",
                                                                     cell_methods = "lev: mean"),
                                                options = options)
        tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, ("lon", "lat", "time"),
                                                 attrib = _var_attrib(units = "kg m-2",
                                                                      long_name = "column model tracer mass per area for $(s)",
                                                                      coordinates = "lon lat",
                                                                      cell_methods = "lev: sum",
                                                                      extra = Dict("description" =>
                                                                          "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                 options = options)
    end

    for (t, frame) in enumerate(frames)
        air[:, :, :, t] = T.(frame.air_mass)
        air_area[:, :, :, t] = T.(layer_mass_per_area(frame.air_mass, mesh))
        col_air[:, :, t] = T.(column_mass_per_area(frame.air_mass, mesh))
        for name in tracer_keys
            tracer_vars[name][:, :, :, t] = T.(mixing_ratio_field(frame.air_mass, frame.tracers[name]))
            tracer_cm_vars[name][:, :, t] = T.(column_mean_mixing_ratio(frame.air_mass, frame.tracers[name]))
            tracer_col_vars[name][:, :, t] = T.(column_mass_per_area(frame.tracers[name], mesh))
        end
    end
    return nothing
end

function _write_snapshot_payload!(ds, mesh::ReducedGaussianMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions)
    T = options.float_type
    air = _def_payload_var(ds, "air_mass", T, ("cell", "lev", "time"),
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = "cell_lon cell_lat"),
                           options = options)
    air_area = _def_payload_var(ds, "air_mass_per_area", T, ("cell", "lev", "time"),
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = "cell_lon cell_lat"),
                                options = options)
    col_air = _def_payload_var(ds, "column_air_mass_per_area", T, ("cell", "time"),
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = "cell_lon cell_lat",
                                                    cell_methods = "lev: sum"),
                               options = options)

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_native_vars = Dict{Symbol, Any}()
    tracer_cm_raster_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    for name in tracer_keys
        s = String(name)
        tracer_vars[name] = _def_payload_var(ds, s, T, ("cell", "lev", "time"),
                                             attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                  long_name = "native per-layer $(s) mixing ratio",
                                                                  coordinates = "cell_lon cell_lat"),
                                             options = options)
        tracer_cm_native_vars[name] = _def_payload_var(ds, "$(s)_column_mean_native", T, ("cell", "time"),
                                                       attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                            long_name = "native air-mass-weighted column-mean $(s) mixing ratio",
                                                                            coordinates = "cell_lon cell_lat",
                                                                            cell_methods = "lev: mean"),
                                                       options = options)
        tracer_cm_raster_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, ("lon", "lat", "time"),
                                                       attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                            long_name = "diagnostic lon-lat raster column-mean $(s) mixing ratio",
                                                                            coordinates = "lon lat",
                                                                            cell_methods = "lev: mean",
                                                                            extra = Dict("regridding" => ds.attrib["regridding"])),
                                                       options = options)
        tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, ("cell", "time"),
                                                 attrib = _var_attrib(units = "kg m-2",
                                                                      long_name = "native column model tracer mass per area for $(s)",
                                                                      coordinates = "cell_lon cell_lat",
                                                                      cell_methods = "lev: sum",
                                                                      extra = Dict("description" =>
                                                                          "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                 options = options)
    end

    nn_map = geometry.nn_map
    for (t, frame) in enumerate(frames)
        air[:, :, t] = T.(frame.air_mass)
        air_area[:, :, t] = T.(layer_mass_per_area(frame.air_mass, mesh))
        col_air[:, t] = T.(column_mass_per_area(frame.air_mass, mesh))
        for name in tracer_keys
            cm = column_mean_mixing_ratio(frame.air_mass, frame.tracers[name])
            tracer_vars[name][:, :, t] = T.(mixing_ratio_field(frame.air_mass, frame.tracers[name]))
            tracer_cm_native_vars[name][:, t] = T.(cm)
            tracer_cm_raster_vars[name][:, :, t] = T.(_rg_rasterize(cm, nn_map))
            tracer_col_vars[name][:, t] = T.(column_mass_per_area(frame.tracers[name], mesh))
        end
    end
    return nothing
end

function _write_snapshot_payload!(ds, mesh::CubedSphereMesh, frames, tracer_keys,
                                  geometry, mass_basis_sym::Symbol,
                                  options::SnapshotWriteOptions)
    T = options.float_type
    dims5 = ("Xdim", "Ydim", "nf", "lev", "time")
    dims4 = ("Xdim", "Ydim", "nf", "time")
    coord = "lons lats"
    air = _def_payload_var(ds, "air_mass", T, dims5,
                           attrib = _var_attrib(units = "kg",
                                                long_name = "stored air mass",
                                                coordinates = coord,
                                                grid_mapping = "cubed_sphere"),
                           options = options)
    air_area = _def_payload_var(ds, "air_mass_per_area", T, dims5,
                                attrib = _var_attrib(units = "kg m-2",
                                                     long_name = "stored layer air mass per area",
                                                     coordinates = coord,
                                                     grid_mapping = "cubed_sphere"),
                                options = options)
    col_air = _def_payload_var(ds, "column_air_mass_per_area", T, dims4,
                               attrib = _var_attrib(units = "kg m-2",
                                                    long_name = "column air mass per area",
                                                    coordinates = coord,
                                                    grid_mapping = "cubed_sphere",
                                                    cell_methods = "lev: sum"),
                               options = options)

    tracer_vars = Dict{Symbol, Any}()
    tracer_cm_vars = Dict{Symbol, Any}()
    tracer_col_vars = Dict{Symbol, Any}()
    for name in tracer_keys
        s = String(name)
        tracer_vars[name] = _def_payload_var(ds, s, T, dims5,
                                             attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                  long_name = "per-layer $(s) mixing ratio",
                                                                  coordinates = coord,
                                                                  grid_mapping = "cubed_sphere"),
                                             options = options)
        tracer_cm_vars[name] = _def_payload_var(ds, "$(s)_column_mean", T, dims4,
                                                attrib = _var_attrib(units = _tracer_units(mass_basis_sym),
                                                                     long_name = "air-mass-weighted column-mean $(s) mixing ratio",
                                                                     coordinates = coord,
                                                                     grid_mapping = "cubed_sphere",
                                                                     cell_methods = "lev: mean"),
                                                options = options)
        tracer_col_vars[name] = _def_payload_var(ds, "$(s)_column_mass_per_area", T, dims4,
                                                 attrib = _var_attrib(units = "kg m-2",
                                                                      long_name = "column model tracer mass per area for $(s)",
                                                                      coordinates = coord,
                                                                      grid_mapping = "cubed_sphere",
                                                                      cell_methods = "lev: sum",
                                                                      extra = Dict("description" =>
                                                                          "Sum of model tracer mass divided by horizontal cell area; no molecular-weight conversion is applied.")),
                                                 options = options)
    end

    for (t, frame) in enumerate(frames)
        air[:, :, :, :, t] = T.(_cs_stack3(frame.air_mass))
        air_area[:, :, :, :, t] = T.(_cs_stack3(layer_mass_per_area(frame.air_mass, mesh)))
        col_air[:, :, :, t] = T.(_cs_stack2(column_mass_per_area(frame.air_mass, mesh)))
        for name in tracer_keys
            tracer_vars[name][:, :, :, :, t] =
                T.(_cs_stack3(mixing_ratio_field(frame.air_mass, frame.tracers[name])))
            tracer_cm_vars[name][:, :, :, t] =
                T.(_cs_stack2(column_mean_mixing_ratio(frame.air_mass, frame.tracers[name])))
            tracer_col_vars[name][:, :, :, t] =
                T.(_cs_stack2(column_mass_per_area(frame.tracers[name], mesh)))
        end
    end
    return nothing
end
