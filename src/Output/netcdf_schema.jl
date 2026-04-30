function _ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    isempty(parent) || mkpath(parent)
    return nothing
end

function _nlevel(frame::SnapshotFrame, ::LatLonMesh)
    return size(frame.air_mass, 3)
end

function _nlevel(frame::SnapshotFrame, ::ReducedGaussianMesh)
    return size(frame.air_mass, 2)
end

function _nlevel(frame::SnapshotFrame, ::CubedSphereMesh)
    return size(frame.air_mass[1], 3)
end

function _define_common_attributes!(ds, mesh, frames, mass_basis_sym::Symbol)
    ds.attrib["Conventions"] = "CF-1.8"
    ds.attrib["title"] = "AtmosTransport runtime snapshot"
    ds.attrib["source"] = "AtmosTransport.jl"
    ds.attrib["grid"] = summary(mesh)
    ds.attrib["grid_type"] = _grid_type_string(mesh)
    ds.attrib["mass_basis"] = String(mass_basis_sym)
    ds.attrib["output_contract"] = "AtmosTransport snapshot v2"
    ds.attrib["history"] = @sprintf("written by AtmosTransport.Output with %d frame(s)", length(frames))
    return nothing
end

_grid_type_string(::LatLonMesh) = "latlon"
_grid_type_string(::ReducedGaussianMesh) = "reduced_gaussian"
_grid_type_string(::CubedSphereMesh) = "cubed_sphere"

function _define_time!(ds, times)
    defDim(ds, "time", length(times))
    v = defVar(ds, "time", Float64, ("time",),
               attrib = Dict("units" => "hours since 2000-01-01 00:00:00",
                             "long_name" => "time",
                             "standard_name" => "time",
                             "calendar" => "proleptic_gregorian",
                             "comment" => "simulation-relative output time; calendar origin is nominal"))
    v[:] = Float64.(times)
    return v
end

function _define_lev!(ds, Nz::Integer)
    defDim(ds, "lev", Int(Nz))
    v = defVar(ds, "lev", Float64, ("lev",),
               attrib = Dict("units" => "1",
                             "long_name" => "model level",
                             "positive" => "down"))
    v[:] = collect(1.0:Float64(Nz))
    return v
end

function _bounds_from_faces(faces)
    n = length(faces) - 1
    out = Array{Float64}(undef, n, 2)
    @inbounds for i in 1:n
        out[i, 1] = Float64(faces[i])
        out[i, 2] = Float64(faces[i + 1])
    end
    return out
end

function _define_latlon_geometry!(ds, mesh::LatLonMesh, Nz::Integer, times)
    defDim(ds, "lon", nx(mesh))
    defDim(ds, "lat", ny(mesh))
    defDim(ds, "nv", 2)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    v_lon = defVar(ds, "lon", Float64, ("lon",),
                   attrib = Dict("units" => "degrees_east",
                                 "long_name" => "longitude",
                                 "standard_name" => "longitude",
                                 "bounds" => "lon_bounds"))
    v_lat = defVar(ds, "lat", Float64, ("lat",),
                   attrib = Dict("units" => "degrees_north",
                                 "long_name" => "latitude",
                                 "standard_name" => "latitude",
                                 "bounds" => "lat_bounds"))
    v_lon[:] = Float64.(mesh.λᶜ)
    v_lat[:] = Float64.(mesh.φᶜ)

    defVar(ds, "lon_bounds", Float64, ("lon", "nv"),
           attrib = Dict("units" => "degrees_east"))[:, :] = _bounds_from_faces(mesh.λᶠ)
    defVar(ds, "lat_bounds", Float64, ("lat", "nv"),
           attrib = Dict("units" => "degrees_north"))[:, :] = _bounds_from_faces(mesh.φᶠ)

    area = Array{Float64}(undef, nx(mesh), ny(mesh))
    @inbounds for j in 1:ny(mesh), i in 1:nx(mesh)
        area[i, j] = Float64(cell_area(mesh, i, j))
    end
    defVar(ds, "cell_area", Float64, ("lon", "lat"),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area"))[:, :] = area
    return (;)
end

function _rg_native_coordinates(mesh::ReducedGaussianMesh)
    lons = Array{Float64}(undef, ncells(mesh))
    lats = Array{Float64}(undef, ncells(mesh))
    areas = Array{Float64}(undef, ncells(mesh))
    lon_bounds = Array{Float64}(undef, ncells(mesh), 4)
    lat_bounds = Array{Float64}(undef, ncells(mesh), 4)
    @inbounds for j in 1:nrings(mesh)
        ring_lons = ring_longitudes(mesh, j)
        dlon = 360.0 / mesh.nlon_per_ring[j]
        lat_s = Float64(mesh.lat_faces[j])
        lat_n = Float64(mesh.lat_faces[j + 1])
        for i in eachindex(ring_lons)
            c = cell_index(mesh, i, j)
            lons[c] = Float64(ring_lons[i])
            lats[c] = Float64(mesh.latitudes[j])
            areas[c] = Float64(cell_area(mesh, c))
            lon_w = (i - 1) * dlon
            lon_e = i * dlon
            lon_bounds[c, :] .= (lon_w, lon_e, lon_e, lon_w)
            lat_bounds[c, :] .= (lat_s, lat_s, lat_n, lat_n)
        end
    end
    return lons, lats, areas, lon_bounds, lat_bounds
end

function _rg_legacy_raster_map(mesh::ReducedGaussianMesh)
    nr = nrings(mesh)
    nlon = maximum(mesh.nlon_per_ring)
    dlon = 360.0 / nlon
    out_lons = [(i - 0.5) * dlon for i in 1:nlon]
    out_lats = Float64.(mesh.latitudes)
    nn_map = zeros(Int, nlon, nr)
    @inbounds for j in 1:nr
        nlon_ring = mesh.nlon_per_ring[j]
        dlon_ring = 360.0 / nlon_ring
        for i in 1:nlon
            i_ring = clamp(round(Int, out_lons[i] / dlon_ring + 0.5), 1, nlon_ring)
            nn_map[i, j] = cell_index(mesh, i_ring, j)
        end
    end
    return out_lons, out_lats, nn_map
end

function _define_rg_geometry!(ds, mesh::ReducedGaussianMesh, Nz::Integer, times)
    defDim(ds, "cell", ncells(mesh))
    defDim(ds, "nv", 4)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    lons, lats, areas, lon_bounds, lat_bounds = _rg_native_coordinates(mesh)
    defVar(ds, "cell", Int32, ("cell",),
           attrib = Dict("long_name" => "native reduced-Gaussian cell index"))[:] =
        Int32.(1:ncells(mesh))
    defVar(ds, "cell_lon", Float64, ("cell",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "native cell longitude",
                         "standard_name" => "longitude",
                         "bounds" => "cell_lon_bounds"))[:] = lons
    defVar(ds, "cell_lat", Float64, ("cell",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "native cell latitude",
                         "standard_name" => "latitude",
                         "bounds" => "cell_lat_bounds"))[:] = lats
    defVar(ds, "cell_lon_bounds", Float64, ("cell", "nv"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "native cell corner longitudes",
                         "comment" => "Corners are ordered SW, SE, NE, NW."))[:, :] = lon_bounds
    defVar(ds, "cell_lat_bounds", Float64, ("cell", "nv"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "native cell corner latitudes",
                         "comment" => "Corners are ordered SW, SE, NE, NW."))[:, :] = lat_bounds
    defVar(ds, "cell_area", Float64, ("cell",),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area",
                         "coordinates" => "cell_lon cell_lat"))[:] = areas

    out_lons, out_lats, nn_map = _rg_legacy_raster_map(mesh)
    defDim(ds, "lon", length(out_lons))
    defDim(ds, "lat", length(out_lats))
    defVar(ds, "lon", Float64, ("lon",),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "diagnostic longitude",
                         "standard_name" => "longitude"))[:] = out_lons
    defVar(ds, "lat", Float64, ("lat",),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "diagnostic latitude",
                         "standard_name" => "latitude"))[:] = out_lats

    ds.attrib["nrings"] = nrings(mesh)
    ds.attrib["regridding"] = "nearest-neighbor reduced-Gaussian diagnostic raster"
    return (; nn_map)
end

function _cs_convention_tag(mesh::CubedSphereMesh)
    mesh.convention isa GnomonicPanelConvention && return "gnomonic"
    mesh.convention isa GEOSNativePanelConvention && return "geos_native"
    return "unknown"
end

function _cs_central_meridian(mesh::CubedSphereMesh)
    return longitude_offset_deg(cs_definition(mesh))
end

function _cs_panel_center_arrays(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    lons = Array{Float64}(undef, Nc, Nc, 6)
    lats = Array{Float64}(undef, Nc, Nc, 6)
    for p in 1:6
        lonp, latp = panel_cell_center_lonlat(mesh, p)
        lons[:, :, p] = lonp
        lats[:, :, p] = latp
    end
    return lons, lats
end

function _cs_panel_corner_arrays(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    lons = Array{Float64}(undef, Nc + 1, Nc + 1, 6)
    lats = Array{Float64}(undef, Nc + 1, Nc + 1, 6)
    for p in 1:6
        lonp, latp = panel_cell_corner_lonlat(mesh, p)
        lons[:, :, p] = lonp
        lats[:, :, p] = latp
    end
    return lons, lats
end

function _define_cs_geometry!(ds, mesh::CubedSphereMesh, Nz::Integer, times)
    Nc = mesh.Nc
    defDim(ds, "Xdim", Nc)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "Xcorner", Nc + 1)
    defDim(ds, "Ycorner", Nc + 1)
    defDim(ds, "nf", 6)
    _define_lev!(ds, Nz)
    _define_time!(ds, times)

    convention_tag = _cs_convention_tag(mesh)
    central_meridian = _cs_central_meridian(mesh)
    coordinate_tag = coordinate_law_tag(coordinate_law(mesh))
    center_tag = center_law_tag(center_law(mesh))
    definition_tag = String(cs_definition_tag(cs_definition(mesh)))
    ds.attrib["Nc"] = Nc
    ds.attrib["cs_definition"] = definition_tag
    ds.attrib["cs_coordinate_law"] = coordinate_tag
    ds.attrib["cs_center_law"] = center_tag
    ds.attrib["panel_convention"] = convention_tag
    ds.attrib["grid_mapping_name"] = "gnomonic cubed-sphere"
    ds.attrib["longitude_of_central_meridian"] = central_meridian

    defVar(ds, "Xdim", Float64, ("Xdim",),
           attrib = Dict("units" => "1",
                         "long_name" => "cubed-sphere panel X index"))[:] =
        collect(1.0:Nc)
    defVar(ds, "Ydim", Float64, ("Ydim",),
           attrib = Dict("units" => "1",
                         "long_name" => "cubed-sphere panel Y index"))[:] =
        collect(1.0:Nc)
    defVar(ds, "nf", Int32, ("nf",),
           attrib = Dict("axis" => "e",
                         "long_name" => "cubed-sphere face"))[:] = Int32.(1:6)

    lons, lats = _cs_panel_center_arrays(mesh)
    clons, clats = _cs_panel_corner_arrays(mesh)
    defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-center longitude",
                         "standard_name" => "longitude"))[:, :, :] = lons
    defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-center latitude",
                         "standard_name" => "latitude"))[:, :, :] = lats
    defVar(ds, "corner_lons", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_east",
                         "long_name" => "cell-corner longitude"))[:, :, :] = clons
    defVar(ds, "corner_lats", Float64, ("Xcorner", "Ycorner", "nf"),
           attrib = Dict("units" => "degrees_north",
                         "long_name" => "cell-corner latitude"))[:, :, :] = clats

    area = Array{Float64}(undef, Nc, Nc, 6)
    @inbounds for p in 1:6, j in 1:Nc, i in 1:Nc
        area[i, j, p] = Float64(cell_area(mesh, i, j))
    end
    defVar(ds, "cell_area", Float64, ("Xdim", "Ydim", "nf"),
           attrib = Dict("units" => "m2",
                         "long_name" => "horizontal cell area",
                         "standard_name" => "cell_area",
                         "coordinates" => "lons lats"))[:, :, :] = area

    gm = defVar(ds, "cubed_sphere", Int32, (),
                attrib = Dict("grid_mapping_name" => "gnomonic cubed-sphere",
                              "cs_definition" => definition_tag,
                              "cs_coordinate_law" => coordinate_tag,
                              "cs_center_law" => center_tag,
                              "panel_convention" => convention_tag,
                              "longitude_of_central_meridian" => central_meridian,
                              "semi_major_axis" => Float64(mesh.radius),
                              "inverse_flattening" => 0.0,
                              "comment" => "Use lons/lats and corner_lons/corner_lats as authoritative native-cell coordinates."))
    gm[] = Int32(0)
    return (;)
end

function _define_geometry!(ds, mesh::LatLonMesh, Nz::Integer, times)
    return _define_latlon_geometry!(ds, mesh, Nz, times)
end

function _define_geometry!(ds, mesh::ReducedGaussianMesh, Nz::Integer, times)
    return _define_rg_geometry!(ds, mesh, Nz, times)
end

function _define_geometry!(ds, mesh::CubedSphereMesh, Nz::Integer, times)
    return _define_cs_geometry!(ds, mesh, Nz, times)
end
