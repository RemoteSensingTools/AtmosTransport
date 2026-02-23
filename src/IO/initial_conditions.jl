# ---------------------------------------------------------------------------
# Initial conditions loader
#
# Load 3D tracer fields from NetCDF files to initialize a simulation.
# Supports both lat-lon and cubed-sphere grids via dispatch.
#
# Used by CATRINE intercomparison (CO2/SF6 from LSCE inversions) and
# extensible to other protocols.
# ---------------------------------------------------------------------------

using NCDatasets
using ..Grids: LatitudeLongitudeGrid, CubedSphereGrid

export load_initial_conditions!

"""
    load_initial_conditions!(tracers, filepath, grid;
                              variable_map=Dict{Symbol,String}(),
                              time_index=1)

Load 3D initial condition fields from `filepath` into `tracers`.

Arguments:
- `tracers`: NamedTuple of 3D arrays (e.g., `(co2=Array{FT,3}, sf6=Array{FT,3}, ...)`)
- `filepath`: path to NetCDF file containing initial fields
- `grid`: model grid (LatitudeLongitudeGrid or CubedSphereGrid)
- `variable_map`: maps tracer symbol → NetCDF variable name.
  For example: `Dict(:co2 => "CO2", :sf6 => "SF6")`.
  Tracers not in the map are skipped (left at their current values).
- `time_index`: time step to read from multi-time files (default: 1)

For lat-lon grids, the loader performs bilinear interpolation from the source
grid to the model grid when dimensions differ, or direct copy when they match.

Tracers not listed in `variable_map` are left unchanged (e.g., fossil_co2 and
rn222 start from zero in CATRINE).
"""
function load_initial_conditions!(tracers::NamedTuple,
                                   filepath::String,
                                   grid::LatitudeLongitudeGrid{FT};
                                   variable_map::Dict{Symbol,String} = Dict{Symbol,String}(),
                                   time_index::Int = 1) where FT
    isfile(filepath) || error("Initial conditions file not found: $filepath")
    ds = NCDataset(filepath)

    # Discover coordinate variables
    lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
    lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
    lev_var = _ic_find_coord(ds, ["lev", "level", "plev", "z", "hybrid", "nhym"])

    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])
    lev_src = Float64.(ds[lev_var][:])

    Nlon_s = length(lon_src)
    Nlat_s = length(lat_src)
    Nlev_s = length(lev_src)

    Nx_m, Ny_m, Nz_m = grid.Nx, grid.Ny, grid.Nz

    # Check if grids match exactly
    grids_match = (Nlon_s == Nx_m && Nlat_s == Ny_m && Nlev_s == Nz_m)

    for (tracer_name, nc_varname) in variable_map
        haskey(tracers, tracer_name) || continue
        haskey(ds, nc_varname) || begin
            @warn "Variable '$nc_varname' not found in $filepath — skipping $tracer_name"
            continue
        end

        # Read the 3D field
        raw_var = ds[nc_varname]
        raw = if ndims(raw_var) == 4
            FT.(nomissing(raw_var[:, :, :, time_index], zero(FT)))
        elseif ndims(raw_var) == 3
            FT.(nomissing(raw_var[:, :, :], zero(FT)))
        else
            @warn "Variable '$nc_varname' is $(ndims(raw_var))D, expected 3D or 4D — skipping"
            continue
        end

        # Handle latitude direction (ensure S→N)
        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1, :]
            lat_use = reverse(lat_src)
        else
            lat_use = lat_src
        end

        # Handle longitude convention (-180:180 → 0:360)
        lon_use = lon_src
        if minimum(lon_src) < 0
            n = length(lon_src)
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:n, 1:split-1)
                lon_use = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :, :]
            end
        end

        # Handle vertical orientation (ensure top→bottom matches model)
        # Convention: model level 1 = top, Nz = surface
        # If source levels are ascending in pressure (top→bottom), keep as-is
        # If descending, flip
        if Nlev_s > 1 && lev_src[1] > lev_src[end]
            raw = raw[:, :, end:-1:1]
        end

        c = tracers[tracer_name]

        if grids_match
            # Direct copy
            c .= raw
        else
            # Regrid via nearest-neighbor interpolation
            _regrid_3d_to_model!(c, raw, lon_use, lat_use, grid, FT)
        end

        total = sum(raw)
        @info "Loaded initial condition for $tracer_name from '$nc_varname': " *
              "source $(Nlon_s)×$(Nlat_s)×$(Nlev_s), sum=$(round(Float64(total), sigdigits=6))"
    end

    close(ds)
    return nothing
end

"""
    load_initial_conditions!(tracers, filepath, grid::CubedSphereGrid; ...)

Load initial conditions for cubed-sphere grids. The source file is expected
to be on a lat-lon grid; the loader regrids to each CS panel via
nearest-neighbor interpolation.

`tracers` should be a NamedTuple where each value is an NTuple{6} of 3D
haloed panel arrays.
"""
function load_initial_conditions!(tracers::NamedTuple,
                                   filepath::String,
                                   grid::CubedSphereGrid{FT};
                                   variable_map::Dict{Symbol,String} = Dict{Symbol,String}(),
                                   time_index::Int = 1) where FT
    isfile(filepath) || error("Initial conditions file not found: $filepath")
    ds = NCDataset(filepath)

    lon_var = _ic_find_coord(ds, ["lon", "longitude", "x"])
    lat_var = _ic_find_coord(ds, ["lat", "latitude", "y"])
    lev_var = _ic_find_coord(ds, ["lev", "level", "plev", "z", "hybrid", "nhym"])

    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])
    lev_src = Float64.(ds[lev_var][:])

    Nlon_s = length(lon_src)
    Nlat_s = length(lat_src)
    Nlev_s = length(lev_src)

    Nc = grid.Nc
    Hp = grid.Hp
    Nz_m = grid.Nz

    for (tracer_name, nc_varname) in variable_map
        haskey(tracers, tracer_name) || continue
        haskey(ds, nc_varname) || begin
            @warn "Variable '$nc_varname' not found in $filepath — skipping $tracer_name"
            continue
        end

        raw_var = ds[nc_varname]
        raw = if ndims(raw_var) == 4
            FT.(nomissing(raw_var[:, :, :, time_index], zero(FT)))
        elseif ndims(raw_var) == 3
            FT.(nomissing(raw_var[:, :, :], zero(FT)))
        else
            @warn "Variable '$nc_varname' is $(ndims(raw_var))D — skipping"
            continue
        end

        # Latitude S→N
        if length(lat_src) > 1 && lat_src[1] > lat_src[end]
            raw = raw[:, end:-1:1, :]
            lat_use = reverse(lat_src)
        else
            lat_use = lat_src
        end

        # Longitude → 0:360
        lon_use = lon_src
        if minimum(lon_src) < 0
            n = length(lon_src)
            split = findfirst(>=(0), lon_src)
            if split !== nothing
                idx = vcat(split:n, 1:split-1)
                lon_use = mod.(lon_src[idx], 360.0)
                raw = raw[idx, :, :]
            end
        end

        # Vertical orientation
        if Nlev_s > 1 && lev_src[1] > lev_src[end]
            raw = raw[:, :, end:-1:1]
        end

        Δlon = lon_use[2] - lon_use[1]
        Δlat = lat_use[2] - lat_use[1]

        panels = tracers[tracer_name]  # NTuple{6} of haloed 3D arrays

        # Number of source vs model levels — use minimum
        Nz_use = min(Nlev_s, Nz_m)

        for p in 1:6
            panel = panels[p]
            for j in 1:Nc, i in 1:Nc
                lon = mod(grid.λᶜ[p][i, j] + 180, 360) - 180
                lat = grid.φᶜ[p][i, j]
                ii = clamp(round(Int, (lon - lon_use[1]) / Δlon) + 1, 1, Nlon_s)
                jj = clamp(round(Int, (lat - lat_use[1]) / Δlat) + 1, 1, Nlat_s)
                for k in 1:Nz_use
                    panel[Hp + i, Hp + j, k] = raw[ii, jj, k]
                end
            end
        end

        @info "Loaded initial condition for $tracer_name (CS C$(Nc)) from '$nc_varname'"
    end

    close(ds)
    return nothing
end


# --- Internal helpers ---

function _ic_find_coord(ds, candidates::Vector{String})
    for name in candidates
        haskey(ds, name) && return name
    end
    # Return nothing instead of error — some dims may be optional
    return nothing
end

"""Nearest-neighbor regridding of a 3D field to lat-lon model grid."""
function _regrid_3d_to_model!(c::AbstractArray{FT,3}, raw::Array{FT,3},
                               lon_src, lat_src,
                               grid::LatitudeLongitudeGrid{FT},
                               ::Type{FT}) where FT
    Nx_m, Ny_m, Nz_m = grid.Nx, grid.Ny, grid.Nz
    Nlon_s = size(raw, 1)
    Nlat_s = size(raw, 2)
    Nlev_s = size(raw, 3)
    Nz_use = min(Nlev_s, Nz_m)

    λᶜ = grid.λᶜ_cpu
    φᶜ = grid.φᶜ_cpu

    for jm in 1:Ny_m
        js = _ic_nearest_idx(φᶜ[jm], lat_src)
        for im in 1:Nx_m
            is = _ic_nearest_idx(λᶜ[im], lon_src)
            for k in 1:Nz_use
                c[im, jm, k] = raw[is, js, k]
            end
        end
    end
end

function _ic_nearest_idx(val, arr)
    best = 1
    best_dist = abs(arr[1] - val)
    for i in 2:length(arr)
        d = abs(arr[i] - val)
        if d < best_dist
            best_dist = d
            best = i
        end
    end
    return best
end
