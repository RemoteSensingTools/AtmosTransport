# ---------------------------------------------------------------------------
# Shared utilities for cubed-sphere → lat-lon regridding and visualization
#
# Usage (from any visualization script):
#   include(joinpath(@__DIR__, "cs_regrid_utils.jl"))
# ---------------------------------------------------------------------------

using NCDatasets

"""
Precomputed nearest-neighbor mapping from cubed-sphere to regular lat-lon grid.
"""
struct CSRegridMap
    nlon::Int
    nlat::Int
    panel::Matrix{Int}      # (nlon, nlat) → panel index 1..6
    ji::Matrix{Int}          # (nlon, nlat) → linear index within panel
    lon::Vector{Float64}     # regular lon centers
    lat::Vector{Float64}     # regular lat centers
end

"""
    build_cs_regrid_map(cs_lons, cs_lats; dlon=1.0, dlat=1.0)

Build a nearest-neighbor regridding map from CS cell centers (Xdim, Ydim, nf)
to a regular lat-lon grid with spacing `dlon` × `dlat`.
"""
function build_cs_regrid_map(cs_lons::Array{Float64,3}, cs_lats::Array{Float64,3};
                              dlon=1.0, dlat=1.0)
    nlon = round(Int, 360.0 / dlon)
    nlat = round(Int, 180.0 / dlat)
    lon = collect(range(-180.0 + dlon/2, 180.0 - dlon/2; length=nlon))
    lat = collect(range(-90.0 + dlat/2, 90.0 - dlat/2; length=nlat))

    nx, ny, nf = size(cs_lons)
    panel_map = zeros(Int, nlon, nlat)
    ji_map    = zeros(Int, nlon, nlat)

    # Shift CS lons to -180..180
    cs_lons_shifted = copy(cs_lons)
    cs_lons_shifted[cs_lons_shifted .> 180.0] .-= 360.0

    # Flatten all CS cells
    n_cells = nf * ny * nx
    flat_lon   = Vector{Float64}(undef, n_cells)
    flat_lat   = Vector{Float64}(undef, n_cells)
    flat_panel = Vector{Int}(undef, n_cells)
    flat_ji    = Vector{Int}(undef, n_cells)

    idx = 0
    for p in 1:nf, j in 1:ny, i in 1:nx
        idx += 1
        flat_lon[idx]   = cs_lons_shifted[i, j, p]
        flat_lat[idx]   = cs_lats[i, j, p]
        flat_panel[idx] = p
        flat_ji[idx]    = (j - 1) * nx + i
    end

    # For each regular grid cell, find nearest CS cell
    deg2rad = π / 180.0
    for jj in 1:nlat
        target_lat = lat[jj]
        lat_mask = abs.(flat_lat .- target_lat) .< 3.0
        if count(lat_mask) == 0
            lat_mask = abs.(flat_lat .- target_lat) .< 10.0
        end
        candidate_idx = findall(lat_mask)

        for ii in 1:nlon
            target_lon = lon[ii]
            best_dist = Inf
            best_k = 1
            for k in candidate_idx
                dlon_k = flat_lon[k] - target_lon
                dlon_k > 180.0 && (dlon_k -= 360.0)
                dlon_k < -180.0 && (dlon_k += 360.0)
                dlat_k = flat_lat[k] - target_lat
                cos_lat = cos(target_lat * deg2rad)
                dist = (dlon_k * cos_lat)^2 + dlat_k^2
                if dist < best_dist
                    best_dist = dist
                    best_k = k
                end
            end
            panel_map[ii, jj] = flat_panel[best_k]
            ji_map[ii, jj]    = flat_ji[best_k]
        end
    end

    return CSRegridMap(nlon, nlat, panel_map, ji_map, lon, lat)
end

"""
    regrid_cs!(out, cs_data, rmap)

Regrid CS panel data (Xdim, Ydim, nf) to regular lat-lon using precomputed map.
"""
function regrid_cs!(out::Matrix{Float32}, cs_data::Array{<:Real,3}, rmap::CSRegridMap)
    nx, _, _ = size(cs_data)
    for jj in 1:rmap.nlat, ii in 1:rmap.nlon
        p  = rmap.panel[ii, jj]
        li = rmap.ji[ii, jj]
        j, i = divrem(li - 1, nx)
        out[ii, jj] = Float32(cs_data[i + 1, j + 1, p])
    end
    return out
end

"""
    load_cs_coordinates(nc_path) -> (cs_lons, cs_lats)

Load CS cell center coordinates from a NetCDF file containing `lons` and `lats` variables.
"""
function load_cs_coordinates(nc_path::String)
    NCDataset(nc_path, "r") do ds
        Float64.(ds["lons"][:, :, :]), Float64.(ds["lats"][:, :, :])
    end
end

"""
    load_cs_daily_nc(dir, pattern, rmap, var_name, levs; date_range, exclude_pattern, scale)

Load daily CS NetCDF files, regrid a variable at specified levels, return regridded fields.

Returns `(; times::Vector{DateTime}, fields::Vector{Array{Float32,3}})` where each
`fields[i]` is `(nlon, nlat, nt)` for level `i`.
"""
function load_cs_daily_nc(dir::String, pattern::String, rmap::CSRegridMap,
                           var_name::String, levs::Vector{Int};
                           date_start::DateTime, date_end::DateTime,
                           exclude_pattern::String="",
                           scale::Float64=1.0, label::String="")
    daily_files = sort(filter(f -> endswith(f, ".nc") && contains(f, pattern), readdir(dir)))
    if !isempty(exclude_pattern)
        daily_files = filter(f -> !contains(f, exclude_pattern), daily_files)
    end

    nl   = length(levs)
    nlon = rmap.nlon
    nlat = rmap.nlat
    buf  = zeros(Float32, nlon, nlat)

    all_times  = DateTime[]
    all_fields = [Vector{Matrix{Float32}}() for _ in 1:nl]

    for fname in daily_files
        NCDataset(joinpath(dir, fname), "r") do ds
            file_times = ds["time"][:]
            data_var = ds[var_name]
            for (ti, dt) in enumerate(file_times)
                date_start <= dt <= date_end || continue
                push!(all_times, dt)
                for (li, lev) in enumerate(levs)
                    data_cs = Float32.(data_var[:, :, :, lev, ti]) .* Float32(scale)
                    regrid_cs!(buf, data_cs, rmap)
                    push!(all_fields[li], copy(buf))
                end
            end
        end
    end

    perm = sortperm(all_times)
    all_times = all_times[perm]
    for li in 1:nl
        all_fields[li] = all_fields[li][perm]
    end

    nt = length(all_times)
    lbl = isempty(label) ? pattern : label
    @info "$lbl: $nt snapshots ($(all_times[1]) → $(all_times[end]))"

    fields = [zeros(Float32, nlon, nlat, nt) for _ in 1:nl]
    for li in 1:nl, ti in 1:nt
        fields[li][:, :, ti] .= all_fields[li][ti]
    end

    return (; times=all_times, fields)
end

"""
    load_geoschem_nc(dir, rmap, var_name, levs; date_start, date_end, scale)

Load GEOS-Chem per-timestep NC4 files matching CATRINE_inst pattern.
"""
function load_geoschem_nc(dir::String, rmap::CSRegridMap,
                           var_name::String, levs::Vector{Int};
                           date_start::DateTime, date_end::DateTime,
                           scale::Float64=1.0)
    all_files = sort(filter(f -> endswith(f, ".nc4") && contains(f, "CATRINE_inst"),
                             readdir(dir)))
    files = String[]
    times = DateTime[]
    for f in all_files
        m = match(r"(\d{8})_(\d{4})z", f)
        m === nothing && continue
        dt = DateTime(m[1] * m[2], dateformat"yyyymmddHHMM")
        if date_start <= dt <= date_end
            push!(files, f)
            push!(times, dt)
        end
    end

    @info "GEOS-Chem: $(length(files)) snapshots ($(times[1]) → $(times[end]))"

    nt  = length(files)
    nl  = length(levs)
    buf = zeros(Float32, rmap.nlon, rmap.nlat)
    fields = [zeros(Float32, rmap.nlon, rmap.nlat, nt) for _ in 1:nl]

    for (ti, fname) in enumerate(files)
        NCDataset(joinpath(dir, fname), "r") do ds
            data = ds[var_name]
            for (li, lev) in enumerate(levs)
                data_cs = Float64.(data[:, :, :, lev, 1]) .* scale
                regrid_cs!(buf, data_cs, rmap)
                fields[li][:, :, ti] .= buf
            end
        end
    end

    return (; times, fields)
end

"""
    lon_lat_meshes(rmap) -> (lon2d, lat2d)

Create 2D coordinate arrays for `surface!` plotting (nlat × nlon, transposed for Makie).
"""
function lon_lat_meshes(rmap::CSRegridMap)
    lon2d = Float32[rmap.lon[i] for _j in 1:rmap.nlat, i in 1:rmap.nlon]
    lat2d = Float32[rmap.lat[j] for  j in 1:rmap.nlat, _i in 1:rmap.nlon]
    return lon2d, lat2d
end

# Clamped sqrt colorscale (handles tiny negative values from numerical noise)
safe_sqrt(x) = sqrt(max(zero(x), x))
