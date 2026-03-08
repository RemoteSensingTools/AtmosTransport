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

export load_initial_conditions!, PendingInitialConditions, apply_pending_ic!,
       finalize_ic_vertical_interp!, has_deferred_ic_vinterp,
       UniformICData, _store_uniform_ic

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
            # Direct copy (handles GPU via broadcast)
            copyto!(c, raw)
        else
            # Regrid via nearest-neighbor interpolation on CPU, then upload
            c_cpu = Array{FT}(undef, Nx_m, Ny_m, Nz_m)
            fill!(c_cpu, zero(FT))
            _regrid_3d_to_model!(c_cpu, raw, lon_use, lat_use, grid, FT)
            copyto!(c, c_cpu)
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

    # Check for hybrid-sigma vertical coordinates (ap/bp/Psurf)
    has_hybrid = haskey(ds, "ap") && haskey(ds, "bp") && haskey(ds, "Psurf")
    ap_src = has_hybrid ? Float64.(ds["ap"][:]) : Float64[]
    bp_src = has_hybrid ? Float64.(ds["bp"][:]) : Float64[]
    psurf_raw = has_hybrid ? Float64.(nomissing(ds["Psurf"][:, :], 101325.0)) : zeros(Float64, 0, 0)

    # Precompute lon/lat reordering for Psurf (same transforms as tracer data)
    lat_use_global = lat_src
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        lat_use_global = reverse(lat_src)
        if has_hybrid
            psurf_raw = psurf_raw[:, end:-1:1]
        end
    end
    lon_use_global = lon_src
    if minimum(lon_src) < 0
        n = length(lon_src)
        split_g = findfirst(>=(0), lon_src)
        if split_g !== nothing
            idx_g = vcat(split_g:n, 1:split_g-1)
            lon_use_global = mod.(lon_src[idx_g], 360.0)
            if has_hybrid
                psurf_raw = psurf_raw[idx_g, :]
            end
        end
    end
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

        # Vertical orientation — ensure level 1 = surface (increasing index = up)
        # LMDZ: level 1 is typically surface (lowest), so check if pressure decreases
        vert_flipped = false
        if Nlev_s > 1 && lev_src[1] > lev_src[end]
            raw = raw[:, :, end:-1:1]
            vert_flipped = true
        end

        Δlon = lon_use[2] - lon_use[1]
        Δlat = lat_use_global[2] - lat_use_global[1]

        panels = tracers[tracer_name]  # NTuple{6} of haloed 3D arrays

        if has_hybrid
            # --- Deferred vertical interpolation path ---
            # Horizontally regrid mixing ratios to CS panels (all source levels)
            panels_mr = ntuple(6) do _
                zeros(FT, Nc, Nc, Nlev_s)
            end
            psurf_panels = ntuple(6) do _
                zeros(Float64, Nc, Nc)
            end

            for p in 1:6
                for j in 1:Nc, i in 1:Nc
                    lon = mod(grid.λᶜ[p][i, j], 360.0)
                    lat = grid.φᶜ[p][i, j]
                    ii = clamp(round(Int, (lon - lon_use[1]) / Δlon) + 1, 1, Nlon_s)
                    jj = clamp(round(Int, (lat - lat_use_global[1]) / Δlat) + 1, 1, Nlat_s)
                    for k in 1:Nlev_s
                        panels_mr[p][i, j, k] = raw[ii, jj, k]
                    end
                    psurf_panels[p][i, j] = psurf_raw[ii, jj]
                end
            end

            # If vertical was flipped, reverse ap/bp to match flipped data
            ap_use = vert_flipped ? reverse(ap_src) : ap_src
            bp_use = vert_flipped ? reverse(bp_src) : bp_src

            # Store for deferred vertical interp + mass conversion
            _store_deferred_ic(DeferredICData{FT}(
                tracer_name, panels_mr, ap_use, bp_use,
                psurf_panels, Nlev_s))

            # Write zeros to tracer panels for now (will be overwritten by finalize)
            cpu_buf = zeros(FT, size(panels[1]))
            for p in 1:6
                copyto!(panels[p], cpu_buf)
            end

            @info "Loaded IC for $tracer_name (CS C$(Nc), deferred vertical interp: " *
                  "$(Nlev_s) source levels → $(Nz_m) target levels)"
        else
            # --- Direct level-copy path (no hybrid coords) ---
            Nz_use = min(Nlev_s, Nz_m)
            cpu_buf = zeros(FT, size(panels[1]))
            for p in 1:6
                fill!(cpu_buf, zero(FT))
                for j in 1:Nc, i in 1:Nc
                    lon = mod(grid.λᶜ[p][i, j], 360.0)
                    lat = grid.φᶜ[p][i, j]
                    ii = clamp(round(Int, (lon - lon_use[1]) / Δlon) + 1, 1, Nlon_s)
                    jj = clamp(round(Int, (lat - lat_use_global[1]) / Δlat) + 1, 1, Nlat_s)
                    for k in 1:Nz_use
                        cpu_buf[Hp + i, Hp + j, k] = raw[ii, jj, k]
                    end
                end
                copyto!(panels[p], cpu_buf)
            end
            @info "Loaded IC for $tracer_name (CS C$(Nc), direct level copy) from '$nc_varname'"
        end
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

# ---------------------------------------------------------------------------
# Deferred IC loading for cubed-sphere grids
#
# CS tracers are allocated as placeholders during build_model_from_config,
# so ICs must be applied later in the run loop after real panel arrays exist.
# ---------------------------------------------------------------------------

"""
    PendingInitialConditions

Stores IC file/variable entries to be applied later (for CS grids).
"""
struct PendingInitialConditions
    entries :: Vector{NamedTuple{(:file, :variable_map, :time_index),
                                  Tuple{String, Dict{Symbol,String}, Int}}}
end
PendingInitialConditions() = PendingInitialConditions(NamedTuple{(:file, :variable_map, :time_index),
    Tuple{String, Dict{Symbol,String}, Int}}[])

"""
    apply_pending_ic!(tracers, pending::PendingInitialConditions, grid)

Apply all pending IC entries to the given (now properly allocated) tracers.
"""
function apply_pending_ic!(tracers, pending::PendingInitialConditions, grid)
    for entry in pending.entries
        if isfile(entry.file)
            load_initial_conditions!(tracers, entry.file, grid;
                                      variable_map=entry.variable_map,
                                      time_index=entry.time_index)
        else
            @warn "IC file not found: $(entry.file)"
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Deferred vertical interpolation + mass conversion for IC data
#
# IC files (LMDZ) use hybrid-sigma levels (ap/bp/Psurf) that differ from
# GEOS levels. Full vertical interpolation requires DELP from the first met
# window. We store horizontally-regridded IC mixing ratios + vertical coords
# here, then `finalize_ic_vertical_interp!` does the pressure-based interp
# and mol/mol → tracer mass conversion after DELP is available.
# ---------------------------------------------------------------------------

"""Per-tracer deferred IC data: horizontally-regridded mixing ratios on source levels."""
struct DeferredICData{FT}
    tracer_name :: Symbol
    # Horizontally regridded mixing ratio on source vertical grid
    # panels_mr[p][i, j, k] = mixing ratio at source level k
    panels_mr :: NTuple{6, Array{FT, 3}}
    # Source vertical coordinate (hybrid sigma)
    ap :: Vector{Float64}   # half-level A coeff [Pa], length Nlev+1
    bp :: Vector{Float64}   # half-level B coeff [1], length Nlev+1
    # Surface pressure regridded to CS panels
    psurf :: NTuple{6, Matrix{Float64}}
    Nlev_src :: Int
end

const _DEFERRED_IC = Ref(DeferredICData[])

"""Per-tracer uniform IC: constant mixing ratio everywhere."""
struct UniformICData
    tracer_name :: Symbol
    value       :: Float64   # mixing ratio [mol/mol]
end

const _DEFERRED_UNIFORM_IC = Ref(UniformICData[])

has_deferred_ic_vinterp() = !isempty(_DEFERRED_IC[]) || !isempty(_DEFERRED_UNIFORM_IC[])

function _store_deferred_ic(data::DeferredICData)
    push!(_DEFERRED_IC[], data)
end

function _store_uniform_ic(data::UniformICData)
    push!(_DEFERRED_UNIFORM_IC[], data)
end

function _clear_deferred_ic()
    _DEFERRED_IC[] = DeferredICData[]
    _DEFERRED_UNIFORM_IC[] = UniformICData[]
end

"""
    finalize_ic_vertical_interp!(tracers, m_panels, delp_panels, grid)

After the first met window loads DELP and air mass is computed, perform
pressure-based vertical interpolation of deferred IC data and convert
from mixing ratio (mol/mol) to tracer mass (rm = q × m).

`delp_panels` should be NTuple{6} of GPU arrays with DELP [Pa].
`m_panels` should be NTuple{6} of GPU arrays with air mass [kg].
"""
function finalize_ic_vertical_interp!(tracers, m_panels, delp_panels,
                                       grid::CubedSphereGrid{FT}) where FT
    deferred = _DEFERRED_IC[]
    has_uniform = !isempty(_DEFERRED_UNIFORM_IC[])
    isempty(deferred) && !has_uniform && return nothing

    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz

    for ic_data in deferred
        tname = ic_data.tracer_name
        haskey(tracers, tname) || continue
        panels = tracers[tname]

        ap = ic_data.ap
        bp = ic_data.bp
        Nlev_src = ic_data.Nlev_src

        # Copy DELP and air mass to CPU for vertical interp
        cpu_delp = [Array(delp_panels[p]) for p in 1:6]
        cpu_m    = [Array(m_panels[p]) for p in 1:6]

        cpu_buf = zeros(FT, size(panels[1]))

        for p in 1:6
            fill!(cpu_buf, zero(FT))
            for j in 1:Nc, i in 1:Nc
                ii = Hp + i
                jj = Hp + j

                # Source pressure levels from hybrid sigma
                ps = ic_data.psurf[p][i, j]
                src_p_half = [ap[k] + bp[k] * ps for k in 1:Nlev_src+1]
                src_p_mid  = [0.5 * (src_p_half[k] + src_p_half[k+1]) for k in 1:Nlev_src]

                # Target (GEOS) pressure levels from DELP
                # Model convention: k=1 = TOA, k=Nz = surface (top-to-bottom).
                # Accumulate DELP from TOA (k=1) downward to build pressure.
                tgt_p_half = zeros(Float64, Nz + 1)
                tgt_p_half[1] = 0.0  # TOA
                for k in 1:Nz
                    tgt_p_half[k + 1] = tgt_p_half[k] + Float64(cpu_delp[p][ii, jj, k])
                end
                tgt_p_mid = [0.5 * (tgt_p_half[k] + tgt_p_half[k+1]) for k in 1:Nz]

                # Source mixing ratio profile
                src_q = [Float64(ic_data.panels_mr[p][i, j, k]) for k in 1:Nlev_src]

                # Log-pressure interpolation from source to target
                for k in 1:Nz
                    p_tgt = tgt_p_mid[k]
                    # Clamp to source range
                    if p_tgt >= src_p_mid[1]
                        cpu_buf[ii, jj, k] = FT(src_q[1])
                    elseif p_tgt <= src_p_mid[end]
                        cpu_buf[ii, jj, k] = FT(src_q[end])
                    else
                        # Find bracketing source levels
                        # src_p_mid is in decreasing order (surface → TOA)
                        ks = 1
                        while ks < Nlev_src && src_p_mid[ks+1] > p_tgt
                            ks += 1
                        end
                        # Linear interp in log-pressure
                        lp1 = log(src_p_mid[ks])
                        lp2 = log(src_p_mid[ks+1])
                        lpt = log(p_tgt)
                        w = (lpt - lp1) / (lp2 - lp1)
                        q_interp = src_q[ks] + w * (src_q[ks+1] - src_q[ks])
                        cpu_buf[ii, jj, k] = FT(q_interp)
                    end

                    # Convert mixing ratio → tracer mass: rm = q × m
                    cpu_buf[ii, jj, k] *= FT(cpu_m[p][ii, jj, k])
                end
            end
            copyto!(panels[p], cpu_buf)
        end

        # Log diagnostics
        q_vals = Float64[]
        for p in 1:6
            cpu_p = Array(panels[p])
            cpu_mp = Array(m_panels[p])
            for k in 1:Nz, j in Hp+1:Hp+Nc, i in Hp+1:Hp+Nc
                m_val = cpu_mp[i, j, k]
                if m_val > 0
                    push!(q_vals, cpu_p[i, j, k] / m_val)
                end
            end
        end
        if !isempty(q_vals)
            @info "IC finalized for $tname: mixing ratio min=$(minimum(q_vals)) " *
                  "max=$(maximum(q_vals)) mean=$(sum(q_vals)/length(q_vals))"
        end
    end

    # Process uniform ICs (flat mixing ratio everywhere)
    for uic in _DEFERRED_UNIFORM_IC[]
        tname = uic.tracer_name
        haskey(tracers, tname) || continue
        panels = tracers[tname]

        cpu_m = [Array(m_panels[p]) for p in 1:6]
        cpu_buf = zeros(FT, size(panels[1]))

        for p in 1:6
            fill!(cpu_buf, zero(FT))
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                ii = Hp + i
                jj = Hp + j
                cpu_buf[ii, jj, k] = FT(uic.value) * FT(cpu_m[p][ii, jj, k])
            end
            copyto!(panels[p], cpu_buf)
        end
        @info "IC finalized for $tname: uniform mixing ratio = $(uic.value)"
    end

    _clear_deferred_ic()
    return nothing
end

"""
    finalize_ic_vertical_interp!(tracers, m_3d, grid::LatitudeLongitudeGrid)

Lat-lon version: applies deferred uniform ICs.  Lat-lon tracers store mixing
ratios (q), so uniform IC simply fills with the constant value.  File-based IC
vertical interpolation for lat-lon is not yet implemented.
"""
function finalize_ic_vertical_interp!(tracers, m_3d,
                                       grid::LatitudeLongitudeGrid{FT}) where FT
    deferred = _DEFERRED_IC[]
    has_uniform = !isempty(_DEFERRED_UNIFORM_IC[])
    isempty(deferred) && !has_uniform && return nothing

    # File-based deferred ICs: not yet supported for lat-lon
    for ic_data in deferred
        @warn "Deferred file-based IC vertical interpolation not implemented for " *
              "lat-lon grids — skipping $(ic_data.tracer_name). Use uniform_value instead."
    end

    # Uniform ICs: lat-lon stores mixing ratio q directly (not rm)
    for uic in _DEFERRED_UNIFORM_IC[]
        tname = uic.tracer_name
        haskey(tracers, tname) || continue
        q = tracers[tname]
        fill!(q, FT(uic.value))
        @info "IC finalized for $tname (LatLon): uniform mixing ratio = $(uic.value)"
    end

    _clear_deferred_ic()
    return nothing
end
