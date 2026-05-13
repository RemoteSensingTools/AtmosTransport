# ---------------------------------------------------------------------------
# Plan 26 P0.D2 — bridge on-disk CSObservationSet to in-memory
# Vector{CSObservation{CSColumnMeanObjective, Float64}} consumed by the
# CS 4D-Var surface-flux path (`cs_surface_flux_4dvar`,
# `cs_surface_flux_jacobian`).
#
# Two mappings:
#
#   * Time. `date_components -> DateTime -> step index` via the left-
#     closed half-open interval `[t_start + (k-1)*dt, t_start + k*dt)`.
#     Observation exactly at `t_start` lands on step 1; an observation
#     at `t_start + nsteps*dt` (one past the end) lands on step
#     `nsteps + 1` and is rejected by the `:reject` policy.
#
#   * Geography. `(lat, lon) -> (panel, i, j)` via brute-force nearest
#     cell center, comparing unit-vector dot products in Cartesian
#     space. O(M * 6 * Nc^2) per call. Cheap for v1 inversion sizes
#     (M ~ 10^3-10^4 obs vs C48 = 13_824 cells); replace with a tree
#     or closed-form panel un-projection if a profile shows this
#     dominating runtime.
#
# Altitude is dropped: v1 always projects to a column-mean objective.
# Layer-stratified handling for satellite verticality can come later.
# ---------------------------------------------------------------------------

import Dates

const _OUT_OF_RANGE_POLICIES = (:reject, :skip, :clamp)

"""
    bind_to_mesh(set::CSObservationSet,
                 mesh::CubedSphereMesh,
                 t_start::Dates.DateTime,
                 dt::Real;
                 nsteps::Union{Nothing, Integer} = nothing,
                 tracer_filter::Union{Nothing, AbstractString} = nothing,
                 out_of_range_policy::Symbol = :reject)
        -> Vector{CSObservation{CSColumnMeanObjective, Float64}}

Map each `CSObservationRecord` in `set` to a `CSObservation` tied to a
model step and a [`CSColumnMeanObjective`](@ref) on `mesh`, ready to
feed `cs_surface_flux_jacobian` / `cs_surface_flux_4dvar`.

- `t_start` is the absolute model start time (UTC).
- `dt` is the model step length in seconds; the step grid is
  `t_start, t_start + dt, ..., t_start + nsteps*dt`.
- `nsteps`, if given, bounds the valid step range to `1:nsteps`.
  When `nothing`, only the lower bound (`step >= 1`) is enforced.
- `tracer_filter` keeps only records whose `tracer` field matches
  exactly (e.g. `"CO2"`). `nothing` keeps every record.
- `out_of_range_policy` chooses what happens when an observation's
  date falls outside the valid step range:
    - `:reject` (default) — throw an `ArgumentError`.
    - `:skip` — drop the record silently.
    - `:clamp` — snap to the nearest valid step (`1` or `nsteps`).

Altitude (`record.alt`) is ignored: every observation becomes a
column-mean objective. The on-disk `set.time_origin` is not consulted
during the computation; it is documentation only.
"""
function bind_to_mesh(set::CSObservationSet,
                      mesh::CubedSphereMesh,
                      t_start::Dates.DateTime,
                      dt::Real;
                      nsteps::Union{Nothing, Integer} = nothing,
                      tracer_filter::Union{Nothing, AbstractString} = nothing,
                      out_of_range_policy::Symbol = :reject)
    out_of_range_policy in _OUT_OF_RANGE_POLICIES || throw(ArgumentError(
        "out_of_range_policy must be one of $(_OUT_OF_RANGE_POLICIES); " *
        "got $(repr(out_of_range_policy))"))
    dt_f = float(dt)
    dt_f > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    if nsteps !== nothing
        nsteps >= 1 || throw(ArgumentError(
            "nsteps must be >= 1 when provided, got $nsteps"))
    end
    nsteps_max = nsteps === nothing ? typemax(Int) : Int(nsteps)
    tracer_match = tracer_filter === nothing ? nothing : String(tracer_filter)

    cache = _build_cs_cell_center_cache(mesh)
    out = Vector{CSObservation{CSColumnMeanObjective, Float64}}()
    sizehint!(out, length(set))

    @inbounds for record in set.records
        tracer_match === nothing || record.tracer == tracer_match || continue

        k = _step_index_from_date(record.date_components, t_start, dt_f)
        in_range = 1 <= k <= nsteps_max
        if !in_range
            if out_of_range_policy === :reject
                throw(ArgumentError(
                    "observation id $(record.id) at " *
                    "$(_date_components_string(record.date_components)) " *
                    "maps to step $k, outside [1, " *
                    "$(nsteps === nothing ? "Inf" : string(nsteps_max))]; " *
                    "pass `out_of_range_policy = :skip` or `:clamp` to handle"))
            elseif out_of_range_policy === :skip
                continue
            else  # :clamp
                k = clamp(k, 1, nsteps_max)
            end
        end

        p, i, j = _locate_cs_cell(Float64(record.lat),
                                  Float64(record.lon), cache)
        push!(out, CSObservation(k,
                                 CSColumnMeanObjective(p, i, j),
                                 record.value, record.value_sigma))
    end
    return out
end

# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------

@inline function _step_index_from_date(dc::NTuple{6, Int16},
                                       t_start::Dates.DateTime,
                                       dt_f::Real)
    t_obs = Dates.DateTime(Int(dc[1]), Int(dc[2]), Int(dc[3]),
                            Int(dc[4]), Int(dc[5]), Int(dc[6]))
    ms = Dates.value(t_obs - t_start)   # Int64 milliseconds (signed)
    seconds = ms / 1000
    return floor(Int, seconds / dt_f) + 1
end

_date_components_string(dc::NTuple{6, Int16}) =
    string(lpad(Int(dc[1]), 4, '0'), "-",
           lpad(Int(dc[2]), 2, '0'), "-",
           lpad(Int(dc[3]), 2, '0'), "T",
           lpad(Int(dc[4]), 2, '0'), ":",
           lpad(Int(dc[5]), 2, '0'), ":",
           lpad(Int(dc[6]), 2, '0'))

# ---------------------------------------------------------------------------
# Geographic mapping
# ---------------------------------------------------------------------------

# Flat unit-vector cache for all 6 * Nc^2 cell centers. Stored as three
# Float64 vectors so the hot loop is a plain SIMD-friendly dot product.
struct _CSCellCenterCache
    Nc::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    zs::Vector{Float64}
end

function _build_cs_cell_center_cache(mesh::CubedSphereMesh)
    Nc = mesh.Nc
    n = 6 * Nc * Nc
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    zs = Vector{Float64}(undef, n)
    @inbounds for p in 1:6
        lons, lats = panel_cell_center_lonlat(mesh, p)
        for j in 1:Nc, i in 1:Nc
            ux, uy, uz = _lonlat_to_unit_xyz(Float64(lons[i, j]),
                                              Float64(lats[i, j]))
            idx = _cs_cache_index(p, i, j, Nc)
            xs[idx] = ux
            ys[idx] = uy
            zs[idx] = uz
        end
    end
    return _CSCellCenterCache(Nc, xs, ys, zs)
end

@inline _cs_cache_index(p::Integer, i::Integer, j::Integer, Nc::Integer) =
    ((Int(p) - 1) * Nc + (Int(j) - 1)) * Nc + Int(i)

@inline function _lonlat_to_unit_xyz(lon_deg::Float64, lat_deg::Float64)
    lon = deg2rad(lon_deg)
    lat = deg2rad(lat_deg)
    cl = cos(lat)
    return (cl * cos(lon), cl * sin(lon), sin(lat))
end

@inline function _locate_cs_cell(lat_deg::Float64, lon_deg::Float64,
                                 cache::_CSCellCenterCache)
    ux, uy, uz = _lonlat_to_unit_xyz(lon_deg, lat_deg)
    xs = cache.xs; ys = cache.ys; zs = cache.zs
    best = -Inf
    best_idx = 1
    @inbounds for idx in eachindex(xs)
        d = xs[idx] * ux + ys[idx] * uy + zs[idx] * uz
        if d > best
            best = d
            best_idx = idx
        end
    end
    Nc = cache.Nc
    idx0 = best_idx - 1
    i = (idx0 % Nc) + 1
    idx0 ÷= Nc
    j = (idx0 % Nc) + 1
    p = (idx0 ÷ Nc) + 1
    return (Int(p), Int(i), Int(j))
end
