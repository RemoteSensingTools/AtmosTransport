# =====================================================================
# Unified regridding utilities for emission sources
#
# All lon/lat normalization, cell area computation, and regridding
# algorithms live here to avoid duplication and ensure consistency.
# =====================================================================

using ..Grids: CubedSphereGrid, LatitudeLongitudeGrid, cell_area, floattype,
               has_gmao_coords, set_coord_status!

# =====================================================================
# Longitude / latitude normalization
# =====================================================================

"""
    ensure_south_to_north(flux, lat_src) → (flux_sn, lat_sorted)

Flip array and latitude vector so latitude increases (south → north).
Returns copies; originals are not modified.
"""
function ensure_south_to_north(flux::Matrix, lat_src)
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        return flux[:, end:-1:1], reverse(lat_src)
    end
    return flux, collect(lat_src)
end

"""
    remap_lon_neg180_to_0_360(lon_src, flux) → (lon_new, flux_new)

Remap longitudes from -180:180 convention to 0:360 by reordering the
longitude axis. If already in [0, 360], returns a copy unchanged.
"""
function remap_lon_neg180_to_0_360(lon_src::AbstractVector, flux::Matrix{FT}) where FT
    if minimum(lon_src) >= 0
        return FT.(lon_src), flux
    end
    n = length(lon_src)
    split = findfirst(>=(0), lon_src)
    if split === nothing
        # All negative — just shift by 360
        return FT.(lon_src .+ 360), flux
    end
    idx = vcat(split:n, 1:split-1)
    lon_new = FT.(mod.(lon_src[idx], 360))
    flux_new = flux[idx, :]
    return lon_new, flux_new
end

"""
    normalize_lons_lats(lons, lats, flux, FT) → (lons_out, lats_out, flux_out)

Ensure south-to-north latitude ordering and [0, 360] longitude convention.
Handles any input convention automatically.
"""
function normalize_lons_lats(lons, lats, flux::Matrix{FT}) where FT
    flux_sn, lat_sorted = ensure_south_to_north(flux, lats)
    lon_out, flux_out = remap_lon_neg180_to_0_360(lons, flux_sn)
    return lon_out, FT.(lat_sorted), flux_out
end

# =====================================================================
# Lat-lon cell area computation
# =====================================================================

"""
    latlon_cell_areas(lons, lats, R) → Vector{FT}

Compute cell areas for a regular lat-lon grid. Returns a vector of length
`Nlat` since areas depend only on latitude for regular grids.
Uses the exact spherical formula: A = R² × Δλ × |sin(φ_n) - sin(φ_s)|
"""
function latlon_cell_areas(lons::AbstractVector{FT}, lats::AbstractVector{FT},
                           R::FT) where FT
    Δlon = abs(lons[2] - lons[1])
    Δlat = abs(lats[2] - lats[1])
    Nlat = length(lats)
    areas = Vector{FT}(undef, Nlat)
    @inbounds for j in 1:Nlat
        φ_s = lats[j] - Δlat / 2
        φ_n = lats[j] + Δlat / 2
        areas[j] = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
    end
    return areas
end

"""
    latlon_cell_area(lon_idx, lat_idx, lons, lats, R) → FT

Compute area of a single lat-lon cell. Convenience wrapper.
"""
function latlon_cell_area(j::Int, lons::AbstractVector{FT}, lats::AbstractVector{FT},
                          R::FT) where FT
    Δlon = abs(lons[2] - lons[1])
    Δlat = abs(lats[2] - lats[1])
    φ_s = lats[j] - Δlat / 2
    φ_n = lats[j] + Δlat / 2
    return R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
end

"""
    _equal_area_sample_lat(lat_south, lat_north, frac) → lat

Sample a latitude inside `[lat_south, lat_north]` using uniform spacing in
`sin(lat)`, which corresponds to uniform sampling in spherical cell area.
`frac` should lie in `[0, 1]`.
"""
@inline function _equal_area_sample_lat(lat_south::Float64,
                                        lat_north::Float64,
                                        frac::Float64)
    sin_s = sind(lat_south)
    sin_n = sind(lat_north)
    sin_lat = muladd(frac, sin_n - sin_s, sin_s)
    return asind(clamp(sin_lat, -1.0, 1.0))
end

"""
    _default_cs_map_subsampling(Δlon_s, Δlat_s, Nc) → (N_sub, ratio)

Heuristic sub-cell sampling resolution for lat-lon -> cubed-sphere overlap
approximation. `ratio` is the maximum source-cell width relative to the target
CS cell width in degrees.
"""
@inline function _default_cs_map_subsampling(Δlon_s, Δlat_s, Nc::Int)
    Δcs_deg = oftype(max(Δlon_s, Δlat_s), 90) / Nc
    ratio = max(Δlon_s, Δlat_s) / Δcs_deg
    N_sub = ratio > 1 ? max(20, ceil(Int, ratio * 10)) : max(5, ceil(Int, ratio * 8))
    return N_sub, ratio
end

# =====================================================================
# Conservative lat-lon → cubed-sphere regridding map
# =====================================================================

"""
    ConservativeCSMap{FT}

Source-indexed conservative mapping from a regular lat-lon grid to
cubed-sphere panels. For each source cell at linear index
`k = (j-1)*Nlon + i`, the CSR entries at `offsets[k] : offsets[k+1]-1`
list the CS cell(s) that the source cell contributes mass to.

Regridding uses mass accumulation: each source cell's mass is scattered
to its target CS cell(s), then divided by exact CS cell area to get density.
Sub-cell sampling ensures both fine→coarse and coarse→fine work correctly.
"""
struct ConservativeCSMap{FT}
    offsets     :: Vector{Int}      # CSR row pointers, length Nlon*Nlat + 1
    target_p    :: Vector{Int32}    # panel index per entry
    target_i    :: Vector{Int32}    # cell i per entry
    target_j    :: Vector{Int32}    # cell j per entry
    weight      :: Vector{FT}      # fraction of source mass going to this CS cell
    native_area :: Vector{FT}      # per-latitude source cell areas [m²]
    eff_area    :: NTuple{6, Matrix{FT}}  # effective area per CS cell (from sampling)
    cs_area     :: NTuple{6, Matrix{FT}}  # exact CS cell areas (from gridspec/grid)
    Nlon :: Int
    Nlat :: Int
end

"""
    _to_m180_180(lon) → lon in [-180, 180]

Normalize longitude to [-180, 180] range.
"""
_to_m180_180(lon) = mod(lon + oftype(lon, 180), oftype(lon, 360)) - oftype(lon, 180)

"""
    _find_nearest_cs_cell(src_lon, src_lat, bins, n_lon_bins, n_lat_bins,
                           bin_size, grid, Nc) → (p, ic, jc)

Find the CS cell whose center is nearest to (src_lon, src_lat) using spatial bins.
"""
function _find_nearest_cs_cell(src_lon::Float64, src_lat::Float64,
                                bins, n_lon_bins::Int, n_lat_bins::Int,
                                bin_size::Float64, grid)
    bi0 = clamp(floor(Int, src_lon / bin_size) + 1, 1, n_lon_bins)
    bj0 = clamp(floor(Int, (src_lat + 90.0) / bin_size) + 1, 1, n_lat_bins)

    best_dist = Inf
    best_p = Int32(0)
    best_ic = Int32(0)
    best_jc = Int32(0)

    for dbj in -1:1, dbi in -1:1
        bi = bi0 + dbi
        bj = bj0 + dbj
        bi < 1 && (bi += n_lon_bins)
        bi > n_lon_bins && (bi -= n_lon_bins)
        (bj < 1 || bj > n_lat_bins) && continue

        for (cp, ci, cj) in bins[(bj - 1) * n_lon_bins + bi]
            cs_lon = mod(Float64(grid.λᶜ[cp][ci, cj]), 360.0)
            cs_lat = Float64(grid.φᶜ[cp][ci, cj])
            dlon = abs(src_lon - cs_lon)
            dlon > 180.0 && (dlon = 360.0 - dlon)
            dlat = abs(src_lat - cs_lat)
            dist = dlon^2 + dlat^2
            if dist < best_dist
                best_dist = dist
                best_p = cp
                best_ic = ci
                best_jc = cj
            end
        end
    end
    return best_p, best_ic, best_jc
end

"""
    build_conservative_cs_map(lons, lats, grid; cs_areas=nothing, N_sub=nothing)
        → ConservativeCSMap

Build a conservative regridding map from a regular lat-lon grid to
cubed-sphere using sub-cell sampling.

For each source cell, `N_sub × N_sub` sample points are placed uniformly
in longitude and uniformly in `sin(lat)` within its bounds. Each sample
finds the nearest CS cell center. The fraction of samples mapping to each
CS cell becomes the weight.

This handles both fine→coarse (most samples → same CS cell, weight ≈ 1.0)
and coarse→fine (samples spread across multiple CS cells).

If `cs_areas` is provided (e.g., from Met_AREAM2 or corner-based computation),
these exact areas are stored in the map for density conversion. Otherwise,
grid.Aᶜ (gnomonic areas) is used as fallback.

`N_sub` optionally overrides the adaptive sub-cell sampling resolution. This is
useful for accuracy studies where the overlap approximation should be tightened.
"""
function build_conservative_cs_map(lons::AbstractVector{FT}, lats::AbstractVector{FT},
                                    grid::CubedSphereGrid{FT};
                                    cs_areas::Union{Nothing, NTuple{6, Matrix{FT}}} = nothing,
                                    N_sub::Union{Nothing, Int} = nothing
                                    ) where FT
    Nc = grid.Nc

    # Auto-generate GMAO coordinates if not already loaded
    if !has_gmao_coords(grid)
        @info "Generating C$Nc gridspec (no GMAO coordinates loaded)..."
        spec = generate_cs_gridspec(Nc; R=Float64(grid.radius))
        for p in 1:6, j in 1:Nc, i in 1:Nc
            grid.λᶜ[p][i, j] = FT(spec.lons[p][i, j])
            grid.φᶜ[p][i, j] = FT(spec.lats[p][i, j])
            grid.Aᶜ[p][i, j] = FT(spec.areas[p][i, j])
        end
        set_coord_status!(grid, :gmao, "generate_cs_gridspec($(Nc))")
        if cs_areas === nothing
            cs_areas = ntuple(p -> FT.(spec.areas[p]), 6)
        end
    end

    Nlon = length(lons)
    Nlat = length(lats)
    R = FT(grid.radius)

    Δlon_s = abs(lons[2] - lons[1])
    Δlat_s = abs(lats[2] - lats[1])

    native_area = latlon_cell_areas(lons, lats, R)

    # Use exact areas if provided, otherwise fall back to grid areas
    area = cs_areas !== nothing ? cs_areas : ntuple(p -> FT.(grid.Aᶜ[p]), 6)

    # Determine sub-cell sampling resolution.
    # Need enough samples that each CS cell gets many hits for accurate weights.
    # For coarse->fine (e.g., 1 degree on C180 ~0.5 degree), each source cell
    # covers multiple CS cells across, so N_sub must be large enough for smooth
    # fractional coverage.
    N_sub_eff, ratio = _default_cs_map_subsampling(Δlon_s, Δlat_s, Nc)
    if N_sub !== nothing
        N_sub_eff = max(1, N_sub)
    end
    @info "build_conservative_cs_map: N_sub=$N_sub_eff ($(Nlon)×$(Nlat) → C$(Nc), ratio=$(round(ratio, digits=2)))" maxlog=1

    # Build spatial index: bin CS cell centers by lon/lat
    bin_size = max(1.0, 90.0 / Nc * 2)
    n_lon_bins = ceil(Int, 360.0 / bin_size)
    n_lat_bins = ceil(Int, 180.0 / bin_size)
    bins = [Tuple{Int32, Int32, Int32}[] for _ in 1:n_lon_bins * n_lat_bins]

    for p in 1:6, jc in 1:Nc, ic in 1:Nc
        cs_lon = mod(Float64(grid.λᶜ[p][ic, jc]), 360.0)
        cs_lat = Float64(grid.φᶜ[p][ic, jc])
        bi = clamp(floor(Int, cs_lon / bin_size) + 1, 1, n_lon_bins)
        bj = clamp(floor(Int, (cs_lat + 90.0) / bin_size) + 1, 1, n_lat_bins)
        push!(bins[(bj - 1) * n_lon_bins + bi], (Int32(p), Int32(ic), Int32(jc)))
    end

    # Sub-cell sampling: for each source cell, generate N_sub² sample points,
    # find nearest CS cell for each, count samples per CS cell → weight
    Ntotal = Nlon * Nlat
    # Temporary storage: list of (p, i, j, weight) per source cell
    all_entries = Vector{Vector{Tuple{Int32, Int32, Int32, FT}}}(undef, Ntotal)

    n_entries_total = 0
    counts = Dict{Tuple{Int32, Int32, Int32}, Int}()
    N_sub_sq = N_sub_eff * N_sub_eff

    for js in 1:Nlat, is in 1:Nlon
        lin = (js - 1) * Nlon + is
        lon_west = Float64(lons[is]) - Float64(Δlon_s) / 2
        lat_south = Float64(lats[js]) - Float64(Δlat_s) / 2
        lat_north = Float64(lats[js]) + Float64(Δlat_s) / 2

        empty!(counts)

        for sj in 1:N_sub_eff, si in 1:N_sub_eff
            lon_s = mod(lon_west + (si - 0.5) / N_sub_eff * Float64(Δlon_s), 360.0)
            # Sample uniformly in sin(lat) so each sub-cell point represents an
            # equal share of the spherical source-cell area.
            lat_s = _equal_area_sample_lat(lat_south, lat_north, (sj - 0.5) / N_sub_eff)
            p, ic, jc = _find_nearest_cs_cell(lon_s, lat_s, bins, n_lon_bins,
                                               n_lat_bins, bin_size, grid)
            p == 0 && continue
            key = (p, ic, jc)
            counts[key] = get(counts, key, 0) + 1
        end

        entries = Vector{Tuple{Int32, Int32, Int32, FT}}(undef, length(counts))
        idx = 0
        for ((p, ic, jc), cnt) in counts
            idx += 1
            entries[idx] = (p, ic, jc, FT(cnt) / FT(N_sub_sq))
        end
        all_entries[lin] = entries
        n_entries_total += length(entries)
    end

    # Flatten to CSR arrays
    offsets = Vector{Int}(undef, Ntotal + 1)
    target_p = Vector{Int32}(undef, n_entries_total)
    target_i = Vector{Int32}(undef, n_entries_total)
    target_j = Vector{Int32}(undef, n_entries_total)
    weight = Vector{FT}(undef, n_entries_total)

    offsets[1] = 1
    k = 1
    for lin in 1:Ntotal
        for (p, ic, jc, w) in all_entries[lin]
            target_p[k] = p
            target_i[k] = ic
            target_j[k] = jc
            weight[k] = w
            k += 1
        end
        offsets[lin + 1] = k
    end

    # Compute effective area per CS cell: sum of weighted source areas
    # eff_area ensures uniform fields map to uniform (exact by construction)
    eff_area = ntuple(_ -> zeros(FT, Nc, Nc), 6)
    for lin in 1:Ntotal
        js = div(lin - 1, Nlon) + 1
        for kidx in offsets[lin] : offsets[lin + 1] - 1
            eff_area[target_p[kidx]][target_i[kidx], target_j[kidx]] +=
                native_area[js] * weight[kidx]
        end
    end

    # Report statistics
    avg_targets = n_entries_total / Ntotal
    @info "Conservative CS map: $(n_entries_total) entries, " *
          "avg $(round(avg_targets, digits=2)) CS cells/source cell"

    return ConservativeCSMap{FT}(offsets, target_p, target_i, target_j, weight,
                                  native_area, eff_area, area, Nlon, Nlat)
end

# Keep old API for backward compatibility
build_latlon_to_cs_map(lons::AbstractVector{FT}, lats::AbstractVector{FT},
                       grid::CubedSphereGrid{FT}; kwargs...) where FT =
    build_conservative_cs_map(lons, lats, grid; kwargs...)

# =====================================================================
# Conservative lat-lon → cubed-sphere regridding
# =====================================================================

"""
    regrid_latlon_to_cs(flux_kgm2s, lons, lats, grid; cs_map=nothing, N_sub=nothing)
        → NTuple{6, Matrix}

Conservative regridding from lat-lon to cubed-sphere via mass accumulation.

Algorithm:
1. For each source cell, compute mass = flux × area
2. Scatter mass to target CS cell(s) using sub-cell sampling weights
3. Divide accumulated mass by exact CS cell area → flux density

Mass conservation is exact by construction (weights sum to 1.0 per source cell).
"""
function regrid_latlon_to_cs(flux_kgm2s::Matrix{FT}, lons::AbstractVector{FT},
                              lats::AbstractVector{FT},
                              grid::CubedSphereGrid{FT};
                              cs_map = nothing,
                              cs_areas = nothing,
                              N_sub::Union{Nothing, Int} = nothing,
                              renormalize = false) where FT
    Nc = grid.Nc
    Nlon = length(lons)
    Nlat = length(lats)

    if !has_gmao_coords(grid)
        @warn "regrid_latlon_to_cs: grid uses gnomonic coordinates — " *
              "load GMAO coordinates for accurate panel placement" maxlog=1
    end

    if cs_map === nothing
        cs_map = build_conservative_cs_map(lons, lats, grid;
                                           cs_areas=cs_areas, N_sub=N_sub)
    end

    (; offsets, target_p, target_i, target_j, weight, native_area, eff_area, cs_area) = cs_map

    # Accumulate mass (kg/s) per CS cell
    mass_acc = ntuple(_ -> zeros(FT, Nc, Nc), 6)

    @inbounds for j in 1:Nlat, i in 1:Nlon
        lin = (j - 1) * Nlon + i
        f = flux_kgm2s[i, j]
        f == zero(FT) && continue
        mass = f * native_area[j]
        for k in offsets[lin] : offsets[lin + 1] - 1
            mass_acc[target_p[k]][target_i[k], target_j[k]] += mass * weight[k]
        end
    end

    # Convert mass → flux density using effective area (from sampling).
    # eff_area = Σ src_area * weight for each CS cell, so for uniform input:
    #   flux_out = (1.0 * eff_area) / eff_area = 1.0  (exact by construction)
    flux_out = ntuple(6) do p
        out = zeros(FT, Nc, Nc)
        for jc in 1:Nc, ic in 1:Nc
            a = eff_area[p][ic, jc]
            if a > zero(FT)
                out[ic, jc] = mass_acc[p][ic, jc] / a
            end
        end
        out
    end

    # Mass conservation diagnostic using exact CS areas (what transport sees)
    total_src = zero(Float64)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        total_src += Float64(flux_kgm2s[i, j]) * Float64(native_area[j])
    end
    total_tgt = sum(sum(Float64.(flux_out[p]) .* Float64.(cs_area[p])) for p in 1:6)

    if abs(total_src) > 1e-30
        mass_err = abs(total_tgt - total_src) / abs(total_src)
        mass_err > 0.01 && @warn "regrid_latlon_to_cs mass error: " *
            "$(round(mass_err * 100, digits=3))% (src=$(total_src), tgt=$(total_tgt))"
    end

    # Optional renormalization to ensure exact mass conservation with transport grid
    if renormalize && abs(total_tgt) > 1e-30 && abs(total_src) > 1e-30
        scale = FT(total_src / total_tgt)
        if FT(0.5) < scale < FT(2.0)
            for p in 1:6
                flux_out[p] .*= scale
            end
        end
    end

    return flux_out
end

# =====================================================================
# Nearest-neighbor regridding (lat-lon → lat-lon)
# =====================================================================

"""
    nearest_neighbor_regrid(flux_src, lon_src, lat_src, grid) → Matrix{FT}

Nearest-neighbor regridding from source lat-lon to model lat-lon grid.
Handles any longitude/latitude convention automatically.

**Not mass-conserving** — use `_conservative_regrid` for high-resolution
sources where mass conservation matters.
"""
function nearest_neighbor_regrid(flux_src::Matrix{FT}, lon_src, lat_src,
                                  grid::LatitudeLongitudeGrid{FT}) where FT
    Nx_m, Ny_m = grid.Nx, grid.Ny
    flux_out = zeros(FT, Nx_m, Ny_m)

    # Normalize: ensure S→N + 0:360
    lon_use, lat_sorted, flux_sorted = normalize_lons_lats(lon_src, lat_src, flux_src)

    λᶜ = grid.λᶜ_cpu
    φᶜ = grid.φᶜ_cpu

    for jm in 1:Ny_m, im in 1:Nx_m
        js = _nearest_idx(φᶜ[jm], lat_sorted)
        is = _nearest_idx(λᶜ[im], lon_use)
        flux_out[im, jm] = flux_sorted[is, js]
    end

    return flux_out
end

"""Find index of nearest value in a sorted-ish array."""
function _nearest_idx(val, arr)
    _, idx = findmin(abs.(arr .- val))
    return idx
end

# =====================================================================
# Conservative lat-lon → lat-lon regridding
# =====================================================================

"""
    conservative_regrid_ll(flux_native, lon_src, lat_src, grid) → Matrix{FT}

Conservative (mass-preserving) regridding from a lat-lon source grid to a
`LatitudeLongitudeGrid` model grid. Computes exact spherical overlap fractions
between source and target cells using the `sin(φ)` latitude formula.

Suitable for any resolution ratio (fine→coarse and coarse→fine).
"""
function conservative_regrid_ll(flux_native::Matrix{FT},
                                 lon_src::AbstractVector, lat_src::AbstractVector,
                                 grid::LatitudeLongitudeGrid{FT}) where FT
    Nx_m, Ny_m = grid.Nx, grid.Ny
    Δlon_s = FT(lon_src[2] - lon_src[1])
    Δlat_s = FT(lat_src[2] - lat_src[1])
    R = FT(grid.radius)

    mass_model = zeros(FT, Nx_m, Ny_m)

    λᶠ_cpu = grid.λᶠ_cpu
    φᶠ_cpu = grid.φᶠ_cpu

    for js in eachindex(lat_src), is in eachindex(lon_src)
        f = flux_native[is, js]
        f == zero(FT) && continue

        φ_s_south = FT(lat_src[js]) - Δlat_s / 2
        φ_s_north = FT(lat_src[js]) + Δlat_s / 2
        area_src = R^2 * deg2rad(Δlon_s) * abs(sind(φ_s_north) - sind(φ_s_south))
        emission_rate = f * area_src

        lon_s_west = FT(lon_src[is]) - Δlon_s / 2
        lon_s_east = FT(lon_src[is]) + Δlon_s / 2

        im_start = _find_model_index_lon(lon_s_west, λᶠ_cpu)
        im_end   = _find_model_index_lon(lon_s_east - FT(1e-10), λᶠ_cpu)
        jm_start = _find_model_index_lat(φ_s_south, φᶠ_cpu)
        jm_end   = _find_model_index_lat(φ_s_north - FT(1e-10), φᶠ_cpu)

        (im_start === nothing || jm_start === nothing) && continue
        (im_end === nothing || jm_end === nothing) && continue

        for jm in jm_start:jm_end, im in im_start:im_end
            (im < 1 || im > Nx_m || jm < 1 || jm > Ny_m) && continue
            overlap_lon = max(zero(FT), min(lon_s_east, λᶠ_cpu[im + 1]) -
                                        max(lon_s_west, λᶠ_cpu[im]))
            overlap_lat_s = max(φ_s_south, φᶠ_cpu[jm])
            overlap_lat_n = min(φ_s_north, φᶠ_cpu[jm + 1])
            overlap_lat_n <= overlap_lat_s && continue
            frac_lon = overlap_lon / Δlon_s
            frac_lat = abs(sind(overlap_lat_n) - sind(overlap_lat_s)) /
                       abs(sind(φ_s_north) - sind(φ_s_south))
            mass_model[im, jm] += emission_rate * frac_lon * frac_lat
        end
    end

    # Divide by model cell area to get flux density
    flux_model = zeros(FT, Nx_m, Ny_m)
    for jm in 1:Ny_m, im in 1:Nx_m
        a = cell_area(im, jm, grid)
        flux_model[im, jm] = mass_model[im, jm] / a
    end

    return flux_model
end

# Keep old name as alias for backward compatibility
const _conservative_regrid = conservative_regrid_ll

function _find_model_index_lon(lon_val, λᶠ)
    n = length(λᶠ) - 1
    for i in 1:n
        if lon_val >= λᶠ[i] && lon_val < λᶠ[i + 1]
            return i
        end
    end
    return nothing
end

function _find_model_index_lat(lat_val, φᶠ)
    n = length(φᶠ) - 1
    for i in 1:n
        if lat_val >= φᶠ[i] && lat_val < φᶠ[i + 1]
            return i
        end
    end
    return nothing
end

# =====================================================================
# Unit conversion helpers
# =====================================================================

"""
    tonnes_per_year_to_kgm2s(raw, lons, lats, R) → Matrix{FT}

Convert Tonnes/year per cell → kg/m²/s using analytical cell areas.
"""
function tonnes_per_year_to_kgm2s(raw::Matrix{FT}, lons, lats, R::FT) where FT
    areas = latlon_cell_areas(FT.(lons), FT.(lats), R)
    sec_per_yr = FT(365.25 * 86400)
    Nlon, Nlat = size(raw)
    flux = similar(raw)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        flux[i, j] = raw[i, j] * FT(1000) / (sec_per_yr * areas[j])
    end
    return flux
end

"""
    mass_per_cell_to_kgm2s(raw, lons, lats, R, sec_per_period) → Matrix{FT}

Convert mass/cell/period → kg/m²/s using analytical cell areas.
"""
function mass_per_cell_to_kgm2s(raw::Matrix{FT}, lons, lats, R::FT,
                                 sec_per_period::FT) where FT
    areas = latlon_cell_areas(FT.(lons), FT.(lats), R)
    Nlon, Nlat = size(raw)
    flux = similar(raw)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        flux[i, j] = raw[i, j] / (sec_per_period * areas[j])
    end
    return flux
end

# =====================================================================
# Area computation from corner coordinates
# =====================================================================

"""
    compute_areas_from_corners(corner_lons, corner_lats, R, Nc) → NTuple{6, Matrix{Float64}}

Compute cell areas from GMAO corner coordinates using spherical quadrilateral
formula (l'Huilier's theorem). Returns areas in m² that match GEOS-Chem's
`Met_AREAM2` to <0.04%.

`corner_lons` and `corner_lats` are (Nc+1, Nc+1, 6) arrays of cell corners
in degrees.
"""
function compute_areas_from_corners(corner_lons::Array{Float64, 3},
                                     corner_lats::Array{Float64, 3},
                                     R::Float64, Nc::Int)
    areas = ntuple(6) do p
        panel_areas = zeros(Float64, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            panel_areas[i, j] = _spherical_quad_area_lonlat(
                corner_lons[i, j, p],     corner_lats[i, j, p],
                corner_lons[i+1, j, p],   corner_lats[i+1, j, p],
                corner_lons[i+1, j+1, p], corner_lats[i+1, j+1, p],
                corner_lons[i, j+1, p],   corner_lats[i, j+1, p],
                R)
        end
        panel_areas
    end
    return areas
end

"""Spherical quadrilateral area from 4 corner lon/lat pairs (degrees)."""
function _spherical_quad_area_lonlat(lon1, lat1, lon2, lat2,
                                      lon3, lat3, lon4, lat4, R)
    v1 = _lonlat_to_xyz(lon1, lat1)
    v2 = _lonlat_to_xyz(lon2, lat2)
    v3 = _lonlat_to_xyz(lon3, lat3)
    v4 = _lonlat_to_xyz(lon4, lat4)
    return R^2 * (_spherical_triangle_area(v1, v2, v3) +
                  _spherical_triangle_area(v1, v3, v4))
end

function _lonlat_to_xyz(lon_deg, lat_deg)
    λ = deg2rad(lon_deg)
    φ = deg2rad(lat_deg)
    return (cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))
end

"""Spherical triangle excess area on unit sphere via l'Huilier's theorem."""
function _spherical_triangle_area(v1, v2, v3)
    a = acos(clamp(v2[1]*v3[1] + v2[2]*v3[2] + v2[3]*v3[3], -1.0, 1.0))
    b = acos(clamp(v1[1]*v3[1] + v1[2]*v3[2] + v1[3]*v3[3], -1.0, 1.0))
    c = acos(clamp(v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3], -1.0, 1.0))
    s = (a + b + c) / 2
    t = max(tan(s/2) * tan((s-a)/2) * tan((s-b)/2) * tan((s-c)/2), 0.0)
    return 4 * atan(sqrt(t))
end

# =====================================================================
# Cubed-sphere grid generation (port of gcpy CSGrid + csgrid_gmao)
#
# Generates cell corner coordinates, centers, and areas for any
# cubed-sphere resolution Nc. Uses the gnomonic projection with GMAO
# face orientation, matching GEOS-Chem/GCHP conventions.
#
# Reference: Jiawei Zhuang's cubedsphere package
#   https://github.com/JiaweiZhuang/cubedsphere
# Ported from gcpy/grid.py (Liam Bindle, Sebastian Eastham)
# =====================================================================

"""
    generate_cs_gridspec(Nc; R=6.371e6) → (corner_lons, corner_lats, lons, lats, areas)

Generate cubed-sphere grid specification for resolution C`Nc`.

Returns NTuple of 6 arrays per quantity:
- `corner_lons[p]`, `corner_lats[p]`: `(Nc+1, Nc+1)` cell corner coordinates [degrees]
- `lons[p]`, `lats[p]`: `(Nc, Nc)` cell center coordinates [degrees]
- `areas[p]`: `(Nc, Nc)` cell areas [m²]

Uses the GMAO gnomonic projection and face orientation, matching
GEOS-Chem/GCHP conventions. Validated against GEOS-Chem Met_AREAM2
to <0.003% per cell for C180.
"""
function generate_cs_gridspec(Nc::Int; R::Float64=6.371e6)
    # Step 1: Generate edges on face 0 via gnomonic projection
    inv_sqrt3 = 1.0 / sqrt(3.0)
    asin_inv_sqrt3 = asin(inv_sqrt3)
    delta_y = 2.0 * asin_inv_sqrt3 / Nc
    nx = Nc + 1

    lambda_rad = zeros(nx, nx)
    theta_rad  = zeros(nx, nx)

    # West and east edges
    lambda_rad[1, :] .= 3π / 4
    lambda_rad[nx, :] .= 5π / 4
    theta_rad[1, :]  .= -asin_inv_sqrt3 .+ delta_y .* (0:Nc)
    theta_rad[nx, :] .= theta_rad[1, :]

    # Mirror points
    lon_mir1, lat_mir1 = lambda_rad[1, 1], theta_rad[1, 1]
    lon_mir2, lat_mir2 = lambda_rad[nx, nx], theta_rad[nx, nx]
    xyz_mir1 = _ll2cart(lon_mir1, lat_mir1)
    xyz_mir2 = _ll2cart(lon_mir2, lat_mir2)
    xyz_cross = _cross3(xyz_mir1, xyz_mir2)
    xyz_cross = xyz_cross ./ sqrt(sum(xyz_cross .^ 2))

    for i in 2:Nc
        lon_ref, lat_ref = lambda_rad[1, i], theta_rad[1, i]
        xyz_ref = collect(_ll2cart(lon_ref, lat_ref))
        xyz_dot = sum(xyz_cross .* xyz_ref)
        xyz_img = xyz_ref .- 2.0 .* xyz_dot .* xyz_cross
        lon_img, lat_img = _cart2ll(xyz_img[1], xyz_img[2], xyz_img[3])
        lambda_rad[i, 1]  = lon_img
        lambda_rad[i, nx] = lon_img
        theta_rad[i, 1]   = lat_img
        theta_rad[i, nx]  = -lat_img
    end

    # Map edges on sphere back to cube (all at x = -1/√3)
    pp = zeros(3, nx, nx)
    for i in (1, nx), j in (1, nx)
        pp[:, i, j] .= _ll2cart(lambda_rad[i, j], theta_rad[i, j])
    end
    for ij in 2:nx
        pp[:, 1, ij] .= _ll2cart(lambda_rad[1, ij], theta_rad[1, ij])
        pp[2, 1, ij] = -pp[2, 1, ij] * inv_sqrt3 / pp[1, 1, ij]
        pp[3, 1, ij] = -pp[3, 1, ij] * inv_sqrt3 / pp[1, 1, ij]

        pp[:, ij, 1] .= _ll2cart(lambda_rad[ij, 1], theta_rad[ij, 1])
        pp[2, ij, 1] = -pp[2, ij, 1] * inv_sqrt3 / pp[1, ij, 1]
        pp[3, ij, 1] = -pp[3, ij, 1] * inv_sqrt3 / pp[1, ij, 1]
    end
    pp[1, :, :] .= -inv_sqrt3
    for i in 2:nx, j in 2:nx
        pp[2, i, j] = pp[2, i, 1]
        pp[3, i, j] = pp[3, 1, j]
    end

    # Convert back to lat/lon
    for j in 1:nx, i in 1:nx
        lo, la = _cart2ll(pp[1, i, j], pp[2, i, j], pp[3, i, j])
        lambda_rad[i, j] = lo
        theta_rad[i, j]  = la
    end

    # Symmetrize along i
    for j in 2:nx, i in 2:nx
        lambda_rad[i, j] = lambda_rad[i, 1]
    end
    for j in 1:nx
        for i in 1:(Nc ÷ 2)
            isymm = Nc + 2 - i
            avg = 0.5 * (lambda_rad[i, j] - lambda_rad[isymm, j])
            lambda_rad[i, j]     = avg + π
            lambda_rad[isymm, j] = π - avg
            avg = 0.5 * (theta_rad[i, j] + theta_rad[isymm, j])
            theta_rad[i, j]     = avg
            theta_rad[isymm, j] = avg
        end
    end

    # Symmetrize along j
    for j in 1:(Nc ÷ 2)
        jsymm = Nc + 2 - j
        for i in 2:nx
            avg = 0.5 * (lambda_rad[i, j] + lambda_rad[i, jsymm])
            lambda_rad[i, j]     = avg
            lambda_rad[i, jsymm] = avg
            avg = 0.5 * (theta_rad[i, j] - theta_rad[i, jsymm])
            theta_rad[i, j]     = avg
            theta_rad[i, jsymm] = -avg
        end
    end

    lambda_rad .-= π

    # Step 2: Mirror to all 6 faces
    xgrid = copy(lambda_rad)
    ygrid = copy(theta_rad)
    new_xgrid = zeros(nx, nx, 6)
    new_ygrid = zeros(nx, nx, 6)
    new_xgrid[:, :, 1] .= xgrid
    new_ygrid[:, :, 1] .= ygrid

    for face in 2:6, j in 1:nx, i in 1:nx
        x, y, z = xgrid[i, j], ygrid[i, j], 1.0
        if face == 2
            nx_, ny_, nz_ = _rotate_sphere_3d(x, y, z, -π/2, 'z')
        elseif face == 3
            tx, ty, tz = _rotate_sphere_3d(x, y, z, -π/2, 'z')
            nx_, ny_, nz_ = _rotate_sphere_3d(tx, ty, tz, π/2, 'x')
        elseif face == 4
            tx, ty, tz = _rotate_sphere_3d(x, y, z, π, 'z')
            nx_, ny_, nz_ = _rotate_sphere_3d(tx, ty, tz, π/2, 'x')
        elseif face == 5
            tx, ty, tz = _rotate_sphere_3d(x, y, z, π/2, 'z')
            nx_, ny_, nz_ = _rotate_sphere_3d(tx, ty, tz, π/2, 'y')
        else  # face == 6
            tx, ty, tz = _rotate_sphere_3d(x, y, z, π/2, 'y')
            nx_, ny_, nz_ = _rotate_sphere_3d(tx, ty, tz, 0.0, 'z')
        end
        new_xgrid[i, j, face] = nx_
        new_ygrid[i, j, face] = ny_
    end

    # Cleanup: wrap longitude to [0, 2π), snap small values
    for f in 1:6, j in 1:nx, i in 1:nx
        lo = new_xgrid[i, j, f]
        lo < 0 && (lo += 2π)
        abs(lo) < 1e-10 && (lo = 0.0)
        new_xgrid[i, j, f] = lo
        abs(new_ygrid[i, j, f]) < 1e-10 && (new_ygrid[i, j, f] = 0.0)
    end

    lon_edge_deg = rad2deg.(new_xgrid)
    lat_edge_deg = rad2deg.(new_ygrid)

    # Step 3: Compute cell centers as centroid of 4 corners on unit sphere
    lon_ctr_deg = zeros(Nc, Nc, 6)
    lat_ctr_deg = zeros(Nc, Nc, 6)
    for f in 1:6, j in 1:Nc, i in 1:Nc
        xyz_corners = [
            collect(_ll2cart(new_xgrid[i, j, f],     new_ygrid[i, j, f])),
            collect(_ll2cart(new_xgrid[i+1, j, f],   new_ygrid[i+1, j, f])),
            collect(_ll2cart(new_xgrid[i+1, j+1, f], new_ygrid[i+1, j+1, f])),
            collect(_ll2cart(new_xgrid[i, j+1, f],   new_ygrid[i, j+1, f]))
        ]
        e_mid = sum(xyz_corners)
        e_abs = sqrt(sum(e_mid .^ 2))
        e_abs > 0 && (e_mid ./= e_abs)
        lo, la = _cart2ll(e_mid[1], e_mid[2], e_mid[3])
        lon_ctr_deg[i, j, f] = rad2deg(lo)
        lat_ctr_deg[i, j, f] = rad2deg(la)
    end

    # Apply GMAO offset (-10°) and wrap negative lons to [0, 360)
    offset = -10.0
    lon_edge_deg .+= offset
    lon_ctr_deg  .+= offset
    @. lon_edge_deg = ifelse(lon_edge_deg < 0, lon_edge_deg + 360, lon_edge_deg)
    @. lon_ctr_deg  = ifelse(lon_ctr_deg  < 0, lon_ctr_deg  + 360, lon_ctr_deg)

    # Step 4: Apply GMAO face orientation (transpose + flips)
    # Following gcpy csgrid_gmao: transpose faces [0,1,3,4] (0-indexed),
    # flip axis 1 for faces [3,4], flip axis 0 for faces [3,4,2,5],
    # then swap faces 2↔5 (north/south pole).
    _apply_gmao_orientation!(lon_edge_deg, lat_edge_deg)
    _apply_gmao_orientation!(lon_ctr_deg, lat_ctr_deg)

    # Step 4b: Convert from Python (Y,X) to Julia/NetCDF (X,Y) convention.
    # Python's csgrid_gmao produces arrays with axis 0=Y, axis 1=X.
    # NCDatasets reads the GC file as (Xdim, Ydim, nf), i.e. dim 1=X, dim 2=Y.
    # We must permute each face to match.
    for f in 1:6
        lon_edge_deg[:, :, f] .= permutedims(lon_edge_deg[:, :, f])
        lat_edge_deg[:, :, f] .= permutedims(lat_edge_deg[:, :, f])
        lon_ctr_deg[:, :, f]  .= permutedims(lon_ctr_deg[:, :, f])
        lat_ctr_deg[:, :, f]  .= permutedims(lat_ctr_deg[:, :, f])
    end

    # Step 5: Pack into per-panel tuples
    corner_lons = ntuple(p -> copy(lon_edge_deg[:, :, p]), 6)
    corner_lats = ntuple(p -> copy(lat_edge_deg[:, :, p]), 6)
    lons = ntuple(p -> copy(lon_ctr_deg[:, :, p]), 6)
    lats = ntuple(p -> copy(lat_ctr_deg[:, :, p]), 6)

    # Step 6: Compute areas from corners
    areas = ntuple(6) do p
        panel_areas = zeros(Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            panel_areas[i, j] = _spherical_quad_area_lonlat(
                corner_lons[p][i, j],     corner_lats[p][i, j],
                corner_lons[p][i+1, j],   corner_lats[p][i+1, j],
                corner_lons[p][i+1, j+1], corner_lats[p][i+1, j+1],
                corner_lons[p][i, j+1],   corner_lats[p][i, j+1],
                R)
        end
        panel_areas
    end

    return (; corner_lons, corner_lats, lons, lats, areas)
end

"""Apply GMAO face orientation: transpose faces 1,2,4,5; flip faces 4,5,3,6.

Matches gcpy `csgrid_gmao`: Python 0-indexed tiles [0,1,3,4] → Julia [1,2,4,5].
Operations on each 2D face `arr[:,:,tile]`:
  1. Transpose faces 1,2,4,5
  2. Flip dim 2 (columns) for faces 4,5  — Python `np.flip(a, 1)`
  3. Flip dim 1 (rows) for faces 4,5,3,6 — Python `np.flip(a, 0)`
  4. Swap faces 3 ↔ 6 (north/south pole)
Each step uses a temporary copy to avoid read-after-write issues.
"""
function _apply_gmao_orientation!(lon_arr, lat_arr)
    for arr in (lon_arr, lat_arr)
        # 1. Transpose
        for tile in (1, 2, 4, 5)
            arr[:, :, tile] .= permutedims(arr[:, :, tile])
        end
        # 2. Flip columns (dim 2) for faces 4,5
        for tile in (4, 5)
            tmp = copy(arr[:, :, tile])
            arr[:, :, tile] .= tmp[:, end:-1:1]
        end
        # 3. Flip rows (dim 1) for faces 4,5,3,6
        for tile in (4, 5, 3, 6)
            tmp = copy(arr[:, :, tile])
            arr[:, :, tile] .= tmp[end:-1:1, :]
        end
        # 4. Swap north & south pole faces (3 ↔ 6)
        tmp = copy(arr[:, :, 3])
        arr[:, :, 3] .= arr[:, :, 6]
        arr[:, :, 6] .= tmp
    end
end

# Helpers for grid generation
function _ll2cart(lon, lat)
    (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
end

function _cart2ll(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    x, y, z = x/r, y/r, z/r
    lon = (abs(x) + abs(y)) < 1e-20 ? 0.0 : atan(y, x)
    lon < 0 && (lon += 2π)
    lat = asin(clamp(z, -1.0, 1.0))
    return (lon, lat)
end

function _cross3(a, b)
    [a[2]*b[3] - a[3]*b[2],
     a[3]*b[1] - a[1]*b[3],
     a[1]*b[2] - a[2]*b[1]]
end

function _spherical_to_cart(θ, φ, r)
    (r * cos(φ) * cos(θ), r * cos(φ) * sin(θ), r * sin(φ))
end

function _cart_to_spherical(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    θ = atan(y, x)
    φ = atan(z, sqrt(x^2 + y^2))
    return (θ, φ, r)
end

function _rotate_sphere_3d(θ, φ, r, rot_ang, axis)
    ca, sa = cos(rot_ang), sin(rot_ang)
    x, y, z = _spherical_to_cart(θ, φ, r)
    if axis == 'x'
        xn, yn, zn = x, ca*y + sa*z, -sa*y + ca*z
    elseif axis == 'y'
        xn, yn, zn = ca*x - sa*z, y, sa*x + ca*z
    else  # 'z'
        xn, yn, zn = ca*x + sa*y, -sa*x + ca*y, z
    end
    return _cart_to_spherical(xn, yn, zn)
end
