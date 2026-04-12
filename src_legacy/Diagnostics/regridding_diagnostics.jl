# ---------------------------------------------------------------------------
# Cubed-sphere → lat-lon regridding for diagnostic output
# ---------------------------------------------------------------------------

"""
$(TYPEDEF)

Precomputed mapping from cubed-sphere panels to a regular lat-lon output grid.
Built once via `build_regrid_mapping`; reused each write call to avoid
redundant coordinate computation.

$(FIELDS)
"""
struct RegridMapping{AT_I, AT_F, FT}
    "per-panel longitude bin indices (1-based, Nc×Nc; 0 = outside region)"
    ii_panels :: NTuple{6, AT_I}
    "per-panel latitude bin indices (1-based, Nc×Nc; 0 = outside region)"
    jj_panels :: NTuple{6, AT_I}
    "normalization weights (1/count, Nlon×Nlat)"
    norm_inv  :: AT_F
    "working accumulator buffer (Nlon×Nlat)"
    scratch   :: AT_F
    "boolean mask: true where count == 0 (gap cells needing fill)"
    gap_mask  :: AT_I
    "output longitude count"
    Nlon :: Int
    "output latitude count"
    Nlat :: Int
    "western boundary [degrees]"
    lon0 :: FT
    "eastern boundary [degrees]"
    lon1 :: FT
    "southern boundary [degrees]"
    lat0 :: FT
    "northern boundary [degrees]"
    lat1 :: FT
    "number of gap cells"
    n_gaps :: Int
end

"""
$(SIGNATURES)

Build a `RegridMapping` for the given cubed-sphere grid to a `Nlon × Nlat`
lat-lon output grid.

Index arrays and normalization weights are computed on CPU (using CPU grid
coordinates) then uploaded to the target device via `AT` (e.g. `CuArray` for
GPU, `Array` for CPU).
"""
function build_regrid_mapping(grid::CubedSphereGrid{FT}, AT, Nlon::Int, Nlat::Int;
                               lon0::FT=FT(-180), lon1::FT=FT(180),
                               lat0::FT=FT(-90),  lat1::FT=FT(90),
                               file_lons=nothing, file_lats=nothing) where FT
    Nc    = grid.Nc
    dlon  = (lon1 - lon0) / Nlon
    dlat  = (lat1 - lat0) / Nlat

    ii_cpu = [zeros(Int32, Nc, Nc) for _ in 1:6]
    jj_cpu = [zeros(Int32, Nc, Nc) for _ in 1:6]
    count  = zeros(Int32, Nlon, Nlat)

    use_file = file_lons !== nothing && file_lats !== nothing

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            lon = if use_file
                mod(FT(file_lons[i, j, p]) + FT(180), FT(360)) - FT(180)
            else
                mod(grid.λᶜ[p][i, j] + FT(180), FT(360)) - FT(180)
            end
            lat = use_file ? FT(file_lats[i, j, p]) : grid.φᶜ[p][i, j]
            # Skip cells outside the bounding box (sentinel index 0)
            if lon < lon0 || lon >= lon1 || lat < lat0 || lat > lat1
                ii_cpu[p][i, j] = Int32(0)
                jj_cpu[p][i, j] = Int32(0)
                continue
            end
            ii  = clamp(floor(Int32, (lon - lon0) / dlon) + Int32(1), Int32(1), Int32(Nlon))
            jj  = clamp(floor(Int32, (lat - lat0) / dlat) + Int32(1), Int32(1), Int32(Nlat))
            ii_cpu[p][i, j] = ii
            jj_cpu[p][i, j] = jj
            count[ii, jj]  += Int32(1)
        end
    end

    norm_inv = zeros(FT, Nlon, Nlat)
    for idx in eachindex(norm_inv)
        norm_inv[idx] = count[idx] > Int32(0) ? FT(1) / FT(count[idx]) : FT(0)
    end

    # Build gap mask: Int32(1) where count == 0 (needs neighbor fill)
    gap_cpu = Int32.(count .== 0)
    n_gaps  = sum(gap_cpu)

    # Upload to device
    ii_panels = ntuple(p -> AT(ii_cpu[p]), 6)
    jj_panels = ntuple(p -> AT(jj_cpu[p]), 6)
    norm_dev  = AT(norm_inv)
    scratch   = AT(zeros(FT, Nlon, Nlat))
    gap_dev   = AT(gap_cpu)

    @info "RegridMapping: C$Nc → $(Nlon)×$(Nlat) lat-lon [$(lon0)°..$(lon1)°, $(lat0)°..$(lat1)°] built ($n_gaps gap cells)"
    return RegridMapping(ii_panels, jj_panels, norm_dev, scratch, gap_dev,
                          Nlon, Nlat, lon0, lon1, lat0, lat1, n_gaps)
end

# ---------------------------------------------------------------------------
# GPU/CPU scatter kernel
# ---------------------------------------------------------------------------

@kernel function _regrid_scatter_panel!(out, @Const(panel_data),
                                         @Const(ii_map), @Const(jj_map))
    i, j = @index(Global, NTuple)
    @inbounds begin
        ii = ii_map[i, j]
        if ii != Int32(0)  # skip cells outside bounding box
            jj  = jj_map[i, j]
            val = panel_data[i, j]
            @atomic out[ii, jj] += val
        end
    end
end

@kernel function _regrid_gap_fill!(out, @Const(gap_mask), Nlon, Nlat)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if gap_mask[i, j] == Int32(1)
            # Average valid neighbors (4-connected)
            s   = zero(eltype(out))
            cnt = Int32(0)
            if i > 1    && gap_mask[i-1, j] == Int32(0); s += out[i-1, j]; cnt += Int32(1); end
            if i < Nlon && gap_mask[i+1, j] == Int32(0); s += out[i+1, j]; cnt += Int32(1); end
            if j > 1    && gap_mask[i, j-1] == Int32(0); s += out[i, j-1]; cnt += Int32(1); end
            if j < Nlat && gap_mask[i, j+1] == Int32(0); s += out[i, j+1]; cnt += Int32(1); end
            if cnt > Int32(0)
                out[i, j] = s / cnt
            end
        end
    end
end

"""
$(SIGNATURES)

Regrid cubed-sphere column-mean panels (6 × Nc × Nc, no halo) to a lat-lon
grid using a precomputed `RegridMapping`.

Operates on the device where `mapping` and `c_col_panels` reside (GPU or CPU).
Returns `mapping.scratch` (device array, Nlon × Nlat) — caller must `Array()`
it for CPU/NetCDF use.
"""
function regrid_cs_to_latlon!(mapping::RegridMapping,
                               c_col_panels::NTuple{6})
    Nc      = size(mapping.ii_panels[1], 1)
    backend = get_backend(mapping.scratch)

    # Reset accumulator
    fill!(mapping.scratch, zero(eltype(mapping.scratch)))

    # Scatter: each CS cell atomically adds its value to the lat-lon bin
    k! = _regrid_scatter_panel!(backend, 256)
    for p in 1:6
        k!(mapping.scratch, c_col_panels[p],
           mapping.ii_panels[p], mapping.jj_panels[p]; ndrange=(Nc, Nc))
    end
    synchronize(backend)

    # Normalize by bin count
    mapping.scratch .*= mapping.norm_inv

    # Fill gap cells from valid neighbors
    if mapping.n_gaps > 0
        gf! = _regrid_gap_fill!(backend, 256)
        gf!(mapping.scratch, mapping.gap_mask,
            Int32(mapping.Nlon), Int32(mapping.Nlat);
            ndrange=(mapping.Nlon, mapping.Nlat))
        synchronize(backend)
    end

    return mapping.scratch
end

# ---------------------------------------------------------------------------
# CPU fallback (keeps backward compatibility for non-GPU paths)
# ---------------------------------------------------------------------------

"""
    regrid_cs_to_latlon(panels_cpu, grid; Nlon=720, Nlat=361)

Regrid cubed-sphere panel data (6 × Nc × Nc) to a regular lat-lon grid
via nearest-neighbor binning. Returns `(data_ll, lons, lats)`.

`panels_cpu` must be an NTuple{6, Matrix{FT}} of CPU arrays.
`grid` must be a `CubedSphereGrid` (used to look up panel center coordinates).
"""
function regrid_cs_to_latlon(panels_cpu::NTuple{6, Matrix{FT}},
                              grid::CubedSphereGrid{FT};
                              Nlon::Int=720, Nlat::Int=361,
                              lon0::FT=FT(-180), lon1::FT=FT(180),
                              lat0::FT=FT(-90),  lat1::FT=FT(90),
                              file_lons=nothing, file_lats=nothing) where FT
    dlon = (lon1 - lon0) / Nlon
    dlat = (lat1 - lat0) / Nlat
    lons_out = range(lon0 + dlon/2, lon1 - dlon/2, length=Nlon)
    lats_out = range(lat0 + dlat/2, lat1 - dlat/2, length=Nlat)
    Nc = grid.Nc
    out   = zeros(FT, Nlon, Nlat)
    count = zeros(Int, Nlon, Nlat)
    use_file = file_lons !== nothing && file_lats !== nothing

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            lon = if use_file
                mod(FT(file_lons[i, j, p]) + FT(180), FT(360)) - FT(180)
            else
                mod(grid.λᶜ[p][i, j] + FT(180), FT(360)) - FT(180)
            end
            lat = use_file ? FT(file_lats[i, j, p]) : grid.φᶜ[p][i, j]
            # Skip cells outside the bounding box
            if lon < lon0 || lon >= lon1 || lat < lat0 || lat > lat1
                continue
            end
            ii = clamp(floor(Int, (lon - lon0) / dlon) + 1, 1, Nlon)
            jj = clamp(floor(Int, (lat - lat0) / dlat) + 1, 1, Nlat)
            out[ii, jj] += panels_cpu[p][i, j]
            count[ii, jj] += 1
        end
    end

    for idx in eachindex(out)
        if count[idx] > 0
            out[idx] /= count[idx]
        end
    end

    # Fill gap cells from valid neighbors
    gap = count .== 0
    if any(gap)
        filled = copy(out)
        for j in 1:Nlat, i in 1:Nlon
            if gap[i, j]
                s, cnt = zero(FT), 0
                if i > 1    && !gap[i-1, j]; s += out[i-1, j]; cnt += 1; end
                if i < Nlon && !gap[i+1, j]; s += out[i+1, j]; cnt += 1; end
                if j > 1    && !gap[i, j-1]; s += out[i, j-1]; cnt += 1; end
                if j < Nlat && !gap[i, j+1]; s += out[i, j+1]; cnt += 1; end
                if cnt > 0; filled[i, j] = s / cnt; end
            end
        end
        out = filled
    end

    return out, collect(lons_out), collect(lats_out)
end
