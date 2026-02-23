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
struct RegridMapping{AT_I, AT_F}
    "per-panel longitude bin indices (1-based, Nc×Nc)"
    ii_panels :: NTuple{6, AT_I}
    "per-panel latitude bin indices (1-based, Nc×Nc)"
    jj_panels :: NTuple{6, AT_I}
    "normalization weights (1/count, Nlon×Nlat)"
    norm_inv  :: AT_F
    "working accumulator buffer (Nlon×Nlat)"
    scratch   :: AT_F
    "output longitude count"
    Nlon :: Int
    "output latitude count"
    Nlat :: Int
end

"""
$(SIGNATURES)

Build a `RegridMapping` for the given cubed-sphere grid to a `Nlon × Nlat`
lat-lon output grid.

Index arrays and normalization weights are computed on CPU (using CPU grid
coordinates) then uploaded to the target device via `AT` (e.g. `CuArray` for
GPU, `Array` for CPU).
"""
function build_regrid_mapping(grid::CubedSphereGrid{FT}, AT, Nlon::Int, Nlat::Int) where FT
    Nc    = grid.Nc
    dlon  = FT(360) / Nlon
    dlat  = FT(180) / (Nlat - 1)
    lon0  = FT(-180)
    lat0  = FT(-90)

    ii_cpu = [zeros(Int32, Nc, Nc) for _ in 1:6]
    jj_cpu = [zeros(Int32, Nc, Nc) for _ in 1:6]
    count  = zeros(Int32, Nlon, Nlat)

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            lon = mod(grid.λᶜ[p][i, j] + FT(180), FT(360)) - FT(180)
            lat = grid.φᶜ[p][i, j]
            ii  = clamp(round(Int32, (lon - lon0) / dlon) + Int32(1), Int32(1), Int32(Nlon))
            jj  = clamp(round(Int32, (lat - lat0) / dlat) + Int32(1), Int32(1), Int32(Nlat))
            ii_cpu[p][i, j] = ii
            jj_cpu[p][i, j] = jj
            count[ii, jj]  += Int32(1)
        end
    end

    norm_inv = zeros(FT, Nlon, Nlat)
    for idx in eachindex(norm_inv)
        norm_inv[idx] = count[idx] > Int32(0) ? FT(1) / FT(count[idx]) : FT(0)
    end

    # Upload to device
    ii_panels = ntuple(p -> AT(ii_cpu[p]), 6)
    jj_panels = ntuple(p -> AT(jj_cpu[p]), 6)
    norm_dev  = AT(norm_inv)
    scratch   = AT(zeros(FT, Nlon, Nlat))

    @info "RegridMapping: C$Nc → $(Nlon)×$(Nlat) lat-lon built"
    return RegridMapping(ii_panels, jj_panels, norm_dev, scratch, Nlon, Nlat)
end

# ---------------------------------------------------------------------------
# GPU/CPU scatter kernel
# ---------------------------------------------------------------------------

@kernel function _regrid_scatter_panel!(out, @Const(panel_data),
                                         @Const(ii_map), @Const(jj_map))
    i, j = @index(Global, NTuple)
    @inbounds begin
        val = panel_data[i, j]
        ii  = ii_map[i, j]
        jj  = jj_map[i, j]
        @atomic out[ii, jj] += val
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
                              Nlon::Int=720, Nlat::Int=361) where FT
    lons_out = range(FT(-180), FT(180) - FT(360) / Nlon, length=Nlon)
    lats_out = range(FT(-90), FT(90), length=Nlat)
    Nc = grid.Nc
    out   = zeros(FT, Nlon, Nlat)
    count = zeros(Int, Nlon, Nlat)

    dlon = lons_out[2] - lons_out[1]
    dlat = lats_out[2] - lats_out[1]

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            lon = mod(grid.λᶜ[p][i, j] + FT(180), FT(360)) - FT(180)
            lat = grid.φᶜ[p][i, j]
            ii = clamp(round(Int, (lon - lons_out[1]) / dlon) + 1, 1, Nlon)
            jj = clamp(round(Int, (lat - lats_out[1]) / dlat) + 1, 1, Nlat)
            out[ii, jj] += panels_cpu[p][i, j]
            count[ii, jj] += 1
        end
    end

    for idx in eachindex(out)
        if count[idx] > 0
            out[idx] /= count[idx]
        end
    end
    return out, collect(lons_out), collect(lats_out)
end
