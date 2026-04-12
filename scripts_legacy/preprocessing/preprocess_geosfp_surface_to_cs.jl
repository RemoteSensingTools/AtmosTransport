#!/usr/bin/env julia
# ===========================================================================
# Regrid GEOS-FP 0.25° lat-lon surface fields → C720 cubed-sphere NetCDF
#
# Reads 0.25° × 0.3125° A1 (PBLH, USTAR, HFLUX, T2M) and A3mstE (CMFMC)
# files, bilinearly interpolates to C720 panel coordinates, and writes
# per-day NetCDF files compatible with read_geosfp_cs_surface_fields() and
# read_geosfp_cs_cmfmc().
#
# Output format:
#   GEOSFP_CS720.YYYYMMDD.A1.nc     — (Xdim=720, Ydim=720, nf=6, time=24)
#   GEOSFP_CS720.YYYYMMDD.A3mstE.nc — (Xdim=720, Ydim=720, nf=6, lev=73, time=8)
#
# Usage (sequential):
#   julia --project=. scripts/preprocess_geosfp_surface_to_cs.jl <config.toml>
#
# Parallel (4 workers):
#   julia --project=. -p 4 scripts/preprocess_geosfp_surface_to_cs.jl <config.toml>
#
# TOML config:
#   [input]      data_dir, start_date, end_date
#   [output]     directory
#   [grid]       coord_file, Nc (default 720)
# ===========================================================================

using Distributed

@everywhere begin
    using AtmosTransport
    using AtmosTransport.IO: read_geosfp_cs_grid_info
    using NCDatasets
    using Dates
    using Printf
    using TOML
end

length(ARGS) >= 1 || error(
    "Usage: julia --project=. [-p N] scripts/preprocess_geosfp_surface_to_cs.jl <config.toml>")

const cfg = TOML.parsefile(ARGS[1])

# --- Config ---
const INPUT_DIR  = expanduser(cfg["input"]["data_dir"])
const START_DATE = Date(cfg["input"]["start_date"])
const END_DATE   = Date(cfg["input"]["end_date"])
const OUTDIR     = expanduser(cfg["output"]["directory"])
const COORD_FILE = expanduser(cfg["grid"]["coord_file"])
const Nc_GRID    = get(cfg["grid"], "Nc", 720)

# ===========================================================================
# Bilinear interpolation weights (precomputed per CS cell)
# ===========================================================================

@everywhere begin

# --- Surface field variable names ---
const A1_VARS    = ["PBLH", "USTAR", "HFLUX", "T2M"]
const A3_VARS    = ["CMFMC"]

"""
Precompute bilinear interpolation weights for mapping a regular lat-lon grid
to cubed-sphere panel cell centers.

Returns `(i0, j0, wx, wy)` — each an NTuple{6} of (Nc, Nc) Int32/Float32 arrays.
For each CS cell (i, j, panel), the interpolated value is:

    (1-wx)*(1-wy)*src[i0,j0] + wx*(1-wy)*src[i0+1,j0] +
    (1-wx)*wy*src[i0,j0+1] + wx*wy*src[i0+1,j0+1]
"""
function precompute_bilinear_weights(cs_lons, cs_lats, src_lons, src_lats, Nc)
    Nlon = length(src_lons)
    Nlat = length(src_lats)
    dlon = Float64(src_lons[2] - src_lons[1])
    dlat = Float64(src_lats[2] - src_lats[1])
    lon0 = Float64(src_lons[1])
    lat0 = Float64(src_lats[1])

    i0_panels = ntuple(_ -> zeros(Int32, Nc, Nc), 6)
    j0_panels = ntuple(_ -> zeros(Int32, Nc, Nc), 6)
    wx_panels = ntuple(_ -> zeros(Float32, Nc, Nc), 6)
    wy_panels = ntuple(_ -> zeros(Float32, Nc, Nc), 6)

    for p in 1:6
        for jj in 1:Nc, ii in 1:Nc
            # cs_lons/cs_lats from read_geosfp_cs_grid_info: (Xdim, Ydim, nf)
            lon = Float64(cs_lons[ii, jj, p])
            lat = Float64(cs_lats[ii, jj, p])

            # Wrap longitude to source grid range
            while lon < lon0;        lon += 360.0; end
            while lon >= lon0 + 360; lon -= 360.0; end

            # Fractional index in source grid
            fi = (lon - lon0) / dlon + 1.0
            fj = (lat - lat0) / dlat + 1.0

            i_lo = clamp(floor(Int32, fi), Int32(1), Int32(Nlon - 1))
            j_lo = clamp(floor(Int32, fj), Int32(1), Int32(Nlat - 1))

            i0_panels[p][ii, jj] = i_lo
            j0_panels[p][ii, jj] = j_lo
            wx_panels[p][ii, jj] = Float32(clamp(fi - i_lo, 0.0, 1.0))
            wy_panels[p][ii, jj] = Float32(clamp(fj - j_lo, 0.0, 1.0))
        end
    end

    return i0_panels, j0_panels, wx_panels, wy_panels
end

"""
Apply precomputed bilinear weights to interpolate a 2D field from lat-lon to
one CS panel.
"""
function apply_bilinear_2d!(out::Matrix{Float32}, src::Matrix,
                             i0::Matrix{Int32}, j0::Matrix{Int32},
                             wx::Matrix{Float32}, wy::Matrix{Float32})
    Nc = size(out, 1)
    @inbounds for jj in 1:Nc, ii in 1:Nc
        il = i0[ii, jj];  jl = j0[ii, jj]
        w_x = wx[ii, jj]; w_y = wy[ii, jj]
        out[ii, jj] = Float32(
            (1 - w_x) * (1 - w_y) * src[il,     jl]     +
                 w_x  * (1 - w_y) * src[il + 1, jl]     +
            (1 - w_x) *      w_y  * src[il,     jl + 1] +
                 w_x  *      w_y  * src[il + 1, jl + 1])
    end
end

"""
Apply bilinear weights to a 3D field (lon, lat, lev) → one CS panel (Nc, Nc, Nz).
"""
function apply_bilinear_3d!(out::Array{Float32, 3}, src::Array,
                             i0::Matrix{Int32}, j0::Matrix{Int32},
                             wx::Matrix{Float32}, wy::Matrix{Float32})
    Nc = size(out, 1)
    Nz = size(out, 3)
    @inbounds for k in 1:Nz, jj in 1:Nc, ii in 1:Nc
        il = i0[ii, jj];  jl = j0[ii, jj]
        w_x = wx[ii, jj]; w_y = wy[ii, jj]
        out[ii, jj, k] = Float32(
            (1 - w_x) * (1 - w_y) * src[il,     jl,     k] +
                 w_x  * (1 - w_y) * src[il + 1, jl,     k] +
            (1 - w_x) *      w_y  * src[il,     jl + 1, k] +
                 w_x  *      w_y  * src[il + 1, jl + 1, k])
    end
end

# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

function process_a1_day(inpath::String, outpath::String,
                        Nc::Int, weights)
    i0_p, j0_p, wx_p, wy_p = weights

    NCDataset(inpath, "r") do ds_in
        Nt = length(ds_in["time"])
        @info "  [A1] $Nt timesteps, regridding to C$(Nc)..."

        NCDataset(outpath, "c") do ds_out
            defDim(ds_out, "Xdim", Nc)
            defDim(ds_out, "Ydim", Nc)
            defDim(ds_out, "nf", 6)
            defDim(ds_out, "time", Nt)

            for varname in A1_VARS
                defVar(ds_out, varname, Float32, ("Xdim", "Ydim", "nf", "time"))
            end

            # Pre-allocate output panel buffer
            panel_buf = zeros(Float32, Nc, Nc)

            for varname in A1_VARS
                haskey(ds_in, varname) || continue
                for t in 1:Nt
                    src_2d = Array{Float64}(ds_in[varname][:, :, t])  # (lon, lat)
                    for p in 1:6
                        apply_bilinear_2d!(panel_buf, src_2d,
                                           i0_p[p], j0_p[p], wx_p[p], wy_p[p])
                        ds_out[varname][:, :, p, t] = panel_buf
                    end
                end
                @info "    $varname done"
            end
        end
    end
end

function process_a3mste_day(inpath::String, outpath::String,
                             Nc::Int, weights)
    i0_p, j0_p, wx_p, wy_p = weights

    NCDataset(inpath, "r") do ds_in
        Nt = length(ds_in["time"])
        # CMFMC dims in source: (lon, lat, lev, time)
        Nz_edge = size(ds_in["CMFMC"], 3)
        @info "  [A3mstE] $Nt timesteps, $Nz_edge levels, regridding to C$(Nc)..."

        NCDataset(outpath, "c") do ds_out
            defDim(ds_out, "Xdim", Nc)
            defDim(ds_out, "Ydim", Nc)
            defDim(ds_out, "nf", 6)
            defDim(ds_out, "lev", Nz_edge)
            defDim(ds_out, "time", Nt)

            defVar(ds_out, "CMFMC", Float32, ("Xdim", "Ydim", "nf", "lev", "time"))

            panel_buf = zeros(Float32, Nc, Nc, Nz_edge)

            for t in 1:Nt
                src_3d = Array{Float64}(ds_in["CMFMC"][:, :, :, t])  # (lon, lat, lev)
                for p in 1:6
                    apply_bilinear_3d!(panel_buf, src_3d,
                                       i0_p[p], j0_p[p], wx_p[p], wy_p[p])
                    ds_out["CMFMC"][:, :, p, :, t] = panel_buf
                end
            end
            @info "    CMFMC done"
        end
    end
end

"""Process one day: regrid A1 + A3mstE."""
function process_one_day(date::Date, pcfg)
    datestr = Dates.format(date, "yyyymmdd")

    a1_in  = joinpath(pcfg.input_dir, "GEOSFP.$(datestr).A1.025x03125.nc")
    a3_in  = joinpath(pcfg.input_dir, "GEOSFP.$(datestr).A3mstE.025x03125.nc")
    a1_out = joinpath(pcfg.outdir, "GEOSFP_CS$(pcfg.Nc).$(datestr).A1.nc")
    a3_out = joinpath(pcfg.outdir, "GEOSFP_CS$(pcfg.Nc).$(datestr).A3mstE.nc")

    results = Symbol[]

    # A1 surface fields
    if isfile(a1_out) && filesize(a1_out) > 1000
        @info "[$datestr] A1 already exists — skipping"
        push!(results, :skipped)
    elseif !isfile(a1_in)
        @warn "[$datestr] A1 input not found: $a1_in"
        push!(results, :missing)
    else
        @info "[$datestr] Processing A1 on worker $(myid())..."
        process_a1_day(a1_in, a1_out, pcfg.Nc, pcfg.weights)
        push!(results, :done)
    end

    # A3mstE convective mass flux
    if isfile(a3_out) && filesize(a3_out) > 1000
        @info "[$datestr] A3mstE already exists — skipping"
        push!(results, :skipped)
    elseif !isfile(a3_in)
        @warn "[$datestr] A3mstE input not found: $a3_in"
        push!(results, :missing)
    else
        @info "[$datestr] Processing A3mstE on worker $(myid())..."
        process_a3mste_day(a3_in, a3_out, pcfg.Nc, pcfg.weights)
        push!(results, :done)
    end

    return (date, results)
end

end # @everywhere

# ===========================================================================
# Main
# ===========================================================================

function main()
    @info "=" ^ 70
    @info "GEOS-FP Surface Field Regridding (0.25° lat-lon → cubed-sphere)"
    @info "=" ^ 70
    @info "  Input:      $INPUT_DIR"
    @info "  Output:     $OUTDIR"
    @info "  Dates:      $START_DATE to $END_DATE"
    @info "  Coord file: $COORD_FILE"
    @info "  Grid:       C$(Nc_GRID)"
    @info "  Workers:    $(nworkers())"

    mkpath(OUTDIR)

    # Read CS grid coordinates from a GEOS-FP C720 file
    @info "Reading cubed-sphere coordinates from $(basename(COORD_FILE))..."
    cs_lons, cs_lats, _, _ = read_geosfp_cs_grid_info(COORD_FILE)
    @info "  Grid shape: $(size(cs_lons))"

    # Read source grid coordinates from first available A1 file
    a1_sample = ""
    for date in START_DATE:Day(1):END_DATE
        datestr = Dates.format(date, "yyyymmdd")
        f = joinpath(INPUT_DIR, "GEOSFP.$(datestr).A1.025x03125.nc")
        if isfile(f)
            a1_sample = f
            break
        end
    end
    isempty(a1_sample) && error("No A1 input files found in $INPUT_DIR")

    src_lons, src_lats = NCDataset(a1_sample, "r") do ds
        Array{Float64}(ds["lon"][:]), Array{Float64}(ds["lat"][:])
    end
    @info "  Source grid: $(length(src_lons)) × $(length(src_lats)) " *
          "($(src_lons[1])..$(src_lons[end])° lon, $(src_lats[1])..$(src_lats[end])° lat)"

    # Precompute bilinear weights (once, reused for all days)
    @info "Precomputing bilinear interpolation weights..."
    t0 = time()
    weights = precompute_bilinear_weights(cs_lons, cs_lats, src_lons, src_lats, Nc_GRID)
    @info @sprintf("  Done in %.1fs", time() - t0)

    # Bundle config for workers
    pcfg = (
        input_dir = INPUT_DIR,
        outdir    = OUTDIR,
        Nc        = Nc_GRID,
        weights   = weights,
    )

    dates = collect(START_DATE:Day(1):END_DATE)
    wall_start = time()

    results = pmap(d -> process_one_day(d, pcfg), dates)

    wall_total = round(time() - wall_start, digits=1)
    n_done = count(r -> any(s -> s === :done, r[2]), results)
    @info "\n" * "=" ^ 70
    @info "Regridding complete!"
    @info "  Wall time: $(wall_total)s"
    @info "  Days processed: $n_done"
    @info "  Workers: $(nworkers())"
    @info "  Output: $OUTDIR"
    @info "=" ^ 70
end

main()
