#!/usr/bin/env julia
# ===========================================================================
# Preprocessing: GEOS CS NetCDF → flat binary v4
#
# v4 extends v3 by embedding QV and PS at both window boundaries.
# This eliminates separate CTM_I1/I3 file loading and guarantees temporal
# coherence between mass fluxes and thermodynamic fields.
#
# Binary layout per window:
#   [6 DELP] [6 AM] [6 BM] [6 CX] [6 CY] [6 XFX] [6 YFX]   ← v3
#   [6 QV_start(Nc×Nc×Nz)] [6 QV_end(Nc×Nc×Nz)]              ← v4
#   [6 PS_start(Nc×Nc)]    [6 PS_end(Nc×Nc)]                  ← v4
#
# QV stored without halos, in native vertical order (bottom-to-top for GEOS-IT).
# QV_end(window w) = QV_start(window w+1) — intentional redundancy.
# PS in Pa (native GEOS units).
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_geosfp_cs_v4.jl \
#         --data_dir ~/data/geosit_c180_catrine \
#         --gridspec ~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc \
#         --output_dir /temp1/catrine/met/geosit_c180/massflux_v4 \
#         --start_date 2021-12-01 --end_date 2021-12-08
# ===========================================================================

using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels,
                          cgrid_to_staggered_panels, GEOS_CS_PRODUCTS
using NCDatasets
using Dates
using Printf
using JSON3
using LinearAlgebra: norm, cross, dot

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
function parse_args()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--") && i < length(ARGS)
            args[ARGS[i][3:end]] = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    data_dir     = expanduser(get(args, "data_dir", "~/data/geosit_c180_catrine"))
    output_dir   = expanduser(get(args, "output_dir", "/temp1/catrine/met/geosit_c180/massflux_v4"))
    gridspec     = expanduser(get(args, "gridspec",
                    "~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc"))
    start_date   = Date(get(args, "start_date", "2021-12-01"))
    end_date     = Date(get(args, "end_date", "2021-12-08"))
    mass_flux_dt = parse(Float64, get(args, "mass_flux_dt", "450.0"))
    product      = get(args, "product", "geosit_c180")
    return (; data_dir, output_dir, gridspec, start_date, end_date, mass_flux_dt, product)
end

const HEADER_SIZE = 8192

# ---------------------------------------------------------------------------
# Spherical geometry utilities (port of GCHP fv_grid_utils.F90)
# ---------------------------------------------------------------------------
# These are identical to v3 — included here to keep the script self-contained.

"""Convert (lon, lat) in degrees to Cartesian (x, y, z) on unit sphere."""
function _ll2cart(lon_deg, lat_deg)
    λ = deg2rad(Float64(lon_deg))
    φ = deg2rad(Float64(lat_deg))
    return (cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ))
end

"""Midpoint of two points on the unit sphere (normalized average)."""
function _mid_pt(p1, p2)
    m = (p1[1]+p2[1], p1[2]+p2[2], p1[3]+p2[3])
    n = sqrt(m[1]^2 + m[2]^2 + m[3]^2)
    return (m[1]/n, m[2]/n, m[3]/n)
end

function cos_angle(p1, p2, p3)
    px = p1[2]*p2[3] - p1[3]*p2[2]
    py = p1[3]*p2[1] - p1[1]*p2[3]
    pz = p1[1]*p2[2] - p1[2]*p2[1]
    qx = p1[2]*p3[3] - p1[3]*p3[2]
    qy = p1[3]*p3[1] - p1[1]*p3[3]
    qz = p1[1]*p3[2] - p1[2]*p3[1]
    pp = px*px + py*py + pz*pz
    qq = qx*qx + qy*qy + qz*qz
    denom = sqrt(pp * qq)
    denom < 1e-30 && return 1.0
    return clamp((px*qx + py*qy + pz*qz) / denom, -1.0, 1.0)
end

function compute_sin_sg(corner_lons, corner_lats, center_lons, center_lats, Nc)
    sin_sg_panels = ntuple(6) do p
        sg = zeros(Float64, Nc, Nc, 4)
        c_lon = corner_lons[p]; c_lat = corner_lats[p]
        a_lon = center_lons[p]; a_lat = center_lats[p]
        for j in 1:Nc, i in 1:Nc
            v_sw = _ll2cart(c_lon[i,j],     c_lat[i,j])
            v_se = _ll2cart(c_lon[i+1,j],   c_lat[i+1,j])
            v_ne = _ll2cart(c_lon[i+1,j+1], c_lat[i+1,j+1])
            v_nw = _ll2cart(c_lon[i,j+1],   c_lat[i,j+1])
            a_c  = _ll2cart(a_lon[i,j], a_lat[i,j])
            mid_e = _mid_pt(v_se, v_ne); cos_e = cos_angle(mid_e, a_c, v_ne)
            sg[i,j,1] = min(1.0, sqrt(max(0.0, 1.0 - cos_e^2)))
            mid_n = _mid_pt(v_nw, v_ne); cos_n = cos_angle(mid_n, a_c, v_ne)
            sg[i,j,2] = min(1.0, sqrt(max(0.0, 1.0 - cos_n^2)))
            mid_w = _mid_pt(v_sw, v_nw); cos_w = cos_angle(mid_w, a_c, v_nw)
            sg[i,j,3] = min(1.0, sqrt(max(0.0, 1.0 - cos_w^2)))
            mid_s = _mid_pt(v_sw, v_se); cos_s = cos_angle(mid_s, a_c, v_se)
            sg[i,j,4] = min(1.0, sqrt(max(0.0, 1.0 - cos_s^2)))
        end
        sg
    end
    return sin_sg_panels
end

function _haversine(lon1_deg, lat1_deg, lon2_deg, lat2_deg, R=1.0)
    λ1, φ1 = deg2rad(Float64(lon1_deg)), deg2rad(Float64(lat1_deg))
    λ2, φ2 = deg2rad(Float64(lon2_deg)), deg2rad(Float64(lat2_deg))
    dλ = λ2 - λ1; dφ = φ2 - φ1
    a = sin(dφ/2)^2 + cos(φ1) * cos(φ2) * sin(dλ/2)^2
    return R * 2 * atan(sqrt(a), sqrt(1 - a))
end

function compute_grid_metrics(corner_lons, corner_lats, center_lons, center_lats, Nc, R)
    conn = (
        ((3,2), (6,0), (2,0), (5,2)),
        ((3,0), (6,2), (4,2), (1,0)),
        ((5,2), (2,0), (4,0), (1,2)),
        ((5,0), (2,2), (6,2), (3,0)),
        ((1,2), (4,0), (6,0), (3,2)),
        ((1,0), (4,2), (2,2), (5,0)),
    )
    function _neighbor_center(p, edge, s)
        nb_panel, orient = conn[p][edge]
        s_src = orient >= 2 ? (Nc + 1 - s) : s
        q_e = 0
        for e in 1:4
            if conn[nb_panel][e][1] == p; q_e = e; break; end
        end
        if     q_e == 1; return (center_lons[nb_panel][s_src, Nc], center_lats[nb_panel][s_src, Nc])
        elseif q_e == 2; return (center_lons[nb_panel][s_src, 1],  center_lats[nb_panel][s_src, 1])
        elseif q_e == 3; return (center_lons[nb_panel][Nc, s_src], center_lats[nb_panel][Nc, s_src])
        else;            return (center_lons[nb_panel][1, s_src],  center_lats[nb_panel][1, s_src])
        end
    end
    dxa = ntuple(6) do p
        d = zeros(Float64, Nc, Nc)
        for j in 1:Nc
            for i in 1:Nc-1
                d[i,j] = _haversine(center_lons[p][i,j], center_lats[p][i,j],
                                     center_lons[p][i+1,j], center_lats[p][i+1,j], R)
            end
            nb_lon, nb_lat = _neighbor_center(p, 3, j)
            d[Nc,j] = _haversine(center_lons[p][Nc,j], center_lats[p][Nc,j], nb_lon, nb_lat, R)
        end; d
    end
    dya = ntuple(6) do p
        d = zeros(Float64, Nc, Nc)
        for i in 1:Nc
            for j in 1:Nc-1
                d[i,j] = _haversine(center_lons[p][i,j], center_lats[p][i,j],
                                     center_lons[p][i,j+1], center_lats[p][i,j+1], R)
            end
            nb_lon, nb_lat = _neighbor_center(p, 1, i)
            d[i,Nc] = _haversine(center_lons[p][i,Nc], center_lats[p][i,Nc], nb_lon, nb_lat, R)
        end; d
    end
    dy_face = ntuple(6) do p
        d = zeros(Float64, Nc+1, Nc)
        for j in 1:Nc, iif in 1:(Nc+1)
            d[iif,j] = _haversine(corner_lons[p][iif,j], corner_lats[p][iif,j],
                                   corner_lons[p][iif,j+1], corner_lats[p][iif,j+1], R)
        end; d
    end
    dx_face = ntuple(6) do p
        d = zeros(Float64, Nc, Nc+1)
        for jf in 1:(Nc+1), i in 1:Nc
            d[i,jf] = _haversine(corner_lons[p][i,jf], corner_lats[p][i,jf],
                                   corner_lons[p][i+1,jf], corner_lats[p][i+1,jf], R)
        end; d
    end
    return (; dxa, dya, dy_face, dx_face)
end

function compute_xfx_yfx(cx_stag, cy_stag, dxa, dya, dy_face, dx_face, sin_sg, Nc, Nz;
                          nb_dxa=nothing, nb_sg=nothing)
    xfx = ntuple(6) do p
        x = zeros(Float32, Nc+1, Nc, Nz)
        for k in 1:Nz, j in 1:Nc, iif in 1:(Nc+1)
            c = cx_stag[p][iif,j,k]
            if c > 0
                if iif == 1 && nb_dxa !== nothing
                    x[iif,j,k] = Float32(c * nb_dxa[p].west_dxa[j] * dy_face[p][iif,j] * nb_sg[p].west_sg[j])
                else
                    i_up = max(1, iif-1)
                    x[iif,j,k] = Float32(c * dxa[p][i_up,j] * dy_face[p][iif,j] * sin_sg[p][i_up,j,3])
                end
            else
                if iif == Nc+1 && nb_dxa !== nothing
                    x[iif,j,k] = Float32(c * nb_dxa[p].east_dxa[j] * dy_face[p][iif,j] * nb_sg[p].east_sg[j])
                else
                    i_up = min(Nc, iif)
                    x[iif,j,k] = Float32(c * dxa[p][i_up,j] * dy_face[p][iif,j] * sin_sg[p][i_up,j,1])
                end
            end
        end; x
    end
    yfx = ntuple(6) do p
        y = zeros(Float32, Nc, Nc+1, Nz)
        for k in 1:Nz, jf in 1:(Nc+1), i in 1:Nc
            c = cy_stag[p][i,jf,k]
            if c > 0
                if jf == 1 && nb_dxa !== nothing
                    y[i,jf,k] = Float32(c * nb_dxa[p].south_dya[i] * dx_face[p][i,jf] * nb_sg[p].south_sg[i])
                else
                    j_up = max(1, jf-1)
                    y[i,jf,k] = Float32(c * dya[p][i,j_up] * dx_face[p][i,jf] * sin_sg[p][i,j_up,4])
                end
            else
                if jf == Nc+1 && nb_dxa !== nothing
                    y[i,jf,k] = Float32(c * nb_dxa[p].north_dya[i] * dx_face[p][i,jf] * nb_sg[p].north_sg[i])
                else
                    j_up = min(Nc, jf)
                    y[i,jf,k] = Float32(c * dya[p][i,j_up] * dx_face[p][i,jf] * sin_sg[p][i,j_up,2])
                end
            end
        end; y
    end
    return (xfx, yfx)
end

function compute_boundary_geometry(dxa, dya, sin_sg, Nc)
    conn = (
        ((3,2), (6,0), (2,0), (5,2)),
        ((3,0), (6,2), (4,2), (1,0)),
        ((5,2), (2,0), (4,0), (1,2)),
        ((5,0), (2,2), (6,2), (3,0)),
        ((1,2), (4,0), (6,0), (3,2)),
        ((1,0), (4,2), (2,2), (5,0)),
    )
    function _recip_edge(p, edge)
        nb_panel = conn[p][edge][1]
        for e in 1:4; conn[nb_panel][e][1] == p && return e; end
        error("No reciprocal edge for P$p edge $edge")
    end
    function _get_boundary_dxa_sg(p, edge)
        nb_panel, orient = conn[p][edge]
        q_e = _recip_edge(p, edge)
        flip = orient >= 2
        use_dya = q_e <= 2
        sg_edge_map = (2, 4, 1, 3)
        sg_edge = sg_edge_map[q_e]
        dxa_vals = zeros(Float64, Nc)
        sg_vals  = zeros(Float64, Nc)
        for s in 1:Nc
            s_src = flip ? (Nc+1-s) : s
            if     q_e == 1; i_nb, j_nb = s_src, Nc
            elseif q_e == 2; i_nb, j_nb = s_src, 1
            elseif q_e == 3; i_nb, j_nb = Nc, s_src
            else;            i_nb, j_nb = 1, s_src
            end
            dxa_vals[s] = use_dya ? dya[nb_panel][i_nb, j_nb] : dxa[nb_panel][i_nb, j_nb]
            sg_vals[s]  = sin_sg[nb_panel][i_nb, j_nb, sg_edge]
        end
        return dxa_vals, sg_vals
    end
    nb_dxa = ntuple(6) do p
        west_dxa, _ = _get_boundary_dxa_sg(p, 4)
        east_dxa, _ = _get_boundary_dxa_sg(p, 3)
        south_dya, _ = _get_boundary_dxa_sg(p, 2)
        north_dya, _ = _get_boundary_dxa_sg(p, 1)
        (; west_dxa, east_dxa, south_dya, north_dya)
    end
    nb_sg = ntuple(6) do p
        _, west_sg  = _get_boundary_dxa_sg(p, 4)
        _, east_sg  = _get_boundary_dxa_sg(p, 3)
        _, south_sg = _get_boundary_dxa_sg(p, 2)
        _, north_sg = _get_boundary_dxa_sg(p, 1)
        (; west_sg, east_sg, south_sg, north_sg)
    end
    return nb_dxa, nb_sg
end

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
function find_ctm_file(data_dir, date, product)
    daydir = joinpath(data_dir, Dates.format(date, "yyyymmdd"))
    isdir(daydir) || return nothing, 0
    info = GEOS_CS_PRODUCTS[product]
    tag = "CTM_A1.C$(info.Nc)"
    for f in readdir(daydir)
        if contains(f, tag) && endswith(f, ".nc")
            fp = joinpath(daydir, f)
            nt = NCDataset(fp, "r") do ds; length(ds["time"]); end
            return fp, nt
        end
    end
    return nothing, 0
end

"""Find CTM_I1 file (hourly QV + PS) for a given date."""
function find_ctm_i1_file(data_dir, date, Nc)
    daydir = joinpath(data_dir, Dates.format(date, "yyyymmdd"))
    isdir(daydir) || return nothing
    datestr = Dates.format(date, "yyyymmdd")
    # Try GEOSIT naming first, then GEOSFP
    for prefix in ["GEOSIT", "GEOSFP"]
        fname = "$(prefix).$(datestr).CTM_I1.C$(Nc).nc"
        fp = joinpath(daydir, fname)
        isfile(fp) && return fp
    end
    return nothing
end

"""
    read_qv_ps_timestep(ctm_a1_path, ctm_i1_path, tidx, Nc, Nz) → (qv_panels, ps_panels)

Read QV from CTM_I1 and PS from CTM_A1 at time index tidx.
Returns NTuple{6} of (Nc,Nc,Nz) for QV and NTuple{6} of (Nc,Nc) for PS.
QV is flipped to model vertical order (top-to-bottom, k=1=TOA) during read.
"""
function read_qv_ps_timestep(ctm_a1_path::String, ctm_i1_path::String,
                              tidx::Int, Nc::Int, Nz::Int)
    FT = Float32
    # Read QV from CTM_I1, flip bottom-to-top → top-to-bottom
    qv_panels = NCDataset(ctm_i1_path, "r") do ds
        qv_raw = FT.(coalesce.(ds["QV"][:, :, :, :, tidx], FT(0)))
        ntuple(6) do p
            buf = Array{FT}(undef, Nc, Nc, Nz)
            for k in 1:Nz, j in 1:Nc, i in 1:Nc
                buf[i, j, Nz - k + 1] = qv_raw[i, j, p, k]  # flip: file k=1(sfc) → model k=Nz
            end
            buf
        end
    end

    # Read PS from CTM_A1
    ps_panels = NCDataset(ctm_a1_path, "r") do ds
        ps_raw = FT.(coalesce.(ds["PS"][:, :, :, tidx], FT(0)))
        ntuple(p -> ps_raw[:, :, p], 6)
    end

    return qv_panels, ps_panels
end

# ---------------------------------------------------------------------------
# Load gridspec
# ---------------------------------------------------------------------------
function load_gridspec(gridspec_path)
    ds = NCDataset(gridspec_path, "r")
    c_lons_raw = Array{Float64}(ds["corner_lons"][:, :, :])
    c_lats_raw = Array{Float64}(ds["corner_lats"][:, :, :])
    a_lons_raw = Array{Float64}(ds["lons"][:, :, :])
    a_lats_raw = Array{Float64}(ds["lats"][:, :, :])
    close(ds)

    ndim1 = size(c_lons_raw, 1)
    if ndim1 == 6
        Nc = size(c_lons_raw, 2) - 1
        corner_lons = ntuple(p -> c_lons_raw[p, :, :], 6)
        corner_lats = ntuple(p -> c_lats_raw[p, :, :], 6)
        center_lons = ntuple(p -> a_lons_raw[p, :, :], 6)
        center_lats = ntuple(p -> a_lats_raw[p, :, :], 6)
    else
        Nc = size(c_lons_raw, 1) - 1
        if size(c_lons_raw, 3) == 6
            corner_lons = ntuple(p -> c_lons_raw[:, :, p], 6)
            corner_lats = ntuple(p -> c_lats_raw[:, :, p], 6)
            center_lons = ntuple(p -> a_lons_raw[:, :, p], 6)
            center_lats = ntuple(p -> a_lats_raw[:, :, p], 6)
        else
            error("Cannot determine gridspec dimension order: sizes = $(size(c_lons_raw))")
        end
    end
    return (; corner_lons, corner_lats, center_lons, center_lats, Nc)
end

# ---------------------------------------------------------------------------
# Preprocess one day (v4)
# ---------------------------------------------------------------------------
function preprocess_day_v4(date::Date, filepath, n_timesteps, output_path, cfg,
                            grid_metrics, sin_sg;
                            nb_dxa=nothing, nb_sg=nothing,
                            ctm_i1_path=nothing, next_ctm_a1_path=nothing,
                            next_ctm_i1_path=nothing)
    FT = Float32
    Hp = 3
    mass_flux_dt = cfg.mass_flux_dt
    Nc = size(sin_sg[1], 1)

    ts0 = read_geosfp_cs_timestep(filepath; FT, convert_to_kgs=true, dt_met=mass_flux_dt)
    Nz = ts0.Nz
    Nt = n_timesteps

    n_delp_panel = (Nc + 2Hp)^2 * Nz
    n_am_panel   = (Nc + 1) * Nc * Nz
    n_bm_panel   = Nc * (Nc + 1) * Nz
    n_qv_panel   = Nc * Nc * Nz
    n_ps_panel   = Nc * Nc
    # v4: v3 fields + QV_start + QV_end + PS_start + PS_end
    elems_per_window = 6 * (n_delp_panel + 3*n_am_panel + 3*n_bm_panel) +
                       2 * 6 * n_qv_panel + 2 * 6 * n_ps_panel

    has_qv_ps = ctm_i1_path !== nothing && isfile(ctm_i1_path)

    header = Dict{String,Any}(
        "magic"              => "CSFLX",
        "version"            => 4,
        "grid_type"          => "cubed_sphere",
        "Nc"                 => Nc,
        "Nz"                 => Nz,
        "Hp"                 => Hp,
        "Nt"                 => Nt,
        "float_type"         => "Float32",
        "float_bytes"        => 4,
        "header_bytes"       => HEADER_SIZE,
        "window_bytes"       => elems_per_window * 4,
        "n_delp_panel"       => n_delp_panel,
        "n_am_panel"         => n_am_panel,
        "n_bm_panel"         => n_bm_panel,
        "n_qv_panel"         => n_qv_panel,
        "n_ps_panel"         => n_ps_panel,
        "n_panels"           => 6,
        "elems_per_window"   => elems_per_window,
        "date"               => Dates.format(date, "yyyy-mm-dd"),
        "dt_met_seconds"     => mass_flux_dt,
        "product"            => cfg.product,
        "include_courant"    => true,
        "include_area_flux"  => true,
        "include_qv"         => has_qv_ps,
        "include_ps"         => has_qv_ps,
        "vertical_order"     => "top_to_bottom",  # QV pre-flipped to model order
    )

    header_json = JSON3.write(header)
    @assert length(header_json) < HEADER_SIZE "Header too large: $(length(header_json)) bytes"

    @info @sprintf("  C%d×%d, %d windows, %.1f MB/win, %.2f GB total (v4, QV/PS=%s)",
                    Nc, Nz, Nt, elems_per_window * 4 / 1e6,
                    (HEADER_SIZE + elems_per_window * 4 * Nt) / 1e9,
                    has_qv_ps ? "yes" : "no")

    open(output_path, "w") do io
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)

        for tidx in 1:Nt
            t0 = time()

            # Read DELP, MFXC, MFYC
            ts = read_geosfp_cs_timestep(filepath; FT, time_index=tidx,
                                          convert_to_kgs=true, dt_met=mass_flux_dt)
            delp_h, mfxc, mfyc = to_haloed_panels(ts; Hp)
            am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

            # Read CX/CY, flip, stagger
            ds = NCDataset(filepath, "r")
            cx_raw = Array{FT}(ds["CX"][:, :, :, :, tidx])
            cy_raw = Array{FT}(ds["CY"][:, :, :, :, tidx])
            close(ds)

            # GEOS-IT is bottom-to-top — always flip
            cx_raw = reverse(cx_raw, dims=4)
            cy_raw = reverse(cy_raw, dims=4)

            cx_panels = ntuple(p -> cx_raw[:, :, p, :], 6)
            cy_panels = ntuple(p -> cy_raw[:, :, p, :], 6)
            cx_stag, cy_stag = cgrid_to_staggered_panels(cx_panels, cy_panels)

            # Compute xfx/yfx with exact sin_sg + cross-panel boundary geometry
            xfx, yfx = compute_xfx_yfx(cx_stag, cy_stag,
                                          grid_metrics.dxa, grid_metrics.dya,
                                          grid_metrics.dy_face, grid_metrics.dx_face,
                                          sin_sg, Nc, ts.Nz;
                                          nb_dxa=nb_dxa, nb_sg=nb_sg)

            # Write v3 fields: DELP, AM, BM, CX, CY, XFX, YFX
            for p in 1:6; write(io, vec(delp_h[p])); end
            for p in 1:6; write(io, vec(am_stag[p])); end
            for p in 1:6; write(io, vec(bm_stag[p])); end
            for p in 1:6; write(io, vec(cx_stag[p])); end
            for p in 1:6; write(io, vec(cy_stag[p])); end
            for p in 1:6; write(io, vec(xfx[p])); end
            for p in 1:6; write(io, vec(yfx[p])); end

            # Write v4 fields: QV_start, QV_end, PS_start, PS_end
            if has_qv_ps
                # QV_start / PS_start: timestep tidx of current day
                qv_start, ps_start = read_qv_ps_timestep(filepath, ctm_i1_path,
                                                           tidx, Nc, Nz)

                # QV_end / PS_end: timestep tidx+1 (or first of next day)
                if tidx < Nt
                    qv_end, ps_end = read_qv_ps_timestep(filepath, ctm_i1_path,
                                                           tidx + 1, Nc, Nz)
                else
                    # Last window: need first timestep of next day
                    if next_ctm_a1_path !== nothing && next_ctm_i1_path !== nothing
                        qv_end, ps_end = read_qv_ps_timestep(next_ctm_a1_path,
                                                               next_ctm_i1_path,
                                                               1, Nc, Nz)
                    else
                        @warn "Last window of $(Dates.format(date, "yyyymmdd")): next day unavailable, using QV/PS_start as QV/PS_end" maxlog=1
                        qv_end, ps_end = qv_start, ps_start
                    end
                end

                for p in 1:6; write(io, vec(qv_start[p])); end
                for p in 1:6; write(io, vec(qv_end[p])); end
                for p in 1:6; write(io, vec(ps_start[p])); end
                for p in 1:6; write(io, vec(ps_end[p])); end
            else
                # No CTM_I1: write zeros as placeholder (header says include_qv=false)
                zero_qv = zeros(FT, Nc, Nc, Nz)
                zero_ps = zeros(FT, Nc, Nc)
                for _ in 1:12; write(io, vec(zero_qv)); end  # 6 start + 6 end
                for _ in 1:12; write(io, vec(zero_ps)); end  # 6 start + 6 end
            end

            elapsed = round(time() - t0, digits=2)
            if tidx <= 2 || tidx == Nt || tidx % 6 == 0
                @info @sprintf("    Window %d/%d: %.2fs", tidx, Nt, elapsed)
            end
        end
    end

    actual   = filesize(output_path)
    expected = HEADER_SIZE + elems_per_window * 4 * Nt
    @info @sprintf("  Done: %.2f GB", actual / 1e9)
    actual == expected || @warn "Size mismatch: expected $expected, got $actual"
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    cfg = parse_args()
    mkpath(cfg.output_dir)

    @info "Preprocessing GEOS CS → binary v4 (v3 + embedded QV/PS at window boundaries)"
    @info "  Input:    $(cfg.data_dir)"
    @info "  Gridspec: $(cfg.gridspec)"
    @info "  Output:   $(cfg.output_dir)"
    @info "  Dates:    $(cfg.start_date) → $(cfg.end_date)"

    # Load gridspec and compute static geometry ONCE
    @info "Loading gridspec and computing grid metrics..."
    gs = load_gridspec(cfg.gridspec)
    Nc = gs.Nc
    R = 6.371e6  # Earth radius [m]

    @info "Computing sin_sg metric factors ($(Nc)×$(Nc)×6 panels)..."
    t0 = time()
    sin_sg = compute_sin_sg(gs.corner_lons, gs.corner_lats,
                             gs.center_lons, gs.center_lats, Nc)
    @info @sprintf("  sin_sg computed in %.1fs", time() - t0)

    mid = div(Nc, 2)
    sg_center = sin_sg[1][mid, mid, 1]
    sg_corner = sin_sg[1][1, 1, 1]
    @info @sprintf("  sin_sg validation: center=%.6f, corner=%.6f", sg_center, sg_corner)

    @info "Computing grid metrics (dxa, dya, dx, dy)..."
    grid_metrics = compute_grid_metrics(gs.corner_lons, gs.corner_lats,
                                         gs.center_lons, gs.center_lats, Nc, R)

    @info "Computing cross-panel boundary geometry..."
    nb_dxa, nb_sg = compute_boundary_geometry(grid_metrics.dxa, grid_metrics.dya, sin_sg, Nc)
    @info "  Done. Boundary XFX/YFX will use exact cross-panel dxa/sin_sg."

    # Build file lookup for CTM_I1 and CTM_A1 (needed for day-boundary crossing)
    all_dates = cfg.start_date:Day(1):cfg.end_date
    ctm_a1_files = Dict{Date, Tuple{String, Int}}()
    ctm_i1_files = Dict{Date, String}()
    @info "Scanning CTM_A1 and CTM_I1 files..."
    for date in (cfg.start_date - Day(1)):(Day(1)):(cfg.end_date + Day(1))
        fp, nt = find_ctm_file(cfg.data_dir, date, cfg.product)
        fp !== nothing && (ctm_a1_files[date] = (fp, nt))
        i1 = find_ctm_i1_file(cfg.data_dir, date, Nc)
        i1 !== nothing && (ctm_i1_files[date] = i1)
    end
    @info "  Found $(length(ctm_a1_files)) CTM_A1 files, $(length(ctm_i1_files)) CTM_I1 files"

    # Process each day
    for date in all_dates
        datestr = Dates.format(date, "yyyymmdd")
        outpath = joinpath(cfg.output_dir, "geosfp_cs_$(datestr)_float32.bin")

        if isfile(outpath) && filesize(outpath) > HEADER_SIZE
            @info "[$datestr] Already exists — skipping"
            continue
        end

        if !haskey(ctm_a1_files, date)
            @warn "[$datestr] No CTM_A1 file found — skipping"
            continue
        end

        filepath, nt = ctm_a1_files[date]
        ctm_i1_path = get(ctm_i1_files, date, nothing)

        # Next day files for day-boundary crossing (last window QV_end/PS_end)
        next_date = date + Day(1)
        next_ctm_a1_path = haskey(ctm_a1_files, next_date) ? ctm_a1_files[next_date][1] : nothing
        next_ctm_i1_path = get(ctm_i1_files, next_date, nothing)

        if ctm_i1_path === nothing
            @warn "[$datestr] No CTM_I1 file — QV/PS will not be embedded"
        end

        @info "\n--- [$datestr] Processing (v4) ---"
        preprocess_day_v4(date, filepath, nt, outpath, cfg, grid_metrics, sin_sg;
                          nb_dxa=nb_dxa, nb_sg=nb_sg,
                          ctm_i1_path=ctm_i1_path,
                          next_ctm_a1_path=next_ctm_a1_path,
                          next_ctm_i1_path=next_ctm_i1_path)
    end

    @info "All done."
end

main()
