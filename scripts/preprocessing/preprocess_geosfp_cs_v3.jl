#!/usr/bin/env julia
# ===========================================================================
# Preprocessing: GEOS CS NetCDF → flat binary v3
#
# v3 adds precomputed area fluxes (xfx/yfx) with exact sin_sg metric
# correction from GEOS-Chem gridspec coordinates.
#
# Binary layout per window:
#   [6 DELP] [6 AM] [6 BM] [6 CX] [6 CY] [6 XFX] [6 YFX]
#
# XFX/YFX are precomputed using exact sin_sg from gridspec corners:
#   xfx = cx * dxa_upwind * dy_face * sin_sg_upwind
#
# This eliminates the sin_sg ≈ 1.0 approximation and removes runtime
# area flux computation entirely.
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_geosfp_cs_v3.jl \
#         --data_dir ~/data/geosit_c180_catrine \
#         --gridspec ~/code/gitHub/AtmosTransportModel/data/grids/cs_c180_gridspec.nc \
#         --output_dir /temp1/catrine/met/geosit_c180/massflux_v3 \
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
    output_dir   = expanduser(get(args, "output_dir", "/temp1/catrine/met/geosit_c180/massflux_v3"))
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

"""
    cos_angle(p1, p2, p3)

Cosine of the angle at p1 between rays to p2 and p3 on the unit sphere.
Port of GCHP fv_grid_utils.F90:2851-2895.

Uses vector cross products: P = p1 × p2, Q = p1 × p3, cos(θ) = P·Q / (|P||Q|).
"""
function cos_angle(p1, p2, p3)
    # P = p1 × p2
    px = p1[2]*p2[3] - p1[3]*p2[2]
    py = p1[3]*p2[1] - p1[1]*p2[3]
    pz = p1[1]*p2[2] - p1[2]*p2[1]
    # Q = p1 × p3
    qx = p1[2]*p3[3] - p1[3]*p3[2]
    qy = p1[3]*p3[1] - p1[1]*p3[3]
    qz = p1[1]*p3[2] - p1[2]*p3[1]

    pp = px*px + py*py + pz*pz
    qq = qx*qx + qy*qy + qz*qz
    denom = sqrt(pp * qq)
    denom < 1e-30 && return 1.0
    return clamp((px*qx + py*qy + pz*qz) / denom, -1.0, 1.0)
end

"""
    compute_sin_sg(corner_lons, corner_lats, center_lons, center_lats, Nc)

Compute sin_sg metric correction factors for all 6 panels.
Returns NTuple{6} of (Nc, Nc, 4) arrays: sin_sg[i,j,edge] for edge ∈ {1=E, 2=N, 3=W, 4=S}.

Port of GCHP fv_grid_utils.F90:375-414.
sin_sg at edge midpoints = sin(angle between grid-line tangent and cell-face normal).
"""
function compute_sin_sg(corner_lons, corner_lats, center_lons, center_lats, Nc)
    # corner_lons[p] is (Nc+1, Nc+1) — vertex coordinates
    # center_lons[p] is (Nc, Nc) — cell center coordinates
    sin_sg_panels = ntuple(6) do p
        sg = zeros(Float64, Nc, Nc, 4)
        c_lon = corner_lons[p]
        c_lat = corner_lats[p]
        a_lon = center_lons[p]
        a_lat = center_lats[p]

        for j in 1:Nc, i in 1:Nc
            # Cell vertices (Cartesian on unit sphere)
            v_sw = _ll2cart(c_lon[i,   j],   c_lat[i,   j])
            v_se = _ll2cart(c_lon[i+1, j],   c_lat[i+1, j])
            v_ne = _ll2cart(c_lon[i+1, j+1], c_lat[i+1, j+1])
            v_nw = _ll2cart(c_lon[i,   j+1], c_lat[i,   j+1])

            # Cell center and neighbor centers
            a_c = _ll2cart(a_lon[i, j], a_lat[i, j])

            # GCHP sin_sg: angle at edge midpoint between the CELL CENTER
            # and the edge endpoint. Port of fv_grid_utils.F90:375-404.
            # sin_sg(i,j,edge) uses the center of cell (i,j) as reference.

            # Edge 1 (East): midpoint of SE-NE, angle between cell center and NE vertex
            mid_e = _mid_pt(v_se, v_ne)
            cos_e = cos_angle(mid_e, a_c, v_ne)
            sg[i, j, 1] = min(1.0, sqrt(max(0.0, 1.0 - cos_e^2)))

            # Edge 2 (North): midpoint of NW-NE, angle between cell center and NE vertex
            mid_n = _mid_pt(v_nw, v_ne)
            cos_n = cos_angle(mid_n, a_c, v_ne)
            sg[i, j, 2] = min(1.0, sqrt(max(0.0, 1.0 - cos_n^2)))

            # Edge 3 (West): midpoint of SW-NW, angle between cell center and NW vertex
            mid_w = _mid_pt(v_sw, v_nw)
            cos_w = cos_angle(mid_w, a_c, v_nw)
            sg[i, j, 3] = min(1.0, sqrt(max(0.0, 1.0 - cos_w^2)))

            # Edge 4 (South): midpoint of SW-SE, angle between cell center and SE vertex
            mid_s = _mid_pt(v_sw, v_se)
            cos_s = cos_angle(mid_s, a_c, v_se)
            sg[i, j, 4] = min(1.0, sqrt(max(0.0, 1.0 - cos_s^2)))
        end
        sg
    end
    return sin_sg_panels
end

"""Compute great-circle distance between two points on sphere of radius R."""
function _haversine(lon1_deg, lat1_deg, lon2_deg, lat2_deg, R=1.0)
    λ1, φ1 = deg2rad(Float64(lon1_deg)), deg2rad(Float64(lat1_deg))
    λ2, φ2 = deg2rad(Float64(lon2_deg)), deg2rad(Float64(lat2_deg))
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    a = sin(dφ/2)^2 + cos(φ1) * cos(φ2) * sin(dλ/2)^2
    return R * 2 * atan(sqrt(a), sqrt(1 - a))
end

"""
    compute_grid_metrics(corner_lons, corner_lats, center_lons, center_lats, Nc, R)

Compute dxa, dya (center-to-center), dx, dy (edge lengths) for all panels.
Returns dxa, dya as NTuple{6} of (Nc, Nc), dx as (Nc, Nc+1), dy as (Nc+1, Nc).
"""
function compute_grid_metrics(corner_lons, corner_lats, center_lons, center_lats, Nc, R)
    # GEOS panel connectivity: (north, south, east, west) neighbors
    # Each entry: (panel, orientation), orientation: 0=aligned, 2=reversed
    # Edge convention: 1=north(j=Nc), 2=south(j=1), 3=east(i=Nc), 4=west(i=1)
    conn = (
        ((3,2), (6,0), (2,0), (5,2)),  # P1: N→P3(rev), S→P6(aln), E→P2(aln), W→P5(rev)
        ((3,0), (6,2), (4,2), (1,0)),  # P2
        ((5,2), (2,0), (4,0), (1,2)),  # P3
        ((5,0), (2,2), (6,2), (3,0)),  # P4
        ((1,2), (4,0), (6,0), (3,2)),  # P5
        ((1,0), (4,2), (2,2), (5,0)),  # P6
    )

    # Helper: get center coords of the cell on neighbor panel that shares a boundary face.
    # For panel p's east boundary (edge 3) at position s ∈ 1:Nc:
    #   The neighbor's boundary cell is at depth=1 along edge q_e.
    function _neighbor_center(p, edge, s)
        nb_panel, orient = conn[p][edge]
        s_src = orient >= 2 ? (Nc + 1 - s) : s
        # Find which edge on the neighbor connects back to p
        q_e = 0
        for e in 1:4
            if conn[nb_panel][e][1] == p
                q_e = e; break
            end
        end
        # Map edge + position to (i, j) on neighbor panel
        if     q_e == 1  # neighbor's north: (s_src, Nc)
            return (center_lons[nb_panel][s_src, Nc], center_lats[nb_panel][s_src, Nc])
        elseif q_e == 2  # neighbor's south: (s_src, 1)
            return (center_lons[nb_panel][s_src, 1], center_lats[nb_panel][s_src, 1])
        elseif q_e == 3  # neighbor's east: (Nc, s_src)
            return (center_lons[nb_panel][Nc, s_src], center_lats[nb_panel][Nc, s_src])
        else             # neighbor's west: (1, s_src)
            return (center_lons[nb_panel][1, s_src], center_lats[nb_panel][1, s_src])
        end
    end

    # dxa[i,j] = center-to-center distance in X direction.
    # GCHP: dxa(i,j) = distance from agrid(i,j) to agrid(i+1,j) (cell spacing).
    dxa = ntuple(6) do p
        d = zeros(Float64, Nc, Nc)
        for j in 1:Nc
            for i in 1:Nc-1
                d[i, j] = _haversine(center_lons[p][i, j], center_lats[p][i, j],
                                      center_lons[p][i+1, j], center_lats[p][i+1, j], R)
            end
            # Boundary: use cross-panel center distance (east neighbor)
            nb_lon, nb_lat = _neighbor_center(p, 3, j)  # edge 3 = east
            d[Nc, j] = _haversine(center_lons[p][Nc, j], center_lats[p][Nc, j],
                                    nb_lon, nb_lat, R)
        end
        d
    end

    # dya[i,j] = center-to-center distance in Y direction.
    dya = ntuple(6) do p
        d = zeros(Float64, Nc, Nc)
        for i in 1:Nc
            for j in 1:Nc-1
                d[i, j] = _haversine(center_lons[p][i, j], center_lats[p][i, j],
                                      center_lons[p][i, j+1], center_lats[p][i, j+1], R)
            end
            # Boundary: use cross-panel center distance (north neighbor)
            nb_lon, nb_lat = _neighbor_center(p, 1, i)  # edge 1 = north
            d[i, Nc] = _haversine(center_lons[p][i, Nc], center_lats[p][i, Nc],
                                    nb_lon, nb_lat, R)
        end
        d
    end

    # dy[iif, j] = Y-direction edge length at X-face iif, cell j
    dy_face = ntuple(6) do p
        d = zeros(Float64, Nc + 1, Nc)
        for j in 1:Nc, iif in 1:(Nc + 1)
            d[iif, j] = _haversine(corner_lons[p][iif, j], corner_lats[p][iif, j],
                                    corner_lons[p][iif, j+1], corner_lats[p][iif, j+1], R)
        end
        d
    end

    # dx[i, jf] = X-direction edge length at Y-face jf, cell i
    dx_face = ntuple(6) do p
        d = zeros(Float64, Nc, Nc + 1)
        for jf in 1:(Nc + 1), i in 1:Nc
            d[i, jf] = _haversine(corner_lons[p][i, jf], corner_lats[p][i, jf],
                                    corner_lons[p][i+1, jf], corner_lats[p][i+1, jf], R)
        end
        d
    end

    return (; dxa, dya, dy_face, dx_face)
end

"""
    compute_xfx_yfx(cx_stag, cy_stag, dxa, dya, dy_face, dx_face, sin_sg, Nc, Nz)

Compute area fluxes from Courant numbers and grid geometry with exact sin_sg.
Port of GCHP fv_tracer2d.F90:400-420.

Returns (xfx, yfx) as NTuple{6} of same shape as (cx_stag, cy_stag).
"""
function compute_xfx_yfx(cx_stag, cy_stag, dxa, dya, dy_face, dx_face, sin_sg, Nc, Nz;
                          nb_dxa=nothing, nb_sg=nothing)
    # nb_dxa[p] / nb_sg[p] provide cross-panel dxa and sin_sg for boundary faces.
    # If not provided, fall back to clamping (backward compat).

    xfx = ntuple(6) do p
        x = zeros(Float32, Nc + 1, Nc, Nz)
        for k in 1:Nz, j in 1:Nc, iif in 1:(Nc + 1)
            c = cx_stag[p][iif, j, k]
            if c > 0
                # Upwind from left: cell (iif-1, j)
                if iif == 1 && nb_dxa !== nothing
                    # Boundary: upwind cell is on west neighbor
                    x[iif, j, k] = Float32(c * nb_dxa[p].west_dxa[j] * dy_face[p][iif, j] * nb_sg[p].west_sg[j])
                else
                    i_up = max(1, iif - 1)
                    x[iif, j, k] = Float32(c * dxa[p][i_up, j] * dy_face[p][iif, j] * sin_sg[p][i_up, j, 3])
                end
            else
                # Upwind from right: cell (iif, j)
                if iif == Nc + 1 && nb_dxa !== nothing
                    # Boundary: upwind cell is on east neighbor
                    x[iif, j, k] = Float32(c * nb_dxa[p].east_dxa[j] * dy_face[p][iif, j] * nb_sg[p].east_sg[j])
                else
                    i_up = min(Nc, iif)
                    x[iif, j, k] = Float32(c * dxa[p][i_up, j] * dy_face[p][iif, j] * sin_sg[p][i_up, j, 1])
                end
            end
        end
        x
    end

    yfx = ntuple(6) do p
        y = zeros(Float32, Nc, Nc + 1, Nz)
        for k in 1:Nz, jf in 1:(Nc + 1), i in 1:Nc
            c = cy_stag[p][i, jf, k]
            if c > 0
                # Upwind from below: cell (i, jf-1)
                if jf == 1 && nb_dxa !== nothing
                    # Boundary: upwind cell is on south neighbor
                    y[i, jf, k] = Float32(c * nb_dxa[p].south_dya[i] * dx_face[p][i, jf] * nb_sg[p].south_sg[i])
                else
                    j_up = max(1, jf - 1)
                    y[i, jf, k] = Float32(c * dya[p][i, j_up] * dx_face[p][i, jf] * sin_sg[p][i, j_up, 4])
                end
            else
                # Upwind from above: cell (i, jf)
                if jf == Nc + 1 && nb_dxa !== nothing
                    # Boundary: upwind cell is on north neighbor
                    y[i, jf, k] = Float32(c * nb_dxa[p].north_dya[i] * dx_face[p][i, jf] * nb_sg[p].north_sg[i])
                else
                    j_up = min(Nc, jf)
                    y[i, jf, k] = Float32(c * dya[p][i, j_up] * dx_face[p][i, jf] * sin_sg[p][i, j_up, 2])
                end
            end
        end
        y
    end

    return (xfx, yfx)
end

"""
    compute_boundary_geometry(dxa, dya, sin_sg, Nc)

Build cross-panel dxa/dya and sin_sg for boundary faces where the upwind cell
is on the neighboring panel. Uses the GEOS panel connectivity.

Returns (nb_dxa, nb_sg) as NTuple{6} of NamedTuples with:
- west_dxa[j], east_dxa[j]: dxa of the upwind cell on the neighbor for X-faces
- south_dya[i], north_dya[i]: dya of the upwind cell on the neighbor for Y-faces
- west_sg[j], east_sg[j]: sin_sg of the upwind cell at the boundary edge
- south_sg[i], north_sg[i]: same for Y boundaries

The sin_sg edge index depends on which edge of the neighbor panel faces the boundary:
- Neighbor's east edge (q_e=3) → sin_sg edge 3 (W) for cx>0, edge 1 (E) for cx<0
- Neighbor's north edge (q_e=1) → sin_sg edge 2 (N) for outgoing
"""
function compute_boundary_geometry(dxa, dya, sin_sg, Nc)
    # Panel connectivity: (north, south, east, west) → (neighbor_panel, orientation)
    conn = (
        ((3,2), (6,0), (2,0), (5,2)),  # P1
        ((3,0), (6,2), (4,2), (1,0)),  # P2
        ((5,2), (2,0), (4,0), (1,2)),  # P3
        ((5,0), (2,2), (6,2), (3,0)),  # P4
        ((1,2), (4,0), (6,0), (3,2)),  # P5
        ((1,0), (4,2), (2,2), (5,0)),  # P6
    )

    # Reciprocal edge: which edge of neighbor connects back to panel p?
    function _recip_edge(p, edge)
        nb_panel = conn[p][edge][1]
        for e in 1:4
            conn[nb_panel][e][1] == p && return e
        end
        error("No reciprocal edge for P$p edge $edge")
    end

    # For an XFX boundary face at iif=1 with cx > 0 (upwind from west neighbor):
    # The west neighbor (edge 4) has reciprocal edge q_e.
    # The upwind cell on the neighbor is at its boundary:
    #   q_e=1 (north): cell (s_src, Nc)   → dya of that cell, sin_sg edge 2 (N)
    #   q_e=2 (south): cell (s_src, 1)    → dya of that cell, sin_sg edge 4 (S)
    #   q_e=3 (east):  cell (Nc, s_src)   → dxa of that cell, sin_sg edge 1 (E)
    #   q_e=4 (west):  cell (1, s_src)    → dxa of that cell, sin_sg edge 3 (W)
    #
    # BUT: the "dxa" we need is in the DIRECTION perpendicular to the boundary.
    # At XFX boundary: the flux direction is panel p's X → this maps to the neighbor's
    # direction perpendicular to its boundary edge.
    # If q_e is the neighbor's north/south edge → perpendicular = Y → use dya
    # If q_e is the neighbor's east/west edge → perpendicular = X → use dxa

    function _get_boundary_dxa_sg(p, edge)
        nb_panel, orient = conn[p][edge]
        q_e = _recip_edge(p, edge)
        flip = orient >= 2

        # Determine which dimension of neighbor is perpendicular to the boundary
        # q_e=1(N) or q_e=2(S): boundary along neighbor's X → perpendicular = Y → dya
        # q_e=3(E) or q_e=4(W): boundary along neighbor's Y → perpendicular = X → dxa
        use_dya = q_e <= 2  # north or south boundary on neighbor

        # sin_sg edge at the boundary face of the upwind cell:
        # The upwind cell exits through edge q_e. The sin_sg at that edge:
        #   q_e=1 (N exit) → edge 2 (N), q_e=2 (S exit) → edge 4 (S)
        #   q_e=3 (E exit) → edge 1 (E), q_e=4 (W exit) → edge 3 (W)
        sg_edge_map = (2, 4, 1, 3)  # N→2, S→4, E→1, W→3
        sg_edge = sg_edge_map[q_e]

        dxa_vals = zeros(Float64, Nc)
        sg_vals  = zeros(Float64, Nc)

        for s in 1:Nc
            s_src = flip ? (Nc + 1 - s) : s
            # Cell coordinates on neighbor at boundary
            if     q_e == 1; i_nb, j_nb = s_src, Nc   # north edge
            elseif q_e == 2; i_nb, j_nb = s_src, 1    # south edge
            elseif q_e == 3; i_nb, j_nb = Nc, s_src   # east edge
            else;            i_nb, j_nb = 1, s_src     # west edge
            end

            dxa_vals[s] = use_dya ? dya[nb_panel][i_nb, j_nb] : dxa[nb_panel][i_nb, j_nb]
            sg_vals[s]  = sin_sg[nb_panel][i_nb, j_nb, sg_edge]
        end

        return dxa_vals, sg_vals
    end

    nb_dxa = ntuple(6) do p
        # West boundary (edge 4): for XFX at iif=1, cx > 0
        west_dxa, _ = _get_boundary_dxa_sg(p, 4)
        # East boundary (edge 3): for XFX at iif=Nc+1, cx < 0
        east_dxa, _ = _get_boundary_dxa_sg(p, 3)
        # South boundary (edge 2): for YFX at jf=1, cy > 0
        south_dya, _ = _get_boundary_dxa_sg(p, 2)
        # North boundary (edge 1): for YFX at jf=Nc+1, cy < 0
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

# ---------------------------------------------------------------------------
# Load gridspec
# ---------------------------------------------------------------------------
function load_gridspec(gridspec_path)
    ds = NCDataset(gridspec_path, "r")
    # Gridspec has (nf, Y, X) ordering
    c_lons_raw = Array{Float64}(ds["corner_lons"][:, :, :])  # (nf, Nc+1, Nc+1)
    c_lats_raw = Array{Float64}(ds["corner_lats"][:, :, :])
    a_lons_raw = Array{Float64}(ds["lons"][:, :, :])          # (nf, Nc, Nc)
    a_lats_raw = Array{Float64}(ds["lats"][:, :, :])
    close(ds)

    # Check dimension order — gridspec may be (nf, Y, X) or (X, Y, nf)
    ndim1 = size(c_lons_raw, 1)
    if ndim1 == 6
        # (nf, Nc+1, Nc+1) — split by first dim
        Nc = size(c_lons_raw, 2) - 1
        corner_lons = ntuple(p -> c_lons_raw[p, :, :], 6)
        corner_lats = ntuple(p -> c_lats_raw[p, :, :], 6)
        center_lons = ntuple(p -> a_lons_raw[p, :, :], 6)
        center_lats = ntuple(p -> a_lats_raw[p, :, :], 6)
    else
        # (Nc+1, Nc+1, nf) or similar — split by last dim
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
# Preprocess one day
# ---------------------------------------------------------------------------
function preprocess_day_v3(date::Date, filepath, n_timesteps, output_path, cfg,
                            grid_metrics, sin_sg; nb_dxa=nothing, nb_sg=nothing)
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
    # v3: DELP + AM + BM + CX + CY + XFX + YFX
    elems_per_window = 6 * (n_delp_panel + 3*n_am_panel + 3*n_bm_panel)

    header = Dict{String,Any}(
        "magic"              => "CSFLX",
        "version"            => 3,
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
        "n_panels"           => 6,
        "elems_per_window"   => elems_per_window,
        "date"               => Dates.format(date, "yyyy-mm-dd"),
        "dt_met_seconds"     => mass_flux_dt,
        "product"            => cfg.product,
        "include_courant"    => true,
        "include_area_flux"  => true,
    )

    header_json = JSON3.write(header)
    @assert length(header_json) < HEADER_SIZE

    @info @sprintf("  C%d×%d, %d windows, %.1f MB/win, %.2f GB total",
                    Nc, Nz, Nt, elems_per_window * 4 / 1e6,
                    (HEADER_SIZE + elems_per_window * 4 * Nt) / 1e9)

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

            # Write: DELP, AM, BM, CX, CY, XFX, YFX
            for p in 1:6; write(io, vec(delp_h[p])); end
            for p in 1:6; write(io, vec(am_stag[p])); end
            for p in 1:6; write(io, vec(bm_stag[p])); end
            for p in 1:6; write(io, vec(cx_stag[p])); end
            for p in 1:6; write(io, vec(cy_stag[p])); end
            for p in 1:6; write(io, vec(xfx[p])); end
            for p in 1:6; write(io, vec(yfx[p])); end

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

    @info "Preprocessing GEOS CS → binary v3 (with precomputed xfx/yfx + exact sin_sg)"
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

    # Validate sin_sg: should be ~1.0 at center, ~0.85-0.95 at corners
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

    # Process each day
    for date in cfg.start_date:Day(1):cfg.end_date
        datestr = Dates.format(date, "yyyymmdd")
        outpath = joinpath(cfg.output_dir, "geosfp_cs_$(datestr)_float32.bin")

        if isfile(outpath) && filesize(outpath) > HEADER_SIZE
            @info "[$datestr] Already exists — skipping"
            continue
        end

        filepath, nt = find_ctm_file(cfg.data_dir, date, cfg.product)
        if filepath === nothing
            @warn "[$datestr] No CTM file found — skipping"
            continue
        end

        @info "\n--- [$datestr] Processing ---"
        preprocess_day_v3(date, filepath, nt, outpath, cfg, grid_metrics, sin_sg;
                          nb_dxa=nb_dxa, nb_sg=nb_sg)
    end

    @info "All done."
end

main()
