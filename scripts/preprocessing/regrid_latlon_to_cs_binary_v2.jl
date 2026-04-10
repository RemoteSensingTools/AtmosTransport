#!/usr/bin/env julia
#
# Regrid ERA5 LatLon transport binary → C90 Cubed-Sphere transport binary
#
# Reads the existing 0.5° LatLon v2 transport binary (Dec 1-2 2021),
# interpolates cell-center fields (m, ps, winds) to C90 panels via
# bilinear interpolation, computes per-panel face mass fluxes from
# the regridded winds, diagnoses cm from continuity, and writes a
# CS transport binary in the same format.
#
# Usage:
#   julia --project=. scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl \
#       --input <latlon_binary.bin> --output <cs_binary.bin> \
#       --Nc 90 [--day 2021-12-01]
#
# Note: This is a FAST PATH for the 24-hour cross-grid validation.
# It uses bilinear interpolation (not conservative regridding) for
# cell-center fields. This is sufficient for C90 (~1°) from 0.5° source
# since the target is coarser than the source.

using Printf
using JSON3

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2

# ---------------------------------------------------------------------------
# Gnomonic cell center lon/lat computation
# ---------------------------------------------------------------------------

"""Compute (lon, lat) in degrees for all cell centers on panel `p`."""
function panel_cell_centers(Nc::Int, p::Int; FT=Float64)
    dα = FT(π) / (2 * Nc)
    α_centers = [FT(-π/4) + (i - 0.5) * dα for i in 1:Nc]

    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        ξ = tan(α_centers[i])
        η = tan(α_centers[j])
        d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
        x, y, z = if p == 1
            (d, ξ*d, η*d)
        elseif p == 2
            (-ξ*d, d, η*d)
        elseif p == 3
            (-d, -ξ*d, η*d)
        elseif p == 4
            (ξ*d, -d, η*d)
        elseif p == 5
            (-η*d, ξ*d, d)
        else
            (η*d, ξ*d, -d)
        end
        lons[i, j] = atand(y, x)
        lats[i, j] = asind(z / sqrt(x^2 + y^2 + z^2))
        # Normalize lon to [0, 360)
        lons[i, j] < 0 && (lons[i, j] += 360)
    end
    return lons, lats
end

"""Compute (lon, lat) for x-face midpoints (between cells i and i+1)."""
function panel_xface_centers(Nc::Int, p::Int; FT=Float64)
    dα = FT(π) / (2 * Nc)
    α_faces   = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    α_centers = [FT(-π/4) + (i - 0.5) * dα for i in 1:Nc]

    lons = zeros(FT, Nc + 1, Nc)
    lats = zeros(FT, Nc + 1, Nc)
    for j in 1:Nc, i in 1:(Nc + 1)
        ξ = tan(α_faces[i])
        η = tan(α_centers[j])
        d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
        x, y, z = if p == 1
            (d, ξ*d, η*d)
        elseif p == 2
            (-ξ*d, d, η*d)
        elseif p == 3
            (-d, -ξ*d, η*d)
        elseif p == 4
            (ξ*d, -d, η*d)
        elseif p == 5
            (-η*d, ξ*d, d)
        else
            (η*d, ξ*d, -d)
        end
        lons[i, j] = atand(y, x)
        lats[i, j] = asind(z / sqrt(x^2 + y^2 + z^2))
        lons[i, j] < 0 && (lons[i, j] += 360)
    end
    return lons, lats
end

"""Compute (lon, lat) for y-face midpoints (between cells j and j+1)."""
function panel_yface_centers(Nc::Int, p::Int; FT=Float64)
    dα = FT(π) / (2 * Nc)
    α_faces   = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    α_centers = [FT(-π/4) + (i - 0.5) * dα for i in 1:Nc]

    lons = zeros(FT, Nc, Nc + 1)
    lats = zeros(FT, Nc, Nc + 1)
    for j in 1:(Nc + 1), i in 1:Nc
        ξ = tan(α_centers[i])
        η = tan(α_faces[j])
        d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
        x, y, z = if p == 1
            (d, ξ*d, η*d)
        elseif p == 2
            (-ξ*d, d, η*d)
        elseif p == 3
            (-d, -ξ*d, η*d)
        elseif p == 4
            (ξ*d, -d, η*d)
        elseif p == 5
            (-η*d, ξ*d, d)
        else
            (η*d, ξ*d, -d)
        end
        lons[i, j] = atand(y, x)
        lats[i, j] = asind(z / sqrt(x^2 + y^2 + z^2))
        lons[i, j] < 0 && (lons[i, j] += 360)
    end
    return lons, lats
end

# ---------------------------------------------------------------------------
# Bilinear interpolation from LatLon grid
# ---------------------------------------------------------------------------

"""
    bilinear_interp_3d!(dst, src, dst_lons, dst_lats, src_lons, src_lats)

Bilinear interpolation of a 3D field `src[Nx, Ny, Nz]` on a regular lat-lon
grid to arbitrary (lon, lat) points stored in `dst_lons`, `dst_lats`.

`src_lons` must be regularly spaced in [0, 360) with periodic wrapping.
`src_lats` must be regularly spaced, south-to-north.
"""
function bilinear_interp_3d!(dst::AbstractArray{FT, 3},
                              src::AbstractArray{FT, 3},
                              dst_lons::AbstractMatrix{FT},
                              dst_lats::AbstractMatrix{FT},
                              src_lons::AbstractVector{FT},
                              src_lats::AbstractVector{FT}) where FT
    Nx_src = length(src_lons)
    Ny_src = length(src_lats)
    Nz = size(src, 3)
    dlon = src_lons[2] - src_lons[1]
    dlat = src_lats[2] - src_lats[1]
    lon0 = src_lons[1]
    lat0 = src_lats[1]

    Ni, Nj = size(dst_lons)
    @assert size(dst) == (Ni, Nj, Nz)

    @inbounds for j in 1:Nj, i in 1:Ni
        # Find fractional position in source grid
        lon = mod(dst_lons[i, j] - lon0, FT(360))
        fi = lon / dlon + one(FT)
        lat = dst_lats[i, j]
        fj = (lat - lat0) / dlat + one(FT)

        # Integer indices with clamping for lat, periodic for lon
        i0 = floor(Int, fi)
        j0 = floor(Int, fj)
        wx = fi - i0
        wy = fj - j0

        i0 = mod1(i0, Nx_src)
        i1 = mod1(i0 + 1, Nx_src)
        j0 = clamp(j0, 1, Ny_src)
        j1 = clamp(j0 + 1, 1, Ny_src)

        # Clamp weights at boundaries
        if j0 == j1
            wy = zero(FT)
        end

        w00 = (one(FT) - wx) * (one(FT) - wy)
        w10 = wx * (one(FT) - wy)
        w01 = (one(FT) - wx) * wy
        w11 = wx * wy

        for k in 1:Nz
            dst[i, j, k] = w00 * src[i0, j0, k] + w10 * src[i1, j0, k] +
                           w01 * src[i0, j1, k] + w11 * src[i1, j1, k]
        end
    end
    return dst
end

"""Bilinear interpolation for 2D fields."""
function bilinear_interp_2d!(dst::AbstractMatrix{FT},
                              src::AbstractMatrix{FT},
                              dst_lons::AbstractMatrix{FT},
                              dst_lats::AbstractMatrix{FT},
                              src_lons::AbstractVector{FT},
                              src_lats::AbstractVector{FT}) where FT
    src_3d = reshape(src, size(src, 1), size(src, 2), 1)
    dst_3d = reshape(dst, size(dst, 1), size(dst, 2), 1)
    bilinear_interp_3d!(dst_3d, src_3d, dst_lons, dst_lats, src_lons, src_lats)
    return dst
end

# ---------------------------------------------------------------------------
# Mass flux computation on CS panels
# ---------------------------------------------------------------------------

"""
Compute face mass fluxes on a CS panel from cell-center winds and pressure.

am[i, j, k] = 0.5 * (u[i-1,j,k] + u[i,j,k]) * dp[i,j,k] * Δy[i,j] / g * dt_factor
bm[i, j, k] = 0.5 * (v[i,j-1,k] + v[i,j,k]) * dp[i,j,k] * Δx[i,j] / g * dt_factor

where dt_factor = dt / (2 * steps_per_window) to produce per-substep fluxes.
"""
function compute_panel_face_fluxes!(am, bm,
                                     u_panel, v_panel, dp_panel,
                                     Δx, Δy, g, dt_factor,
                                     Nc, Nz)
    # am: x-face flux (Nc+1, Nc, Nz)
    @inbounds for k in 1:Nz, j in 1:Nc
        # Western boundary face (i=1)
        am[1, j, k] = u_panel[1, j, k] * dp_panel[1, j, k] * Δy[1, j] / g * dt_factor
        # Interior faces
        for i in 2:Nc
            u_face = 0.5 * (u_panel[i-1, j, k] + u_panel[i, j, k])
            dp_face = 0.5 * (dp_panel[i-1, j, k] + dp_panel[i, j, k])
            am[i, j, k] = u_face * dp_face * Δy[min(i, Nc), j] / g * dt_factor
        end
        # Eastern boundary face (i=Nc+1)
        am[Nc+1, j, k] = u_panel[Nc, j, k] * dp_panel[Nc, j, k] * Δy[Nc, j] / g * dt_factor
    end

    # bm: y-face flux (Nc, Nc+1, Nz)
    @inbounds for k in 1:Nz, i in 1:Nc
        # Southern boundary face (j=1)
        bm[i, 1, k] = v_panel[i, 1, k] * dp_panel[i, 1, k] * Δx[i, 1] / g * dt_factor
        # Interior faces
        for j in 2:Nc
            v_face = 0.5 * (v_panel[i, j-1, k] + v_panel[i, j, k])
            dp_face = 0.5 * (dp_panel[i, j-1, k] + dp_panel[i, j, k])
            bm[i, j, k] = v_face * dp_face * Δx[i, min(j, Nc)] / g * dt_factor
        end
        # Northern boundary face (j=Nc+1)
        bm[i, Nc+1, k] = v_panel[i, Nc, k] * dp_panel[i, Nc, k] * Δx[i, Nc] / g * dt_factor
    end

    return nothing
end

"""
Diagnose vertical mass flux cm from horizontal flux divergence and mass tendency.

cm[i,j,k+1] = cm[i,j,k] + am[i,j,k] - am[i+1,j,k] + bm[i,j,k] - bm[i,j+1,k] - dm[i,j,k]

where dm = (m_next - m_curr) / (2 * steps_per_window).
cm[:,:,1] = 0 (TOA), cm[:,:,Nz+1] = 0 (surface, enforced by construction if balanced).
"""
function diagnose_cm!(cm, am, bm, dm, m, Nc, Nz)
    @inbounds for j in 1:Nc, i in 1:Nc
        # Pass 1: raw cm from cumulative sum TOA→surface
        cm[i, j, 1] = 0.0  # TOA boundary
        for k in 1:Nz
            div_h = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            cm[i, j, k+1] = cm[i, j, k] + div_h - dm[i, j, k]
        end

        # Pass 2: redistribute surface residual to enforce cm[Nz+1] = 0
        # The residual arises because interpolated am/bm don't satisfy
        # exact global mass balance on the CS grid. We distribute the
        # residual proportionally to cell mass (proxy for dp).
        residual = cm[i, j, Nz + 1]
        if abs(residual) > 0
            total_m = 0.0
            for k in 1:Nz
                total_m += m[i, j, k]
            end
            if total_m > 0
                cum_fix = 0.0
                for k in 1:Nz
                    frac = m[i, j, k] / total_m
                    cum_fix += frac * residual
                    cm[i, j, k + 1] -= cum_fix
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

function main()
    # Parse arguments
    args = Dict{String, String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            val = i < length(ARGS) ? ARGS[i+1] : ""
            args[key] = val
            i += 2
        else
            i += 1
        end
    end

    input_path = get(args, "input", "")
    output_path = get(args, "output", "")
    Nc = parse(Int, get(args, "Nc", "90"))

    isempty(input_path) && error("--input <latlon_binary.bin> required")
    isempty(output_path) && error("--output <cs_binary.bin> required")
    isfile(input_path) || error("Input file not found: $input_path")

    FT = Float64
    g = 9.80665  # standard gravity [m/s²]
    R_earth = 6.371e6

    println("=" ^ 72)
    println("ERA5 LatLon → C$Nc Cubed-Sphere transport binary")
    println("  Input:  $input_path")
    println("  Output: $output_path")
    println("=" ^ 72)

    # --- Load LatLon binary ---
    println("\n[1/4] Reading LatLon binary header...")
    reader = TransportBinaryReader(input_path; FT=FT)
    h = reader.header
    Nx_ll = h.Nx
    Ny_ll = h.Ny
    Nz = h.nlevel
    nwindow = h.nwindow

    src_lons = FT.(h.lons_f64)
    src_lats = FT.(h.lats_f64)
    A_ifc = h.A_ifc
    B_ifc = h.B_ifc

    println("  LatLon: $(Nx_ll)×$(Ny_ll)×$(Nz), $nwindow windows")
    println("  Steps/window: $(h.steps_per_window), dt_met: $(h.dt_met_seconds)s")

    # --- Build CS mesh and geometry ---
    println("\n[2/4] Building C$Nc cubed-sphere geometry...")
    mesh = CubedSphereMesh(Nc=Nc, FT=FT)
    Δx = mesh.Δx
    Δy = mesh.Δy
    areas = mesh.cell_areas

    # Precompute panel cell centers and face centers
    cell_lons = [panel_cell_centers(Nc, p; FT=FT)[1] for p in 1:6]
    cell_lats = [panel_cell_centers(Nc, p; FT=FT)[2] for p in 1:6]
    xface_lons = [panel_xface_centers(Nc, p; FT=FT)[1] for p in 1:6]
    xface_lats = [panel_xface_centers(Nc, p; FT=FT)[2] for p in 1:6]
    yface_lons = [panel_yface_centers(Nc, p; FT=FT)[1] for p in 1:6]
    yface_lats = [panel_yface_centers(Nc, p; FT=FT)[2] for p in 1:6]

    println("  C$Nc: 6×$(Nc)×$(Nc)×$(Nz) cells")
    println("  Δx range: $(round(minimum(Δx)/1e3, digits=1))–$(round(maximum(Δx)/1e3, digits=1)) km")

    # --- Process windows ---
    println("\n[3/4] Processing $nwindow windows...")

    steps_per_window = h.steps_per_window
    dt_met = h.dt_met_seconds
    dt_factor = dt_met / (2 * steps_per_window)

    # Preallocate per-panel arrays
    m_panels  = [zeros(FT, Nc, Nc, Nz) for _ in 1:6]
    ps_panels = [zeros(FT, Nc, Nc) for _ in 1:6]
    am_panels = [zeros(FT, Nc+1, Nc, Nz) for _ in 1:6]
    bm_panels = [zeros(FT, Nc, Nc+1, Nz) for _ in 1:6]
    cm_panels = [zeros(FT, Nc, Nc, Nz+1) for _ in 1:6]

    # Temp arrays
    dm_panel = zeros(FT, Nc, Nc, Nz)

    # Preallocate LatLon flux density arrays (reused per window)
    am_cc = zeros(FT, Nx_ll, Ny_ll, Nz)
    bm_cc = zeros(FT, Nx_ll, Ny_ll, Nz)

    # LatLon grid spacings
    Δlat_ll = abs(src_lats[2] - src_lats[1]) * FT(π) / 180
    Δlon_ll = abs(src_lons[2] - src_lons[1]) * FT(π) / 180
    Δy_ll = R_earth * Δlat_ll

    # Storage for all windows
    windows_data = Vector{NamedTuple}(undef, nwindow)

    for win in 1:nwindow
        t0 = time()

        # Load LatLon window: returns (m, ps, fluxes::StructuredFaceFluxState)
        m_ll, ps_ll, fluxes_ll = load_window!(reader, win)
        am_ll = fluxes_ll.am  # (Nx+1, Ny, Nz)
        bm_ll = fluxes_ll.bm  # (Nx, Ny+1, Nz)

        # Regrid cell mass and surface pressure to CS panels
        for p in 1:6
            bilinear_interp_3d!(m_panels[p], m_ll,
                                cell_lons[p], cell_lats[p], src_lons, src_lats)
            bilinear_interp_2d!(ps_panels[p], ps_ll,
                                cell_lons[p], cell_lats[p], src_lons, src_lats)
        end

        # Build cell-center flux density [kg/m/substep] from LatLon face fluxes.
        # Average adjacent face values to cell centers, divide by edge length.
        @inbounds for k in 1:Nz, j in 1:Ny_ll, i in 1:Nx_ll
            am_cc[i, j, k] = FT(0.5) * (am_ll[i, j, k] + am_ll[i+1, j, k]) / Δy_ll
        end
        @inbounds for k in 1:Nz, j in 1:Ny_ll, i in 1:Nx_ll
            cos_lat = cosd(src_lats[j])
            Δx_ll = R_earth * Δlon_ll * max(cos_lat, FT(1e-6))
            bm_cc[i, j, k] = FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, min(j+1, Ny_ll+1), k]) / Δx_ll
        end

        # Interpolate flux densities to CS face positions, scale by CS edge lengths
        for p in 1:6
            u_xface = zeros(FT, Nc+1, Nc, Nz)
            bilinear_interp_3d!(u_xface, am_cc,
                                xface_lons[p], xface_lats[p], src_lons, src_lats)
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:(Nc+1)
                am_panels[p][i, j, k] = u_xface[i, j, k] * Δy[min(i, Nc), j]
            end

            v_yface = zeros(FT, Nc, Nc+1, Nz)
            bilinear_interp_3d!(v_yface, bm_cc,
                                yface_lons[p], yface_lats[p], src_lons, src_lats)
            @inbounds for k in 1:Nz, j in 1:(Nc+1), i in 1:Nc
                bm_panels[p][i, j, k] = v_yface[i, j, k] * Δx[i, min(j, Nc)]
            end
        end

        # Load next window's mass for dm computation
        if win < nwindow
            m_next_ll, _, _ = load_window!(reader, win + 1)
        else
            m_next_ll = m_ll  # last window: dm = 0
        end

        # Diagnose cm from continuity on each panel
        for p in 1:6
            # Compute dm = (m_next - m_curr) / (2 * steps_per_window) on CS
            m_next_p = zeros(FT, Nc, Nc, Nz)
            bilinear_interp_3d!(m_next_p, m_next_ll,
                                cell_lons[p], cell_lats[p], src_lons, src_lats)
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                dm_panel[i, j, k] = (m_next_p[i, j, k] - m_panels[p][i, j, k]) / (2 * steps_per_window)
            end
            diagnose_cm!(cm_panels[p], am_panels[p], bm_panels[p], dm_panel, m_panels[p], Nc, Nz)
        end

        # Store window data
        windows_data[win] = (
            m  = ntuple(p -> copy(m_panels[p]), 6),
            ps = ntuple(p -> copy(ps_panels[p]), 6),
            am = ntuple(p -> copy(am_panels[p]), 6),
            bm = ntuple(p -> copy(bm_panels[p]), 6),
            cm = ntuple(p -> copy(cm_panels[p]), 6),
        )

        elapsed = time() - t0
        @printf("  Window %2d/%d: %.1fs\n", win, nwindow, elapsed)
    end

    close(reader.io)

    # --- Write CS binary ---
    println("\n[4/4] Writing CS transport binary...")

    payload_sections = [:m, :am, :bm, :cm, :ps]
    n_m  = 6 * Nc * Nc * Nz
    n_am = 6 * (Nc + 1) * Nc * Nz
    n_bm = 6 * Nc * (Nc + 1) * Nz
    n_cm = 6 * Nc * Nc * (Nz + 1)
    n_ps = 6 * Nc * Nc
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps

    header = Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 1,
        "header_bytes" => 65536,
        "float_type" => "Float64",
        "float_bytes" => 8,
        "grid_type" => "cubed_sphere",
        "horizontal_topology" => "StructuredDirectional",
        "ncell" => 6 * Nc * Nc,
        "nface_h" => 6 * 2 * Nc * (Nc + 1),
        "nlevel" => Nz,
        "nwindow" => nwindow,
        "dt_met_seconds" => h.dt_met_seconds,
        "half_dt_seconds" => h.half_dt_seconds,
        "steps_per_window" => steps_per_window,
        "source_flux_sampling" => String(h.source_flux_sampling),
        "air_mass_sampling" => String(h.air_mass_sampling),
        "flux_sampling" => "window_constant",
        "flux_kind" => String(h.flux_kind),
        "humidity_sampling" => "none",
        "delta_semantics" => "none",
        "mass_basis" => String(h.mass_basis),
        "poisson_balance_target_scale" => 1.0 / (2 * steps_per_window),
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
        "A_ifc" => A_ifc,
        "B_ifc" => B_ifc,
        "payload_sections" => String.(payload_sections),
        "elems_per_window" => elems_per_window,
        "include_qv" => false,
        "include_qv_endpoints" => false,
        "include_flux_delta" => false,
        "n_qv" => 0,
        "n_qv_start" => 0,
        "n_qv_end" => 0,
        "n_geometry_elems" => 0,
        "Nc" => Nc,
        "npanel" => 6,
        "Hp" => 0,
        "panel_convention" => "GEOSFP_file",
        "n_m" => n_m,
        "n_am" => n_am,
        "n_bm" => n_bm,
        "n_cm" => n_cm,
        "n_ps" => n_ps,
        "Nx" => Nc,  # for reader compatibility (not real LL coords)
        "Ny" => Nc,
        # Dummy lons/lats to satisfy reader parser (CS doesn't have global lon/lat)
        "lons" => collect(range(0.0, 360.0 - 360.0/Nc, length=Nc)),
        "lats" => collect(range(-90.0 + 90.0/Nc, 90.0 - 90.0/Nc, length=Nc)),
        "longitude_interval" => [0.0, 360.0],
        "latitude_interval" => [-90.0, 90.0],
        "source_binary" => input_path,
        "regrid_method" => "bilinear_cell_center",
        "creation_time" => string(Dates.now()),
    )

    header_json = JSON3.write(header)
    header_bytes = 65536
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("Header exceeds $header_bytes bytes (need $(ncodeunits(header_json)))")

    open(output_path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))

        # Write payload: for each window, panels concatenated per section
        payload = Vector{FT}(undef, elems_per_window)
        for win in 1:nwindow
            wd = windows_data[win]
            offset = 0

            # m: 6 panels of (Nc, Nc, Nz)
            for p in 1:6
                n = Nc * Nc * Nz
                copyto!(payload, offset + 1, vec(wd.m[p]), 1, n)
                offset += n
            end

            # am: 6 panels of (Nc+1, Nc, Nz)
            for p in 1:6
                n = (Nc + 1) * Nc * Nz
                copyto!(payload, offset + 1, vec(wd.am[p]), 1, n)
                offset += n
            end

            # bm: 6 panels of (Nc, Nc+1, Nz)
            for p in 1:6
                n = Nc * (Nc + 1) * Nz
                copyto!(payload, offset + 1, vec(wd.bm[p]), 1, n)
                offset += n
            end

            # cm: 6 panels of (Nc, Nc, Nz+1)
            for p in 1:6
                n = Nc * Nc * (Nz + 1)
                copyto!(payload, offset + 1, vec(wd.cm[p]), 1, n)
                offset += n
            end

            # ps: 6 panels of (Nc, Nc)
            for p in 1:6
                n = Nc * Nc
                copyto!(payload, offset + 1, vec(wd.ps[p]), 1, n)
                offset += n
            end

            @assert offset == elems_per_window
            write(io, payload)
        end
    end

    filesize_gb = filesize(output_path) / 1e9
    println("  Written: $output_path ($(round(filesize_gb, digits=2)) GB)")
    println("  Header: $header_bytes bytes")
    println("  Payload: $nwindow windows × $elems_per_window elements × 8 bytes")
    println("\nDone.")
end

using Dates
main()
