#!/usr/bin/env julia
#
# Preprocess ERA5 LatLon transport binary → CS transport binary via
# true conservative (spherical polygon intersection) regridding.
#
# This is the conservative-regridding counterpart to
# regrid_latlon_to_cs_binary_v2.jl (which uses bilinear cell-center
# interpolation). All cell-center fields (m, ps, recovered winds) go
# through ConservativeRegridding.jl's spherical polygon intersection
# instead of bilinear cell-center sampling. Face mass fluxes on the
# destination grid are then recomputed from the regridded winds +
# dp, Poisson-balanced against dm_dt, and cm is diagnosed by
# continuity — same structure as the bilinear script.
#
# Mass-conservation guarantee: for the `m` field, the source total
# air mass `sum(m_ll .* src_areas)` equals the destination total
# `sum(m_panels .* dst_areas)` to machine precision. Winds are
# intensive quantities so conservative regridding gives area-weighted
# averages (the desired behavior).
#
# Panel convention: forced to GnomonicPanelConvention so the output
# binary layout matches regrid_latlon_to_cs_binary_v2.jl exactly and
# is drop-in compatible with existing CS consumers.
#
# Usage:
#   julia --project=. scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl \
#       --input <latlon_binary.bin> --output <cs_binary.bin> \
#       --Nc 90 [--cache-dir <path>]
#
# See: /home/cfranken/.claude/plans/luminous-prancing-firefly.md (Tier 3)

using Printf
using Dates
using JSON3
using FFTW

include(joinpath(@__DIR__, "..", "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2.Regridding

# ---------------------------------------------------------------------------
# Helpers duplicated from regrid_latlon_to_cs_binary_v2.jl
#
# These are verbatim copies of the bilinear script's downstream helpers
# (face-flux recomputation from winds + dp, Poisson balancing, cm
# diagnosis). Copying instead of `include`ing because the bilinear
# script is a standalone entry point with `main()` and cannot be
# imported as a module.
# ---------------------------------------------------------------------------

"""
    balance_panel_mass_fluxes!(am, bm, dm_dt, Nc, Nz)

FFT-based Poisson balance of am/bm on one CS panel so horizontal flux
convergence matches the prescribed mass tendency `dm_dt`. Panels are
treated as doubly-periodic in gnomonic coordinates — approximate at
panel boundaries but correct in the interior.
"""
function balance_panel_mass_fluxes!(am::Array{FT, 3}, bm::Array{FT, 3},
                                    dm_dt::Array{FT, 3}, Nc::Int, Nz::Int;
                                    n_iterations::Int = 5) where FT
    fac = Array{Float64}(undef, Nc, Nc)
    @inbounds for j in 1:Nc, i in 1:Nc
        fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nc) + cos(2π * (j - 1) / Nc) - 2.0)
    end
    fac[1, 1] = 1.0

    residual = Array{Float64}(undef, Nc, Nc)
    psi      = Array{Float64}(undef, Nc, Nc)

    for _ in 1:n_iterations
        for k in 1:Nz
            @inbounds for j in 1:Nc, i in 1:Nc
                conv = (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
                       (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k]))
                residual[i, j] = conv - Float64(dm_dt[i, j, k])
            end
            maximum(abs, residual) < 1e-10 && continue

            A = FFTW.fft(complex.(residual))
            @inbounds for j in 1:Nc, i in 1:Nc
                A[i, j] /= fac[i, j]
            end
            A[1, 1] = 0.0 + 0.0im
            psi .= real.(FFTW.ifft(A))

            @inbounds for j in 1:Nc
                for i in 2:Nc
                    am[i, j, k] += FT(psi[i, j] - psi[i - 1, j])
                end
                am[1,      j, k] += FT(psi[1, j] - psi[Nc, j])
                am[Nc + 1, j, k] += FT(psi[1, j] - psi[Nc, j])
            end
            @inbounds for i in 1:Nc
                for j in 2:Nc
                    bm[i, j, k] += FT(psi[i, j] - psi[i, j - 1])
                end
                bm[i, 1,      k] += FT(psi[i, 1] - psi[i, Nc])
                bm[i, Nc + 1, k] += FT(psi[i, 1] - psi[i, Nc])
            end
        end
    end
    return nothing
end

"""
    diagnose_cm!(cm, am, bm, dm, m, Nc, Nz; max_cfl=40.0)

Diagnose vertical mass flux `cm` from horizontal flux divergence and
mass tendency. Column residual at the surface is redistributed
proportionally to cell mass. Max vertical CFL is capped at `max_cfl`
by uniform column scaling.
"""
function diagnose_cm!(cm, am, bm, dm, m, Nc, Nz; max_cfl::Float64 = 40.0)
    @inbounds for j in 1:Nc, i in 1:Nc
        cm[i, j, 1] = 0.0
        for k in 1:Nz
            div_h = am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
            cm[i, j, k + 1] = cm[i, j, k] + div_h - dm[i, j, k]
        end

        worst_cfl = 0.0
        for k in 2:Nz
            m_thin = min(m[i, j, k - 1], m[i, j, k])
            m_thin > 0 || continue
            worst_cfl = max(worst_cfl, abs(cm[i, j, k]) / m_thin)
        end
        if worst_cfl > max_cfl
            scale = max_cfl / worst_cfl
            for k in 2:Nz + 1
                cm[i, j, k] *= scale
            end
        end

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
# Flat ↔ per-panel helpers
#
# The regridder output is a flat vector of 6 × Nc² cells (column-major
# per-panel, panels concatenated in gnomonic order 1..6). Transport
# binaries expect (Nc, Nc, Nz) per-panel arrays, so we shuttle between
# representations via reshape + view.
# ---------------------------------------------------------------------------

@inline function _panel_flat_range(p::Int, Nc::Int)
    return (p - 1) * Nc * Nc + 1 : p * Nc * Nc
end

"""
    unpack_panels_3d!(panels, flat, Nc, Nz)

Copy a flat `(6·Nc², Nz)` matrix into 6 per-panel `(Nc, Nc, Nz)` arrays.
"""
function unpack_panels_3d!(panels::NTuple{6, Array{FT, 3}},
                           flat::AbstractMatrix{FT}, Nc::Int, Nz::Int) where FT
    for p in 1:6
        r = _panel_flat_range(p, Nc)
        for k in 1:Nz
            @inbounds for (linear, flat_idx) in enumerate(r)
                j, i = fldmod1(linear, Nc)
                panels[p][i, j, k] = flat[flat_idx, k]
            end
        end
    end
    return panels
end

"""
    unpack_panels_2d!(panels, flat, Nc)

Copy a flat `6·Nc²` vector into 6 per-panel `(Nc, Nc)` arrays.
"""
function unpack_panels_2d!(panels::NTuple{6, Matrix{FT}},
                           flat::AbstractVector{FT}, Nc::Int) where FT
    for p in 1:6
        r = _panel_flat_range(p, Nc)
        @inbounds for (linear, flat_idx) in enumerate(r)
            j, i = fldmod1(linear, Nc)
            panels[p][i, j] = flat[flat_idx]
        end
    end
    return panels
end

# ---------------------------------------------------------------------------
# Argument parsing (tiny)
# ---------------------------------------------------------------------------

function parse_args(argv)
    args = Dict{String, String}()
    i = 1
    while i <= length(argv)
        if startswith(argv[i], "--")
            key = argv[i][3:end]
            val = i < length(argv) ? argv[i + 1] : ""
            args[key] = val
            i += 2
        else
            i += 1
        end
    end
    return args
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    args = parse_args(ARGS)
    input_path  = get(args, "input", "")
    output_path = get(args, "output", "")
    Nc          = parse(Int, get(args, "Nc", "90"))
    cache_dir   = get(args, "cache-dir",
                      joinpath(homedir(), ".cache", "AtmosTransport", "cr_regridding"))

    isempty(input_path)  && error("--input <latlon_binary.bin> required")
    isempty(output_path) && error("--output <cs_binary.bin> required")
    isfile(input_path)   || error("Input file not found: $input_path")

    FT       = Float64
    g        = 9.80665
    R_earth  = 6.371e6

    println("=" ^ 72)
    println("ERA5 LatLon → C$Nc (CR.jl conservative regridding)")
    println("  Input:     $input_path")
    println("  Output:    $output_path")
    println("  Cache dir: $cache_dir")
    println("=" ^ 72)

    # --- Load LatLon binary header ---
    println("\n[1/5] Reading LatLon binary header...")
    reader  = TransportBinaryReader(input_path; FT = FT)
    h       = reader.header
    Nx_ll   = h.Nx
    Ny_ll   = h.Ny
    Nz      = h.nlevel
    nwindow = h.nwindow
    src_lons = FT.(h.lons_f64)
    src_lats = FT.(h.lats_f64)
    A_ifc    = h.A_ifc
    B_ifc    = h.B_ifc
    println("  LatLon:        $(Nx_ll)×$(Ny_ll)×$(Nz), $nwindow windows")
    println("  Steps/window:  $(h.steps_per_window)  dt_met: $(h.dt_met_seconds)s")

    # --- Build meshes and regridder ---
    println("\n[2/5] Building src & dst meshes and (cached) regridder...")
    # Reconstruct the LatLon *face* vectors from the binary's cell-center list.
    # ERA5 0.5° / 0.25° binaries store cell centers in `lons_f64`, `lats_f64`.
    # Uniform-spacing reconstruction of face edges:
    dlon = src_lons[2] - src_lons[1]
    dlat = src_lats[2] - src_lats[1]
    lon_west  = src_lons[1]   - dlon / 2
    lon_east  = src_lons[end] + dlon / 2
    lat_south = src_lats[1]   - dlat / 2
    lat_north = src_lats[end] + dlat / 2
    # Clamp latitude to [-90, 90] for pole-adjacent cells (cell centers sit
    # inside the grid; half-Δ extension can slightly exceed the physical pole).
    lat_south = max(lat_south, -90.0)
    lat_north = min(lat_north,  90.0)

    src_mesh = LatLonMesh(
        FT = FT, Nx = Nx_ll, Ny = Ny_ll,
        longitude = (lon_west, lon_east),
        latitude  = (lat_south, lat_north),
        radius    = FT(R_earth),
    )
    # Force gnomonic panel convention so output binary layout matches the
    # existing bilinear regridder script bit-for-bit.
    dst_mesh = CubedSphereMesh(
        Nc = Nc, FT = FT,
        radius = FT(R_earth),
        convention = GnomonicPanelConvention(),
    )
    println("  src: ", summary(src_mesh))
    println("  dst: ", summary(dst_mesh))

    regridder = build_regridder(src_mesh, dst_mesh;
                                normalize = false,
                                cache_dir = cache_dir)
    nnz_count = length(regridder.intersections.nzval)
    println("  regridder: $(size(regridder.intersections, 1))×",
            "$(size(regridder.intersections, 2))  nnz=$nnz_count")

    # --- Build CS metric arrays (for face-flux reconstruction) ---
    Δx = dst_mesh.Δx
    Δy = dst_mesh.Δy

    # --- Preallocate per-window CS fields ---
    m_panels  = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     6)
    ps_panels = ntuple(_ -> zeros(FT, Nc, Nc),         6)
    am_panels = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), 6)
    bm_panels = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), 6)
    cm_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), 6)
    u_cs_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz),   6)
    v_cs_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz),   6)
    dm_panel  = zeros(FT, Nc, Nc, Nz)

    # --- Preallocate LatLon cell-center wind recovery arrays ---
    u_cc = zeros(FT, Nx_ll, Ny_ll, Nz)
    v_cc = zeros(FT, Nx_ll, Ny_ll, Nz)

    # --- Regridder scratch buffers ---
    n_src    = length(regridder.src_areas)
    n_dst    = length(regridder.dst_areas)
    src_flat_3d = zeros(FT, n_src, Nz)
    dst_flat_3d = zeros(FT, n_dst, Nz)
    src_flat_2d = zeros(FT, n_src)
    dst_flat_2d = zeros(FT, n_dst)

    # --- LatLon → per-panel CS: a single helper closure ---
    regrid_3d_panels! = function (out_panels, src_3d)
        # src_3d shape (Nx, Ny, Nz) → (n_src, Nz)
        copyto!(src_flat_3d, reshape(src_3d, n_src, Nz))
        apply_regridder!(dst_flat_3d, regridder, src_flat_3d)
        unpack_panels_3d!(out_panels, dst_flat_3d, Nc, Nz)
    end
    regrid_2d_panels! = function (out_panels, src_2d)
        copyto!(src_flat_2d, reshape(src_2d, n_src))
        apply_regridder!(dst_flat_2d, regridder, src_flat_2d)
        unpack_panels_2d!(out_panels, dst_flat_2d, Nc)
    end

    # LatLon grid spacings for wind recovery
    Δlat_ll  = abs(dlat) * π / 180
    Δlon_ll  = abs(dlon) * π / 180
    Δy_ll    = R_earth * Δlat_ll
    steps_per_window = h.steps_per_window
    dt_met    = h.dt_met_seconds
    dt_factor = dt_met / (2 * steps_per_window)

    # --- Window loop ---
    println("\n[3/5] Processing $nwindow windows (conservative regrid)...")
    windows_data = Vector{NamedTuple}(undef, nwindow)

    for win in 1:nwindow
        t0 = time()

        # Load source window
        m_ll, ps_ll, fluxes_ll = load_window!(reader, win)
        am_ll = fluxes_ll.am
        bm_ll = fluxes_ll.bm

        # ---- Conservative regrid: m (extensive, total mass preserved) ----
        regrid_3d_panels!(m_panels, m_ll)
        # ---- Conservative regrid: ps (intensive, area-weighted average) ----
        regrid_2d_panels!(ps_panels, ps_ll)

        # ---- Recover cell-center winds on LatLon from am/bm ----
        @inbounds for k in 1:Nz, j in 1:Ny_ll, i in 1:Nx_ll
            dp_ll = abs((A_ifc[k] - A_ifc[k + 1]) +
                        (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
            area_factor = Δy_ll * dp_ll / g * dt_factor
            u_cc[i, j, k] = area_factor > 1e-10 ?
                FT(0.5) * (am_ll[i, j, k] + am_ll[i + 1, j, k]) / area_factor :
                zero(FT)
        end
        @inbounds for k in 1:Nz, j in 1:Ny_ll, i in 1:Nx_ll
            cos_lat = cosd(src_lats[j])
            Δx_ll_loc = R_earth * Δlon_ll * max(cos_lat, FT(1e-6))
            dp_ll = abs((A_ifc[k] - A_ifc[k + 1]) +
                        (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
            area_factor = Δx_ll_loc * dp_ll / g * dt_factor
            jn = min(j + 1, Ny_ll + 1)
            v_cc[i, j, k] = area_factor > 1e-10 ?
                FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, jn, k]) / area_factor :
                zero(FT)
        end

        # ---- Conservative regrid recovered winds to CS cell centers ----
        regrid_3d_panels!(u_cs_panels, u_cc)
        regrid_3d_panels!(v_cs_panels, v_cc)

        # ---- Build CS dp from regridded ps and compute CS face fluxes ----
        for p in 1:6
            dp_cs = zeros(FT, Nc, Nc, Nz)
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                dp_cs[i, j, k] = abs((A_ifc[k] - A_ifc[k + 1]) +
                                     (B_ifc[k] - B_ifc[k + 1]) * ps_panels[p][i, j])
            end
            # am: x-face, bm: y-face (identical formulation to bilinear script)
            @inbounds for k in 1:Nz, j in 1:Nc
                am_panels[p][1, j, k] = u_cs_panels[p][1, j, k] *
                                         dp_cs[1, j, k] * Δy[1, j] / g * dt_factor
                for i in 2:Nc
                    u_face  = FT(0.5) * (u_cs_panels[p][i - 1, j, k] + u_cs_panels[p][i, j, k])
                    dp_face = FT(0.5) * (dp_cs[i - 1, j, k] + dp_cs[i, j, k])
                    am_panels[p][i, j, k] = u_face * dp_face * Δy[i, j] / g * dt_factor
                end
                am_panels[p][Nc + 1, j, k] = u_cs_panels[p][Nc, j, k] *
                                              dp_cs[Nc, j, k] * Δy[Nc, j] / g * dt_factor
            end
            @inbounds for k in 1:Nz, i in 1:Nc
                bm_panels[p][i, 1, k] = v_cs_panels[p][i, 1, k] *
                                         dp_cs[i, 1, k] * Δx[i, 1] / g * dt_factor
                for j in 2:Nc
                    v_face  = FT(0.5) * (v_cs_panels[p][i, j - 1, k] + v_cs_panels[p][i, j, k])
                    dp_face = FT(0.5) * (dp_cs[i, j - 1, k] + dp_cs[i, j, k])
                    bm_panels[p][i, j, k] = v_face * dp_face * Δx[i, j] / g * dt_factor
                end
                bm_panels[p][i, Nc + 1, k] = v_cs_panels[p][i, Nc, k] *
                                              dp_cs[i, Nc, k] * Δx[i, Nc] / g * dt_factor
            end
        end

        # ---- Next-window mass for dm_dt ----
        if win < nwindow
            m_next_ll, _, _ = load_window!(reader, win + 1)
        else
            m_next_ll = m_ll  # last window: dm_dt = 0
        end
        m_next_panels = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
        regrid_3d_panels!(m_next_panels, m_next_ll)

        # ---- Poisson balance am/bm + diagnose cm per panel ----
        for p in 1:6
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                dm_panel[i, j, k] = (m_next_panels[p][i, j, k] - m_panels[p][i, j, k]) /
                                    (2 * steps_per_window)
            end
            balance_panel_mass_fluxes!(am_panels[p], bm_panels[p], dm_panel, Nc, Nz)
            diagnose_cm!(cm_panels[p], am_panels[p], bm_panels[p], dm_panel, m_panels[p], Nc, Nz)
        end

        # ---- Mass-conservation sanity: global src mass vs global dst mass ----
        if win == 1
            src_total = sum(m_ll .* reshape(regridder.src_areas, Nx_ll, Ny_ll))
            dst_total = 0.0
            for p in 1:6
                @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                    global_idx = (p - 1) * Nc * Nc + i + (j - 1) * Nc
                    dst_total += m_panels[p][i, j, k] * regridder.dst_areas[global_idx]
                end
            end
            rel_err = abs(dst_total - src_total) / max(abs(src_total), 1e-30)
            @printf("  [conservation] window 1: src=%.6e  dst=%.6e  rel_err=%.2e\n",
                    src_total, dst_total, rel_err)
        end

        # ---- Store window ----
        windows_data[win] = (
            m  = ntuple(p -> copy(m_panels[p]),  6),
            ps = ntuple(p -> copy(ps_panels[p]), 6),
            am = ntuple(p -> copy(am_panels[p]), 6),
            bm = ntuple(p -> copy(bm_panels[p]), 6),
            cm = ntuple(p -> copy(cm_panels[p]), 6),
        )

        @printf("  Window %2d/%d: %.1fs\n", win, nwindow, time() - t0)
    end

    close(reader.io)

    # --- Write CS transport binary ---
    println("\n[4/5] Writing CS transport binary...")
    payload_sections = [:m, :am, :bm, :cm, :ps]
    n_m  = 6 * Nc * Nc * Nz
    n_am = 6 * (Nc + 1) * Nc * Nz
    n_bm = 6 * Nc * (Nc + 1) * Nz
    n_cm = 6 * Nc * Nc * (Nz + 1)
    n_ps = 6 * Nc * Nc
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps

    header = Dict{String, Any}(
        "magic"                               => "MFLX",
        "format_version"                      => 1,
        "header_bytes"                        => 65536,
        "float_type"                          => "Float64",
        "float_bytes"                         => 8,
        "grid_type"                           => "cubed_sphere",
        "horizontal_topology"                 => "StructuredDirectional",
        "ncell"                               => 6 * Nc * Nc,
        "nface_h"                             => 6 * 2 * Nc * (Nc + 1),
        "nlevel"                              => Nz,
        "nwindow"                             => nwindow,
        "dt_met_seconds"                      => h.dt_met_seconds,
        "half_dt_seconds"                     => h.half_dt_seconds,
        "steps_per_window"                    => steps_per_window,
        "source_flux_sampling"                => String(h.source_flux_sampling),
        "air_mass_sampling"                   => String(h.air_mass_sampling),
        "flux_sampling"                       => "window_constant",
        "flux_kind"                           => String(h.flux_kind),
        "humidity_sampling"                   => "none",
        "delta_semantics"                     => "none",
        "mass_basis"                          => String(h.mass_basis),
        "poisson_balance_target_scale"        => 1.0 / (2 * steps_per_window),
        "poisson_balance_target_semantics"    => "forward_window_mass_difference / (2 * steps_per_window)",
        "A_ifc"                               => A_ifc,
        "B_ifc"                               => B_ifc,
        "payload_sections"                    => String.(payload_sections),
        "elems_per_window"                    => elems_per_window,
        "include_qv"                          => false,
        "include_qv_endpoints"                => false,
        "include_flux_delta"                  => false,
        "n_qv"                                => 0,
        "n_qv_start"                          => 0,
        "n_qv_end"                            => 0,
        "n_geometry_elems"                    => 0,
        "Nc"                                  => Nc,
        "npanel"                              => 6,
        "Hp"                                  => 0,
        "panel_convention"                    => "Gnomonic",
        "n_m"                                 => n_m,
        "n_am"                                => n_am,
        "n_bm"                                => n_bm,
        "n_cm"                                => n_cm,
        "n_ps"                                => n_ps,
        "Nx"                                  => Nc,
        "Ny"                                  => Nc,
        "lons"                                => collect(range(0.0, 360.0 - 360.0 / Nc, length = Nc)),
        "lats"                                => collect(range(-90.0 + 90.0 / Nc, 90.0 - 90.0 / Nc, length = Nc)),
        "longitude_interval"                  => [0.0, 360.0],
        "latitude_interval"                   => [-90.0, 90.0],
        "source_binary"                       => input_path,
        "regrid_method"                       => "conservative_crjl",
        "regridder_nnz"                       => nnz_count,
        "regridder_cache_dir"                 => cache_dir,
        "creation_time"                       => string(Dates.now()),
    )

    header_json  = JSON3.write(header)
    header_bytes = 65536
    pad          = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("Header exceeds $header_bytes bytes (need $(ncodeunits(header_json)))")

    open(output_path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))

        payload = Vector{FT}(undef, elems_per_window)
        for win in 1:nwindow
            wd = windows_data[win]
            offset = 0
            for p in 1:6
                n = Nc * Nc * Nz
                copyto!(payload, offset + 1, vec(wd.m[p]), 1, n);   offset += n
            end
            for p in 1:6
                n = (Nc + 1) * Nc * Nz
                copyto!(payload, offset + 1, vec(wd.am[p]), 1, n);  offset += n
            end
            for p in 1:6
                n = Nc * (Nc + 1) * Nz
                copyto!(payload, offset + 1, vec(wd.bm[p]), 1, n);  offset += n
            end
            for p in 1:6
                n = Nc * Nc * (Nz + 1)
                copyto!(payload, offset + 1, vec(wd.cm[p]), 1, n);  offset += n
            end
            for p in 1:6
                n = Nc * Nc
                copyto!(payload, offset + 1, vec(wd.ps[p]), 1, n);  offset += n
            end
            @assert offset == elems_per_window
            write(io, payload)
        end
    end

    filesize_gb = filesize(output_path) / 1e9
    println("  Written: $output_path  (", round(filesize_gb, digits = 2), " GB)")
    println("\n[5/5] Done.")
end

main()
