# Bilinear LatLon -> cubed-sphere transport-binary target.
#
# Audience:
# - users normally call the CLI wrapper or
#   `build_transport_binary_v2_target(:cubed_sphere_bilinear, ...)`
# - contributors can use this as the second concrete implementation of the
#   stable transport-binary v2 target interface
#
# Stability: the target type and builder are stable in-repo API. The helper
# functions below remain private implementation details.

using Dates
using Printf
using JSON3
using FFTW

using .AtmosTransport

const CS_BILINEAR_HEADER_BYTES = 65536
const CS_BILINEAR_PANEL_COUNT = 6

"""
    CubedSphereBilinearTransportBinaryTarget

Stable target for bilinear LatLon -> cubed-sphere transport-binary
regridding.

This is the fast, non-conservative counterpart to
`CubedSphereConservativeTransportBinaryTarget`. It preserves the historical
cell-center bilinear interpolation path and reconstructs CS face fluxes from
recovered winds plus local pressure thickness.
"""
struct CubedSphereBilinearTransportBinaryTarget{FT <: AbstractFloat} <: AbstractTransportBinaryV2Target
    input_path :: String
    output_path :: String
    Nc :: Int
    radius :: FT
    gravity :: FT
end

target_input_path(target::CubedSphereBilinearTransportBinaryTarget) = target.input_path
target_output_path(target::CubedSphereBilinearTransportBinaryTarget) = target.output_path
target_float_type(::CubedSphereBilinearTransportBinaryTarget{FT}) where FT = FT
target_summary(target::CubedSphereBilinearTransportBinaryTarget) =
    "ERA5 LatLon -> C$(target.Nc) (bilinear cell-center regridding)"

"""
    build_transport_binary_v2_target(::Val{:cubed_sphere_bilinear}, argv; FT=Float64)

Internal dispatch entry backing
`build_transport_binary_v2_target(:cubed_sphere_bilinear, argv; FT=...)`.

CLI example:
```bash
julia --project=. scripts/preprocessing/regrid_latlon_to_cs_binary_v2.jl \
    --input era5_latlon.bin --output era5_cs.bin --Nc 90
```
"""
function build_transport_binary_v2_target(::Val{:cubed_sphere_bilinear},
                                          argv::AbstractVector{<:AbstractString};
                                          FT::Type{T} = Float64) where T <: AbstractFloat
    args = parse_flag_args(argv)
    input_path = _require_flag(args, "input")
    output_path = _require_flag(args, "output")
    Nc = parse(Int, get(args, "Nc", "90"))
    Nc > 0 || error("--Nc must be positive, got $Nc")

    return CubedSphereBilinearTransportBinaryTarget{T}(
        input_path,
        output_path,
        Nc,
        T(6.371e6),
        T(9.80665),
    )
end

struct CubedSphereBilinearWorkspace{FT}
    m_panels :: NTuple{CS_BILINEAR_PANEL_COUNT, Array{FT, 3}}
    ps_panels :: NTuple{CS_BILINEAR_PANEL_COUNT, Matrix{FT}}
    am_panels :: NTuple{CS_BILINEAR_PANEL_COUNT, Array{FT, 3}}
    bm_panels :: NTuple{CS_BILINEAR_PANEL_COUNT, Array{FT, 3}}
    cm_panels :: NTuple{CS_BILINEAR_PANEL_COUNT, Array{FT, 3}}
    dm_panel :: Array{FT, 3}
    am_cc :: Array{FT, 3}
    bm_cc :: Array{FT, 3}
    u_cs :: Array{FT, 3}
    v_cs :: Array{FT, 3}
    dp_cs :: Array{FT, 3}
    m_next_panel :: Array{FT, 3}
    poisson_fac :: Matrix{Float64}
    poisson_residual :: Matrix{Float64}
    poisson_psi :: Matrix{Float64}
end

struct CubedSphereBilinearContext{FT, M}
    mesh :: M
    cell_lons :: NTuple{CS_BILINEAR_PANEL_COUNT, Matrix{FT}}
    cell_lats :: NTuple{CS_BILINEAR_PANEL_COUNT, Matrix{FT}}
    src_lons :: Vector{FT}
    src_lats :: Vector{FT}
    A_ifc :: Vector{FT}
    B_ifc :: Vector{FT}
    Nx_ll :: Int
    Ny_ll :: Int
    Nz :: Int
    Nc :: Int
    nwindow :: Int
    steps_per_window :: Int
    dt_factor :: FT
    Δx :: Matrix{FT}
    Δy :: Matrix{FT}
    Δlon_ll :: FT
    Δy_ll :: FT
    workspace :: CubedSphereBilinearWorkspace{FT}
end

@inline function _copy_panel_field_tuple_bilinear(panels)
    return ntuple(p -> copy(panels[p]), CS_BILINEAR_PANEL_COUNT)
end

function _cs_bilinear_allocate_workspace(Nc::Int, Nx_ll::Int, Ny_ll::Int, Nz::Int,
                                         ::Type{FT}) where FT
    poisson_fac = Array{Float64}(undef, Nc, Nc)
    @inbounds for j in 1:Nc, i in 1:Nc
        poisson_fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nc) + cos(2π * (j - 1) / Nc) - 2.0)
    end
    poisson_fac[1, 1] = 1.0

    return CubedSphereBilinearWorkspace(
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_BILINEAR_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc), CS_BILINEAR_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_BILINEAR_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_BILINEAR_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_BILINEAR_PANEL_COUNT),
        zeros(FT, Nc, Nc, Nz),
        zeros(FT, Nx_ll, Ny_ll, Nz),
        zeros(FT, Nx_ll, Ny_ll, Nz),
        zeros(FT, Nc, Nc, Nz),
        zeros(FT, Nc, Nc, Nz),
        zeros(FT, Nc, Nc, Nz),
        zeros(FT, Nc, Nc, Nz),
        poisson_fac,
        zeros(Float64, Nc, Nc),
        zeros(Float64, Nc, Nc),
    )
end

"""Compute `(lon, lat)` in degrees for all gnomonic cell centers on panel `p`."""
function _cs_bilinear_panel_cell_centers(Nc::Int, p::Int; FT=Float64)
    dα = FT(π) / (2 * Nc)
    α_centers = [FT(-π / 4) + (i - 0.5) * dα for i in 1:Nc]

    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        ξ = tan(α_centers[i])
        η = tan(α_centers[j])
        d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
        x, y, z = if p == 1
            (d, ξ * d, η * d)
        elseif p == 2
            (-ξ * d, d, η * d)
        elseif p == 3
            (-d, -ξ * d, η * d)
        elseif p == 4
            (ξ * d, -d, η * d)
        elseif p == 5
            (-η * d, ξ * d, d)
        else
            (η * d, ξ * d, -d)
        end
        lons[i, j] = atand(y, x)
        lats[i, j] = asind(z / sqrt(x^2 + y^2 + z^2))
        lons[i, j] < 0 && (lons[i, j] += 360)
    end
    return lons, lats
end

function _cs_bilinear_interp_3d!(dst::AbstractArray{FT, 3},
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
        lon = mod(dst_lons[i, j] - lon0, FT(360))
        fi = lon / dlon + one(FT)
        lat = dst_lats[i, j]
        fj = (lat - lat0) / dlat + one(FT)

        i0 = floor(Int, fi)
        j0 = floor(Int, fj)
        wx = fi - i0
        wy = fj - j0

        i0 = mod1(i0, Nx_src)
        i1 = mod1(i0 + 1, Nx_src)
        j0 = clamp(j0, 1, Ny_src)
        j1 = clamp(j0 + 1, 1, Ny_src)
        j0 == j1 && (wy = zero(FT))

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

function _cs_bilinear_interp_2d!(dst::AbstractMatrix{FT},
                                 src::AbstractMatrix{FT},
                                 dst_lons::AbstractMatrix{FT},
                                 dst_lats::AbstractMatrix{FT},
                                 src_lons::AbstractVector{FT},
                                 src_lats::AbstractVector{FT}) where FT
    src_3d = reshape(src, size(src, 1), size(src, 2), 1)
    dst_3d = reshape(dst, size(dst, 1), size(dst, 2), 1)
    _cs_bilinear_interp_3d!(dst_3d, src_3d, dst_lons, dst_lats, src_lons, src_lats)
    return dst
end

function _cs_bilinear_balance_panel_mass_fluxes!(am::AbstractArray{FT, 3},
                                                 bm::AbstractArray{FT, 3},
                                                 dm_dt::AbstractArray{FT, 3},
                                                 fac::AbstractMatrix{Float64},
                                                 residual::AbstractMatrix{Float64},
                                                 psi::AbstractMatrix{Float64},
                                                 Nc::Int,
                                                 Nz::Int;
                                                 n_iterations::Int = 5) where FT
    for iter in 1:n_iterations
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
                am[1, j, k] += FT(psi[1, j] - psi[Nc, j])
                am[Nc + 1, j, k] += FT(psi[1, j] - psi[Nc, j])
            end

            @inbounds for i in 1:Nc
                for j in 2:Nc
                    bm[i, j, k] += FT(psi[i, j] - psi[i, j - 1])
                end
                bm[i, 1, k] += FT(psi[i, 1] - psi[i, Nc])
                bm[i, Nc + 1, k] += FT(psi[i, 1] - psi[i, Nc])
            end
        end
    end
    return nothing
end

function _cs_bilinear_max_balance_residual(am::AbstractArray{FT, 3},
                                           bm::AbstractArray{FT, 3},
                                           dm_dt::AbstractArray{FT, 3},
                                           Nc::Int,
                                           Nz::Int) where FT
    max_residual = 0.0
    @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
        conv = (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
               (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k]))
        max_residual = max(max_residual, abs(conv - Float64(dm_dt[i, j, k])))
    end
    return max_residual
end

function _cs_bilinear_diagnose_cm!(cm::AbstractArray{FT, 3},
                                   am::AbstractArray{FT, 3},
                                   bm::AbstractArray{FT, 3},
                                   dm::AbstractArray{FT, 3},
                                   m::AbstractArray{FT, 3},
                                   Nc::Int,
                                   Nz::Int;
                                   max_cfl::Float64 = 40.0) where FT
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
            cfl_k = abs(cm[i, j, k]) / m_thin
            worst_cfl = max(worst_cfl, cfl_k)
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

function prepare_transport_binary_v2_target(target::CubedSphereBilinearTransportBinaryTarget{FT},
                                            reader::TransportBinaryReader{FT}) where FT
    h = reader.header
    println("=" ^ 72)
    println(target_summary(target))
    println("  Input:  ", target.input_path)
    println("  Output: ", target.output_path)
    println("=" ^ 72)

    println("\n[1/4] Reading source transport binary header...")
    Nx_ll = h.Nx
    Ny_ll = h.Ny
    Nz = h.nlevel
    println("  LatLon: $(Nx_ll)×$(Ny_ll)×$(Nz), $(h.nwindow) windows")
    println("  Steps/window: $(h.steps_per_window), dt_met: $(h.dt_met_seconds)s")

    println("\n[2/4] Building C$(target.Nc) cubed-sphere geometry...")
    mesh = CubedSphereMesh(Nc=target.Nc, FT=FT)
    centers = ntuple(p -> _cs_bilinear_panel_cell_centers(target.Nc, p; FT=FT),
                     CS_BILINEAR_PANEL_COUNT)
    cell_lons = ntuple(p -> centers[p][1], CS_BILINEAR_PANEL_COUNT)
    cell_lats = ntuple(p -> centers[p][2], CS_BILINEAR_PANEL_COUNT)
    println("  C$(target.Nc): $(CS_BILINEAR_PANEL_COUNT)×$(target.Nc)×$(target.Nc)×$(Nz) cells")
    println("  Δx range: $(round(minimum(mesh.Δx)/1e3, digits=1))–$(round(maximum(mesh.Δx)/1e3, digits=1)) km")

    src_lons = FT.(h.lons_f64)
    src_lats = FT.(h.lats_f64)
    Δlat_ll = abs(src_lats[2] - src_lats[1]) * FT(π) / 180
    Δlon_ll = abs(src_lons[2] - src_lons[1]) * FT(π) / 180
    Δy_ll = target.radius * Δlat_ll

    workspace = _cs_bilinear_allocate_workspace(target.Nc, Nx_ll, Ny_ll, Nz, FT)
    dt_factor = FT(h.dt_met_seconds / (2 * h.steps_per_window))

    return CubedSphereBilinearContext(
        mesh,
        cell_lons,
        cell_lats,
        src_lons,
        src_lats,
        FT.(h.A_ifc),
        FT.(h.B_ifc),
        Nx_ll,
        Ny_ll,
        Nz,
        target.Nc,
        h.nwindow,
        h.steps_per_window,
        dt_factor,
        mesh.Δx,
        mesh.Δy,
        Δlon_ll,
        Δy_ll,
        workspace,
    )
end

function collect_transport_binary_v2_windows(target::CubedSphereBilinearTransportBinaryTarget{FT},
                                             ctx::CubedSphereBilinearContext{FT},
                                             reader::TransportBinaryReader{FT}) where FT
    println("\n[3/4] Processing $(ctx.nwindow) windows...")
    ws = ctx.workspace
    windows_data = Vector{NamedTuple}(undef, ctx.nwindow)

    for win in 1:ctx.nwindow
        t0 = time()
        m_ll, ps_ll, fluxes_ll = load_window!(reader, win)
        am_ll = fluxes_ll.am
        bm_ll = fluxes_ll.bm

        for p in 1:CS_BILINEAR_PANEL_COUNT
            _cs_bilinear_interp_3d!(ws.m_panels[p], m_ll,
                                    ctx.cell_lons[p], ctx.cell_lats[p], ctx.src_lons, ctx.src_lats)
            _cs_bilinear_interp_2d!(ws.ps_panels[p], ps_ll,
                                    ctx.cell_lons[p], ctx.cell_lats[p], ctx.src_lons, ctx.src_lats)
        end

        @inbounds for k in 1:ctx.Nz, j in 1:ctx.Ny_ll, i in 1:ctx.Nx_ll
            dp = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) + (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ps_ll[i, j])
            area_factor = ctx.Δy_ll * dp / target.gravity * ctx.dt_factor
            ws.am_cc[i, j, k] = area_factor > FT(1e-10) ?
                FT(0.5) * (am_ll[i, j, k] + am_ll[i + 1, j, k]) / area_factor : zero(FT)
        end

        @inbounds for k in 1:ctx.Nz, j in 1:ctx.Ny_ll, i in 1:ctx.Nx_ll
            cos_lat = cosd(ctx.src_lats[j])
            Δx_ll = target.radius * ctx.Δlon_ll * max(cos_lat, FT(1e-6))
            dp = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) + (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ps_ll[i, j])
            area_factor = Δx_ll * dp / target.gravity * ctx.dt_factor
            ws.bm_cc[i, j, k] = area_factor > FT(1e-10) ?
                FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, min(j + 1, ctx.Ny_ll + 1), k]) / area_factor : zero(FT)
        end

        for p in 1:CS_BILINEAR_PANEL_COUNT
            _cs_bilinear_interp_3d!(ws.u_cs, ws.am_cc,
                                    ctx.cell_lons[p], ctx.cell_lats[p], ctx.src_lons, ctx.src_lats)
            _cs_bilinear_interp_3d!(ws.v_cs, ws.bm_cc,
                                    ctx.cell_lons[p], ctx.cell_lats[p], ctx.src_lons, ctx.src_lats)

            @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc, i in 1:ctx.Nc
                ws.dp_cs[i, j, k] = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) +
                                        (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ws.ps_panels[p][i, j])
            end

            @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc
                ws.am_panels[p][1, j, k] = ws.u_cs[1, j, k] * ws.dp_cs[1, j, k] *
                                           ctx.Δy[1, j] / target.gravity * ctx.dt_factor
                for i in 2:ctx.Nc
                    u_face = FT(0.5) * (ws.u_cs[i - 1, j, k] + ws.u_cs[i, j, k])
                    dp_face = FT(0.5) * (ws.dp_cs[i - 1, j, k] + ws.dp_cs[i, j, k])
                    ws.am_panels[p][i, j, k] = u_face * dp_face * ctx.Δy[i, j] /
                                               target.gravity * ctx.dt_factor
                end
                ws.am_panels[p][ctx.Nc + 1, j, k] = ws.u_cs[ctx.Nc, j, k] * ws.dp_cs[ctx.Nc, j, k] *
                                                    ctx.Δy[ctx.Nc, j] / target.gravity * ctx.dt_factor
            end

            @inbounds for k in 1:ctx.Nz, i in 1:ctx.Nc
                ws.bm_panels[p][i, 1, k] = ws.v_cs[i, 1, k] * ws.dp_cs[i, 1, k] *
                                           ctx.Δx[i, 1] / target.gravity * ctx.dt_factor
                for j in 2:ctx.Nc
                    v_face = FT(0.5) * (ws.v_cs[i, j - 1, k] + ws.v_cs[i, j, k])
                    dp_face = FT(0.5) * (ws.dp_cs[i, j - 1, k] + ws.dp_cs[i, j, k])
                    ws.bm_panels[p][i, j, k] = v_face * dp_face * ctx.Δx[i, j] /
                                               target.gravity * ctx.dt_factor
                end
                ws.bm_panels[p][i, ctx.Nc + 1, k] = ws.v_cs[i, ctx.Nc, k] * ws.dp_cs[i, ctx.Nc, k] *
                                                    ctx.Δx[i, ctx.Nc] / target.gravity * ctx.dt_factor
            end
        end

        if win < ctx.nwindow
            m_next_ll, _, _ = load_window!(reader, win + 1)
        else
            m_next_ll = m_ll
        end

        first_panel_residual = 0.0
        for p in 1:CS_BILINEAR_PANEL_COUNT
            _cs_bilinear_interp_3d!(ws.m_next_panel, m_next_ll,
                                    ctx.cell_lons[p], ctx.cell_lats[p], ctx.src_lons, ctx.src_lats)
            @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc, i in 1:ctx.Nc
                ws.dm_panel[i, j, k] = (ws.m_next_panel[i, j, k] - ws.m_panels[p][i, j, k]) /
                                       (2 * ctx.steps_per_window)
            end

            _cs_bilinear_balance_panel_mass_fluxes!(
                ws.am_panels[p], ws.bm_panels[p], ws.dm_panel,
                ws.poisson_fac, ws.poisson_residual, ws.poisson_psi,
                ctx.Nc, ctx.Nz,
            )
            p == 1 && (first_panel_residual = _cs_bilinear_max_balance_residual(
                ws.am_panels[p], ws.bm_panels[p], ws.dm_panel, ctx.Nc, ctx.Nz))

            _cs_bilinear_diagnose_cm!(
                ws.cm_panels[p], ws.am_panels[p], ws.bm_panels[p],
                ws.dm_panel, ws.m_panels[p], ctx.Nc, ctx.Nz,
            )
        end

        if win == 1
            @printf("  [balance] win 1: panel-1 post residual %.2e\n", first_panel_residual)
        end

        windows_data[win] = (
            m = _copy_panel_field_tuple_bilinear(ws.m_panels),
            ps = _copy_panel_field_tuple_bilinear(ws.ps_panels),
            am = _copy_panel_field_tuple_bilinear(ws.am_panels),
            bm = _copy_panel_field_tuple_bilinear(ws.bm_panels),
            cm = _copy_panel_field_tuple_bilinear(ws.cm_panels),
        )
        @printf("  Window %2d/%d: %.1fs\n", win, ctx.nwindow, time() - t0)
    end

    return windows_data
end

function build_transport_binary_v2_header(target::CubedSphereBilinearTransportBinaryTarget,
                                          ctx::CubedSphereBilinearContext,
                                          reader,
                                          windows)
    h = reader.header
    payload_sections = [:m, :am, :bm, :cm, :ps]
    n_m  = CS_BILINEAR_PANEL_COUNT * ctx.Nc * ctx.Nc * ctx.Nz
    n_am = CS_BILINEAR_PANEL_COUNT * (ctx.Nc + 1) * ctx.Nc * ctx.Nz
    n_bm = CS_BILINEAR_PANEL_COUNT * ctx.Nc * (ctx.Nc + 1) * ctx.Nz
    n_cm = CS_BILINEAR_PANEL_COUNT * ctx.Nc * ctx.Nc * (ctx.Nz + 1)
    n_ps = CS_BILINEAR_PANEL_COUNT * ctx.Nc * ctx.Nc
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 1,
        "header_bytes" => CS_BILINEAR_HEADER_BYTES,
        "float_type" => string(target_float_type(target)),
        "float_bytes" => sizeof(target_float_type(target)),
        "grid_type" => "cubed_sphere",
        "horizontal_topology" => "StructuredDirectional",
        "ncell" => CS_BILINEAR_PANEL_COUNT * ctx.Nc * ctx.Nc,
        "nface_h" => CS_BILINEAR_PANEL_COUNT * 2 * ctx.Nc * (ctx.Nc + 1),
        "nlevel" => ctx.Nz,
        "nwindow" => length(windows),
        "dt_met_seconds" => h.dt_met_seconds,
        "half_dt_seconds" => h.half_dt_seconds,
        "steps_per_window" => ctx.steps_per_window,
        "source_flux_sampling" => String(h.source_flux_sampling),
        "air_mass_sampling" => String(h.air_mass_sampling),
        "flux_sampling" => "window_constant",
        "flux_kind" => String(h.flux_kind),
        "humidity_sampling" => "none",
        "delta_semantics" => "none",
        "mass_basis" => String(h.mass_basis),
        "poisson_balance_target_scale" => 1.0 / (2 * ctx.steps_per_window),
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
        "poisson_balance_method" => "per_panel_fft_periodic_gnomonic",
        "A_ifc" => ctx.A_ifc,
        "B_ifc" => ctx.B_ifc,
        "payload_sections" => String.(payload_sections),
        "elems_per_window" => elems_per_window,
        "include_qv" => false,
        "include_qv_endpoints" => false,
        "include_flux_delta" => false,
        "n_qv" => 0,
        "n_qv_start" => 0,
        "n_qv_end" => 0,
        "n_geometry_elems" => 0,
        "Nc" => ctx.Nc,
        "npanel" => CS_BILINEAR_PANEL_COUNT,
        "Hp" => 0,
        "panel_convention" => "GEOSFP_file",
        "n_m" => n_m,
        "n_am" => n_am,
        "n_bm" => n_bm,
        "n_cm" => n_cm,
        "n_ps" => n_ps,
        "Nx" => ctx.Nc,
        "Ny" => ctx.Nc,
        "lons" => collect(range(0.0, 360.0 - 360.0 / ctx.Nc, length=ctx.Nc)),
        "lats" => collect(range(-90.0 + 90.0 / ctx.Nc, 90.0 - 90.0 / ctx.Nc, length=ctx.Nc)),
        "longitude_interval" => [0.0, 360.0],
        "latitude_interval" => [-90.0, 90.0],
        "source_binary" => target.input_path,
        "regrid_method" => "bilinear_cell_center",
        "creation_time" => string(Dates.now()),
    )
end

function write_transport_binary_v2_output(target::CubedSphereBilinearTransportBinaryTarget{FT},
                                          ctx::CubedSphereBilinearContext{FT},
                                          reader,
                                          header::Dict{String, Any},
                                          windows) where FT
    println("\n[4/4] Writing CS transport binary...")
    out_dir = dirname(target.output_path)
    isempty(out_dir) || mkpath(out_dir)

    header_json = JSON3.write(header)
    pad = CS_BILINEAR_HEADER_BYTES - ncodeunits(header_json)
    pad >= 0 || error("Header exceeds $(CS_BILINEAR_HEADER_BYTES) bytes (need $(ncodeunits(header_json)))")

    elems_per_window = Int(header["elems_per_window"])
    payload = Vector{FT}(undef, elems_per_window)
    bytes_written = Int64(0)

    open(target.output_path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        bytes_written += CS_BILINEAR_HEADER_BYTES

        for win in windows
            offset = 0
            for p in 1:CS_BILINEAR_PANEL_COUNT
                n = ctx.Nc * ctx.Nc * ctx.Nz
                copyto!(payload, offset + 1, vec(win.m[p]), 1, n)
                offset += n
            end
            for p in 1:CS_BILINEAR_PANEL_COUNT
                n = (ctx.Nc + 1) * ctx.Nc * ctx.Nz
                copyto!(payload, offset + 1, vec(win.am[p]), 1, n)
                offset += n
            end
            for p in 1:CS_BILINEAR_PANEL_COUNT
                n = ctx.Nc * (ctx.Nc + 1) * ctx.Nz
                copyto!(payload, offset + 1, vec(win.bm[p]), 1, n)
                offset += n
            end
            for p in 1:CS_BILINEAR_PANEL_COUNT
                n = ctx.Nc * ctx.Nc * (ctx.Nz + 1)
                copyto!(payload, offset + 1, vec(win.cm[p]), 1, n)
                offset += n
            end
            for p in 1:CS_BILINEAR_PANEL_COUNT
                n = ctx.Nc * ctx.Nc
                copyto!(payload, offset + 1, vec(win.ps[p]), 1, n)
                offset += n
            end
            @assert offset == elems_per_window
            write(io, payload)
            bytes_written += elems_per_window * sizeof(FT)
        end
    end

    println("  Written: ", target.output_path, "  (", round(filesize(target.output_path) / 1e9, digits=2), " GB)")
    return bytes_written
end
