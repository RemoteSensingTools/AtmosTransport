# Conservative LatLon -> cubed-sphere transport-binary target.
#
# Audience:
# - users normally call the CLI wrapper or
#   `build_transport_binary_v2_target(:cubed_sphere_conservative, ...)`
# - contributors can use this as the reference implementation of the stable
#   transport-binary v2 target interface
#
# Stability: the target type and builder are stable in-repo API. The helper
# functions below remain private implementation details.

using Dates
using Printf
using JSON3

using .AtmosTransport
using .AtmosTransport.Regridding

include(joinpath(@__DIR__, "cs_global_poisson_balance.jl"))

const CS_TRANSPORT_HEADER_BYTES = 65536
const CS_PANEL_COUNT = 6

"""
    CubedSphereConservativeTransportBinaryTarget

Stable target for conservative LatLon -> cubed-sphere transport-binary
regridding.

Users normally construct this via
`build_transport_binary_v2_target(:cubed_sphere_conservative, argv)`.
Required flags are `--input` and `--output`; `--Nc` and `--cache-dir` are
optional.
"""
struct CubedSphereConservativeTransportBinaryTarget{FT <: AbstractFloat} <: AbstractTransportBinaryV2Target
    input_path :: String
    output_path :: String
    Nc :: Int
    cache_dir :: String
    radius :: FT
    gravity :: FT
end

target_input_path(target::CubedSphereConservativeTransportBinaryTarget) = target.input_path
target_output_path(target::CubedSphereConservativeTransportBinaryTarget) = target.output_path
target_float_type(::CubedSphereConservativeTransportBinaryTarget{FT}) where FT = FT
target_summary(target::CubedSphereConservativeTransportBinaryTarget) =
    "ERA5 LatLon -> C$(target.Nc) (CR.jl conservative regridding)"

"""
    build_transport_binary_v2_target(::Val{:cubed_sphere_conservative}, argv; FT=Float64)

Internal dispatch entry backing
`build_transport_binary_v2_target(:cubed_sphere_conservative, argv; FT=...)`.

CLI example:
```bash
julia --project=. scripts/preprocessing/preprocess_era5_cs_conservative_v2.jl \
    --input era5_latlon.bin --output era5_cs.bin --Nc 90
```

Programmatic example:
```julia
target = build_transport_binary_v2_target(
    :cubed_sphere_conservative,
    ["--input", "era5_latlon.bin", "--output", "era5_cs.bin", "--Nc", "90"],
)
println(target_summary(target))
run_transport_binary_v2_preprocessor(target)
```
"""
function build_transport_binary_v2_target(::Val{:cubed_sphere_conservative},
                                          argv::AbstractVector{<:AbstractString};
                                          FT::Type{T} = Float64) where T <: AbstractFloat
    args = parse_flag_args(argv)
    input_path = _require_flag(args, "input")
    output_path = _require_flag(args, "output")
    Nc = parse(Int, get(args, "Nc", "90"))
    Nc > 0 || error("--Nc must be positive, got $Nc")
    cache_dir = get(args, "cache-dir",
                    joinpath(homedir(), ".cache", "AtmosTransport", "cr_regridding"))

    return CubedSphereConservativeTransportBinaryTarget{T}(
        input_path,
        output_path,
        Nc,
        cache_dir,
        T(6.371e6),
        T(9.80665),
    )
end

@inline function _panel_flat_range(p::Int, Nc::Int)
    return (p - 1) * Nc * Nc + 1 : p * Nc * Nc
end

function unpack_panels_3d!(panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                           flat::AbstractMatrix{FT}, Nc::Int, Nz::Int) where FT
    for p in 1:CS_PANEL_COUNT
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

function unpack_panels_2d!(panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                           flat::AbstractVector{FT}, Nc::Int) where FT
    for p in 1:CS_PANEL_COUNT
        r = _panel_flat_range(p, Nc)
        @inbounds for (linear, flat_idx) in enumerate(r)
            j, i = fldmod1(linear, Nc)
            panels[p][i, j] = flat[flat_idx]
        end
    end
    return panels
end

struct CubedSphereConservativeWorkspace{FT}
    m_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    ps_panels :: NTuple{CS_PANEL_COUNT, Matrix{FT}}
    am_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    bm_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    cm_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    u_cs_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    v_cs_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    dm_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    m_next_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    dp_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}
    u_cc :: Array{FT, 3}
    v_cc :: Array{FT, 3}
    src_flat_3d :: Matrix{FT}
    dst_flat_3d :: Matrix{FT}
    src_flat_2d :: Vector{FT}
    dst_flat_2d :: Vector{FT}
end

struct CubedSphereConservativeContext{FT, SM, DM, R, DX, DY}
    src_mesh :: SM
    dst_mesh :: DM
    regridder :: R
    Δx :: DX
    Δy :: DY
    cs_ft :: CSGlobalFaceTable
    cs_deg :: Vector{Int}
    cs_scratch :: CSPoissonScratch
    source_lons :: Vector{FT}
    source_lats :: Vector{FT}
    A_ifc :: Vector{FT}
    B_ifc :: Vector{FT}
    Nx_ll :: Int
    Ny_ll :: Int
    Nz :: Int
    Nc :: Int
    nwindow :: Int
    steps_per_window :: Int
    dt_met_seconds :: FT
    dt_factor :: FT
    Δlon_ll :: FT
    Δy_ll :: FT
    nnz_count :: Int
    workspace :: CubedSphereConservativeWorkspace{FT}
end

function _allocate_cs_workspace(Nc::Int, Nx_ll::Int, Ny_ll::Int, Nz::Int,
                                n_src::Int, n_dst::Int, ::Type{FT}) where FT
    return CubedSphereConservativeWorkspace(
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT),
        zeros(FT, Nx_ll, Ny_ll, Nz),
        zeros(FT, Nx_ll, Ny_ll, Nz),
        zeros(FT, n_src, Nz),
        zeros(FT, n_dst, Nz),
        zeros(FT, n_src),
        zeros(FT, n_dst),
    )
end

function _build_latlon_mesh_from_header(h, radius)
    src_lons = Float64.(h.lons_f64)
    src_lats = Float64.(h.lats_f64)
    dlon = src_lons[2] - src_lons[1]
    dlat = src_lats[2] - src_lats[1]
    lon_west  = src_lons[1]   - dlon / 2
    lon_east  = src_lons[end] + dlon / 2
    lat_south = max(src_lats[1]   - dlat / 2, -90.0)
    lat_north = min(src_lats[end] + dlat / 2,  90.0)

    mesh = LatLonMesh(
        FT = typeof(radius),
        Nx = h.Nx,
        Ny = h.Ny,
        longitude = (lon_west, lon_east),
        latitude = (lat_south, lat_north),
        radius = radius,
    )
    return mesh, typeof(radius).(src_lons), typeof(radius).(src_lats)
end

function _regrid_3d_panels!(ctx::CubedSphereConservativeContext{FT},
                            out_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                            src_3d::AbstractArray{FT, 3}) where FT
    ws = ctx.workspace
    copyto!(ws.src_flat_3d, reshape(src_3d, size(ws.src_flat_3d)...))
    apply_regridder!(ws.dst_flat_3d, ctx.regridder, ws.src_flat_3d)
    return unpack_panels_3d!(out_panels, ws.dst_flat_3d, ctx.Nc, ctx.Nz)
end

function _regrid_2d_panels!(ctx::CubedSphereConservativeContext{FT},
                            out_panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                            src_2d::AbstractArray{FT, 2}) where FT
    ws = ctx.workspace
    copyto!(ws.src_flat_2d, reshape(src_2d, size(ws.src_flat_2d)...))
    apply_regridder!(ws.dst_flat_2d, ctx.regridder, ws.src_flat_2d)
    return unpack_panels_2d!(out_panels, ws.dst_flat_2d, ctx.Nc)
end

@inline function _copy_panel_field_tuple(panels)
    return ntuple(p -> copy(panels[p]), CS_PANEL_COUNT)
end

function prepare_transport_binary_v2_target(target::CubedSphereConservativeTransportBinaryTarget{FT},
                                            reader::TransportBinaryReader{FT}) where FT
    h = reader.header
    println("=" ^ 72)
    println(target_summary(target))
    println("  Input:     ", target.input_path)
    println("  Output:    ", target.output_path)
    println("  Cache dir: ", target.cache_dir)
    println("=" ^ 72)

    println("\n[1/4] Reading source transport binary header...")
    Nx_ll = h.Nx
    Ny_ll = h.Ny
    Nz = h.nlevel
    println("  LatLon:        $(Nx_ll)×$(Ny_ll)×$(Nz), $(h.nwindow) windows")
    println("  Steps/window:  $(h.steps_per_window)  dt_met: $(h.dt_met_seconds)s")

    println("\n[2/4] Building meshes, regridder, and CS balance context...")
    src_mesh, source_lons, source_lats = _build_latlon_mesh_from_header(h, target.radius)
    dst_mesh = CubedSphereMesh(
        Nc = target.Nc,
        FT = FT,
        radius = target.radius,
        convention = GnomonicPanelConvention(),
    )
    println("  src: ", summary(src_mesh))
    println("  dst: ", summary(dst_mesh))

    regridder = build_regridder(src_mesh, dst_mesh;
                                normalize = false,
                                cache_dir = target.cache_dir)
    nnz_count = length(regridder.intersections.nzval)
    println("  regridder: $(size(regridder.intersections, 1))×$(size(regridder.intersections, 2))  nnz=$nnz_count")

    cs_conn = default_panel_connectivity()
    cs_ft = build_cs_global_face_table(target.Nc, cs_conn)
    cs_deg = cs_cell_face_degree(cs_ft)
    cs_scratch = CSPoissonScratch(cs_ft.nc)
    @printf("  Face table: %d faces, %d cells, degree range [%d, %d]\n",
            cs_ft.nf, cs_ft.nc, minimum(cs_deg), maximum(cs_deg))

    Δlon_ll = FT(abs(source_lons[2] - source_lons[1]) * π / 180)
    Δlat_ll = FT(abs(source_lats[2] - source_lats[1]) * π / 180)
    Δy_ll = target.radius * Δlat_ll
    dt_factor = FT(h.dt_met_seconds / (2 * h.steps_per_window))
    workspace = _allocate_cs_workspace(target.Nc, Nx_ll, Ny_ll, Nz,
                                       length(regridder.src_areas),
                                       length(regridder.dst_areas),
                                       FT)

    return CubedSphereConservativeContext(
        src_mesh,
        dst_mesh,
        regridder,
        dst_mesh.Δx,
        dst_mesh.Δy,
        cs_ft,
        cs_deg,
        cs_scratch,
        source_lons,
        source_lats,
        FT.(h.A_ifc),
        FT.(h.B_ifc),
        Nx_ll,
        Ny_ll,
        Nz,
        target.Nc,
        h.nwindow,
        h.steps_per_window,
        FT(h.dt_met_seconds),
        dt_factor,
        Δlon_ll,
        Δy_ll,
        nnz_count,
        workspace,
    )
end

function _recover_latlon_cell_center_winds!(target::CubedSphereConservativeTransportBinaryTarget{FT},
                                            ctx::CubedSphereConservativeContext{FT},
                                            ps_ll::AbstractArray{FT, 2},
                                            fluxes_ll) where FT
    ws = ctx.workspace
    am_ll = fluxes_ll.am
    bm_ll = fluxes_ll.bm

    @inbounds for k in 1:ctx.Nz, j in 1:ctx.Ny_ll, i in 1:ctx.Nx_ll
        dp_ll = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) +
                    (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = ctx.Δy_ll * dp_ll / target.gravity * ctx.dt_factor
        ws.u_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (am_ll[i, j, k] + am_ll[i + 1, j, k]) / area_factor :
            zero(FT)
    end

    @inbounds for k in 1:ctx.Nz, j in 1:ctx.Ny_ll, i in 1:ctx.Nx_ll
        cos_lat = cosd(ctx.source_lats[j])
        Δx_ll_loc = target.radius * ctx.Δlon_ll * max(cos_lat, FT(1e-6))
        dp_ll = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) +
                    (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = Δx_ll_loc * dp_ll / target.gravity * ctx.dt_factor
        jn = min(j + 1, ctx.Ny_ll + 1)
        ws.v_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, jn, k]) / area_factor :
            zero(FT)
    end

    return nothing
end

function _reconstruct_cs_horizontal_fluxes!(target::CubedSphereConservativeTransportBinaryTarget{FT},
                                            ctx::CubedSphereConservativeContext{FT}) where FT
    ws = ctx.workspace

    for p in 1:CS_PANEL_COUNT
        @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc, i in 1:ctx.Nc
            ws.dp_panels[p][i, j, k] = abs((ctx.A_ifc[k] - ctx.A_ifc[k + 1]) +
                                           (ctx.B_ifc[k] - ctx.B_ifc[k + 1]) * ws.ps_panels[p][i, j])
        end

        @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc
            ws.am_panels[p][1, j, k] = ws.u_cs_panels[p][1, j, k] *
                                       ws.dp_panels[p][1, j, k] * ctx.Δy[1, j] /
                                       target.gravity * ctx.dt_factor
            for i in 2:ctx.Nc
                u_face  = FT(0.5) * (ws.u_cs_panels[p][i - 1, j, k] + ws.u_cs_panels[p][i, j, k])
                dp_face = FT(0.5) * (ws.dp_panels[p][i - 1, j, k] + ws.dp_panels[p][i, j, k])
                ws.am_panels[p][i, j, k] = u_face * dp_face * ctx.Δy[i, j] /
                                           target.gravity * ctx.dt_factor
            end
            ws.am_panels[p][ctx.Nc + 1, j, k] = ws.u_cs_panels[p][ctx.Nc, j, k] *
                                                ws.dp_panels[p][ctx.Nc, j, k] * ctx.Δy[ctx.Nc, j] /
                                                target.gravity * ctx.dt_factor
        end

        @inbounds for k in 1:ctx.Nz, i in 1:ctx.Nc
            ws.bm_panels[p][i, 1, k] = ws.v_cs_panels[p][i, 1, k] *
                                       ws.dp_panels[p][i, 1, k] * ctx.Δx[i, 1] /
                                       target.gravity * ctx.dt_factor
            for j in 2:ctx.Nc
                v_face  = FT(0.5) * (ws.v_cs_panels[p][i, j - 1, k] + ws.v_cs_panels[p][i, j, k])
                dp_face = FT(0.5) * (ws.dp_panels[p][i, j - 1, k] + ws.dp_panels[p][i, j, k])
                ws.bm_panels[p][i, j, k] = v_face * dp_face * ctx.Δx[i, j] /
                                           target.gravity * ctx.dt_factor
            end
            ws.bm_panels[p][i, ctx.Nc + 1, k] = ws.v_cs_panels[p][i, ctx.Nc, k] *
                                                ws.dp_panels[p][i, ctx.Nc, k] * ctx.Δx[i, ctx.Nc] /
                                                target.gravity * ctx.dt_factor
        end
    end

    return nothing
end

function collect_transport_binary_v2_windows(target::CubedSphereConservativeTransportBinaryTarget{FT},
                                             ctx::CubedSphereConservativeContext{FT},
                                             reader::TransportBinaryReader{FT}) where FT
    println("\n[3/4] Processing $(ctx.nwindow) windows...")
    ws = ctx.workspace
    windows_data = Vector{NamedTuple}(undef, ctx.nwindow)

    for win in 1:ctx.nwindow
        t0 = time()
        m_ll, ps_ll, fluxes_ll = load_window!(reader, win)
        _regrid_3d_panels!(ctx, ws.m_panels, m_ll)
        _regrid_2d_panels!(ctx, ws.ps_panels, ps_ll)

        _recover_latlon_cell_center_winds!(target, ctx, ps_ll, fluxes_ll)
        _regrid_3d_panels!(ctx, ws.u_cs_panels, ws.u_cc)
        _regrid_3d_panels!(ctx, ws.v_cs_panels, ws.v_cc)
        _reconstruct_cs_horizontal_fluxes!(target, ctx)

        if win < ctx.nwindow
            m_next_ll, _, _ = load_window!(reader, win + 1)
        else
            m_next_ll = m_ll
        end
        _regrid_3d_panels!(ctx, ws.m_next_panels, m_next_ll)

        bal_diag = balance_cs_global_mass_fluxes!(
            ws.am_panels, ws.bm_panels, ws.m_panels, ws.m_next_panels,
            ctx.cs_ft, ctx.cs_deg, ctx.steps_per_window, ctx.cs_scratch;
            tol=1e-14, max_iter=5000,
        )
        if win == 1
            @printf("  [balance] win 1: pre=%.2e post=%.2e (mean=%.2e) proj=%.2e iter=%d\n",
                    bal_diag.max_pre_residual, bal_diag.max_post_residual,
                    bal_diag.max_rhs_mean, bal_diag.max_post_projected,
                    bal_diag.max_cg_iter)
        end

        for p in 1:CS_PANEL_COUNT
            @inbounds for k in 1:ctx.Nz, j in 1:ctx.Nc, i in 1:ctx.Nc
                ws.dm_panels[p][i, j, k] = (ws.m_next_panels[p][i, j, k] - ws.m_panels[p][i, j, k]) /
                                           (2 * ctx.steps_per_window)
            end
        end
        diagnose_cs_cm!(ws.cm_panels, ws.am_panels, ws.bm_panels, ws.dm_panels,
                        ws.m_panels, ctx.Nc, ctx.Nz)

        if win == 1
            src_total = sum(m_ll .* reshape(ctx.regridder.src_areas, ctx.Nx_ll, ctx.Ny_ll))
            dst_total = zero(FT)
            for p in 1:CS_PANEL_COUNT, k in 1:ctx.Nz, j in 1:ctx.Nc, i in 1:ctx.Nc
                dst_total += ws.m_panels[p][i, j, k] * ctx.regridder.dst_areas[_global_cell(i, j, p, ctx.Nc)]
            end
            rel_err = abs(dst_total - src_total) / max(abs(src_total), FT(1e-30))
            @printf("  [conservation] window 1: src=%.6e  dst=%.6e  rel_err=%.2e\n",
                    src_total, dst_total, rel_err)
        end

        windows_data[win] = (
            m = _copy_panel_field_tuple(ws.m_panels),
            ps = _copy_panel_field_tuple(ws.ps_panels),
            am = _copy_panel_field_tuple(ws.am_panels),
            bm = _copy_panel_field_tuple(ws.bm_panels),
            cm = _copy_panel_field_tuple(ws.cm_panels),
        )
        @printf("  Window %2d/%d: %.1fs\n", win, ctx.nwindow, time() - t0)
    end

    return windows_data
end

function build_transport_binary_v2_header(target::CubedSphereConservativeTransportBinaryTarget,
                                          ctx::CubedSphereConservativeContext,
                                          reader,
                                          windows)
    h = reader.header
    payload_sections = [:m, :am, :bm, :cm, :ps]
    n_m  = CS_PANEL_COUNT * ctx.Nc * ctx.Nc * ctx.Nz
    n_am = CS_PANEL_COUNT * (ctx.Nc + 1) * ctx.Nc * ctx.Nz
    n_bm = CS_PANEL_COUNT * ctx.Nc * (ctx.Nc + 1) * ctx.Nz
    n_cm = CS_PANEL_COUNT * ctx.Nc * ctx.Nc * (ctx.Nz + 1)
    n_ps = CS_PANEL_COUNT * ctx.Nc * ctx.Nc
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 1,
        "header_bytes" => CS_TRANSPORT_HEADER_BYTES,
        "float_type" => string(target_float_type(target)),
        "float_bytes" => sizeof(target_float_type(target)),
        "grid_type" => "cubed_sphere",
        "horizontal_topology" => "StructuredDirectional",
        "ncell" => CS_PANEL_COUNT * ctx.Nc * ctx.Nc,
        "nface_h" => CS_PANEL_COUNT * 2 * ctx.Nc * (ctx.Nc + 1),
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
        "poisson_balance_method" => "global_cg_graph_laplacian",
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
        "npanel" => CS_PANEL_COUNT,
        "Hp" => 0,
        "panel_convention" => "Gnomonic",
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
        "regrid_method" => "conservative_crjl",
        "regridder_nnz" => ctx.nnz_count,
        "regridder_cache_dir" => target.cache_dir,
        "creation_time" => string(Dates.now()),
    )
end

function write_transport_binary_v2_output(target::CubedSphereConservativeTransportBinaryTarget{FT},
                                          ctx::CubedSphereConservativeContext{FT},
                                          reader,
                                          header::Dict{String, Any},
                                          windows) where FT
    println("\n[4/4] Writing CS transport binary...")
    out_dir = dirname(target.output_path)
    isempty(out_dir) || mkpath(out_dir)

    header_json = JSON3.write(header)
    pad = CS_TRANSPORT_HEADER_BYTES - ncodeunits(header_json)
    pad >= 0 || error("Header exceeds $(CS_TRANSPORT_HEADER_BYTES) bytes (need $(ncodeunits(header_json)))")

    elems_per_window = Int(header["elems_per_window"])
    payload = Vector{FT}(undef, elems_per_window)
    bytes_written = Int64(0)

    open(target.output_path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        bytes_written += CS_TRANSPORT_HEADER_BYTES

        for win in windows
            offset = 0
            for p in 1:CS_PANEL_COUNT
                n = ctx.Nc * ctx.Nc * ctx.Nz
                copyto!(payload, offset + 1, vec(win.m[p]), 1, n)
                offset += n
            end
            for p in 1:CS_PANEL_COUNT
                n = (ctx.Nc + 1) * ctx.Nc * ctx.Nz
                copyto!(payload, offset + 1, vec(win.am[p]), 1, n)
                offset += n
            end
            for p in 1:CS_PANEL_COUNT
                n = ctx.Nc * (ctx.Nc + 1) * ctx.Nz
                copyto!(payload, offset + 1, vec(win.bm[p]), 1, n)
                offset += n
            end
            for p in 1:CS_PANEL_COUNT
                n = ctx.Nc * ctx.Nc * (ctx.Nz + 1)
                copyto!(payload, offset + 1, vec(win.cm[p]), 1, n)
                offset += n
            end
            for p in 1:CS_PANEL_COUNT
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
