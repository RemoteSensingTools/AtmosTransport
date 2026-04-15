# ---------------------------------------------------------------------------
# Cubed-sphere transport binary preprocessing helpers.
#
# Ported from scripts_legacy/preprocessing/transport_binary_v2_cs_conservative.jl
# and adapted for the modern streaming pipeline in src/Preprocessing/.
#
# Key functions:
#   - regrid_scalar_to_cs_panels!  — conservative LL→CS for 3D and 2D fields
#   - recover_ll_cell_center_winds! — extract u,v from LL am,bm fluxes
#   - reconstruct_cs_fluxes!       — build CS am,bm from regridded u,v
#   - CubedSpherePreprocessWorkspace — all per-window scratch arrays
# ---------------------------------------------------------------------------

const CS_PANEL_COUNT = 6

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

"""
    CubedSpherePreprocessWorkspace{FT}

Pre-allocated workspace for one or two CS transport windows. Holds all per-panel
arrays needed by the regrid → wind recovery → flux reconstruction → balance → cm
pipeline.
"""
struct CubedSpherePreprocessWorkspace{FT}
    # Current window CS fields
    m_panels      :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    ps_panels     :: NTuple{CS_PANEL_COUNT, Matrix{FT}}      # (Nc, Nc)
    am_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc+1, Nc, Nz)
    bm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc+1, Nz)
    cm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz+1)
    # Next-window mass for Poisson balance look-ahead
    m_next_panels :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Regridded cell-center winds on CS panels
    u_cs_panels   :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    v_cs_panels   :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Pressure thickness on CS panels
    dp_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # Mass tendency for cm diagnosis
    dm_panels     :: NTuple{CS_PANEL_COUNT, Array{FT, 3}}   # (Nc, Nc, Nz)
    # LL cell-center winds (staging grid)
    u_cc          :: Array{FT, 3}   # (Nx_stg, Ny_stg, Nz)
    v_cc          :: Array{FT, 3}   # (Nx_stg, Ny_stg, Nz)
    # Regridder flat I/O buffers
    src_flat_3d   :: Matrix{FT}     # (n_src, Nz)
    dst_flat_3d   :: Matrix{FT}     # (n_dst, Nz)
    src_flat_2d   :: Vector{FT}     # (n_src,)
    dst_flat_2d   :: Vector{FT}     # (n_dst,)
end

function allocate_cs_preprocess_workspace(Nc::Int, Nx_stg::Int, Ny_stg::Int,
                                          Nz::Int, n_src::Int, n_dst::Int,
                                          ::Type{FT}) where FT <: AbstractFloat
    return CubedSpherePreprocessWorkspace{FT}(
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
        zeros(FT, Nx_stg, Ny_stg, Nz),
        zeros(FT, Nx_stg, Ny_stg, Nz),
        zeros(FT, n_src, Nz),
        zeros(FT, n_dst, Nz),
        zeros(FT, n_src),
        zeros(FT, n_dst),
    )
end

# ---------------------------------------------------------------------------
# Panel packing/unpacking
# ---------------------------------------------------------------------------

@inline _cs_panel_flat_range(p::Int, Nc::Int) = (p - 1) * Nc * Nc + 1 : p * Nc * Nc

"""
    unpack_flat_to_panels_3d!(panels, flat, Nc, Nz)

Unpack a flat `(6Nc², Nz)` matrix into 6 panel arrays `(Nc, Nc, Nz)`.
"""
function unpack_flat_to_panels_3d!(panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                    flat::AbstractMatrix{FT}, Nc::Int, Nz::Int) where FT
    for p in 1:CS_PANEL_COUNT
        r = _cs_panel_flat_range(p, Nc)
        for k in 1:Nz
            @inbounds for (linear, flat_idx) in enumerate(r)
                j, i = fldmod1(linear, Nc)  # (div, mod) = (j, i) for column-major (j-1)*Nc+i
                panels[p][i, j, k] = flat[flat_idx, k]
            end
        end
    end
    return panels
end

"""
    unpack_flat_to_panels_2d!(panels, flat, Nc)

Unpack a flat `(6Nc²,)` vector into 6 panel arrays `(Nc, Nc)`.
"""
function unpack_flat_to_panels_2d!(panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                    flat::AbstractVector{FT}, Nc::Int) where FT
    for p in 1:CS_PANEL_COUNT
        r = _cs_panel_flat_range(p, Nc)
        @inbounds for (linear, flat_idx) in enumerate(r)
            j, i = fldmod1(linear, Nc)  # (div, mod) = (j, i) for column-major (j-1)*Nc+i
            panels[p][i, j] = flat[flat_idx]
        end
    end
    return panels
end

"""
    pack_panels_3d_to_flat!(flat, panels, Nc, Nz)

Pack 6 panel arrays `(Nc, Nc, Nz)` into flat `(6Nc², Nz)`.
Inverse of `unpack_flat_to_panels_3d!`.
"""
function pack_panels_3d_to_flat!(flat::AbstractMatrix{FT},
                                  panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                  Nc::Int, Nz::Int) where FT
    for p in 1:CS_PANEL_COUNT
        r = _cs_panel_flat_range(p, Nc)
        for k in 1:Nz
            @inbounds for (linear, flat_idx) in enumerate(r)
                j, i = fldmod1(linear, Nc)  # (div, mod) = (j, i) for column-major (j-1)*Nc+i
                flat[flat_idx, k] = panels[p][i, j, k]
            end
        end
    end
    return flat
end

# ---------------------------------------------------------------------------
# Conservative LL → CS regridding
# ---------------------------------------------------------------------------

"""
    regrid_3d_to_cs_panels!(panels, regridder, src_3d, ws, Nc)

Conservatively regrid a 3D LL field `(Nx, Ny, Nz)` to 6 CS panels `(Nc, Nc, Nz)`.
Uses `ws.src_flat_3d` and `ws.dst_flat_3d` as scratch buffers.
"""
function regrid_3d_to_cs_panels!(panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                  regridder,
                                  src_3d::AbstractArray{FT, 3},
                                  ws::CubedSpherePreprocessWorkspace{FT},
                                  Nc::Int) where FT
    Nz = size(src_3d, 3)
    copyto!(ws.src_flat_3d, reshape(src_3d, size(ws.src_flat_3d)...))
    apply_regridder!(ws.dst_flat_3d, regridder, ws.src_flat_3d)
    return unpack_flat_to_panels_3d!(panels, ws.dst_flat_3d, Nc, Nz)
end

"""
    regrid_2d_to_cs_panels!(panels, regridder, src_2d, ws, Nc)

Conservatively regrid a 2D LL field `(Nx, Ny)` to 6 CS panels `(Nc, Nc)`.
"""
function regrid_2d_to_cs_panels!(panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                  regridder,
                                  src_2d::AbstractArray{FT, 2},
                                  ws::CubedSpherePreprocessWorkspace{FT},
                                  Nc::Int) where FT
    copyto!(ws.src_flat_2d, reshape(src_2d, size(ws.src_flat_2d)...))
    apply_regridder!(ws.dst_flat_2d, regridder, ws.src_flat_2d)
    return unpack_flat_to_panels_2d!(panels, ws.dst_flat_2d, Nc)
end

# ---------------------------------------------------------------------------
# Wind recovery from LL mass fluxes
# ---------------------------------------------------------------------------

"""
    recover_ll_cell_center_winds!(u_cc, v_cc, am_ll, bm_ll, ps_ll,
                                   A_ifc, B_ifc, lats_deg,
                                   Δy_ll, Δlon_ll, radius, gravity, dt_factor)

Recover cell-center u,v wind components from LL mass fluxes by dividing
by the cross-sectional area factor: `am / (Δy × dp/g × dt_factor)`.

`dt_factor = dt_met / (2 × steps_per_window)` is the flux scaling used
in the binary.
"""
function recover_ll_cell_center_winds!(u_cc::Array{FT, 3},
                                        v_cc::Array{FT, 3},
                                        am_ll::AbstractArray{FT, 3},
                                        bm_ll::AbstractArray{FT, 3},
                                        ps_ll::AbstractArray{FT, 2},
                                        A_ifc::AbstractVector, B_ifc::AbstractVector,
                                        lats_deg::AbstractVector,
                                        Δy_ll::FT, Δlon_ll::FT,
                                        radius::FT, gravity::FT,
                                        dt_factor::FT) where FT
    Nx, Ny, Nz = size(u_cc)

    # u from am: average face fluxes to cell centers, divide by area factor
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp = abs((A_ifc[k] - A_ifc[k + 1]) +
                 (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = Δy_ll * dp / gravity * dt_factor
        u_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (am_ll[i, j, k] + am_ll[i + 1, j, k]) / area_factor :
            zero(FT)
    end

    # v from bm: average face fluxes to cell centers, divide by area factor
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        cos_lat = cosd(lats_deg[j])
        Δx_loc = radius * Δlon_ll * max(cos_lat, FT(1e-6))
        dp = abs((A_ifc[k] - A_ifc[k + 1]) +
                 (B_ifc[k] - B_ifc[k + 1]) * ps_ll[i, j])
        area_factor = Δx_loc * dp / gravity * dt_factor
        jn = min(j + 1, Ny + 1)
        v_cc[i, j, k] = area_factor > FT(1e-10) ?
            FT(0.5) * (bm_ll[i, j, k] + bm_ll[i, jn, k]) / area_factor :
            zero(FT)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# CS flux reconstruction from cell-center winds
# ---------------------------------------------------------------------------

"""
    reconstruct_cs_fluxes!(am_panels, bm_panels, u_cs, v_cs, dp_panels,
                            ps_panels, A_ifc, B_ifc, Δx, Δy,
                            gravity, dt_factor, Nc, Nz)

Reconstruct per-panel horizontal mass fluxes from regridded cell-center winds.

Face fluxes are built by averaging adjacent cell-center winds and
pressure thickness, then multiplying by the face length and scaling:

    am[i,j,k] = ū × d̄p × Δy[i,j] / g × dt_factor
    bm[i,j,k] = v̄ × d̄p × Δx[i,j] / g × dt_factor

Boundary faces (i=1, i=Nc+1, j=1, j=Nc+1) use the adjacent cell center
value without averaging — these are corrected by the Poisson balance.
"""
function reconstruct_cs_fluxes!(am_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 bm_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 u_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 v_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 dp_panels::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                 ps_panels::NTuple{CS_PANEL_COUNT, Matrix{FT}},
                                 A_ifc::AbstractVector, B_ifc::AbstractVector,
                                 Δx, Δy, gravity::FT, dt_factor::FT,
                                 Nc::Int, Nz::Int) where FT

    for p in 1:CS_PANEL_COUNT
        # Compute dp per cell
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            dp_panels[p][i, j, k] = abs(FT(A_ifc[k] - A_ifc[k + 1]) +
                                        FT(B_ifc[k] - B_ifc[k + 1]) * ps_panels[p][i, j])
        end

        # am: x-direction face fluxes
        @inbounds for k in 1:Nz, j in 1:Nc
            # West boundary face
            am_panels[p][1, j, k] = u_cs[p][1, j, k] *
                dp_panels[p][1, j, k] * Δy[1, j] / gravity * dt_factor
            # Interior faces
            for i in 2:Nc
                u_face  = FT(0.5) * (u_cs[p][i - 1, j, k] + u_cs[p][i, j, k])
                dp_face = FT(0.5) * (dp_panels[p][i - 1, j, k] + dp_panels[p][i, j, k])
                am_panels[p][i, j, k] = u_face * dp_face * Δy[i, j] / gravity * dt_factor
            end
            # East boundary face
            am_panels[p][Nc + 1, j, k] = u_cs[p][Nc, j, k] *
                dp_panels[p][Nc, j, k] * Δy[Nc, j] / gravity * dt_factor
        end

        # bm: y-direction face fluxes
        @inbounds for k in 1:Nz, i in 1:Nc
            # South boundary face
            bm_panels[p][i, 1, k] = v_cs[p][i, 1, k] *
                dp_panels[p][i, 1, k] * Δx[i, 1] / gravity * dt_factor
            # Interior faces
            for j in 2:Nc
                v_face  = FT(0.5) * (v_cs[p][i, j - 1, k] + v_cs[p][i, j, k])
                dp_face = FT(0.5) * (dp_panels[p][i, j - 1, k] + dp_panels[p][i, j, k])
                bm_panels[p][i, j, k] = v_face * dp_face * Δx[i, j] / gravity * dt_factor
            end
            # North boundary face
            bm_panels[p][i, Nc + 1, k] = v_cs[p][i, Nc, k] *
                dp_panels[p][i, Nc, k] * Δx[i, Nc] / gravity * dt_factor
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Per-level mass consistency correction
# ---------------------------------------------------------------------------

"""
    _enforce_perlevel_mass_consistency!(m_cs, m_ll, Nc, Nz)

Adjust regridded CS mass panels so that the global sum at each level matches
the source LL grid's sum. Conservative regridding preserves total mass but can
shift the per-level distribution by O(10⁻⁶) relative. This small uniform
correction ensures Σ(m_next - m_cur) = 0 per level, which is required for the
Poisson balance on a closed sphere (where Σ div_h = 0 topologically).

The correction is applied to the regridded m BEFORE it's stored, so the
binary's m and the balance target are consistent (no hidden projection).
"""
function _enforce_perlevel_mass_consistency!(m_cs::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                              m_ll::AbstractArray{FT, 3},
                                              Nc::Int, Nz::Int) where FT
    nc_cs = CS_PANEL_COUNT * Nc * Nc
    for k in 1:Nz
        # Target: sum of LL mass at this level
        ll_sum = sum(view(m_ll, :, :, k))

        # Current: sum of CS mass at this level
        cs_sum = zero(FT)
        for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
            cs_sum += m_cs[p][i, j, k]
        end

        # Uniform additive correction per CS cell
        offset = (ll_sum - cs_sum) / nc_cs
        for p in 1:CS_PANEL_COUNT, j in 1:Nc, i in 1:Nc
            m_cs[p][i, j, k] += offset
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# East/north → panel-local wind rotation
# ---------------------------------------------------------------------------

"""
    rotate_winds_to_panel_local!(u_panel, v_panel, u_east, v_north,
                                  Nc, Nz)

Rotate geographic (east, north) wind components to panel-local (x, y)
components for all 6 CS panels.

On the gnomonic cubed sphere, each panel's local (x, y) axes are related
to geographic (east, north) by a rotation that depends on the panel and
cell position. For each cell center (lon, lat), the panel-local unit
vectors ê_x and ê_y are computed from the gnomonic projection, and the
wind is projected: u_x = u_east·cos(α) + v_north·sin(α), etc.

This uses the analytical gnomonic-to-geographic Jacobian at each cell center.
"""
function rotate_winds_to_panel_local!(u_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_panel::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       u_east::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       v_north::NTuple{CS_PANEL_COUNT, Array{FT, 3}},
                                       Nc::Int, Nz::Int) where FT
    # Gnomonic cell centers: uniform in α ∈ [-π/4, π/4]
    dα = FT(π) / (2 * Nc)
    α_centers = FT[-π/4 + (i - FT(0.5)) * dα for i in 1:Nc]

    for p in 1:6
        for j in 1:Nc, i in 1:Nc
            ξ = tan(α_centers[i])
            η = tan(α_centers[j])

            # Geographic coordinates (lon, lat) of cell center on panel p
            lon, lat = _gnomonic_to_lonlat(ξ, η, p)
            cos_lat = max(cos(lat), FT(1e-10))

            # Jacobian of gnomonic → geographic mapping
            # ∂lon/∂ξ, ∂lon/∂η, ∂lat/∂ξ, ∂lat/∂η
            dlon_dξ, dlon_dη, dlat_dξ, dlat_dη = _gnomonic_jacobian(ξ, η, p, cos_lat)

            # Panel-local x-direction in geographic components:
            #   ê_x = (∂lon/∂ξ · cos_lat, ∂lat/∂ξ)  (unnormalized)
            # Panel-local y-direction:
            #   ê_y = (∂lon/∂η · cos_lat, ∂lat/∂η)  (unnormalized)
            ex_east  = dlon_dξ * cos_lat
            ex_north = dlat_dξ
            ey_east  = dlon_dη * cos_lat
            ey_north = dlat_dη

            # Normalize
            nx = max(sqrt(ex_east^2 + ex_north^2), FT(1e-30))
            ny = max(sqrt(ey_east^2 + ey_north^2), FT(1e-30))
            ex_east  /= nx; ex_north /= nx
            ey_east  /= ny; ey_north /= ny

            # Project winds
            @inbounds for k in 1:Nz
                ue = u_east[p][i, j, k]
                vn = v_north[p][i, j, k]
                u_panel[p][i, j, k] = ue * ex_east + vn * ex_north
                v_panel[p][i, j, k] = ue * ey_east + vn * ey_north
            end
        end
    end
    return nothing
end

"""
    _gnomonic_to_lonlat(ξ, η, panel) -> (lon, lat)

Convert gnomonic coordinates (ξ, η) on `panel` to geographic (lon, lat) in radians.
Uses the standard gnomonic projection for equidistant cubed sphere.
"""
function _gnomonic_to_lonlat(ξ::FT, η::FT, panel::Int) where FT
    r = sqrt(one(FT) + ξ^2 + η^2)
    if panel == 1      # +x face
        x, y, z = one(FT)/r, ξ/r, η/r
    elseif panel == 2  # +y face
        x, y, z = -ξ/r, one(FT)/r, η/r
    elseif panel == 3  # -x face
        x, y, z = -one(FT)/r, -ξ/r, η/r
    elseif panel == 4  # -y face
        x, y, z = ξ/r, -one(FT)/r, η/r
    elseif panel == 5  # north pole (+z face)
        x, y, z = -η/r, ξ/r, one(FT)/r
    else               # south pole (-z face)
        x, y, z = η/r, ξ/r, -one(FT)/r
    end
    lon = atan(y, x)
    lat = asin(clamp(z, -one(FT), one(FT)))
    return (lon, lat)
end

"""
    _gnomonic_jacobian(ξ, η, panel, cos_lat) -> (dlon_dξ, dlon_dη, dlat_dξ, dlat_dη)

Analytical Jacobian of the gnomonic-to-geographic mapping at (ξ, η) on `panel`.
"""
function _gnomonic_jacobian(ξ::FT, η::FT, panel::Int, cos_lat::FT) where FT
    r2 = one(FT) + ξ^2 + η^2
    r = sqrt(r2)
    r3 = r2 * r

    # Cartesian (x, y, z) and their derivatives w.r.t. (ξ, η)
    if panel == 1
        x, y, z = one(FT)/r, ξ/r, η/r
        dx_dξ = -ξ/r3;     dx_dη = -η/r3
        dy_dξ = (one(FT) + η^2)/r3; dy_dη = -ξ*η/r3
        dz_dξ = -ξ*η/r3;  dz_dη = (one(FT) + ξ^2)/r3
    elseif panel == 2
        x, y, z = -ξ/r, one(FT)/r, η/r
        dx_dξ = -(one(FT) + η^2)/r3; dx_dη = ξ*η/r3
        dy_dξ = -ξ/r3;              dy_dη = -η/r3  # y=1/r → dy/dξ = -ξ/r³
        dz_dξ = -ξ*η/r3;           dz_dη = (one(FT) + ξ^2)/r3
    elseif panel == 3  # -x face
        x, y, z = -one(FT)/r, -ξ/r, η/r
        dx_dξ = ξ/r3;                dx_dη = η/r3
        dy_dξ = -(one(FT) + η^2)/r3; dy_dη = ξ*η/r3
        dz_dξ = -ξ*η/r3;            dz_dη = (one(FT) + ξ^2)/r3
    elseif panel == 4  # -y face
        x, y, z = ξ/r, -one(FT)/r, η/r
        dx_dξ = (one(FT) + η^2)/r3; dx_dη = -ξ*η/r3
        dy_dξ = ξ/r3;               dy_dη = η/r3  # y=-1/r → dy/dξ = +ξ/r³
        dz_dξ = -ξ*η/r3;           dz_dη = (one(FT) + ξ^2)/r3
    elseif panel == 5  # north pole (+z face)
        x, y, z = -η/r, ξ/r, one(FT)/r
        dx_dξ = ξ*η/r3;             dx_dη = -(one(FT) + ξ^2)/r3
        dy_dξ = (one(FT) + η^2)/r3; dy_dη = -ξ*η/r3
        dz_dξ = -ξ/r3;              dz_dη = -η/r3
    else  # panel 6
        x, y, z = η/r, ξ/r, -one(FT)/r
        dx_dξ = -ξ*η/r3;            dx_dη = (one(FT) + ξ^2)/r3
        dy_dξ = (one(FT) + η^2)/r3; dy_dη = -ξ*η/r3
        dz_dξ = ξ/r3;               dz_dη = η/r3
    end

    # lon = atan(y, x), lat = asin(z)
    rxy2 = x^2 + y^2
    rxy2 = max(rxy2, FT(1e-30))
    dlon_dξ = (x * dy_dξ - y * dx_dξ) / rxy2
    dlon_dη = (x * dy_dη - y * dx_dη) / rxy2
    dlat_dξ = dz_dξ / max(cos_lat, FT(1e-10))
    dlat_dη = dz_dη / max(cos_lat, FT(1e-10))

    return (dlon_dξ, dlon_dη, dlat_dξ, dlat_dη)
end

# ---------------------------------------------------------------------------
# Utility: copy panel tuple (for snapshot storage)
# ---------------------------------------------------------------------------

@inline copy_panel_tuple(panels) = ntuple(p -> copy(panels[p]), CS_PANEL_COUNT)
