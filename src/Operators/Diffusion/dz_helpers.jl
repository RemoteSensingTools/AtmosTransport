# ---------------------------------------------------------------------------
# Layer-thickness fill for the implicit-vertical-diffusion workspace.
#
# `apply_vertical_diffusion!` consumes `workspace.dz_scratch` as the current
# layer thickness in metres. The workspace allocator initializes that array
# to zeros, so the runtime must populate it before each diffusion call —
# otherwise the kernel divides by zero and the entire tracer field NaNs out
# from the next snapshot onward.
#
# The fill uses a constant-`T_ref` hydrostatic approximation,
#
#     delp[i,j,k] = (ak[k+1] - ak[k]) + (bk[k+1] - bk[k]) * ps[i,j]
#     p_ctr[i,j,k] = ½·((ak[k] + ak[k+1]) + (bk[k] + bk[k+1])·ps[i,j])
#     dz[i,j,k]   = R · T_ref / g · delp[i,j,k] / p_ctr[i,j,k]
#
# This matches the preprocessing-side `dz_hydrostatic_constT!`
# (`src/Preprocessing/tm5_convection_conversion.jl`). A virtual-temperature
# variant can be added when humidity is wired through the runtime workspace;
# `T_ref = 260 K` is sufficient for boundary-layer Kz in the same way TM5
# uses it.
#
# `dz` only depends on `ps + ak/bk`, so the fill runs once per met window
# (constant within the window). The runtime hooks it from
# `DrivenSimulation` at window-load time.
# ---------------------------------------------------------------------------

const _DZ_T_REF_DEFAULT = 260.0
const _DZ_R_DEFAULT     = 287.04
const _DZ_G_DEFAULT     = 9.81

@kernel function _dz_hydrostatic_constT_kernel!(dz, @Const(ps),
                                                @Const(ak_ifc), @Const(bk_ifc),
                                                T_ref, R, g, Nz::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ak_lo = ak_ifc[k]
        ak_hi = ak_ifc[k + 1]
        bk_lo = bk_ifc[k]
        bk_hi = bk_ifc[k + 1]
        delp  = (ak_hi - ak_lo) + (bk_hi - bk_lo) * ps[i, j]
        p_ctr = (ak_lo + ak_hi + (bk_lo + bk_hi) * ps[i, j]) * (1 // 2)
        dz[i, j, k] = R * T_ref / g * delp / p_ctr
    end
end

"""
    fill_dz_hydrostatic_constT!(dz, ps, ak_ifc, bk_ifc;
                                 T_ref = 260, R = 287.04, gravity = 9.81)

Populate a 3D `(Nx, Ny, Nz)` `dz` array (host or device) from surface
pressure `ps` and hybrid sigma-pressure interface coefficients
`ak_ifc, bk_ifc` (length `Nz + 1`). Backend follows `dz`.
"""
function fill_dz_hydrostatic_constT!(dz::AbstractArray{<:AbstractFloat, 3},
                                      ps::AbstractArray{<:AbstractFloat, 2},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      T_ref::Real    = _DZ_T_REF_DEFAULT,
                                      R::Real        = _DZ_R_DEFAULT,
                                      gravity::Real  = _DZ_G_DEFAULT)
    Nx, Ny, Nz = size(dz)
    size(ps) == (Nx, Ny) || throw(DimensionMismatch(
        "ps shape $(size(ps)) ≠ ($Nx, $Ny) expected from dz"))
    length(ak_ifc) == Nz + 1 || throw(DimensionMismatch(
        "ak_ifc length $(length(ak_ifc)) ≠ Nz+1=$(Nz+1)"))
    length(bk_ifc) == Nz + 1 || throw(DimensionMismatch(
        "bk_ifc length $(length(bk_ifc)) ≠ Nz+1=$(Nz+1)"))
    FT = eltype(dz)
    backend = get_backend(dz)
    # Stage ak/bk on whatever device `dz` lives on. `similar(dz, FT, n)`
    # gives an Array on CPU or a CuArray on GPU; `copyto!` does the
    # H→D transfer once per fill (cheaper than per-thread broadcast).
    ak_dev = similar(dz, FT, Nz + 1)
    bk_dev = similar(dz, FT, Nz + 1)
    copyto!(ak_dev, FT.(ak_ifc))
    copyto!(bk_dev, FT.(bk_ifc))
    kernel = _dz_hydrostatic_constT_kernel!(backend, (8, 8, 1))
    kernel(dz, ps, ak_dev, bk_dev,
           FT(T_ref), FT(R), FT(gravity), Nz;
           ndrange = (Nx, Ny, Nz))
    synchronize(backend)
    return dz
end

"""
    fill_dz_hydrostatic_constT!(dz_panels::NTuple{6}, ps_panels::NTuple{6},
                                 ak_ifc, bk_ifc; ...)

Cubed-sphere variant: per-panel 3D `(Nc, Nc, Nz)` `dz` arrays are filled
from per-panel `ps` arrays of shape `(Nc, Nc)` (interior only — the
runtime stores `surface_pressure` without the advection halo).
"""
function fill_dz_hydrostatic_constT!(dz_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 3}},
                                      ps_panels::NTuple{6, <:AbstractArray{<:AbstractFloat, 2}},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      kwargs...)
    @inbounds for p in 1:6
        fill_dz_hydrostatic_constT!(dz_panels[p], ps_panels[p],
                                    ak_ifc, bk_ifc; kwargs...)
    end
    return dz_panels
end

@kernel function _dz_hydrostatic_constT_face_kernel!(dz, @Const(ps),
                                                     @Const(ak_ifc), @Const(bk_ifc),
                                                     T_ref, R, g, Nz::Int)
    c, k = @index(Global, NTuple)
    @inbounds begin
        ak_lo = ak_ifc[k]
        ak_hi = ak_ifc[k + 1]
        bk_lo = bk_ifc[k]
        bk_hi = bk_ifc[k + 1]
        delp  = (ak_hi - ak_lo) + (bk_hi - bk_lo) * ps[c]
        p_ctr = (ak_lo + ak_hi + (bk_lo + bk_hi) * ps[c]) * (1 // 2)
        dz[c, k] = R * T_ref / g * delp / p_ctr
    end
end

"""
    fill_dz_hydrostatic_constT!(dz::AbstractArray{<:Any, 2},
                                 ps::AbstractArray{<:Any, 1},
                                 ak_ifc, bk_ifc; ...)

Face-indexed variant for ReducedGaussian topology: `dz` shape `(ncells, Nz)`,
`ps` shape `(ncells,)`. Same constant-`T_ref` formula as the structured/CS
overloads, just unrolled over the face-indexed cell axis.
"""
function fill_dz_hydrostatic_constT!(dz::AbstractArray{<:AbstractFloat, 2},
                                      ps::AbstractArray{<:AbstractFloat, 1},
                                      ak_ifc::AbstractVector,
                                      bk_ifc::AbstractVector;
                                      T_ref::Real    = _DZ_T_REF_DEFAULT,
                                      R::Real        = _DZ_R_DEFAULT,
                                      gravity::Real  = _DZ_G_DEFAULT)
    ncells, Nz = size(dz)
    length(ps) == ncells || throw(DimensionMismatch(
        "ps length $(length(ps)) ≠ ncells=$(ncells) expected from dz"))
    length(ak_ifc) == Nz + 1 || throw(DimensionMismatch(
        "ak_ifc length $(length(ak_ifc)) ≠ Nz+1=$(Nz + 1)"))
    length(bk_ifc) == Nz + 1 || throw(DimensionMismatch(
        "bk_ifc length $(length(bk_ifc)) ≠ Nz+1=$(Nz + 1)"))
    FT = eltype(dz)
    backend = get_backend(dz)
    ak_dev = similar(dz, FT, Nz + 1)
    bk_dev = similar(dz, FT, Nz + 1)
    copyto!(ak_dev, FT.(ak_ifc))
    copyto!(bk_dev, FT.(bk_ifc))
    kernel = _dz_hydrostatic_constT_face_kernel!(backend, (256, 1))
    kernel(dz, ps, ak_dev, bk_dev,
           FT(T_ref), FT(R), FT(gravity), Nz;
           ndrange = (ncells, Nz))
    synchronize(backend)
    return dz
end
