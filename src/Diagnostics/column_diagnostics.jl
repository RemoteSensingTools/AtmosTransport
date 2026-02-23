# ---------------------------------------------------------------------------
# Column-mean and surface-slice diagnostic kernels (KernelAbstractions)
#
# Two dispatch paths:
#   Lat-lon:       3D arrays (Nx, Ny, Nz)
#   Cubed-sphere:  NTuple{6} of haloed 3D panels
# ---------------------------------------------------------------------------

# =====================================================================
# Lat-lon kernels
# =====================================================================

@kernel function _column_mean_kernel!(c_col, @Const(c), @Const(m), Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        sum_cm = zero(FT)
        sum_m  = zero(FT)
        for k in 1:Nz
            sum_cm += c[i, j, k] * m[i, j, k]
            sum_m  += m[i, j, k]
        end
        c_col[i, j] = sum_m > zero(FT) ? sum_cm / sum_m : zero(FT)
    end
end

@kernel function _surface_slice_kernel!(c_sfc, @Const(c), Nz)
    i, j = @index(Global, NTuple)
    @inbounds c_sfc[i, j] = c[i, j, Nz]
end

"""
    column_mean!(c_col, c, m)

Compute mass-weighted column mean of 3D tracer field `c` weighted by
air mass `m`. Result stored in 2D array `c_col`.

Works on both CPU and GPU via KernelAbstractions.
"""
function column_mean!(c_col::AbstractMatrix{FT}, c::AbstractArray{FT,3},
                      m::AbstractArray{FT,3}) where FT
    backend = get_backend(c)
    Nx, Ny, Nz = size(c)
    k! = _column_mean_kernel!(backend, 256)
    k!(c_col, c, m, Nz; ndrange=(Nx, Ny))
    synchronize(backend)
end

"""
    surface_slice!(c_sfc, c)

Extract the surface layer (k = Nz) from a 3D field into a 2D array.
"""
function surface_slice!(c_sfc::AbstractMatrix{FT}, c::AbstractArray{FT,3}) where FT
    backend = get_backend(c)
    Nx, Ny, Nz = size(c)
    k! = _surface_slice_kernel!(backend, 256)
    k!(c_sfc, c, Nz; ndrange=(Nx, Ny))
    synchronize(backend)
end

"""
    compute_diagnostics!(c_col, c_sfc, c, m)

Compute both column mean and surface slice in one call (lat-lon).
"""
function compute_diagnostics!(c_col::AbstractMatrix, c_sfc::AbstractMatrix,
                               c::AbstractArray{<:Any,3}, m::AbstractArray{<:Any,3})
    column_mean!(c_col, c, m)
    surface_slice!(c_sfc, c)
end

# =====================================================================
# Cubed-sphere panel kernels
# =====================================================================

@kernel function _cs_column_mean_kernel!(c_col, @Const(rm), @Const(m), Hp, Nz)
    i, j = @index(Global, NTuple)
    FT_k = eltype(rm)
    @inbounds begin
        sum_cm = zero(FT_k)
        sum_m  = zero(FT_k)
        for k in 1:Nz
            mk = m[Hp + i, Hp + j, k]
            sum_cm += rm[Hp + i, Hp + j, k]
            sum_m  += mk
        end
        c_col[i, j] = sum_m > zero(FT_k) ? sum_cm / sum_m : zero(FT_k)
    end
end

"""
    column_mean!(c_col_panels, rm_panels, m_panels, Nc, Nz, Hp)

Compute mass-weighted column mean for cubed-sphere panels.
`rm_panels` and `m_panels` are NTuple{6} of haloed 3D arrays.
`c_col_panels` is NTuple{6} of (Nc × Nc) 2D arrays.
"""
function column_mean!(c_col_panels::NTuple{6}, rm_panels::NTuple{6},
                      m_panels::NTuple{6}, Nc::Int, Nz::Int, Hp::Int)
    backend = get_backend(rm_panels[1])
    k! = _cs_column_mean_kernel!(backend, 256)
    for p in 1:6
        k!(c_col_panels[p], rm_panels[p], m_panels[p], Hp, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

@kernel function _cs_surface_slice_kernel!(c_sfc, @Const(c), Hp, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c_sfc[i, j] = c[Hp + i, Hp + j, Nz]
end

"""
    surface_slice!(c_sfc_panels, c_panels, Nc, Nz, Hp)

Extract surface layer from cubed-sphere panels.
"""
function surface_slice!(c_sfc_panels::NTuple{6}, c_panels::NTuple{6},
                         Nc::Int, Nz::Int, Hp::Int)
    backend = get_backend(c_panels[1])
    k! = _cs_surface_slice_kernel!(backend, 256)
    for p in 1:6
        k!(c_sfc_panels[p], c_panels[p], Hp, Nz; ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

"""
    compute_diagnostics!(c_col_panels, c_sfc_panels, rm_panels, m_panels, Nc, Nz, Hp)

Compute both column mean and surface slice for cubed-sphere panels.
"""
function compute_diagnostics!(c_col_panels::NTuple{6}, c_sfc_panels::NTuple{6},
                               rm_panels::NTuple{6}, m_panels::NTuple{6},
                               Nc::Int, Nz::Int, Hp::Int)
    column_mean!(c_col_panels, rm_panels, m_panels, Nc, Nz, Hp)
    surface_slice!(c_sfc_panels, rm_panels, Nc, Nz, Hp)
end
