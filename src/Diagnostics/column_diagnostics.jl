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

# =====================================================================
# Column mass kernels (total tracer mass per column, kg)
# =====================================================================

@kernel function _column_mass_kernel!(cm_col, @Const(c), @Const(m), Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        total = zero(FT)
        for k in 1:Nz
            total += c[i, j, k] * m[i, j, k]
        end
        cm_col[i, j] = total
    end
end

"""
    column_mass!(cm_col, c, m)

Compute total tracer mass per column: cm_col[i,j] = Σ_k(c[i,j,k] × m[i,j,k]).
Result is in kg (tracer mass per grid cell column). Divide by cell area for kg/m².
"""
function column_mass!(cm_col::AbstractMatrix{FT}, c::AbstractArray{FT,3},
                      m::AbstractArray{FT,3}) where FT
    backend = get_backend(c)
    Nx, Ny, Nz = size(c)
    k! = _column_mass_kernel!(backend, 256)
    k!(cm_col, c, m, Nz; ndrange=(Nx, Ny))
    synchronize(backend)
end

@kernel function _cs_column_mass_kernel!(cm_col, @Const(rm), Hp, Nz)
    i, j = @index(Global, NTuple)
    FT_k = eltype(rm)
    @inbounds begin
        total = zero(FT_k)
        for k in 1:Nz
            total += rm[Hp + i, Hp + j, k]
        end
        cm_col[i, j] = total
    end
end

"""
    column_mass!(cm_col_panels, rm_panels, Nc, Nz, Hp)

Compute total tracer mass per column for cubed-sphere panels.
`rm_panels` are NTuple{6} of haloed 3D arrays (tracer mass per cell).
Result is in kg per column. Divide by cell area for kg/m².
"""
function column_mass!(cm_col_panels::NTuple{6}, rm_panels::NTuple{6},
                      Nc::Int, Nz::Int, Hp::Int)
    backend = get_backend(rm_panels[1])
    k! = _cs_column_mass_kernel!(backend, 256)
    for p in 1:6
        k!(cm_col_panels[p], rm_panels[p], Hp, Nz; ndrange=(Nc, Nc))
    end
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

# =====================================================================
# Sigma-level extraction kernels
# =====================================================================

@kernel function _sigma_level_kernel!(c_out, @Const(c), @Const(m), Nz, sigma)
    i, j = @index(Global, NTuple)
    FT = eltype(c)
    @inbounds begin
        total_m = zero(FT)
        for k in 1:Nz
            total_m += m[i, j, k]
        end
        target = FT(sigma) * total_m
        cum_m = zero(FT)
        found = false
        for k in 1:Nz
            cum_m += m[i, j, k]
            if !found && cum_m >= target
                c_out[i, j] = c[i, j, k]
                found = true
            end
        end
        if !found
            c_out[i, j] = c[i, j, Nz]
        end
    end
end

"""
    sigma_level_slice!(c_out, c, m, sigma)

Extract tracer at the model level closest to `sigma × p_surface`.
`c` is 3D (Nx, Ny, Nz), `m` is air mass (same shape), `c_out` is 2D (Nx, Ny).
"""
function sigma_level_slice!(c_out::AbstractMatrix{FT}, c::AbstractArray{FT,3},
                             m::AbstractArray{FT,3}, sigma::Float64) where FT
    backend = get_backend(c)
    Nx, Ny, Nz = size(c)
    k! = _sigma_level_kernel!(backend, 256)
    k!(c_out, c, m, Nz, FT(sigma); ndrange=(Nx, Ny))
    synchronize(backend)
end

@kernel function _cs_sigma_level_kernel!(c_out, @Const(rm), @Const(m), Hp, Nz, sigma)
    i, j = @index(Global, NTuple)
    FT_k = eltype(rm)
    @inbounds begin
        total_m = zero(FT_k)
        for k in 1:Nz
            total_m += m[Hp + i, Hp + j, k]
        end
        target = sigma * total_m
        cum_m = zero(FT_k)
        found = false
        for k in 1:Nz
            mk = m[Hp + i, Hp + j, k]
            cum_m += mk
            if !found && cum_m >= target
                c_out[i, j] = mk > zero(FT_k) ? rm[Hp + i, Hp + j, k] / mk : zero(FT_k)
                found = true
            end
        end
        if !found
            mk = m[Hp + i, Hp + j, Nz]
            c_out[i, j] = mk > zero(FT_k) ? rm[Hp + i, Hp + j, Nz] / mk : zero(FT_k)
        end
    end
end

# =====================================================================
# Column tracer flux kernels (for time-integrated flux diagnostics)
#
# Compute Σ_k mf[face_k] × qc[k] where qc = rm/m (mixing ratio).
# Output is in kg/s (instantaneous column tracer flux through face).
# For time-integrated CATRINE FE/FN: accumulate over windows.
# =====================================================================

@kernel function _cs_column_tracer_flux_x_kernel!(flux_col, @Const(am), @Const(rm), @Const(m), Hp, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        total = zero(FT)
        for k in 1:Nz
            mk = m[Hp + i, Hp + j, k]
            qc = mk > zero(FT) ? rm[Hp + i, Hp + j, k] / mk : zero(FT)
            # East face flux: am[i+1,j,k] is the flux leaving cell (i,j) eastward
            total += am[i + 1, j, k] * qc
        end
        flux_col[i, j] = total
    end
end

@kernel function _cs_column_tracer_flux_y_kernel!(flux_col, @Const(bm), @Const(rm), @Const(m), Hp, Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(rm)
    @inbounds begin
        total = zero(FT)
        for k in 1:Nz
            mk = m[Hp + i, Hp + j, k]
            qc = mk > zero(FT) ? rm[Hp + i, Hp + j, k] / mk : zero(FT)
            # North face flux: bm[i,j+1,k] is the flux leaving cell (i,j) northward
            total += bm[i, j + 1, k] * qc
        end
        flux_col[i, j] = total
    end
end

"""
    column_tracer_flux!(flux_panels, mf_panels, rm_panels, m_panels, Nc, Nz, Hp, direction)

Compute vertically-integrated column tracer flux for cubed-sphere panels.

`mf_panels` are mass flux panels: pass `am` for `direction = :east`,
`bm` for `direction = :north`. `rm_panels` is tracer mass (kg),
`m_panels` is air mass (kg). `flux_panels` is NTuple{6} of (Nc × Nc)
output arrays, filled with Σ_k mf[face_k] × (rm[k]/m[k]) in kg/s.

Note: the mass fluxes may already be scaled (e.g. by `half_dt`).
The caller is responsible for unscaling when accumulating.
"""
function column_tracer_flux!(flux_panels::NTuple{6}, mf_panels::NTuple{6},
                              rm_panels::NTuple{6}, m_panels::NTuple{6},
                              Nc::Int, Nz::Int, Hp::Int, direction::Symbol)
    backend = get_backend(rm_panels[1])
    if direction === :east
        k! = _cs_column_tracer_flux_x_kernel!(backend, 256)
    else
        k! = _cs_column_tracer_flux_y_kernel!(backend, 256)
    end
    for p in 1:6
        k!(flux_panels[p], mf_panels[p], rm_panels[p], m_panels[p], Hp, Nz;
           ndrange=(Nc, Nc))
    end
    synchronize(backend)
end

"""
    sigma_level_slice!(c_out_panels, rm_panels, m_panels, Nc, Nz, Hp, sigma)

Extract tracer mixing ratio at the sigma level for cubed-sphere panels.
`rm_panels` is tracer mass, `m_panels` is air mass (both NTuple{6} of haloed 3D arrays).
`c_out_panels` is NTuple{6} of (Nc × Nc) 2D arrays.
"""
function sigma_level_slice!(c_out_panels::NTuple{6}, rm_panels::NTuple{6},
                             m_panels::NTuple{6}, Nc::Int, Nz::Int, Hp::Int,
                             sigma::Float64)
    backend = get_backend(rm_panels[1])
    FT = eltype(rm_panels[1])
    k! = _cs_sigma_level_kernel!(backend, 256)
    for p in 1:6
        k!(c_out_panels[p], rm_panels[p], m_panels[p], Hp, Nz, FT(sigma);
           ndrange=(Nc, Nc))
    end
    synchronize(backend)
end
