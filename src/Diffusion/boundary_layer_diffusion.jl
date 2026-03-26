# ---------------------------------------------------------------------------
# Boundary-layer vertical diffusion — forward
#
# Implicit solve: (I - Δt * D) * c^{n+1} = c^n
# where D is a tridiagonal diffusion operator built from Kz profiles.
# Solved column-by-column with the Thomas algorithm.
#
# Two KernelAbstractions @kernel implementations (work on both CPU and GPU):
#   1. Lat-lon: factored Thomas solve on (Nx, Ny) arrays
#   2. Cubed-sphere: factored Thomas solve on haloed panel arrays
#      with rm ↔ mixing-ratio conversion
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid, CubedSphereGrid
using KernelAbstractions: @kernel, @index, synchronize, get_backend

"""
$(SIGNATURES)

Return the modifiable 3D array for a tracer (Field or raw array).
"""
tracer_data(t) = t isa AbstractField ? interior(t) : t

"""
$(SIGNATURES)

Exponential Kz profile at interface between level k and k+1.
Largest near surface (k close to Nz), decaying upward with e-folding
depth `H_scale` levels.

- `k`: interface index (between levels k and k+1)
- `Nz`: total number of vertical levels
- `Kz_max`: maximum diffusivity [Pa²/s]
- `H_scale`: e-folding depth in levels from surface
"""
function default_Kz_interface(k, Nz, Kz_max, H_scale, ::Type{FT}) where {FT}
    frac = FT(Nz - k) / FT(H_scale)
    return clamp(Kz_max * exp(-frac), zero(FT), Kz_max)
end

"""
$(SIGNATURES)

Solve tridiagonal system A*x = d in-place using Thomas algorithm.
a = sub-diagonal, b = main diagonal, c = super-diagonal.
a[1] and c[N] are not used (boundary).
"""
function thomas_solve!(a, b, c, d, x, w, g, N)
    @inbounds begin
        w[1] = c[1] / b[1]
        g[1] = d[1] / b[1]
        for k in 2:(N - 1)
            denom = b[k] - a[k] * w[k - 1]
            w[k] = c[k] / denom
            g[k] = (d[k] - a[k] * g[k - 1]) / denom
        end
        denom = b[N] - a[N] * w[N - 1]
        g[N] = (d[N] - a[N] * g[N - 1]) / denom
        x[N] = g[N]
        for k in (N - 1):-1:1
            x[k] = g[k] - w[k] * x[k + 1]
        end
    end
end

# =====================================================================
# Pre-compute tridiagonal coefficients (column-independent at ref p_s)
# =====================================================================

"""
Pre-compute the (a, b, c) tridiagonal coefficient vectors for the implicit
diffusion solve `(I - Δt·D) c_new = c_old`. Because the Kz profile and Δz
depend only on level index (at reference surface pressure), the coefficients
are the same for every (i,j) column and can be computed once.

Works with any grid type that defines `grid_size(grid).Nz` and `Δz(k, grid)`.
"""
function build_diffusion_coefficients(grid::AbstractGrid{FT},
                                      Kz_max::FT, H_scale::FT, Δt) where FT
    gs = grid_size(grid)
    Nz = gs.Nz
    a_v = Vector{FT}(undef, Nz)
    b_v = Vector{FT}(undef, Nz)
    c_v = Vector{FT}(undef, Nz)
    @inbounds for k in 1:Nz
        Δz_k = Δz(k, grid)
        D_below = zero(FT)
        D_above = zero(FT)
        if k > 1
            Δz_int_below = (Δz(k - 1, grid) + Δz_k) / 2
            Kz_below = default_Kz_interface(k - 1, Nz, Kz_max, H_scale, FT)
            D_below = Kz_below / (Δz_k * Δz_int_below)
        end
        if k < Nz
            Δz_int_above = (Δz_k + Δz(k + 1, grid)) / 2
            Kz_above = default_Kz_interface(k, Nz, Kz_max, H_scale, FT)
            D_above = Kz_above / (Δz_k * Δz_int_above)
        end
        D_kk = -(D_below + D_above)
        a_v[k] = k > 1  ? -Δt * D_below : zero(FT)
        b_v[k] = one(FT) - Δt * D_kk
        c_v[k] = k < Nz ? -Δt * D_above : zero(FT)
    end
    return a_v, b_v, c_v
end

# =====================================================================
# GPU kernel: one thread per (i,j) column, Thomas solve in registers
# =====================================================================

"""
Pre-factor the tridiagonal matrix into values needed for Thomas back-sub.
For each level k, store `w[k]` (the modified super-diagonal ratio) and
a normalization factor `inv_denom[k] = 1 / (b[k] - a[k]*w[k-1])`.
The kernel then only does:
  g[1] = d[1] * inv_denom[1]
  g[k] = (d[k] - a[k]*g[k-1]) * inv_denom[k]   for k=2..Nz
  x[Nz] = g[Nz]
  x[k] = g[k] - w[k]*x[k+1]                     for k=Nz-1..1
"""
function build_thomas_factors(a_v::Vector{FT}, b_v::Vector{FT},
                              c_v::Vector{FT}) where FT
    Nz = length(a_v)
    w = Vector{FT}(undef, Nz)
    inv_denom = Vector{FT}(undef, Nz)
    w[1] = c_v[1] / b_v[1]
    inv_denom[1] = one(FT) / b_v[1]
    for k in 2:Nz
        denom = b_v[k] - a_v[k] * w[k - 1]
        w[k] = k < Nz ? c_v[k] / denom : zero(FT)
        inv_denom[k] = one(FT) / denom
    end
    return w, inv_denom
end

@kernel function _diffuse_factored_kernel!(arr, @Const(a_v), @Const(w_v),
                                           @Const(inv_d), Nz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Forward sweep: g[k] = (d[k] - a[k]*g[k-1]) * inv_denom[k]
        g_prev = arr[i, j, 1] * inv_d[1]
        arr[i, j, 1] = g_prev
        for k in 2:Nz
            g_prev = (arr[i, j, k] - a_v[k] * g_prev) * inv_d[k]
            arr[i, j, k] = g_prev
        end
        # Back substitution: x[k] = g[k] - w[k]*x[k+1]
        for k in (Nz - 1):-1:1
            arr[i, j, k] -= w_v[k] * arr[i, j, k + 1]
        end
    end
end

# =====================================================================
# CS kernel: Thomas solve on haloed panels with rm ↔ mixing ratio
# =====================================================================

"""
GPU kernel for cubed-sphere boundary-layer diffusion.
Operates on haloed panel arrays where `rm = air_mass × mixing_ratio`.

For each (i,j) column:
1. Convert rm → mixing ratio (c = rm/m)
2. Apply implicit Thomas solve to c
3. Convert back rm = c_new × m
"""
@kernel function _diffuse_cs_panel_kernel!(rm, @Const(m), @Const(a_v), @Const(w_v),
                                           @Const(inv_d), Hp, Nz)
    i, j = @index(Global, NTuple)
    ii = Hp + i
    jj = Hp + j
    @inbounds begin
        # Forward sweep: convert rm→c, apply Thomas elimination
        # Store g values (modified mixing ratios) temporarily in rm
        c = rm[ii, jj, 1] / m[ii, jj, 1]
        g_prev = c * inv_d[1]
        rm[ii, jj, 1] = g_prev
        for k in 2:Nz
            c = rm[ii, jj, k] / m[ii, jj, k]
            g_prev = (c - a_v[k] * g_prev) * inv_d[k]
            rm[ii, jj, k] = g_prev
        end

        # Back substitution in mixing-ratio space
        # rm currently holds g values (modified mixing ratios)
        for k in (Nz - 1):-1:1
            rm[ii, jj, k] -= w_v[k] * rm[ii, jj, k + 1]
        end

        # Convert back: rm = c_new × m
        for k in 1:Nz
            rm[ii, jj, k] *= m[ii, jj, k]
        end
    end
end

# =====================================================================
# Diffusion workspace — pre-computed device arrays for GPU path
# =====================================================================

struct DiffusionWorkspace{FT, V <: AbstractVector{FT}}
    a_v     :: V
    w_v     :: V
    inv_d   :: V
    Nz      :: Int
end

function DiffusionWorkspace(grid::AbstractGrid{FT}, Kz_max::FT, H_scale::FT,
                            Δt, arr_template::AbstractArray{FT}) where FT
    a_v, b_v, c_v = build_diffusion_coefficients(grid, Kz_max, H_scale, FT(Δt))
    w, inv_denom  = build_thomas_factors(a_v, b_v, c_v)
    backend = get_backend(arr_template)
    to_dev(v) = typeof(similar(arr_template, length(v)))(v)
    return DiffusionWorkspace{FT, typeof(to_dev(a_v))}(
        to_dev(a_v), to_dev(w), to_dev(inv_denom), grid_size(grid).Nz)
end

function diffuse_gpu!(tracers::NamedTuple, dw::DiffusionWorkspace)
    for tracer in values(tracers)
        arr = tracer_data(tracer)
        backend = get_backend(arr)
        Nx, Ny = size(arr, 1), size(arr, 2)
        kernel! = _diffuse_factored_kernel!(backend, 256)
        kernel!(arr, dw.a_v, dw.w_v, dw.inv_d, dw.Nz; ndrange=(Nx, Ny))
        synchronize(backend)
    end
    return nothing
end

"""
    diffuse_cs_panels!(rm_panels, m_panels, dw, Nc, Nz, Hp)

Apply boundary-layer vertical diffusion to cubed-sphere tracer panels.
`rm_panels` is NTuple{6} of haloed 3D arrays (air_mass × mixing_ratio).
`m_panels` is NTuple{6} of haloed 3D arrays (air mass in kg).
"""
function diffuse_cs_panels!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                            dw::DiffusionWorkspace, Nc::Int, Nz::Int, Hp::Int)
    for_panels_nosync() do p
        backend = get_backend(rm_panels[p])
        kernel! = _diffuse_cs_panel_kernel!(backend, 256)
        kernel!(rm_panels[p], m_panels[p], dw.a_v, dw.w_v, dw.inv_d,
                Hp, Nz; ndrange=(Nc, Nc))
    end
    return nothing
end

# =====================================================================
# Unified path: build workspace on-the-fly and use KA kernel
# (works on both CPU and GPU via get_backend)
# =====================================================================

function diffuse!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid,
                  diff::BoundaryLayerDiffusion, Δt)
    FT = floattype(grid)
    # Use first tracer array as template for device placement
    arr_template = tracer_data(first(values(tracers)))
    dw = DiffusionWorkspace(grid, FT(diff.Kz_max), FT(diff.H_scale), FT(Δt), arr_template)
    diffuse_gpu!(tracers, dw)
    return nothing
end
