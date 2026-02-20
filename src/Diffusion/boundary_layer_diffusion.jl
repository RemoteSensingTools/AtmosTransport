# ---------------------------------------------------------------------------
# Boundary-layer vertical diffusion — forward
#
# Implicit solve: (I - Δt * D) * c^{n+1} = c^n
# where D is a tridiagonal diffusion operator built from Kz profiles.
# Solved column-by-column with the Thomas algorithm.
#
# Two implementations:
#   1. CPU: column-by-column loop with preallocated workspace
#   2. GPU: KernelAbstractions @kernel — one (i,j) thread per column,
#      pre-computed tridiagonal coefficients passed as device vectors
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, Δz, floattype, LatitudeLongitudeGrid
using KernelAbstractions: @kernel, @index, synchronize, get_backend

"""
$(SIGNATURES)

Return the modifiable 3D array for a tracer (Field or raw array).
"""
tracer_data(t) = t isa AbstractField ? interior(t) : t

"""
$(SIGNATURES)

Default exponential Kz profile at interface between level k and k+1.
Decreases with height (smaller at top, larger near surface).
Clamped to [0, Kz_max].
"""
function default_Kz_interface(k, Nz, Kz_max, ::Type{FT}) where {FT}
    scale_height = FT(Nz)
    Kz = Kz_max * exp(-(Nz - k - FT(0.5)) / scale_height)
    return clamp(Kz, zero(FT), Kz_max)
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
"""
function build_diffusion_coefficients(grid::LatitudeLongitudeGrid{FT},
                                      Kz_max::FT, Δt) where FT
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
            Kz_below = default_Kz_interface(k - 1, Nz, Kz_max, FT)
            D_below = Kz_below / (Δz_k * Δz_int_below)
        end
        if k < Nz
            Δz_int_above = (Δz_k + Δz(k + 1, grid)) / 2
            Kz_above = default_Kz_interface(k, Nz, Kz_max, FT)
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
# Diffusion workspace — pre-computed device arrays for GPU path
# =====================================================================

struct DiffusionWorkspace{FT, V <: AbstractVector{FT}}
    a_v     :: V
    w_v     :: V
    inv_d   :: V
    Nz      :: Int
end

function DiffusionWorkspace(grid::LatitudeLongitudeGrid{FT}, Kz_max::FT,
                            Δt, arr_template::AbstractArray{FT}) where FT
    a_v, b_v, c_v = build_diffusion_coefficients(grid, Kz_max, FT(Δt))
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

# =====================================================================
# CPU path (original column-by-column)
# =====================================================================

function diffuse!(tracers::NamedTuple, met, grid::LatitudeLongitudeGrid, diff::BoundaryLayerDiffusion, Δt)
    gs = grid_size(grid)
    Nx, Ny, Nz = gs.Nx, gs.Ny, gs.Nz
    FT = floattype(grid)
    Kz_max = diff.Kz_max

    a = Vector{FT}(undef, Nz)
    b = Vector{FT}(undef, Nz)
    c = Vector{FT}(undef, Nz)
    w = Vector{FT}(undef, Nz)
    g = Vector{FT}(undef, Nz)
    col = Vector{FT}(undef, Nz)

    for tracer in values(tracers)
        arr = tracer_data(tracer)
        for j in 1:Ny, i in 1:Nx
            @inbounds for k in 1:Nz
                col[k] = arr[i, j, k]
            end

            @inbounds for k in 1:Nz
                Δz_k = Δz(k, grid)
                D_below = zero(FT)
                D_above = zero(FT)
                if k > 1
                    Δz_int_below = (Δz(k - 1, grid) + Δz_k) / 2
                    Kz_below = default_Kz_interface(k - 1, Nz, Kz_max, FT)
                    D_below = Kz_below / (Δz_k * Δz_int_below)
                end
                if k < Nz
                    Δz_int_above = (Δz_k + Δz(k + 1, grid)) / 2
                    Kz_above = default_Kz_interface(k, Nz, Kz_max, FT)
                    D_above = Kz_above / (Δz_k * Δz_int_above)
                end
                D_kk = -(D_below + D_above)

                a[k] = k > 1 ? -Δt * D_below : zero(FT)
                b[k] = one(FT) - Δt * D_kk
                c[k] = k < Nz ? -Δt * D_above : zero(FT)
            end

            thomas_solve!(a, b, c, col, col, w, g, Nz)

            @inbounds for k in 1:Nz
                arr[i, j, k] = col[k]
            end
        end
    end
    return nothing
end
