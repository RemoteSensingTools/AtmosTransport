# ---------------------------------------------------------------------------
# Cell divergence from face fluxes
#
# Computes the discrete divergence of face-centered fluxes into cell updates.
# For structured grids, this is straightforward differencing.
# For unstructured grids (Phase 2+), it uses the CSR connectivity.
# ---------------------------------------------------------------------------

using KernelAbstractions: @kernel, @index, get_backend, synchronize

"""
    diagnose_cm!(cm, am, bm, bt, Nz)

Diagnose vertical mass flux `cm` from horizontal convergence and the
B-coefficient tendency, ensuring column mass conservation.

This is the continuity equation:
    cm[k+1] = cm[k] + (am[i,j,k] - am[i+1,j,k] + bm[i,j,k] - bm[i,j+1,k]) - bt[k] * pit

where `pit = Σ_k (horizontal convergence)` is the total column pressure tendency.
"""
@kernel function _diagnose_cm_kernel!(cm, @Const(am), @Const(bm), @Const(bt), Nz)
    i, j = @index(Global, NTuple)
    FT = eltype(cm)
    @inbounds begin
        pit = zero(FT)
        for k in 1:Nz
            pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        end
        acc = zero(FT)
        cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            acc += conv_k - bt[k] * pit
            cm[i, j, k+1] = acc
        end
    end
end

"""
    diagnose_cm!(cm, am, bm, bt)

Host entry point for vertical flux diagnosis.
"""
function diagnose_cm!(cm::AbstractArray{FT,3}, am::AbstractArray{FT,3},
                      bm::AbstractArray{FT,3}, bt::AbstractVector{FT}) where FT
    Nx, Ny, Nz = size(am, 1) - 1, size(am, 2), size(am, 3)
    backend = get_backend(cm)
    kernel! = _diagnose_cm_kernel!(backend, 256)
    kernel!(cm, am, bm, bt, Int32(Nz); ndrange=(Nx, Ny))
    synchronize(backend)
    return nothing
end

export diagnose_cm!
