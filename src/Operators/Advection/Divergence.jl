# ---------------------------------------------------------------------------
# Backward-compatible vertical-closure wrapper.
#
# The single implementation lives in MetDrivers now. Operators keep a thin
# `diagnose_cm!` shim so existing structured-grid tests and call sites do not
# break while the generic runtime is being built out.
# ---------------------------------------------------------------------------

"""
    diagnose_cm!(cm, am, bm, bt)

Backward-compatible wrapper for structured-grid continuity closure.
"""
function diagnose_cm!(cm::AbstractArray{FT, 3},
                      am::AbstractArray{FT, 3},
                      bm::AbstractArray{FT, 3},
                      bt::AbstractVector{FT}) where FT
    Nx, Ny, Nz = size(am, 1) - 1, size(am, 2), size(am, 3)
    diagnose_cm_from_continuity!(cm, am, bm, bt, Nx, Ny, Nz)
    return nothing
end

export diagnose_cm!
