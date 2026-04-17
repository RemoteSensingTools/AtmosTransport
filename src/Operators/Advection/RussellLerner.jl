"""
    RussellLernerAdvection <: AbstractLinearReconstruction

Legacy van Leer slopes advection wrapper (TM5-compatible). All dispatch
was migrated to `SlopesScheme{L}` in plan 12 — this type is retained
only until the struct-removal step for a clean two-step diff.
"""
struct RussellLernerAdvection <: AbstractLinearReconstruction
    use_limiter :: Bool
end

RussellLernerAdvection(; use_limiter::Bool = true) = RussellLernerAdvection(use_limiter)

export RussellLernerAdvection
