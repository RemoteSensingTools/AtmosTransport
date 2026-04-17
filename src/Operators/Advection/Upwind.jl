"""
    UpwindAdvection <: AbstractConstantReconstruction

Legacy first-order upwind advection wrapper. All dispatch was migrated to
`UpwindScheme` in plan 12 — this type is retained only until the
struct-removal step for a clean two-step diff.
"""
struct UpwindAdvection <: AbstractConstantReconstruction end

const FirstOrderUpwindAdvection = UpwindAdvection

export UpwindAdvection, FirstOrderUpwindAdvection
