"""
    Chemistry

Atmospheric chemistry interface (stub for future extension).

The chemistry module follows the same forward/adjoint pairing as other operators.
Currently only `NoChemistry` is provided. Future implementations (CBM4, mCB05,
MOGUNTIA) subtype `AbstractChemistry` and implement the interface.

# Interface contract

    apply_chemistry!(tracers, grid, chem::AbstractChemistry, Δt)
    adjoint_chemistry!(adj_tracers, grid, chem::AbstractChemistry, Δt)
"""
module Chemistry

export AbstractChemistry, NoChemistry
export apply_chemistry!, adjoint_chemistry!

"""
    AbstractChemistry

Supertype for chemistry schemes. Future extensions (CBM4, mCB05, MOGUNTIA,
aerosol modules) subtype this and implement forward + adjoint methods.
"""
abstract type AbstractChemistry end

"""No chemistry (inert tracers). Both forward and adjoint are no-ops."""
struct NoChemistry <: AbstractChemistry end

apply_chemistry!(tracers, grid, ::NoChemistry, Δt) = nothing
adjoint_chemistry!(adj_tracers, grid, ::NoChemistry, Δt) = nothing

end # module Chemistry
