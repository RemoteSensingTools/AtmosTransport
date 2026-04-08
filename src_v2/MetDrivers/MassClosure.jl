# ---------------------------------------------------------------------------
# Mass closure strategies
#
# Different met sources provide different information for closing the
# vertical mass budget. The closure strategy determines how cm is obtained.
# ---------------------------------------------------------------------------

"""
    AbstractMassClosure

Strategy for ensuring column dry mass conservation.
"""
abstract type AbstractMassClosure end

"""
    DiagnoseVerticalFromHorizontal <: AbstractMassClosure

Diagnose vertical fluxes from horizontal convergence + pressure tendency.
This is the default for ERA5 (spectral or gridded) and the standard
approach in TM5: cm is not read from data but computed from am, bm, and
the B coefficients of the vertical coordinate.
"""
struct DiagnoseVerticalFromHorizontal <: AbstractMassClosure end

"""
    PressureTendencyClosure <: AbstractMassClosure

Use the surface pressure tendency dp/dt to close the vertical budget.
"""
struct PressureTendencyClosure <: AbstractMassClosure end

"""
    NativeVerticalFluxClosure <: AbstractMassClosure

Use natively provided vertical fluxes (e.g., omega or etadot from the
host model). No diagnosis needed.
"""
struct NativeVerticalFluxClosure <: AbstractMassClosure end

export AbstractMassClosure
export DiagnoseVerticalFromHorizontal, PressureTendencyClosure, NativeVerticalFluxClosure
