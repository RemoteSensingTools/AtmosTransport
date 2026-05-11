#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 25 Commit 6 — LinRood end-to-end adjoint integration test.
#
# Verifies that `cs_surface_emission_footprint(scheme=LinRoodPPMScheme())`
# returns gradients consistent with centered finite-difference probes on
# a small 6-panel CS problem (Nc=4, Hp=3, Nz=3). Exercises the cross-panel
# halo adjoint (`_adjoint_fill_panel_halos!`) via the LinRood tape
# integration shipped in this commit.
# ---------------------------------------------------------------------------

using Test
using Random

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport
const Adv = AtmosTransport.Operators.Advection

# Reuse the existing test helpers.
include(joinpath(@__DIR__, "test_cs_ppm_adjoint_footprint.jl"))

@testset "Plan 25 Commit 6 — LinRoodPPMScheme footprint vs FD" begin
    mesh, panels_m, panels_rm, panels_am, panels_bm, panels_cm =
        _transport_cs_problem(Nc=3, Nz=4, nsteps=2)
    dt = 1.5
    scheme = AT.LinRoodPPMScheme()
    obj = AT.CSColumnMeanObjective(1, 2, 2)

    result = AT.cs_surface_emission_footprint(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt, epsilon=1e-6,
    )

    rates = [ntuple(6) do p
        [sin(0.31step + 0.17p + 0.23i - 0.19j) for i in 1:mesh.Nc, j in 1:mesh.Nc]
    end for step in 1:2]

    eps_dir = 2e-6
    j_plus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, eps_dir))
    j_minus = AT.run_cs_footprint_forward(
        panels_rm, panels_m, panels_am, panels_bm, panels_cm, mesh, obj;
        scheme=scheme, dt=dt,
        emission_rates=_scaled_rates(rates, -eps_dir))

    fd = (j_plus - j_minus) / (2eps_dir)
    predicted = _dot_footprint(result, rates)

    # LinRood is more nonlinear than the linear schemes; tolerance
    # accommodates centered-FD truncation at eps_dir=2e-6 and the
    # branch-rich PPM chain.
    @test predicted ≈ fd rtol=2e-4 atol=1e-7
end
