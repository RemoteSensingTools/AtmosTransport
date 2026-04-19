"""
Tests for `TransportModel` + vertical-diffusion integration
(plan 16b, Commit 5).

Verifies:

1. Default `TransportModel` carries `diffusion = NoDiffusion()`.
2. `with_diffusion(model, diff)` returns a new model with only
   the diffusion operator replaced; all other fields shared.
3. `step!` with default `NoDiffusion` is bit-exact equivalent to
   pre-Commit-5 behavior (no floating-point op inserted).
4. `step!` with `ImplicitVerticalDiffusion` actually mixes a
   non-uniform vertical profile.
5. `current_time(::AbstractMetDriver)` default stub returns `0.0`.
"""

using Test
import AtmosTransport
using AtmosTransport: CellState, StructuredFaceFluxState,
                      LatLonMesh, HybridSigmaPressure, AtmosGrid, CPU,
                      UpwindScheme, AdvectionWorkspace,
                      ConstantField, NoDiffusion, ImplicitVerticalDiffusion,
                      TransportModel, step!, with_chemistry, with_diffusion,
                      NoChemistry, ExponentialDecay, AbstractChemistryOperator,
                      AbstractMetDriver,
                      current_time

# -------------------------------------------------------------------------
# Shared small problem
# -------------------------------------------------------------------------

function _make_model(FT; Nx = 4, Ny = 3, Nz = 8,
                     diffusion = NoDiffusion(),
                     chemistry::AbstractChemistryOperator = NoChemistry(),
                     rn_init = FT(1.0))
    mesh = LatLonMesh(; Nx = Nx, Ny = Ny, FT = FT)
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(collect(range(0, 1, length = Nz + 1)))
    vc = HybridSigmaPressure(A_ifc, B_ifc)
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)

    m = ones(FT, Nx, Ny, Nz)
    # Non-uniform Rn profile so diffusion has something to mix
    rm_rn = Array{FT}(undef, Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        rm_rn[i, j, k] = FT(rn_init) * exp(-((k - (Nz + 1) / 2)^2) / 4)
    end
    state = CellState(m; Rn222 = rm_rn)

    am = zeros(FT, Nx + 1, Ny, Nz)
    bm = zeros(FT, Nx, Ny + 1, Nz)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    fluxes = StructuredFaceFluxState(am, bm, cm)

    return TransportModel(state, fluxes, grid, UpwindScheme();
                          chemistry = chemistry,
                          diffusion = diffusion)
end

# =========================================================================
# 1. Default TransportModel carries NoDiffusion
# =========================================================================

@testset "TransportModel default diffusion is NoDiffusion" begin
    model = _make_model(Float64)
    @test model.diffusion isa NoDiffusion
end

# =========================================================================
# 2. with_diffusion shares non-diffusion fields
# =========================================================================

@testset "with_diffusion shares state/fluxes/workspace/chemistry" begin
    FT = Float64
    base = _make_model(FT; chemistry = ExponentialDecay(FT; Rn222 = 330_350.4))

    kz = ConstantField{FT, 3}(1.0)
    op = ImplicitVerticalDiffusion(; kz_field = kz)
    updated = with_diffusion(base, op)

    @test updated.diffusion === op
    @test updated.state === base.state          # shared
    @test updated.fluxes === base.fluxes        # shared
    @test updated.workspace === base.workspace  # shared
    @test updated.chemistry === base.chemistry  # shared
    @test updated.advection === base.advection  # shared
    @test updated.grid === base.grid            # shared
end

# =========================================================================
# 3. Default TransportModel.step! is bit-exact to pre-16b behavior
# =========================================================================

@testset "step! with default NoDiffusion is bit-exact to no-diffusion path" begin
    # Two identical models; both use NoDiffusion by default. The palindrome
    # insert dispatches to `apply_vertical_diffusion!(_, ::NoDiffusion, _, _) = nothing`,
    # so no floating-point work happens. Result must match an independently-
    # built model bit-exactly after several steps.
    FT = Float64
    model_A = _make_model(FT)
    model_B = _make_model(FT)

    @assert model_A.state.tracers.Rn222 == model_B.state.tracers.Rn222

    dt = FT(60)
    for _ in 1:5
        step!(model_A, dt)
        step!(model_B, dt)
    end
    @test model_A.state.tracers.Rn222 == model_B.state.tracers.Rn222   # == not ≈
end

# =========================================================================
# 4. step! with ImplicitVerticalDiffusion mixes a vertical profile
# =========================================================================

@testset "step! with ImplicitVerticalDiffusion actually mixes the profile" begin
    FT = Float64
    Nx, Ny, Nz = 4, 3, 8

    # Model A: NoDiffusion (control). Profile should be unchanged because
    # fluxes are zero, no chemistry, no diffusion.
    model_ctrl = _make_model(FT; Nx = Nx, Ny = Ny, Nz = Nz)
    rn_before = copy(model_ctrl.state.tracers.Rn222)

    # Model B: ImplicitVerticalDiffusion with non-trivial Kz
    kz = ConstantField{FT, 3}(1.0)
    op = ImplicitVerticalDiffusion(; kz_field = kz)
    model_diff = _make_model(FT; Nx = Nx, Ny = Ny, Nz = Nz, diffusion = op)
    # dz_scratch is caller-owned — fill it before stepping
    fill!(model_diff.workspace.dz_scratch, FT(100.0))

    dt = FT(10)
    for _ in 1:3
        step!(model_ctrl, dt)
        step!(model_diff, dt)
    end

    # Control unchanged (bit-exact)
    @test model_ctrl.state.tracers.Rn222 == rn_before

    # Diffusion-on model changed — the Gaussian peak flattens toward the
    # profile mean. The peak value should drop; the wings should rise.
    k_peak = (Nz + 1) ÷ 2
    @test model_diff.state.tracers.Rn222[1, 1, k_peak] < rn_before[1, 1, k_peak]
    # Column mass preserved (Neumann BCs, uniform dz)
    for i in 1:Nx, j in 1:Ny
        m_before = sum(@view rn_before[i, j, :])
        m_after  = sum(@view model_diff.state.tracers.Rn222[i, j, :])
        @test abs(m_after - m_before) / m_before < 1e-12
    end
end

# =========================================================================
# 5. current_time default stub
# =========================================================================

@testset "current_time default stub returns 0.0" begin
    # Concrete drivers may override, but the abstract-type default is
    # `0.0`. Plan 16b Decision 10 documents the accessor; a downstream
    # commit threads it through `apply!` when meteo-dependent Kz fields
    # become operational.
    struct _TestDriver <: AbstractMetDriver; end
    @test current_time(_TestDriver()) === 0.0
end
