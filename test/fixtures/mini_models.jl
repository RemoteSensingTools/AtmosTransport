# ---------------------------------------------------------------------------
# Shared mini-model fixtures for core tests (BACKLOG #13).
#
# Lives OUTSIDE the tier folders on purpose: the tier runner executes every
# `test/<tier>/*.jl` as a standalone test, and this file is an include-only
# helper. Include via:
#     include(joinpath(@__DIR__, "..", "fixtures", "mini_models.jl"))
#
# Contract: the including module must already have an `AtmosTransport`
# binding (either `include(src/AtmosTransport.jl)` or `using AtmosTransport`
# — both bind the name, and `using .AtmosTransport` below resolves it). The builders return NamedTuples so call sites
# can destructure only what they need.
#
# Keep these MINIMAL and parameter-stable: tests pin numerical behavior on
# the exact values these produce. Add knobs, do not change defaults.
# ---------------------------------------------------------------------------

using .AtmosTransport
using .AtmosTransport.State: CellState, CubedSphereState, DryBasis
using .AtmosTransport.Grids: LatLonMesh, ReducedGaussianMesh, CubedSphereMesh

"""
    fixture_cs_panels(value, Np, Nz; FT = Float32)

Six identical halo-padded panels filled with `value`.
"""
fixture_cs_panels(value, Np, Nz; FT = Float32) =
    ntuple(_ -> fill(FT(value), Np, Np, Nz), 6)

"""
    fixture_cs_ramp_panels(base, slope, Np, Nz; FT = Float32)

Six identical panels with a vertical ramp `base + slope * k` (constant per
layer) — a structured tracer whose anomaly is non-trivial after a
global-mean subtraction.
"""
fixture_cs_ramp_panels(base, slope, Np, Nz; FT = Float32) =
    ntuple(_ -> FT(base) .+ FT(slope) .* reshape(collect(FT, 1:Nz), 1, 1, :) .*
                ones(FT, Np, Np, Nz), 6)

"""
    fixture_cs_state(; Nc = 4, Hp = 1, Nz = 3, FT = Float32, tracers...)

Mini cubed-sphere state on unit air mass. With no tracer kwargs you get the
historic two-tracer default (`co2` uniform 400 ppm, `sf6` uniform 10 ppt).
Returns `(; state, mesh, air)`.
"""
function fixture_cs_state(; Nc = 4, Hp = 1, Nz = 3, FT = Float32, tracers...)
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp)
    Np = Nc + 2Hp
    air = ntuple(_ -> ones(FT, Np, Np, Nz), 6)
    tr = isempty(tracers) ?
        (co2 = fixture_cs_panels(400e-6, Np, Nz; FT),
         sf6 = fixture_cs_panels(10e-12, Np, Nz; FT)) :
        NamedTuple(tracers)
    state = CubedSphereState(DryBasis, mesh, air; tr...)
    return (; state, mesh, air)
end

"""
    fixture_cs_model(; Nc = 4, Hp = 1, Nz = 3, FT = Float32,
                       scheme = LinRoodPPMScheme{7}(), tracers...)

Mini cubed-sphere `TransportModel` (state per [`fixture_cs_state`](@ref) +
simple 4-interface hybrid vertical + face fluxes). Returns
`(; model, grid, state, mesh)`.
"""
function fixture_cs_model(; Nc = 4, Hp = 1, Nz = 3, FT = Float32,
                            scheme = LinRoodPPMScheme{7}(), tracers...)
    fx = fixture_cs_state(; Nc, Hp, Nz, FT, tracers...)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(fx.mesh, vc, CPU(); FT = FT)
    fluxes = allocate_face_fluxes(fx.mesh, Nz; FT = FT, basis = DryBasis)
    model = TransportModel(fx.state, fluxes, grid, scheme)
    return (; model, grid, state = fx.state, mesh = fx.mesh)
end

"""
    fixture_ll_model(; Nx = 6, Ny = 4, Nz = 3, FT = Float32,
                       scheme = UpwindScheme(), tracers...)

Mini lat-lon `TransportModel` on ramped air mass. With no tracer kwargs you
get `co2 = 4e-4 * air` and `sf6 = 1e-11 * air`. Returns
`(; model, grid, state, air)`.
"""
function fixture_ll_model(; Nx = 6, Ny = 4, Nz = 3, FT = Float32,
                            scheme = UpwindScheme(), tracers...)
    mesh = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vc = HybridSigmaPressure(FT[0, 100, 300, 600], FT[0, 0, 0.5, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    air = FT(1e16) .+ FT(1e14) .* reshape(1:(Nx * Ny * Nz), Nx, Ny, Nz)
    tr = isempty(tracers) ?
        (co2 = FT(4e-4) .* air, sf6 = FT(1e-11) .* air) : NamedTuple(tracers)
    state = CellState(DryBasis, air; tr...)
    fluxes = allocate_face_fluxes(grid.horizontal, Nz; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, grid, scheme)
    return (; model, grid, state, air)
end

"""
    fixture_rg_model(; Nz = 2, FT = Float64, tracers...)

Mini face-indexed reduced-Gaussian `TransportModel` (two latitude rings of
four cells) on unit air mass; default single tracer `CO2 = 400e-6 * air`.
Returns `(; model, grid, state, mesh)`.
"""
function fixture_rg_model(; Nz = 2, FT = Float64, tracers...)
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vc = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)
    air = ones(FT, ncells(mesh), Nz)
    tr = isempty(tracers) ? (CO2 = FT(400e-6) .* air,) : NamedTuple(tracers)
    state = CellState(DryBasis, air; tr...)
    fluxes = allocate_face_fluxes(mesh, Nz; FT = FT, basis = DryBasis)
    model = TransportModel(state, fluxes, grid, UpwindScheme())
    return (; model, grid, state, mesh)
end
