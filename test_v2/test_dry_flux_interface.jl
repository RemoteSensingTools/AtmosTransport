#!/usr/bin/env julia
"""
Validation tests for the v2 dry-mass face-flux transport architecture.

Tests:
1. Module loading and type construction
2. Geometry API correctness for LatLonMesh
3. CellState and FaceFluxState construction
4. Uniform field invariance through Strang splitting
5. Mass conservation (total tracer mass preserved)
6. Vertical flux diagnosis (cm from horizontal continuity)
7. ERA5 dry flux builder
8. Numerical equivalence with src/ mass-flux advection
"""

using Test

# Load v2 module
include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2: Grids, State, Operators, MetDrivers

# For numerical equivalence test, also load v1
include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))

# =========================================================================
# Test 1: Module loading and type construction
# =========================================================================
@testset "v2 module loading" begin
    @test isdefined(AtmosTransportV2, :LatLonMesh)
    @test isdefined(AtmosTransportV2, :CellState)
    @test isdefined(AtmosTransportV2, :AbstractFaceFluxState)
    @test isdefined(AtmosTransportV2, :AbstractStructuredFaceFluxState)
    @test isdefined(AtmosTransportV2, :StructuredFaceFluxState)
    @test isdefined(AtmosTransportV2, :FaceIndexedFluxState)
    @test isdefined(AtmosTransportV2, :RussellLernerAdvection)
    @test isdefined(AtmosTransportV2, :AtmosGrid)
    @test isdefined(AtmosTransportV2, :HybridSigmaPressure)
    @test isdefined(AtmosTransportV2, :PreprocessedERA5Driver)
    @test isdefined(AtmosTransportV2, :DiagnoseVerticalFromHorizontal)
    @test isdefined(AtmosTransportV2, :StructuredFluxTopology)
    @test isdefined(AtmosTransportV2, :FaceIndexedFluxTopology)
    @test isdefined(AtmosTransportV2, :flux_topology)
end

@testset "Flux topology trait" begin
    mesh_ll = LatLonMesh(; Nx=10, Ny=8, FT=Float64)
    @test flux_topology(mesh_ll) isa StructuredFluxTopology

    sfs = StructuredFaceFluxState(zeros(11,8,4), zeros(10,9,4), zeros(10,8,5))
    @test sfs isa AbstractStructuredFaceFluxState
    @test sfs isa AbstractFaceFluxState

    @test face_flux_x(sfs, 1, 1, 1) == 0.0
    @test face_flux_y(sfs, 1, 1, 1) == 0.0
    @test face_flux_z(sfs, 1, 1, 1) == 0.0
end

# =========================================================================
# Test 1b: Abstraction boundary — mock flux state proves dispatch is on
#           AbstractStructuredFaceFluxState, not the concrete type
# =========================================================================
struct MockStructuredFluxState{AX, AY, AZ} <: AbstractStructuredFaceFluxState
    am :: AX
    bm :: AY
    cm :: AZ
end

@testset "Abstraction boundary: mock flux state" begin
    FT = Float64
    Nx, Ny, Nz = 10, 8, 4

    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A = FT[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B = FT[0.0, 0.0,   0.1,    0.5,     1.0]
    vc = HybridSigmaPressure(A, B)
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.Grids.CPU(); FT=FT)

    g = grid.gravity
    ps_val = grid.reference_pressure
    areas = cell_areas_by_latitude(mesh)

    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m[i, j, k] = dp_k * areas[j] / g
    end
    rm = m .* FT(400e-6)
    state = CellState(m; CO2=rm)

    am = zeros(FT, Nx+1, Ny, Nz)
    bm = zeros(FT, Nx, Ny+1, Nz)
    cm = zeros(FT, Nx, Ny, Nz+1)

    mock_fluxes = MockStructuredFluxState(am, bm, cm)

    @test mock_fluxes isa AbstractStructuredFaceFluxState
    @test mock_fluxes isa AbstractFaceFluxState
    @test !(mock_fluxes isa StructuredFaceFluxState)

    @test face_flux_x(mock_fluxes, 1, 1, 1) == 0.0
    @test face_flux_y(mock_fluxes, 1, 1, 1) == 0.0
    @test face_flux_z(mock_fluxes, 1, 1, 1) == 0.0

    scheme = RussellLernerAdvection(use_limiter=true)
    ws = AdvectionWorkspace(m)

    m_before = sum(state.air_dry_mass)
    rm_before = total_mass(state, :CO2)

    strang_split!(state, mock_fluxes, grid, scheme; workspace=ws)

    @test sum(state.air_dry_mass) ≈ m_before atol=eps(FT)*m_before*1000
    @test total_mass(state, :CO2)  ≈ rm_before atol=eps(FT)*rm_before*1000
end

# =========================================================================
# Test 1c: Universal face geometry API
# =========================================================================
@testset "Universal face_length and face_normal" begin
    mesh = LatLonMesh(; Nx=10, Ny=8, FT=Float64)
    R = mesh.radius

    n_xfaces = (mesh.Nx + 1) * mesh.Ny
    total = nfaces(mesh)
    @test total == n_xfaces + mesh.Nx * (mesh.Ny + 1)

    fl_x = face_length(mesh, 1)
    @test fl_x ≈ R * deg2rad(mesh.Δφ)

    fl_y = face_length(mesh, n_xfaces + 1)
    @test fl_y > 0.0

    nx_x, ny_x = face_normal(mesh, 1)
    @test nx_x == 1.0
    @test ny_x == 0.0

    nx_y, ny_y = face_normal(mesh, n_xfaces + 1)
    @test nx_y == 0.0
    @test ny_y == 1.0
end

# =========================================================================
# Test 2: Geometry API correctness for LatLonMesh
# =========================================================================
@testset "LatLonMesh geometry" begin
    mesh = LatLonMesh(; Nx=60, Ny=45, FT=Float64)

    @test nx(mesh) == 60
    @test ny(mesh) == 45
    @test ncells(mesh) == 60 * 45

    # Cell area at equator should be approximately R² Δλ Δφ
    R = 6.371e6
    Δλ_rad = deg2rad(mesh.Δλ)
    area_eq_j = div(mesh.Ny, 2) + 1  # near equator
    a = cell_area(mesh, 1, area_eq_j)
    @test a > 0.0
    # Rough check: equatorial area ~ R² × Δλ_rad × sin(Δφ/2)*2 ≈ R² Δλ Δφ (cos≈1)
    Δφ_rad = deg2rad(mesh.Δφ)
    expected_equator = R^2 * Δλ_rad * abs(sind(mesh.φᶠ[area_eq_j+1]) - sind(mesh.φᶠ[area_eq_j]))
    @test a ≈ expected_equator

    # Total area should approximate Earth's surface (4πR²)
    total_area = sum(cell_area(mesh, 1, j) for j in 1:mesh.Ny) * mesh.Nx
    earth_area = 4π * R^2
    @test abs(total_area - earth_area) / earth_area < 1e-10

    # dx at equator should be R × Δλ_rad × cos(0) ≈ R × Δλ_rad
    dx_eq = Grids.dx(mesh, area_eq_j)
    @test dx_eq > 0.0

    # dy should be uniform
    dy_val = Grids.dy(mesh)
    @test dy_val ≈ R * Δφ_rad

    # face_cells connectivity
    f = 1
    lc, rc = face_cells(mesh, f)
    @test lc isa Integer
    @test rc isa Integer
end

# =========================================================================
# Test 3: AtmosGrid composite construction
# =========================================================================
@testset "AtmosGrid construction" begin
    mesh = LatLonMesh(; Nx=10, Ny=8, FT=Float64)

    A = Float64[0.0, 1000.0, 5000.0, 20000.0, 50000.0]
    B = Float64[1.0, 0.9, 0.6, 0.2, 0.0]
    vc = HybridSigmaPressure(A, B)
    @test n_levels(vc) == 4

    grid = AtmosGrid(mesh, vc, AtmosTransportV2.Grids.CPU();
                     FT=Float64, radius=6.371e6, gravity=9.80665)
    @test nlevels(grid) == 4
    @test floattype(grid) == Float64
end

# =========================================================================
# Helpers: create test grid and state
# =========================================================================
function make_test_grid(; Nx=60, Ny=30, Nz=4, FT=Float64)
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)

    # Simple 4-level vertical: A + B*ps = pressure at interfaces
    A = FT[0.0, 5000.0, 20000.0, 50000.0, 80000.0]
    B = FT[1.0, 0.75, 0.4, 0.1, 0.0]
    # Normalize so p_surface = A[end]+B[end]*ps matches
    # With these values: p_top = A[1]+B[1]*ps = ps, p_bot = A[5]+B[5]*ps = 80000
    # Let's use proper hybrid: TOA is index 1
    A_proper = FT[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B_proper = FT[0.0, 0.0,   0.1,    0.5,     1.0]
    vc = HybridSigmaPressure(A_proper, B_proper)

    # Use a simple CPU() arch. Since v2 doesn't re-export src/ Architectures,
    # we create a minimal stand-in.
    grid = AtmosGrid(mesh, vc, AtmosTransportV2.Grids.CPU();
                     FT=FT, radius=FT(6.371e6), gravity=FT(9.80665))
    return grid
end

function make_test_state(grid; χ=400e-6)
    FT = floattype(grid)
    mesh = grid.horizontal
    vc = grid.vertical
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(vc)
    g = grid.gravity
    ps_val = grid.reference_pressure

    areas = cell_areas_by_latitude(mesh)

    # Compute dry air mass (no humidity for simplicity)
    m = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc, k, ps_val)
        m[i, j, k] = dp_k * areas[j] / g
    end

    # Tracer: uniform mixing ratio
    rm = m .* FT(χ)

    state = CellState(m; CO2=rm)
    return state
end

function make_zero_fluxes(grid)
    FT = floattype(grid)
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)
    return allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT)
end

function make_simple_fluxes(grid; scale=1e8)
    FT = floattype(grid)
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)

    fluxes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz; FT=FT)

    # Simple zonal flow: am is uniform positive, small enough for CFL < 1
    m_state = make_test_state(grid)
    m_min = minimum(m_state.air_dry_mass)
    am_val = FT(0.1) * m_min  # CFL ~ 0.1

    for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
        fluxes.am[i, j, k] = am_val
    end
    # Zero out polar am at j=1 and j=Ny boundaries
    fluxes.am[:, 1, :] .= zero(FT)

    # Diagnose consistent cm from am (bm=0)
    bt = FT[Grids.b_diff(grid.vertical, k) for k in 1:Nz]
    for j in 1:Ny, i in 1:Nx
        pit = zero(FT)
        for k in 1:Nz
            pit += fluxes.am[i, j, k] - fluxes.am[i+1, j, k] +
                   fluxes.bm[i, j, k] - fluxes.bm[i, j+1, k]
        end
        acc = zero(FT)
        fluxes.cm[i, j, 1] = acc
        for k in 1:Nz
            conv_k = fluxes.am[i, j, k] - fluxes.am[i+1, j, k] +
                     fluxes.bm[i, j, k] - fluxes.bm[i, j+1, k]
            acc += conv_k - bt[k] * pit
            fluxes.cm[i, j, k+1] = acc
        end
    end

    return fluxes
end

# =========================================================================
# Test 4: CellState and FaceFluxState construction
# =========================================================================
@testset "State construction" begin
    grid = make_test_grid()
    state = make_test_state(grid)

    @test size(state.air_dry_mass) == (60, 30, 4)
    @test :CO2 in tracer_names(state)
    @test total_air_mass(state) > 0.0
    @test total_mass(state, :CO2) > 0.0

    # Mixing ratio should be uniform
    χ = mixing_ratio(state, :CO2)
    @test maximum(χ) ≈ 400e-6
    @test minimum(χ) ≈ 400e-6

    fluxes = make_zero_fluxes(grid)
    @test size(fluxes.am) == (61, 30, 4)
    @test size(fluxes.bm) == (60, 31, 4)
    @test size(fluxes.cm) == (60, 30, 5)
end

# =========================================================================
# Test 5: Uniform field invariance (zero fluxes)
# =========================================================================
@testset "Uniform field invariance (zero flux)" begin
    grid = make_test_grid()
    state = make_test_state(grid; χ=400e-6)
    fluxes = make_zero_fluxes(grid)

    scheme = RussellLernerAdvection(use_limiter=true)
    ws = AdvectionWorkspace(state.air_dry_mass)

    m_before = sum(state.air_dry_mass)
    rm_before = total_mass(state, :CO2)

    strang_split!(state, fluxes, grid, scheme; workspace=ws)

    m_after = sum(state.air_dry_mass)
    rm_after = total_mass(state, :CO2)

    @test m_after ≈ m_before atol=eps(Float64)*m_before*1000
    @test rm_after ≈ rm_before atol=eps(Float64)*rm_before*1000

    # Mixing ratio should remain uniform
    χ = mixing_ratio(state, :CO2)
    @test maximum(χ) ≈ minimum(χ) atol=1e-15
end

# =========================================================================
# Test 6: Mass conservation with non-zero fluxes
# =========================================================================
@testset "Mass conservation (zonal flow)" begin
    grid = make_test_grid()
    state = make_test_state(grid; χ=400e-6)
    fluxes = make_simple_fluxes(grid)

    scheme = RussellLernerAdvection(use_limiter=true)
    ws = AdvectionWorkspace(state.air_dry_mass)

    m_before = sum(state.air_dry_mass)
    rm_before = total_mass(state, :CO2)

    strang_split!(state, fluxes, grid, scheme; workspace=ws)

    m_after = sum(state.air_dry_mass)
    rm_after = total_mass(state, :CO2)

    # Air mass should be conserved to machine precision
    @test abs(m_after - m_before) / m_before < 1e-12

    # Tracer mass should be conserved to machine precision
    @test abs(rm_after - rm_before) / rm_before < 1e-12
end

# =========================================================================
# Test 7: Vertical flux diagnosis
# =========================================================================
@testset "Vertical flux diagnosis (cm from continuity)" begin
    grid = make_test_grid()
    FT = Float64
    mesh = grid.horizontal
    Nx, Ny = nx(mesh), ny(mesh)
    Nz = n_levels(grid.vertical)

    # Random am, bm
    am = randn(FT, Nx+1, Ny, Nz) .* 1e6
    bm = randn(FT, Nx, Ny+1, Nz) .* 1e6
    cm = zeros(FT, Nx, Ny, Nz+1)

    bt = FT[Grids.b_diff(grid.vertical, k) for k in 1:Nz]

    # Diagnose cm
    Operators.Advection.diagnose_cm!(cm, am, bm, bt)

    # Verify: cm[1] = 0 (TOA), cm[Nz+1] ≈ 0 (surface, if bt sums properly)
    for j in 1:Ny, i in 1:Nx
        @test cm[i, j, 1] == 0.0
    end

    # Verify continuity: for each cell, horizontal convergence + vertical flux
    # divergence should equal bt[k] * pit
    for j in 1:Ny, i in 1:Nx
        pit = zero(FT)
        for k in 1:Nz
            pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
        end
        for k in 1:Nz
            conv_h = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            div_v = cm[i, j, k+1] - cm[i, j, k]
            # conv_h - div_v should = bt[k] * pit
            @test abs(conv_h - div_v - bt[k] * pit) < 1e-10 * max(abs(pit), 1.0)
        end
    end
end

# =========================================================================
# Test 8: Numerical equivalence with src/ mass-flux advection
# =========================================================================
@testset "Numerical equivalence v1 vs v2" begin
    FT = Float64
    Nx, Ny, Nz = 60, 30, 4

    # Create matching state in v1 format
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A = FT[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B = FT[0.0, 0.0,   0.1,    0.5,     1.0]
    vc_v2 = HybridSigmaPressure(A, B)
    grid_v2 = AtmosGrid(mesh, vc_v2, AtmosTransportV2.Grids.CPU();
                         FT=FT, radius=FT(6.371e6), gravity=FT(9.80665))

    areas = cell_areas_by_latitude(mesh)
    ps_val = FT(101325.0)
    g = FT(9.80665)

    m_init = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dp_k = level_thickness(vc_v2, k, ps_val)
        m_init[i, j, k] = dp_k * areas[j] / g
    end
    rm_init = m_init .* FT(400e-6)

    # Create non-trivial am (pure zonal, periodic)
    m_min = minimum(m_init)
    am = zeros(FT, Nx+1, Ny, Nz)
    for k in 1:Nz, j in 2:Ny-1, i in 1:Nx+1
        am[i, j, k] = FT(0.05) * m_min
    end
    bm = zeros(FT, Nx, Ny+1, Nz)
    cm = zeros(FT, Nx, Ny, Nz+1)

    # Diagnose cm
    bt = FT[B[k+1] - B[k] for k in 1:Nz]
    for j in 1:Ny, i in 1:Nx
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

    # --- V2 path ---
    m_v2 = copy(m_init)
    rm_v2 = copy(rm_init)
    state_v2 = CellState(m_v2; CO2=rm_v2)
    fluxes_v2 = StructuredFaceFluxState(copy(am), copy(bm), copy(cm))
    scheme_v2 = RussellLernerAdvection(use_limiter=true)
    ws_v2 = AdvectionWorkspace(m_v2)

    strang_split!(state_v2, fluxes_v2, grid_v2, scheme_v2; workspace=ws_v2)

    # --- V1 path ---
    # Build v1 grid
    vc_v1 = AtmosTransport.Grids.HybridSigmaPressure(A, B)
    grid_v1 = AtmosTransport.Grids.LatitudeLongitudeGrid(
        AtmosTransport.Architectures.CPU();
        FT=FT, size=(Nx, Ny, Nz),
        longitude=(-180, 180), latitude=(-90, 90),
        vertical=vc_v1, halo=(3, 3, 1))

    m_v1 = copy(m_init)
    rm_v1 = copy(rm_init)
    tracers_v1 = (CO2=rm_v1,)

    ws_v1 = AtmosTransport.Advection.allocate_massflux_workspace(m_v1, am, bm, cm)

    AtmosTransport.Advection.strang_split_massflux!(
        tracers_v1, m_v1, copy(am), copy(bm), copy(cm),
        grid_v1, true, ws_v1; cfl_limit=FT(1.0))

    # Compare results
    @test maximum(abs.(rm_v2 .- rm_v1)) / maximum(abs.(rm_v1)) < 1e-12
    @test maximum(abs.(m_v2 .- m_v1)) / maximum(abs.(m_v1)) < 1e-12

    println("  Max relative rm difference: ", maximum(abs.(rm_v2 .- rm_v1)) / maximum(abs.(rm_v1)))
    println("  Max relative m  difference: ", maximum(abs.(m_v2 .- m_v1)) / maximum(abs.(m_v1)))
end

println("\n✓ All v2 dry-flux interface tests passed!")
