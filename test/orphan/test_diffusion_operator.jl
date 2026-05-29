"""
Integration tests for the `ImplicitVerticalDiffusion` operator
(plan 16b, Commit 3).

These tests validate the `apply!` wrapper around the Commit-2
kernel: `CellState` / `tracers_raw` integration, workspace
requirement and validation, and end-to-end agreement with the
lower-level kernel on matched inputs.
"""

using Test
using AtmosTransport: CellState, AdvectionWorkspace,
    ConstantField, PreComputedKzField, ProfileKzField,
    NoDiffusion, ImplicitVerticalDiffusion, CubedSphereField,
    AbstractDiffusion, apply!, get_tracer,
    ReducedGaussianMesh, HybridSigmaPressure, AtmosGrid, CPU,
    ncells,
    field_value
using AtmosTransport.Operators.Diffusion: _vertical_diffusion_kernel!,
                                           build_diffusion_coefficients,
                                           solve_tridiagonal!
using KernelAbstractions: get_backend, synchronize

# =========================================================================
# Test setup helpers
# =========================================================================

"Construct a CellState with two tracers on an (Nx, Ny, Nz) column grid."
function _make_diffusion_state(::Type{FT}; Nx = 3, Ny = 2, Nz = 5,
                               co2 = FT(1.0), n2o = FT(2.0)) where FT
    m = ones(FT, Nx, Ny, Nz)
    rm_co2 = fill(FT(co2), Nx, Ny, Nz)
    rm_n2o = fill(FT(n2o), Nx, Ny, Nz)
    return CellState(m; CO2 = rm_co2, N2O = rm_n2o)
end

"Construct a face-indexed CellState on a small reduced-Gaussian mesh."
function _make_face_diffusion_state(::Type{FT}; Nz = 5,
                                    co2 = FT(1.0), n2o = FT(2.0)) where FT
    mesh = ReducedGaussianMesh(FT[-45, 45], [4, 4]; FT = FT)
    vc = HybridSigmaPressure(zeros(FT, Nz + 1), FT.(collect(range(0, 1, length = Nz + 1))))
    grid = AtmosGrid(mesh, vc, CPU(); FT = FT)

    ncell = ncells(mesh)
    m = ones(FT, ncell, Nz)
    rm_co2 = fill(FT(co2), ncell, Nz)
    rm_n2o = fill(FT(n2o), ncell, Nz)
    return CellState(m; CO2 = rm_co2, N2O = rm_n2o), grid
end

"Pre-fill workspace.dz_scratch with a known 3D profile."
function _fill_dz!(workspace, dz_arr::AbstractArray)
    size(workspace.dz_scratch) == size(dz_arr) ||
        error("dz profile shape mismatch: workspace $(size(workspace.dz_scratch)) vs $(size(dz_arr))")
    copyto!(workspace.dz_scratch, dz_arr)
    return workspace
end

function _reference_face_diffusion!(q_raw::AbstractArray{FT, 3},
                                    kz_field, dz_arr::AbstractArray{FT, 2},
                                    dt) where FT
    ncell, Nz, Nt = size(q_raw)
    for c in 1:ncell
        Kz_col = FT[field_value(kz_field, (c, k)) for k in 1:Nz]
        dz_col = collect(@view dz_arr[c, :])
        a, b, cdiag = build_diffusion_coefficients(Kz_col, dz_col, dt)
        w = similar(dz_col)
        for t in 1:Nt
            d = collect(@view q_raw[c, :, t])
            x = similar(d)
            solve_tridiagonal!(x, a, b, cdiag, d, w)
            q_raw[c, :, t] .= x
        end
    end
    return q_raw
end

# =========================================================================
# Type hierarchy
# =========================================================================

@testset "Diffusion type hierarchy" begin
    @test NoDiffusion() isa AbstractDiffusion
    @test ImplicitVerticalDiffusion(kz_field = ConstantField{Float64, 3}(1.0)) isa AbstractDiffusion
end

# =========================================================================
# NoDiffusion is an identity
# =========================================================================

@testset "NoDiffusion is a no-op" begin
    FT = Float64
    state = _make_diffusion_state(FT; co2 = 1.0, n2o = 2.0)
    co2_before = copy(state.tracers_raw[:, :, :, 1])
    n2o_before = copy(state.tracers_raw[:, :, :, 2])

    apply!(state, nothing, nothing, NoDiffusion(), 1.0)

    @test state.tracers_raw[:, :, :, 1] == co2_before
    @test state.tracers_raw[:, :, :, 2] == n2o_before
end

# =========================================================================
# Construction: type assertions
# =========================================================================

@testset "ImplicitVerticalDiffusion constructor validation" begin
    FT = Float64
    # Rank-2 and rank-3 Kz fields are both accepted. Structured vs
    # face-indexed compatibility is enforced at apply-time.
    good_face = ConstantField{FT, 2}(1.0)
    op_face = ImplicitVerticalDiffusion(; kz_field = good_face)
    @test op_face.kz_field === good_face

    good = ConstantField{FT, 3}(1.0)
    op = ImplicitVerticalDiffusion(; kz_field = good)
    @test op.kz_field === good
    @test op isa ImplicitVerticalDiffusion{Float64}

    good_cs = CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(1.0), 6))
    op_cs = ImplicitVerticalDiffusion(; kz_field = good_cs)
    @test op_cs.kz_field === good_cs
end

# =========================================================================
# Workspace requirement
# =========================================================================

@testset "apply! requires workspace" begin
    FT = Float64
    state = _make_diffusion_state(FT)
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(1.0))

    # workspace = nothing (default kwarg) → ArgumentError
    @test_throws ArgumentError apply!(state, nothing, nothing, op, 1.0;
                                      workspace = nothing)
end

@testset "apply! with ConstantField Kz matches face-indexed reference" begin
    FT = Float64
    Nz = 5
    dt = FT(0.5)
    Kz_val = FT(1.25)

    state_op, grid = _make_face_diffusion_state(FT; Nz = Nz)
    ncell = size(state_op.air_mass, 1)
    state_op.tracers_raw .= reshape(FT.(1:(ncell * Nz * 2)), ncell, Nz, 2)

    q_ref = copy(state_op.tracers_raw)
    dz_arr = fill(FT(100.0), ncell, Nz)

    ws = AdvectionWorkspace(state_op)
    _fill_dz!(ws, dz_arr)
    kz_field = ConstantField{FT, 2}(Kz_val)
    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    apply!(state_op, nothing, grid, op, dt; workspace = ws)

    _reference_face_diffusion!(q_ref, kz_field, dz_arr, dt)

    @test get_tracer(state_op, :CO2) ≈ q_ref[:, :, 1] atol=1e-12 rtol=1e-12
    @test get_tracer(state_op, :N2O) ≈ q_ref[:, :, 2] atol=1e-12 rtol=1e-12
end

@testset "apply! with PreComputedKzField face-indexed matches reference" begin
    FT = Float64
    Nz = 5
    dt = FT(0.5)

    state_op, grid = _make_face_diffusion_state(FT; Nz = Nz)
    ncell = size(state_op.air_mass, 1)
    state_op.tracers_raw[:, :, 1] .= reshape(FT.(1:(ncell * Nz)), ncell, Nz)

    q_ref = copy(state_op.tracers_raw[:, :, 1:1])
    dz_arr = FT.(80 .+ reshape(1:(ncell * Nz), ncell, Nz))
    Kz_arr = FT.(0.5 .+ reshape(1:(ncell * Nz), ncell, Nz) ./ 100)

    ws = AdvectionWorkspace(state_op)
    _fill_dz!(ws, dz_arr)
    kz_field = PreComputedKzField(copy(Kz_arr))
    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    apply!(state_op, nothing, grid, op, dt; workspace = ws)

    _reference_face_diffusion!(q_ref, kz_field, dz_arr, dt)

    @test get_tracer(state_op, :CO2) ≈ q_ref[:, :, 1] atol=1e-12 rtol=1e-12
end

@testset "apply! with ProfileKzField{2} matches equivalent precomputed field" begin
    FT = Float64
    Nz = 5
    dt = FT(0.5)

    state, grid = _make_face_diffusion_state(FT; Nz = Nz)
    ncell = size(state.air_mass, 1)
    state.tracers_raw[:, :, 1] .= reshape(FT.(1:(ncell * Nz)), ncell, Nz)

    state_ref = CellState(copy(state.air_mass); CO2 = copy(state.tracers_raw[:, :, 1]))
    profile = FT[0.5, 1.0, 1.5, 1.0, 0.5]
    Kz_arr = repeat(reshape(profile, 1, Nz), ncell, 1)
    dz_arr = fill(FT(100.0), ncell, Nz)

    ws = AdvectionWorkspace(state)
    ws_ref = AdvectionWorkspace(state_ref)
    _fill_dz!(ws, dz_arr)
    _fill_dz!(ws_ref, dz_arr)

    op = ImplicitVerticalDiffusion(; kz_field = ProfileKzField(copy(profile); spatial_rank = 2))
    op_ref = ImplicitVerticalDiffusion(; kz_field = PreComputedKzField(copy(Kz_arr)))

    apply!(state, nothing, grid, op, dt; workspace = ws)
    apply!(state_ref, nothing, grid, op_ref, dt; workspace = ws_ref)

    @test get_tracer(state, :CO2) ≈ get_tracer(state_ref, :CO2) atol=1e-12 rtol=1e-12
end

@testset "apply! catches size mismatches" begin
    FT = Float64
    Nx, Ny, Nz = 3, 2, 5
    state = _make_diffusion_state(FT; Nx = Nx, Ny = Ny, Nz = Nz)
    # Build a workspace whose air_mass is a DIFFERENT shape than state:
    m_wrong = ones(FT, Nx + 1, Ny, Nz)
    ws_wrong = AdvectionWorkspace(m_wrong; n_tracers = 2)
    op = ImplicitVerticalDiffusion(; kz_field = ConstantField{FT, 3}(1.0))

    @test_throws DimensionMismatch apply!(state, nothing, nothing, op, 1.0;
                                           workspace = ws_wrong)
end

# =========================================================================
# End-to-end: apply! matches direct kernel launch
# =========================================================================

@testset "apply! with ConstantField Kz matches direct kernel" begin
    FT = Float64
    Nx, Ny, Nz = 3, 2, 6
    dt = 0.5
    Kz_val = 1.5

    # Two independent states with identical initial conditions
    rm_co2 = FT.(reshape(collect(1.0:Nx*Ny*Nz), Nx, Ny, Nz))
    rm_n2o = FT.(reshape(collect(Nx*Ny*Nz+1.0:2*Nx*Ny*Nz), Nx, Ny, Nz))

    state_op = CellState(ones(FT, Nx, Ny, Nz);
                         CO2 = copy(rm_co2), N2O = copy(rm_n2o))
    state_kernel_co2 = reshape(copy(rm_co2), Nx, Ny, Nz, 1)
    state_kernel_n2o = reshape(copy(rm_n2o), Nx, Ny, Nz, 1)

    # Shared dz profile
    dz_arr = fill(FT(100.0), Nx, Ny, Nz)

    # apply! path
    ws = AdvectionWorkspace(state_op)
    _fill_dz!(ws, dz_arr)
    kz_field = ConstantField{FT, 3}(Kz_val)
    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    apply!(state_op, nothing, nothing, op, dt; workspace = ws)

    # Direct kernel path (one tracer at a time for the reference)
    for (buf, tname) in zip((state_kernel_co2, state_kernel_n2o), (:CO2, :N2O))
        Kz_ref_field = ConstantField{FT, 3}(Kz_val)
        w_scratch = similar(dz_arr)
        backend = get_backend(buf)
        kernel = _vertical_diffusion_kernel!(backend, (4, 4, 1))
        kernel(buf, Kz_ref_field, dz_arr, w_scratch, FT(dt), Nz;
               ndrange = (Nx, Ny, 1))
        synchronize(backend)
    end

    # Compare through the accessor API (plan-14 test discipline)
    @test get_tracer(state_op, :CO2) ≈ state_kernel_co2[:, :, :, 1] atol=1e-12 rtol=1e-12
    @test get_tracer(state_op, :N2O) ≈ state_kernel_n2o[:, :, :, 1] atol=1e-12 rtol=1e-12
end

@testset "apply! with PreComputedKzField Kz matches direct kernel" begin
    FT = Float64
    Nx, Ny, Nz = 3, 2, 6
    dt = 0.5

    Kz_arr = FT.(0.5 .+ rand(Nx, Ny, Nz) .* 1.5)    # [0.5, 2.0]
    dz_arr = FT.(80.0 .+ rand(Nx, Ny, Nz) .* 40.0)  # [80, 120]

    rm_co2 = FT.(1.0 .+ rand(Nx, Ny, Nz))

    # apply! path
    state_op = CellState(ones(FT, Nx, Ny, Nz); CO2 = copy(rm_co2))
    ws = AdvectionWorkspace(state_op)
    _fill_dz!(ws, dz_arr)
    kz_field = PreComputedKzField(copy(Kz_arr))
    op = ImplicitVerticalDiffusion(; kz_field = kz_field)
    apply!(state_op, nothing, nothing, op, dt; workspace = ws)

    # Direct kernel path
    q_ref = reshape(copy(rm_co2), Nx, Ny, Nz, 1)
    w_scratch = similar(dz_arr)
    kz_ref_field = PreComputedKzField(copy(Kz_arr))
    backend = get_backend(q_ref)
    kernel = _vertical_diffusion_kernel!(backend, (4, 4, 1))
    kernel(q_ref, kz_ref_field, dz_arr, w_scratch, FT(dt), Nz;
           ndrange = (Nx, Ny, 1))
    synchronize(backend)

    @test get_tracer(state_op, :CO2) ≈ q_ref[:, :, :, 1] atol=1e-12 rtol=1e-12
end

# =========================================================================
# Mass conservation through apply!
# =========================================================================

@testset "apply! conserves column mass (uniform dz, Neumann BCs)" begin
    FT = Float64
    Nx, Ny, Nz = 2, 2, 10
    dt = 1.0

    state = _make_diffusion_state(FT; Nx = Nx, Ny = Ny, Nz = Nz,
                                   co2 = 1.0, n2o = 2.0)
    # Randomize initial tracer fields so the test exercises a non-trivial
    # diffusion
    state.tracers_raw .= FT.(0.5 .+ rand(Nx, Ny, Nz, 2))
    co2_mass_before = [sum(state.tracers_raw[i, j, :, 1]) for i in 1:Nx, j in 1:Ny]
    n2o_mass_before = [sum(state.tracers_raw[i, j, :, 2]) for i in 1:Nx, j in 1:Ny]

    Kz_arr = FT.(0.2 .+ 1.5 .* rand(Nx, Ny, Nz))
    dz_arr = fill(FT(100.0), Nx, Ny, Nz)     # uniform dz
    ws = AdvectionWorkspace(state)
    _fill_dz!(ws, dz_arr)
    op = ImplicitVerticalDiffusion(; kz_field = PreComputedKzField(Kz_arr))
    apply!(state, nothing, nothing, op, dt; workspace = ws)

    co2_mass_after = [sum(state.tracers_raw[i, j, :, 1]) for i in 1:Nx, j in 1:Ny]
    n2o_mass_after = [sum(state.tracers_raw[i, j, :, 2]) for i in 1:Nx, j in 1:Ny]

    @test maximum(abs.(co2_mass_after .- co2_mass_before) ./ co2_mass_before) < 1e-12
    @test maximum(abs.(n2o_mass_after .- n2o_mass_before) ./ n2o_mass_before) < 1e-12
end

# =========================================================================
# Kz field type variation
# =========================================================================

@testset "apply! with ProfileKzField (CPU)" begin
    FT = Float64
    Nx, Ny, Nz = 2, 2, 5
    dt = 0.5

    profile = FT[0.5, 1.0, 1.5, 1.0, 0.5]
    dz_arr  = fill(FT(100.0), Nx, Ny, Nz)
    rm_co2  = FT.(1.0 .+ rand(Nx, Ny, Nz))

    # apply! path
    state = CellState(ones(FT, Nx, Ny, Nz); CO2 = copy(rm_co2))
    ws = AdvectionWorkspace(state)
    _fill_dz!(ws, dz_arr)
    op = ImplicitVerticalDiffusion(;
        kz_field = ProfileKzField(copy(profile)))
    apply!(state, nothing, nothing, op, dt; workspace = ws)

    # Reference: since ProfileKzField is horizontally uniform, compare
    # against a PreComputedKzField with the same profile broadcast.
    Kz_arr = zeros(FT, Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny
        Kz_arr[i, j, :] .= profile
    end
    state_ref = CellState(ones(FT, Nx, Ny, Nz); CO2 = copy(rm_co2))
    ws_ref = AdvectionWorkspace(state_ref)
    _fill_dz!(ws_ref, dz_arr)
    op_ref = ImplicitVerticalDiffusion(;
        kz_field = PreComputedKzField(Kz_arr))
    apply!(state_ref, nothing, nothing, op_ref, dt; workspace = ws_ref)

    @test get_tracer(state, :CO2) ≈ get_tracer(state_ref, :CO2) atol=1e-12 rtol=1e-12
end
