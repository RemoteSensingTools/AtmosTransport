#!/usr/bin/env julia
"""
End-to-end test for the ERA5 LatLon path.

Tests the full chain:
  disk → ERA5BinaryReader → moist flux state → build_dry_fluxes!
       → dry flux state → strang_split! → mass conservation check

1. Synthetic binary round-trip: write a minimal v4 binary, read it back,
   convert moist→dry, advect, verify mass conservation.
2. Vertical closure: verify cm diagnosed from am/bm satisfies continuity.
3. Basis safety: confirm moist fluxes cannot be passed to advection.
"""

using Test
using JSON3

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport: Grids, State, Operators, MetDrivers

# =========================================================================
# Helper: write a synthetic v4 binary file
# =========================================================================

const HEADER_SIZE = 16384

"""
    write_synthetic_binary(path; Nx, Ny, Nz, Nt, include_qv, FT) -> header_dict

Write a minimal v4-format binary with known synthetic data.
Returns the header dict for verification.
"""
function write_synthetic_binary(path::String;
                                Nx::Int = 10, Ny::Int = 8, Nz::Int = 4, Nt::Int = 1,
                                include_qv::Bool = true,
                                FT::Type{<:AbstractFloat} = Float32)
    n_m  = Nx * Ny * Nz
    n_am = (Nx + 1) * Ny * Nz
    n_bm = Nx * (Ny + 1) * Nz
    n_cm = Nx * Ny * (Nz + 1)
    n_ps = Nx * Ny
    n_qv = include_qv ? n_m : 0
    n_dam = n_am
    n_dbm = n_bm
    n_dm  = n_m

    dt = 3600.0
    half_dt = dt / 2

    A_ifc = Float64[0.0, 500.0, 5000.0, 30000.0, 0.0]
    B_ifc = Float64[0.0, 0.0,   0.1,    0.5,     1.0]
    lons = collect(range(-180.0 + 360.0 / (2 * Nx), step=360.0 / Nx, length=Nx))
    lats = collect(range(-90.0 + 180.0 / (2 * Ny), step=180.0 / Ny, length=Ny))

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps + n_qv + n_dam + n_dbm + n_dm

    header = Dict{String,Any}(
        "magic" => "MFLX", "version" => 4, "header_bytes" => HEADER_SIZE,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nt" => Nt,
        "float_type" => string(FT), "float_bytes" => sizeof(FT),
        "window_bytes" => elems_per_window * sizeof(FT),
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => n_qv, "n_cmfmc" => 0,
        "n_entu" => 0, "n_detu" => 0, "n_entd" => 0, "n_detd" => 0,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "n_dam" => n_dam, "n_dbm" => n_dbm, "n_dm" => n_dm,
        "include_flux_delta" => true,
        "include_qv" => include_qv, "include_cmfmc" => false,
        "include_tm5conv" => false, "include_surface" => false,
        "include_temperature" => false,
        "dt_seconds" => dt, "half_dt_seconds" => half_dt,
        "steps_per_met_window" => 4,
        "level_top" => 1, "level_bot" => Nz,
        "lons" => lons, "lats" => lats,
        "A_ifc" => A_ifc, "B_ifc" => B_ifc,
        # Plan 39 Commit D: the 8 self-describing contract fields.
        # Synthetic fixture follows the canonical window_constant path.
        "source_flux_sampling" => "window_start_endpoint",
        "air_mass_sampling"    => "window_start_endpoint",
        "flux_sampling"        => "window_constant",
        "flux_kind"            => "substep_mass_amount",
        "delta_semantics"      => "forward_window_endpoint_difference",
        "humidity_sampling"    => (include_qv ? "window_endpoints" : "none"),
        "poisson_balance_target_scale"     => 1.0 / (2 * 4),  # matches steps_per_met_window
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
    )

    ps_val = FT(101325.0)

    g_val = FT(9.80665)
    R = FT(6.371e6)
    Δλ = FT(360.0 / Nx)
    Δφ = FT(180.0 / Ny)
    Δλ_rad = deg2rad(Δλ)
    φ_faces = FT.(collect(range(-90.0, 90.0, length=Ny + 1)))

    m_arr  = zeros(FT, Nx, Ny, Nz)
    am_arr = zeros(FT, Nx + 1, Ny, Nz)
    bm_arr = zeros(FT, Nx, Ny + 1, Nz)
    cm_arr = zeros(FT, Nx, Ny, Nz + 1)
    ps_arr = fill(ps_val, Nx, Ny)
    qv_arr = fill(FT(0.005), Nx, Ny, Nz)  # 0.5% humidity

    for k in 1:Nz, j in 1:Ny
        dp_k = (A_ifc[k+1] - A_ifc[k]) + (B_ifc[k+1] - B_ifc[k]) * ps_val
        area_j = R^2 * Δλ_rad * abs(sind(φ_faces[j+1]) - sind(φ_faces[j]))
        for i in 1:Nx
            m_arr[i, j, k] = dp_k * area_j / g_val
        end
    end

    # Small uniform zonal flow (CFL ~ 0.05)
    m_min = minimum(m_arr)
    am_val = FT(0.05) * m_min
    for k in 1:Nz, j in 2:Ny-1, i in 1:Nx+1
        am_arr[i, j, k] = am_val
    end

    # Flux deltas: zero (steady state for this test)
    dam_arr = zeros(FT, Nx + 1, Ny, Nz)
    dbm_arr = zeros(FT, Nx, Ny + 1, Nz)
    dm_arr  = zeros(FT, Nx, Ny, Nz)

    open(path, "w") do io
        header_json = JSON3.write(header)
        @assert length(header_json) < HEADER_SIZE
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)

        for _ in 1:Nt
            write(io, vec(m_arr))
            write(io, vec(am_arr))
            write(io, vec(bm_arr))
            write(io, vec(cm_arr))
            write(io, vec(ps_arr))
            if include_qv
                write(io, vec(qv_arr))
            end
            write(io, vec(dam_arr))
            write(io, vec(dbm_arr))
            write(io, vec(dm_arr))
        end
    end

    return header
end

"""
    write_synthetic_binary_with_basis(path; mass_basis_str, kwargs...) -> header_dict

Write a minimal v5 binary with explicit `mass_basis` in the header.
Delegates to `write_synthetic_binary` internals for data generation.
"""
function write_synthetic_binary_with_basis(path::String;
                                            mass_basis_str::String = "moist",
                                            Nx::Int = 6, Ny::Int = 4, Nz::Int = 3, Nt::Int = 1,
                                            FT::Type{<:AbstractFloat} = Float32)
    n_m  = Nx * Ny * Nz
    n_am = (Nx + 1) * Ny * Nz
    n_bm = Nx * (Ny + 1) * Nz
    n_cm = Nx * Ny * (Nz + 1)
    n_ps = Nx * Ny

    dt = 3600.0
    half_dt = dt / 2

    A_ifc = Float64[0.0, 500.0, 5000.0, 0.0]
    B_ifc = Float64[0.0, 0.0,   0.5,    1.0]
    lons = collect(range(-180.0 + 360.0 / (2 * Nx), step=360.0 / Nx, length=Nx))
    lats = collect(range(-90.0 + 180.0 / (2 * Ny), step=180.0 / Ny, length=Ny))

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps

    header = Dict{String,Any}(
        "magic" => "MFLX", "version" => 5, "header_bytes" => HEADER_SIZE,
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "Nt" => Nt,
        "float_type" => string(FT), "float_bytes" => sizeof(FT),
        "window_bytes" => elems_per_window * sizeof(FT),
        "n_m" => n_m, "n_am" => n_am, "n_bm" => n_bm, "n_cm" => n_cm, "n_ps" => n_ps,
        "n_qv" => 0, "n_cmfmc" => 0,
        "n_entu" => 0, "n_detu" => 0, "n_entd" => 0, "n_detd" => 0,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "include_flux_delta" => false,
        "include_qv" => false, "include_cmfmc" => false,
        "include_tm5conv" => false, "include_surface" => false,
        "include_temperature" => false,
        "mass_basis" => mass_basis_str,
        "dt_seconds" => dt, "half_dt_seconds" => half_dt,
        "steps_per_met_window" => 4,
        "level_top" => 1, "level_bot" => Nz,
        "lons" => lons, "lats" => lats,
        "A_ifc" => A_ifc, "B_ifc" => B_ifc,
    )

    ps_val = FT(101325.0)
    g_val = FT(9.80665)
    R = FT(6.371e6)
    Δλ_rad = deg2rad(FT(360.0 / Nx))
    φ_faces = FT.(collect(range(-90.0, 90.0, length=Ny + 1)))

    m_arr  = zeros(FT, Nx, Ny, Nz)
    am_arr = zeros(FT, Nx + 1, Ny, Nz)
    bm_arr = zeros(FT, Nx, Ny + 1, Nz)
    cm_arr = zeros(FT, Nx, Ny, Nz + 1)
    ps_arr = fill(ps_val, Nx, Ny)

    for k in 1:Nz, j in 1:Ny
        dp_k = (A_ifc[k+1] - A_ifc[k]) + (B_ifc[k+1] - B_ifc[k]) * ps_val
        area_j = R^2 * Δλ_rad * abs(sind(φ_faces[j+1]) - sind(φ_faces[j]))
        for i in 1:Nx
            m_arr[i, j, k] = dp_k * area_j / g_val
        end
    end

    open(path, "w") do io
        header_json = JSON3.write(header)
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)
        for _ in 1:Nt
            write(io, vec(m_arr))
            write(io, vec(am_arr))
            write(io, vec(bm_arr))
            write(io, vec(cm_arr))
            write(io, vec(ps_arr))
        end
    end
    return header
end

# =========================================================================
# Test 1: Synthetic binary round-trip
# =========================================================================
@testset "Synthetic binary round-trip" begin
    bin_path = tempname() * ".bin"
    try
        Nx, Ny, Nz = 10, 8, 4

        hdr = write_synthetic_binary(bin_path; Nx=Nx, Ny=Ny, Nz=Nz, Nt=1, include_qv=true)

        reader = ERA5BinaryReader(bin_path; FT=Float64)
        @test Nx == reader.header.Nx
        @test Ny == reader.header.Ny
        @test Nz == reader.header.Nz
        @test window_count(reader) == 1
        @test has_qv(reader) == true
        @test has_flux_delta(reader) == true
        @test length(A_ifc(reader)) == Nz + 1
        @test length(B_ifc(reader)) == Nz + 1

        m, ps, moist_fluxes = load_window!(reader, 1)
        @test size(m) == (Nx, Ny, Nz)
        @test size(ps) == (Nx, Ny)
        @test moist_fluxes isa StructuredFaceFluxState{MoistMassFluxBasis}
        @test flux_basis(moist_fluxes) isa MoistMassFluxBasis

        # Verify loaded values are reasonable
        @test all(m .> 0)
        @test all(ps .> 0)

        qv = load_qv_window!(reader, 1)
        @test qv !== nothing
        @test size(qv) == (Nx, Ny, Nz)
        @test all(isapprox.(qv, 0.005; atol=1e-6))

        deltas = load_flux_delta_window!(reader, 1)
        @test deltas !== nothing
        @test all(deltas.dam .== 0)

        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 2: Full end-to-end: binary → moist → dry → advection → conservation
# =========================================================================
@testset "End-to-end: binary → dry fluxes → advection" begin
    bin_path = tempname() * ".bin"
    try
        Nx, Ny, Nz = 10, 8, 4
        FT = Float64

        write_synthetic_binary(bin_path; Nx=Nx, Ny=Ny, Nz=Nz, Nt=1,
                               include_qv=true, FT=Float32)

        reader = ERA5BinaryReader(bin_path; FT=FT)

        # --- Read phase ---
        m_moist, ps, moist_fluxes = load_window!(reader, 1)
        qv = load_qv_window!(reader, 1)
        close(reader)

        # --- Build grid ---
        A = FT.(reader.header.A_ifc)
        B = FT.(reader.header.B_ifc)
        mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
        vc = HybridSigmaPressure(A, B)
        grid = AtmosGrid(mesh, vc, AtmosTransport.Grids.CPU(); FT=FT)

        # --- Convert phase: moist → dry ---
        dry_fluxes = allocate_face_fluxes(StructuredFluxTopology(), Nx, Ny, Nz;
                                           FT=FT, basis=DryMassFluxBasis)
        cell_mass = zeros(FT, Nx, Ny, Nz)

        driver = PreprocessedERA5Driver(1, reader.header.dt_seconds, 4)
        closure = DiagnoseVerticalFromHorizontal()

        build_dry_fluxes!(dry_fluxes, cell_mass, moist_fluxes, m_moist, qv,
                          grid, driver, closure)

        @test dry_fluxes isa StructuredFaceFluxState{DryMassFluxBasis}
        @test all(cell_mass .> 0)

        # Dry mass should be less than moist mass (by factor ~1-qv)
        @test all(cell_mass .< m_moist .* 1.001)
        @test all(cell_mass .> m_moist .* 0.99)

        # Verify cm satisfies continuity: cm[1] = 0, cm[Nz+1] ≈ 0
        cm = dry_fluxes.cm
        for j in 1:Ny, i in 1:Nx
            @test cm[i, j, 1] ≈ zero(FT) atol=eps(FT)
        end
        Δb = FT[B[k+1] - B[k] for k in 1:Nz]
        for j in 1:Ny, i in 1:Nx
            pit = zero(FT)
            for k in 1:Nz
                pit += dry_fluxes.am[i,j,k] - dry_fluxes.am[i+1,j,k] +
                       dry_fluxes.bm[i,j,k] - dry_fluxes.bm[i,j+1,k]
            end
            for k in 1:Nz
                conv_h = dry_fluxes.am[i,j,k] - dry_fluxes.am[i+1,j,k] +
                         dry_fluxes.bm[i,j,k] - dry_fluxes.bm[i,j+1,k]
                div_v = cm[i,j,k+1] - cm[i,j,k]
                @test abs(conv_h - div_v - Δb[k] * pit) < 1e-10 * max(abs(pit), FT(1))
            end
        end

        # --- Transport phase ---
        rm_init = cell_mass .* FT(400e-6)
        state = CellState(cell_mass; CO2=copy(rm_init))

        scheme = SlopesScheme(MonotoneLimiter())
        ws = AdvectionWorkspace(cell_mass)

        m_total_before = sum(state.air_dry_mass)
        rm_total_before = total_mass(state, :CO2)

        strang_split!(state, dry_fluxes, grid, scheme; workspace=ws)

        m_total_after = sum(state.air_dry_mass)
        rm_total_after = total_mass(state, :CO2)

        # Mass conservation to machine precision
        @test abs(m_total_after - m_total_before) / m_total_before < 1e-12
        @test abs(rm_total_after - rm_total_before) / rm_total_before < 1e-12

        println("  Air mass drift:   ", (m_total_after - m_total_before) / m_total_before)
        println("  Tracer mass drift: ", (rm_total_after - rm_total_before) / rm_total_before)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 3: Basis safety — moist fluxes rejected by strang_split!
# =========================================================================
@testset "Basis safety: moist fluxes rejected by strang_split!" begin
    Nx, Ny, Nz = 6, 4, 3
    FT = Float64
    mesh = LatLonMesh(; Nx=Nx, Ny=Ny, FT=FT)
    A = FT[0.0, 500.0, 5000.0, 0.0]
    B = FT[0.0, 0.0,   0.5,    1.0]
    vc = HybridSigmaPressure(A, B)
    grid = AtmosGrid(mesh, vc, AtmosTransport.Grids.CPU(); FT=FT)

    m = ones(FT, Nx, Ny, Nz) .* FT(1e10)
    state = CellState(m; CO2=m .* FT(400e-6))

    am = zeros(FT, Nx+1, Ny, Nz)
    bm = zeros(FT, Nx, Ny+1, Nz)
    cm = zeros(FT, Nx, Ny, Nz+1)

    moist = StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm)
    scheme = SlopesScheme(MonotoneLimiter())
    ws = AdvectionWorkspace(m)

    @test_throws MethodError strang_split!(state, moist, grid, scheme; workspace=ws)
end

# =========================================================================
# Test 4: Vertical closure (KA kernel) matches CPU loop
# =========================================================================
@testset "Vertical closure: KA kernel matches CPU loop" begin
    FT = Float64
    Nx, Ny, Nz = 20, 15, 6

    am = randn(FT, Nx+1, Ny, Nz) .* FT(1e6)
    bm = randn(FT, Nx, Ny+1, Nz) .* FT(1e6)

    A = vcat(0.0, sort(rand(FT, Nz - 1)) .* 80000, 0.0)
    B = vcat(0.0, sort(rand(FT, Nz - 1)), 1.0)
    Δb = FT[B[k+1] - B[k] for k in 1:Nz]

    cm_cpu = zeros(FT, Nx, Ny, Nz+1)
    cm_ka  = zeros(FT, Nx, Ny, Nz+1)

    diagnose_cm_from_continuity!(cm_cpu, am, bm, Δb, Nx, Ny, Nz)
    diagnose_cm_from_continuity_ka!(cm_ka, am, bm, Δb, Nx, Ny, Nz)

    @test maximum(abs.(cm_cpu .- cm_ka)) < eps(FT) * maximum(abs.(cm_cpu)) * 100

    # Both should have cm[:,:,1] = 0
    @test all(cm_cpu[:, :, 1] .== 0)
    @test all(cm_ka[:, :, 1] .== 0)
end

# =========================================================================
# Test 5: No-QV binary still works (qv = 0 approximation)
# =========================================================================
@testset "Binary without QV" begin
    bin_path = tempname() * ".bin"
    try
        Nx, Ny, Nz = 6, 4, 3

        write_synthetic_binary(bin_path; Nx=Nx, Ny=Ny, Nz=Nz, Nt=1,
                               include_qv=false)

        reader = ERA5BinaryReader(bin_path; FT=Float64)
        @test has_qv(reader) == false

        _, _, moist_fluxes = load_window!(reader, 1)
        qv_result = load_qv_window!(reader, 1)
        @test qv_result === nothing

        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 6: Multi-window binary
# =========================================================================
@testset "Multi-window binary" begin
    bin_path = tempname() * ".bin"
    try
        Nx, Ny, Nz, Nt = 6, 4, 3, 3

        write_synthetic_binary(bin_path; Nx=Nx, Ny=Ny, Nz=Nz, Nt=Nt,
                               include_qv=true)

        reader = ERA5BinaryReader(bin_path; FT=Float64)
        @test window_count(reader) == 3

        for win in 1:Nt
            m, ps, fluxes = load_window!(reader, win)
            @test size(m) == (Nx, Ny, Nz)
            @test all(m .> 0)
        end

        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 7: mass_basis dispatch — reader tags fluxes with correct basis
# =========================================================================
@testset "mass_basis dispatch" begin
    # --- Default (v4, no mass_basis key) → moist ---
    bin_path = tempname() * ".bin"
    try
        write_synthetic_binary(bin_path; Nx=6, Ny=4, Nz=3, Nt=1, include_qv=false)
        reader = ERA5BinaryReader(bin_path; FT=Float64)
        @test mass_basis(reader) === :moist
        _, _, fluxes = load_window!(reader, 1)
        @test fluxes isa StructuredFaceFluxState{MoistMassFluxBasis}
        @test flux_basis(fluxes) isa MoistMassFluxBasis
        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end

    # --- Explicit mass_basis="dry" in header → dry ---
    bin_path2 = tempname() * ".bin"
    try
        write_synthetic_binary_with_basis(bin_path2; mass_basis_str="dry")
        reader = ERA5BinaryReader(bin_path2; FT=Float64)
        @test mass_basis(reader) === :dry
        _, _, fluxes = load_window!(reader, 1)
        @test fluxes isa StructuredFaceFluxState{DryMassFluxBasis}
        @test flux_basis(fluxes) isa DryMassFluxBasis
        close(reader)
    finally
        isfile(bin_path2) && rm(bin_path2)
    end

    # --- Explicit mass_basis="moist" → moist ---
    bin_path3 = tempname() * ".bin"
    try
        write_synthetic_binary_with_basis(bin_path3; mass_basis_str="moist")
        reader = ERA5BinaryReader(bin_path3; FT=Float64)
        @test mass_basis(reader) === :moist
        _, _, fluxes = load_window!(reader, 1)
        @test fluxes isa StructuredFaceFluxState{MoistMassFluxBasis}
        close(reader)
    finally
        isfile(bin_path3) && rm(bin_path3)
    end
end

# =========================================================================
# Test 8: Float64 on-disk binary
# =========================================================================
@testset "Float64 on-disk binary" begin
    bin_path = tempname() * ".bin"
    try
        write_synthetic_binary(bin_path; Nx=6, Ny=4, Nz=3, Nt=1,
                               include_qv=true, FT=Float64)
        reader = ERA5BinaryReader(bin_path; FT=Float64)
        @test reader.header.on_disk_float_type === :Float64
        @test reader.header.float_bytes == 8

        m, ps, fluxes = load_window!(reader, 1)
        @test eltype(m) == Float64
        @test all(m .> 0)
        @test fluxes isa StructuredFaceFluxState{MoistMassFluxBasis}

        qv = load_qv_window!(reader, 1)
        @test qv !== nothing
        @test all(isapprox.(qv, 0.005; atol=1e-10))

        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 9: Tier 2 loaders return nothing on core-only binary
# =========================================================================
@testset "Tier 2 loaders return nothing when absent" begin
    bin_path = tempname() * ".bin"
    try
        write_synthetic_binary(bin_path; Nx=6, Ny=4, Nz=3, Nt=1, include_qv=false)
        reader = ERA5BinaryReader(bin_path; FT=Float64)

        @test has_cmfmc(reader) == false
        @test has_surface(reader) == false
        @test has_tm5conv(reader) == false
        @test has_temperature(reader) == false

        @test load_cmfmc_window!(reader, 1) === nothing
        @test load_surface_window!(reader, 1) === nothing
        @test load_tm5conv_window!(reader, 1) === nothing
        @test load_temperature_window!(reader, 1) === nothing

        close(reader)
    finally
        isfile(bin_path) && rm(bin_path)
    end
end

# =========================================================================
# Test 10: Real production binary — header metadata checks
# =========================================================================
@testset "Real binary metadata" begin
    bin = expanduser("~/data/AtmosTransport/met/era5/spectral_v4_tropo34_dec2021/era5_v4_20211201_merged1000Pa_float32.bin")
    isfile(bin) || @info("Skipping real binary test — file not found"); isfile(bin) || return

    reader = ERA5BinaryReader(bin; FT=Float64)

    @test reader.header.on_disk_float_type === :Float32
    @test reader.header.float_bytes == 4
    @test mass_basis(reader) === :moist
    @test reader.header.Nx == 720
    @test reader.header.Ny == 361
    @test reader.header.Nz == 34
    @test has_flux_delta(reader) == true
    @test has_qv(reader) == false
    @test has_cmfmc(reader) == false
    @test has_surface(reader) == false
    @test has_tm5conv(reader) == false
    @test has_temperature(reader) == false

    _, _, fluxes = load_window!(reader, 1)
    @test fluxes isa StructuredFaceFluxState{MoistMassFluxBasis}

    @test load_cmfmc_window!(reader, 1) === nothing
    @test load_surface_window!(reader, 1) === nothing
    @test load_tm5conv_window!(reader, 1) === nothing
    @test load_temperature_window!(reader, 1) === nothing

    close(reader)
end

println("\n✓ All ERA5 LatLon end-to-end tests passed!")
