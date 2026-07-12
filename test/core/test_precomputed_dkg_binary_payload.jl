#!/usr/bin/env julia

using Test
using Random
using LinearAlgebra: dot
using KernelAbstractions: get_backend, synchronize

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Operators: ImplicitVerticalDiffusion
using .AtmosTransport.Operators.Diffusion: apply_vertical_diffusion_vmr!
using .AtmosTransport.State: PrecomputedCSDkgField, panel_field
using .AtmosTransport.State.Fields: refresh_precomputed_cs_dkg_cache!, field_value
using .AtmosTransport.Adjoints: _vertical_diffusion_cs_single_dkg_adjoint_kernel!

@testset "Exact TM5 dkg binary and runtime path" begin
    FT = Float64
    Nc, Nz, np = 2, 4, 6
    vc = HybridSigmaPressure(FT[0, 20, 100, 400, 1000], FT[0, 0.05, 0.3, 0.7, 1])
    panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
    panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)

    mktemp() do path, io
        close(io)
        writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1, vc; FT, include_flux_delta = true,
            include_precomputed_dkg = true, mass_basis = :dry)
        dkg = ntuple(p -> FT[(k == Nz ? 0 : 100p + 10k + i + j)
                             for i in 1:Nc, j in 1:Nc, k in 1:Nz], np)
        window = (m = panels3(1e6),
                  am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
                  bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
                  cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
                  ps = panels2(9e4), dm = panels3(0), dkg)
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, np)
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

        reader = AtmosTransport.MetDrivers.CubedSphereBinaryReader(path; FT)
        try
            @test :dkg in reader.header.payload_sections
            @test !(:kz in reader.header.payload_sections)
            @test reader.header.raw_header["precomputed_dkg_payload"] ==
                  "tm5_bldiff_interface_dkg_dry_v1"
            loaded = AtmosTransport.MetDrivers.load_cs_window(reader, 1)
            @test loaded.kz === nothing
            @test loaded.dkg == dkg

            cache = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np)
            field = PrecomputedCSDkgField(cache)
            refresh_precomputed_cs_dkg_cache!(field, loaded.dkg)
            @test field_value(panel_field(field, 4), (2, 1, 3)) == dkg[4][2, 1, 3]
            @test_throws ArgumentError refresh_precomputed_cs_dkg_cache!(field, nothing)

            spec = AtmosTransport.Models.diffusion_spec(Dict("kind" => "precomputed_kz"))
            op = AtmosTransport.Models.materialize(
                spec, AtmosTransport.Models.CubedSphereRuntimeRecipeStyle(), FT, reader)
            @test op.kz_field isa PrecomputedCSDkgField
            driver = AtmosTransport.MetDrivers.CubedSphereTransportDriver(reader; Hp = 1)
            @test AtmosTransport.MetDrivers.supports_diffusion(driver)
        finally
            close(reader.io)
        end
    end

    @test_throws ArgumentError AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        tempname(), Nc, np, Nz, 1, vc; FT,
        include_precomputed_kz = true, include_precomputed_dkg = true)
    @test_throws ArgumentError AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
        tempname(), Nc, np, Nz, 1, vc; FT,
        include_precomputed_dkg = true, mass_basis = :moist)

    # Direct-dkg solve conserves column tracer mass without consulting dz.
    Hp = 1
    N = Nc + 2Hp
    panels_m = ntuple(_ -> fill(FT(1e6), N, N, Nz), np)
    panels_rm = ntuple(_ -> zeros(FT, N, N, Nz), np)
    rng = MersenneTwister(42)
    for p in 1:np, k in 1:Nz, j in 1:Nc, i in 1:Nc
        panels_rm[p][i + Hp, j + Hp, k] = rand(rng)
    end
    x_rm = ntuple(p -> copy(panels_rm[p]), np)
    y_seed = ntuple(_ -> zeros(FT, N, N, Nz), np)
    for p in 1:np, k in 1:Nz, j in 1:Nc, i in 1:Nc
        y_seed[p][i + Hp, j + Hp, k] = randn(rng)
    end
    dkg = ntuple(_ -> FT[(k == Nz ? 0 : 2e3 + 100k)
                          for i in 1:Nc, j in 1:Nc, k in 1:Nz], np)
    field = PrecomputedCSDkgField(ntuple(p -> copy(dkg[p]), np))
    op = ImplicitVerticalDiffusion(; kz_field = field)
    ws = (w_scratch = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np),
          dz_scratch = ntuple(_ -> fill(FT(NaN), Nc, Nc, Nz), np))
    before = ntuple(p -> dropdims(sum(@view(panels_rm[p][2:3, 2:3, :]); dims=3); dims=3), np)
    apply_vertical_diffusion_vmr!(panels_rm, panels_m, op, ws, FT(300);
                                  halo_width = Hp)
    after = ntuple(p -> dropdims(sum(@view(panels_rm[p][2:3, 2:3, :]); dims=3); dims=3), np)
    @test all(p -> all(isfinite, @view(panels_rm[p][2:3, 2:3, :])), 1:np)
    for p in 1:np
        @test before[p] ≈ after[p] rtol=1e-13
    end

    lambda = ntuple(p -> copy(y_seed[p]), np)
    backend = get_backend(lambda[1])
    kernel = _vertical_diffusion_cs_single_dkg_adjoint_kernel!(backend, (8, 8))
    for p in 1:np
        kernel(lambda[p], panels_m[p], panel_field(field, p), ws.w_scratch[p],
               FT(300), Nz, Hp; ndrange = (Nc, Nc))
        synchronize(backend)
    end
    inner(a, b) = sum(dot(@view(a[p][2:3, 2:3, :]), @view(b[p][2:3, 2:3, :]))
                      for p in 1:np)
    @test inner(y_seed, panels_rm) ≈ inner(lambda, x_rm) rtol=1e-12
end

@testset "TM5 bldiff emits the vendored dkg formula" begin
    using .AtmosTransport.Preprocessing: BLDiffConstants, BLDiffColumnScratch,
        tm5_bldiff_dkg_column!
    FT = Float64
    Nz = 6
    A = collect(range(0.0, 2000.0; length=Nz + 1))
    B = collect(range(0.0, 1.0; length=Nz + 1))
    ps = FT(1e5)
    T = collect(range(220.0, 288.0; length=Nz))
    q = collect(range(0.0, 0.01; length=Nz))
    u = collect(range(12.0, 3.0; length=Nz)); v = zeros(FT, Nz)
    m = FT[1, 2, 4, 7, 10, 12] .* 1e13
    scratch = BLDiffColumnScratch{FT}(Nz)
    dkg = zeros(FT, Nz)
    tm5_bldiff_dkg_column!(dkg, T, q, u, v, m, ps, 200.0, 100.0, 0.4,
                           A, B, BLDiffConstants{FT}(), scratch)
    for k in 1:Nz-1
        l = Nz - k
        expected = max(scratch.kvh[l], 0) * 2 * (m[k] + m[k+1]) /
                   (scratch.dz[k] + scratch.dz[k+1])^2
        @test dkg[k] ≈ expected rtol=2e-15
    end
    @test dkg[end] == 0
end
