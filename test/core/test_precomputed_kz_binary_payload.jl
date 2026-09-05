#!/usr/bin/env julia
# Round-trip test for the precomputed TM5-diffusion `:kz` CS binary section:
# layer-centre eddy diffusivity (m² s⁻¹) written by the preprocessor and read
# back into the `kz` field of a loaded window for the `precomputed_kz` runtime
# diffusion path.

using Test

using AtmosTransport

@testset "Precomputed Kz CS binary payload roundtrips" begin
    FT = Float32
    Nc, Nz, np = 2, 3, 6
    vc = AtmosTransport.HybridSigmaPressure(FT[0, 10, 50, 100], FT[0, 0.1, 0.5, 1])

    mktemp() do path, io
        close(io)
        writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
            path, Nc, np, Nz, 1, vc;
            FT = FT,
            dt_met_seconds = 3600,
            steps_per_window = 4,
            include_flux_delta = true,
            include_precomputed_kz = true,
            mass_basis = :dry,
        )

        panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
        panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)
        # Distinct per-panel, per-level values so any offset/permutation error
        # in the round-trip is caught.
        kz = ntuple(p -> FT[0.1 * p + k for i in 1:Nc, j in 1:Nc, k in 1:Nz], np)

        window = (
            m  = panels3(10),
            am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
            bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
            cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
            ps = panels2(90000),
            dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np),
            kz = kz,
        )
        AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, np)
        AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

        reader = AtmosTransport.MetDrivers.CubedSphereBinaryReader(path; FT = FT)
        try
            @test :kz in reader.header.payload_sections
            @test reader.header.raw_header["include_precomputed_kz"] == true

            loaded = AtmosTransport.MetDrivers.load_cs_window(reader, 1)
            @test loaded.kz !== nothing
            for p in 1:np
                @test loaded.kz[p] == window.kz[p]   # exact byte round-trip
            end

            # Runtime refresh: a PrecomputedCSKzField filled from the loaded
            # window exposes the values through field_value/panel_field.
            host_cache = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np)
            field = AtmosTransport.State.Fields.PrecomputedCSKzField(host_cache)
            AtmosTransport.State.Fields.refresh_precomputed_cs_kz_cache!(field, loaded.kz)
            for p in 1:np, k in 1:Nz, j in 1:Nc, i in 1:Nc
                got = AtmosTransport.State.Fields.field_value(
                    AtmosTransport.State.Fields.panel_field(field, p), (i, j, k))
                @test got == window.kz[p][i, j, k]
            end
            @test_throws ArgumentError AtmosTransport.State.Fields.refresh_precomputed_cs_kz_cache!(
                field, nothing)
        finally
            close(reader.io)
        end
    end

    @testset "kz absent when not written" begin
        mktemp() do path, io
            close(io)
            writer = AtmosTransport.MetDrivers.open_streaming_cs_transport_binary(
                path, Nc, np, Nz, 1, vc;
                FT = FT, dt_met_seconds = 3600, steps_per_window = 4,
                include_flux_delta = true, mass_basis = :dry,
            )
            panels3(x) = ntuple(p -> fill(FT(x + p), Nc, Nc, Nz), np)
            panels2(x) = ntuple(p -> fill(FT(x + p), Nc, Nc), np)
            window = (
                m  = panels3(10),
                am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), np),
                bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), np),
                cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), np),
                ps = panels2(90000),
                dm = ntuple(_ -> zeros(FT, Nc, Nc, Nz), np),
            )
            AtmosTransport.MetDrivers.write_streaming_cs_window!(writer, window, Nc, np)
            AtmosTransport.MetDrivers.close_streaming_transport_binary!(writer)

            reader = AtmosTransport.MetDrivers.CubedSphereBinaryReader(path; FT = FT)
            try
                @test !(:kz in reader.header.payload_sections)
                @test AtmosTransport.MetDrivers.load_cs_window(reader, 1).kz === nothing
            finally
                close(reader.io)
            end
        end
    end
end
