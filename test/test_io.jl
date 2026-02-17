using Test
using AtmosTransportModel.IO

@testset "IO" begin
    @testset "TOML config loading" begin
        # Load built-in configs
        geosfp_config = default_met_config("geosfp")
        @test geosfp_config.name == "GEOS-FP"
        @test haskey(geosfp_config.variables, :u_wind)
        @test geosfp_config.variables[:u_wind].native_name == "u"
        @test haskey(geosfp_config.collections, "asm_Nv_inst")

        merra2_config = default_met_config("merra2")
        @test merra2_config.name == "MERRA-2"
        @test merra2_config.variables[:u_wind].native_name == "U"

        era5_config = default_met_config("era5")
        @test era5_config.name == "ERA5"
        @test era5_config.variables[:specific_humidity].native_name == "q"
    end

    @testset "MetDataSource construction" begin
        # GEOS-FP
        met_geosfp = GEOSFPMetData()
        @test met_geosfp isa MetDataSource{Float64}
        @test met_geosfp isa AbstractMetData{Float64}
        @test source_name(met_geosfp) == "GEOS-FP"
        @test protocol(met_geosfp) == "opendap"
        @test has_variable(met_geosfp, :u_wind)
        @test has_variable(met_geosfp, :diffusivity)
        @test !has_variable(met_geosfp, :nonexistent_var)

        # MERRA-2
        met_merra = MERRAMetData()
        @test met_merra isa MetDataSource{Float64}
        @test source_name(met_merra) == "MERRA-2"
        @test time_interval(met_merra) == 10800.0

        # ERA5
        met_era5 = ERA5MetData()
        @test met_era5 isa MetDataSource{Float64}
        @test source_name(met_era5) == "ERA5"
        @test time_interval(met_era5) == 3600.0

        # Float32
        met32 = GEOSFPMetData(FT=Float32)
        @test met32 isa MetDataSource{Float32}

        # With local path
        met_local = GEOSFPMetData(local_path="/tmp/test_met")
        @test met_local.local_path == "/tmp/test_met"
    end

    @testset "Variable mapping" begin
        met = GEOSFPMetData()
        @test native_name(met, :u_wind) == "u"
        @test native_name(met, :temperature) == "t"
        @test native_name(met, :diffusivity) == "kh"

        coll = collection_for(met, :u_wind)
        @test coll.dataset == "inst3_3d_asm_Nv"

        coll_trb = collection_for(met, :diffusivity)
        @test coll_trb.dataset == "tavg3_3d_trb_Ne"
        @test coll_trb.levels == 73
    end

    @testset "MERRA-2 stream codes" begin
        @test merra2_stream(1985) == 100
        @test merra2_stream(1995) == 200
        @test merra2_stream(2005) == 300
        @test merra2_stream(2024) == 400
    end

    @testset "URL builders" begin
        config = default_met_config("geosfp")
        url = build_opendap_url(config, "asm_Nv_inst")
        @test contains(url, "opendap.nccs.nasa.gov")
        @test contains(url, "inst3_3d_asm_Nv")

        merra_config = default_met_config("merra2")
        murl = build_merra2_file_url(merra_config, "asm_Nv_inst", 2024, 3, 1)
        @test contains(murl, "MERRA2_400")
        @test contains(murl, "20240301")
        @test contains(murl, "M2I3NVASM")
    end

    @testset "Canonical variables from TOML" begin
        vars = canonical_variables()
        @test :u_wind in vars
        @test :temperature in vars
        @test :surface_pressure in vars
        @test :pressure_thickness in vars

        units = canonical_units()
        @test units[:u_wind] == "m/s"
        @test units[:surface_pressure] == "Pa"

        required = canonical_required()
        @test :u_wind in required
        @test :surface_pressure in required
    end

    @testset "Config validation" begin
        config = default_met_config("geosfp")
        @test validate_met_config(config)
    end

    @testset "Configuration" begin
        @test hasmethod(load_configuration, Tuple{String})
    end

    # Network-dependent tests — only run when ATMOS_TEST_NETWORK=1
    if get(ENV, "ATMOS_TEST_NETWORK", "") == "1"
        @testset "GEOS-FP OPeNDAP live read" begin
            using NCDatasets

            met = GEOSFPMetData()
            url = build_opendap_url(met.config, "asm_Nv_inst")
            @test contains(url, "inst3_3d_asm_Nv")

            # Open and read a single 2D field (surface pressure)
            ds = NCDataset(url)
            ps = ds["ps"][:, :, ds.dim["time"]]
            close(ds)

            @test size(ps) == (1152, 721)
            @test minimum(ps) > 30000.0   # minimum ps > 300 hPa
            @test maximum(ps) < 110000.0  # maximum ps < 1100 hPa
        end
    end
end
