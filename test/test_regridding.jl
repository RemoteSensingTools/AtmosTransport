using Test
using AtmosTransport.Regridding
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids: CubedSphereGrid
using AtmosTransport.IO: default_met_config, build_vertical_coordinate
using AtmosTransport.Sources

@testset "Regridding" begin
    @test IdentityRegridder() isa AbstractRegridder

    # Identity regridder should just copy
    src = rand(10)
    dst = zeros(10)
    regrid!(dst, src, IdentityRegridder())
    @test dst == src

    @testset "Conservative CS map tunable sub-sampling" begin
        function _panel_rms_error(a, b)
            n = 0
            s = 0.0
            for p in 1:6
                @test size(a[p]) == size(b[p])
                for idx in eachindex(a[p])
                    d = Float64(a[p][idx] - b[p][idx])
                    s += d^2
                    n += 1
                end
            end
            return sqrt(s / n)
        end

        vc = build_vertical_coordinate(default_met_config("geosfp"); FT=Float64)
        grid = CubedSphereGrid(CPU(); FT=Float64, Nc=24, vertical=vc)

        lons = collect(15.0:30.0:345.0)
        lats = collect(-75.0:30.0:75.0)
        flux = [1.0 + 0.35 * cosd(lat) + 0.2 * sind(lon) * cosd(lat)^2
                for lon in lons, lat in lats]

        panels_ref = regrid_latlon_to_cs(flux, lons, lats, grid; N_sub=24)
        panels_low = regrid_latlon_to_cs(flux, lons, lats, grid; N_sub=2)
        panels_mid = regrid_latlon_to_cs(flux, lons, lats, grid; N_sub=8)

        err_low = _panel_rms_error(panels_low, panels_ref)
        err_mid = _panel_rms_error(panels_mid, panels_ref)
        @test err_mid < err_low

        cs_map_mid = build_conservative_cs_map(lons, lats, grid; N_sub=8)
        panels_mid_cached = regrid_latlon_to_cs(flux, lons, lats, grid; cs_map=cs_map_mid)
        @test _panel_rms_error(panels_mid_cached, panels_mid) ≈ 0 atol=1e-12
    end
end
