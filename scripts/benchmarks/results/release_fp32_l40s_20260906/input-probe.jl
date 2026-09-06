using AtmosTransport, TOML, Test
path="/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
driver=TransportBinaryDriver(path; FT=Float32, Hp=3, validate_windows=false)
try
    window=load_transport_window(driver,1)
    air,ps=window.air_mass,window.surface_pressure
    @test size(air[1])==(96,96,66)
    @test eltype(air[1])===Float32
    @test air[1][4,4,33]>0
    println("PROBE air=",size(air[1])," ",eltype(air[1])," mass33=",air[1][4,4,33]," kg; ps=",ps[1][1,1]," Pa")
finally
    close(driver)
end
