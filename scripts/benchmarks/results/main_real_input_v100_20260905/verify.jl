using NCDatasets, TOML, Test
max_error = 0.0
checks = 0
for nt in (6,32), sample in 0:2
    path="/tmp/atmos-main-real-profile/tracers$(nt)_sample$(sample).nc"
    NCDataset(path) do ds
        @test ds["time"][:] == [0,2]
        for name in keys(ds)
            endswith(name,"_total_mass") || continue
            mass=ds[name][:]
            @test length(mass)==2 && all(isfinite,mass)
            relative=abs(mass[2]-mass[1])/abs(mass[1])
            global max_error=max(max_error,relative)
            @test relative < 1e-5
            global checks+=2
        end
        for name in keys(ds)
            startswith(name,"tracer") || continue
            @test all(isfinite,ds[name][:])
            global checks+=1
        end
    end
end
println("Maximum relative total-mass drift: ",max_error)
println("Tracer field/mass checks: ",checks," plus six time-axis checks")
