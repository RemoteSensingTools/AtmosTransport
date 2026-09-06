using TOML, Test
monthly = TOML.parsefile(ARGS[1])
weekly = TOML.parsefile(ARGS[2])
@testset "Monthly run reproduces all six weekly total series" begin
    for t in 1:6
        key = "tracer$(lpad(t,2,'0'))_total_mass"
        @test monthly[key][1:8] == weekly[key]
    end
end
