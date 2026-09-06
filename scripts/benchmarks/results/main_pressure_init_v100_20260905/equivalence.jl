using AtmosTransport, Test, Random
const IC = AtmosTransport.Models.InitialConditionIO
# Run from the repository root; compare against the preserved prechange method.
source = read(`git show 9544a3d3:src/Models/initial_conditions/cubed_sphere.jl`, String)
start_index = findfirst("function _build_cs_pressure_layer_ic(", source).start
end_index = findnext("\nend", source, start_index).stop
reference = replace(source[start_index:end_index], "_build_cs_pressure_layer_ic(" => "_pressure_layer_before(")
Base.include_string(IC, reference, "pressure_layer_at_9544a3d3.jl")
Random.seed!(48017)
@testset "Pressure-layer prechange exact equivalence" begin
    for FT in (Float32, Float64), Nz in (4, 66), Hp in (1,3)
        Nc = 4
        mesh = CubedSphereMesh(; Nc, Hp, FT)
        B = FT.(range(0,1;length=Nz+1))
        A = FT.(2000 .* sinpi.(B))
        grid = AtmosGrid(mesh, HybridSigmaPressure(A,B), CPU(); FT)
        air = ntuple(_ -> FT.(1000 .+ 500rand(Nc+2Hp,Nc+2Hp,Nz)), 6)
        ps = ntuple(_ -> FT.(65000 .+ 35000rand(Nc,Nc)), 6)
        for lowest in (false,true), fraction in (0.001, 0.2, 0.5, 0.999, 1.0)
            cfg = Dict{String,Any}("kind"=>"pressure_layer", "lowest_layer"=>lowest,
                "psurf_fraction"=>fraction,"total_molecules"=>1e28)
            old = IC._pressure_layer_before(air,grid,cfg,FT,ps)
            new = IC.build_initial_mixing_ratio(air,grid,cfg;surface_pressure=ps)
            @test all(isequal.(old,new))
        end
    end
end
