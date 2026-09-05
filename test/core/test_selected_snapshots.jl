using Test, AtmosTransport, NCDatasets
using AtmosTransport.Output
using AtmosTransport.Grids: ncells

@testset "Selected capture preserves full-capture NetCDF values" begin
    for FT in (Float32, Float64), topology in (:ll, :rg, :cs)
        Nz = 4
        mesh = topology === :ll ? LatLonMesh(; Nx=6, Ny=4, FT) :
               topology === :rg ? ReducedGaussianMesh(FT[-45,45], [4,8]; FT) :
               CubedSphereMesh(; Nc=4, Hp=3, FT)
        vc = HybridSigmaPressure(zeros(FT,Nz+1), collect(range(zero(FT),one(FT);length=Nz+1)))
        grid = AtmosGrid(mesh, vc, CPU(); FT)
        shape = topology === :ll ? (6,4,Nz) : topology === :rg ? (ncells(mesh),Nz) : (10,10,Nz)
        a = reshape(FT.(1:prod(shape)), shape)
        q = reshape(FT.(range(0.0003,0.0006;length=prod(shape))), shape)
        air = topology === :cs ? ntuple(p -> a .* FT(p),6) : a
        rm = topology === :cs ? map(m -> m .* q,air) : a .* q
        state = topology === :cs ? CubedSphereState(DryBasis,air; co2=rm,unused=rm,halo_width=3) :
                CellState(DryBasis,air; co2=rm,unused=rm)
        model = (; state, grid)
        halo_width = topology === :cs ? 3 : 0
        full = capture_snapshot(model; halo_width)
        for mode in ("none","selected","full")
            fields = output_field_spec(Dict{String,Any}(
                "tracers"=>["co2"], "layers"=>mode, "levels"=>[1,3],
                "air_mass_layers"=>mode, "air_mass"=>true,
                "air_mass_per_area"=>true, "column_air_mass_per_area"=>true,
                "column_mean"=>true, "column_mass_per_area"=>true))
            captured = capture_snapshot(model; halo_width, fields)
            @test Set(keys(captured.tracers)) == Set([:co2])
            @test captured.nlevel == Nz
            mode == "none" && @test isempty(captured.levels)
            mode == "selected" && @test captured.levels == [1,3]
            mktempdir() do dir
                options = SnapshotWriteOptions(;float_type=Float64)
                p = write_snapshot_netcdf(joinpath(dir,"full.nc"),[full],grid;fields,options)
                q = write_snapshot_netcdf(joinpath(dir,"selected.nc"),[captured],grid;fields,options)
                NCDataset(p) do x
                    NCDataset(q) do y
                        @test Set(keys(x)) == Set(keys(y))
                        for name in keys(x)
                            @test isequal(x[name][:], y[name][:])
                        end
                    end
                end
            end
        end
    end
end

@testset "Physics preflight fails before opening binaries" begin
    mktemp() do path, io
        base = Dict{String,Any}("input"=>Dict("binary_paths"=>[path]),
                               "architecture"=>Dict("use_gpu"=>false))
        @test first(AtmosTransport.Models.validate_config(base))
        for (section, options) in (
            ("adveciton",Dict("scheme"=>"ppm")),
            ("run",Dict("scheem"=>"ppm")),
            ("advection",Dict("scheem"=>"ppm")),
            ("advection",Dict("scheme"=>"bad")),
            ("advection",Dict("scheme"=>"linrood","ppm_order"=>6)),
            ("diffusion",Dict("kind"=>"bad")),
            ("diffusion",Dict("kind"=>"constant","value"=>-1)),
            ("convection",Dict("kind"=>"bad")),
            ("chemistry",Dict("kind"=>"bad")))
            cfg = merge(base,Dict(section=>options))
            ok, errors = AtmosTransport.Models.validate_config(cfg)
            @test !ok
            @test any(e -> occursin(section,e),errors)
        end
        for scheme in ("slopes","ppm")
            @test_throws ArgumentError AtmosTransport.Models.build_runtime_advection(
                Dict("advection"=>Dict("scheme"=>scheme)),
                AtmosTransport.Models.ReducedGaussianRuntimeRecipeStyle())
        end
    end
end
