using Pkg, TOML
root = pwd()
dir = mktempdir()
mkpath(dir)
cp(joinpath(root, "benchmarking", "src"), joinpath(dir, "src"); force=true)
project = TOML.parsefile(joinpath(root, "benchmarking", "Project.toml"))
project["sources"]["AtmosTransport"]["path"] = root
open(joinpath(dir, "Project.toml"), "w") do io
    TOML.print(io, project)
end
Pkg.activate(dir)
Pkg.develop(path=root)
Pkg.instantiate()
empty!(ARGS)
append!(ARGS, ["--backend=cpu", "--float-type=Float32,Float64", "--grid=C4",
    "--levels=32", "--tracers=1,4", "--operator=advection,diffusion,convection,full,io,adjoint",
    "--steps=1", "--warmup-steps=1", "--repeats=1", "--group=CPU release smoke",
    "--output=" * joinpath(@__DIR__, "benchmark-results.json")])
include(joinpath(root, "benchmarking", "run_benchmarks.jl"))
println("BENCHMARK_SMOKE_PASSED")
