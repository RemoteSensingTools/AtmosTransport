using Pkg, TOML
base = @__DIR__
root = joinpath(base,"current")
for label in ("current", "baseline")
    dir=joinpath(base,label*"-env")
    mkpath(dir)
    p=TOML.parsefile(joinpath(root,"test","Project.toml"))
    p["sources"]["AtmosTransport"]["path"]=joinpath(base,label)
    p["deps"]["CUDA"]="052768ef-5323-5732-b1bb-66c8b64840ba"
    p["deps"]["CUDA_Runtime_jll"]="76a88914-d11a-5bdc-97e0-2f5a05c973a2"
    p["compat"]["CUDA"]="=5.11.3"
    open(joinpath(dir,"Project.toml"),"w") do io
        TOML.print(io,p)
    end
    write(joinpath(dir,"LocalPreferences.toml"), "[CUDA_Runtime_jll]\nversion = \"12.6\"\n")
    Pkg.activate(dir)
    Pkg.develop(path=joinpath(base,label))
    Pkg.instantiate()
    println("ENVIRONMENT_READY ",label);flush(stdout)
end
