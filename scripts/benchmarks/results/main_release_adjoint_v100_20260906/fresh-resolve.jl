using Pkg, TOML
root = pwd()
p = TOML.parsefile("test/Project.toml")
version = VersionNumber(TOML.parsefile("Project.toml")["version"])
compat = get(p["compat"], "AtmosTransport", "*")
println("Package version: ", version, "; test compat: ", compat)
println("Version admitted: ", version in Pkg.Types.semver_spec(compat))
mktempdir() do dir
    p["sources"]["AtmosTransport"]["path"] = root
    open(joinpath(dir, "Project.toml"), "w") do io
        TOML.print(io, p)
    end
    Pkg.activate(dir)
    try
        Pkg.develop(PackageSpec(path=root))
        println("FRESH TEST ENVIRONMENT RESOLVED")
    catch err
        println("FRESH TEST ENVIRONMENT FAILED: ", sprint(showerror, err))
    end
end
