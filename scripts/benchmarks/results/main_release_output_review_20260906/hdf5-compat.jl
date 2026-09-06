using Pkg, TOML
root = pwd()
dir = mktempdir()
mkpath(dir)
project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
project["sources"]["AtmosTransport"]["path"] = root
project["compat"]["HDF5_jll"] = "1.14"
open(joinpath(dir, "Project.toml"), "w") do io
    TOML.print(io, project)
end
Pkg.activate(dir)
Pkg.develop(path=root)
Pkg.add(PackageSpec(name="HDF5", version="0.17"))
using HDF5, HDF5_jll, NCDatasets, AtmosTransport, Test
println("HDF5=", pkgversion(HDF5), " HDF5_jll=", pkgversion(HDF5_jll),
        " NCDatasets=", pkgversion(NCDatasets),
        " NetCDF_jll=", pkgversion(NCDatasets.NetCDF_jll))
@test NCDatasets.NetCDF_jll.HDF5_jll === HDF5_jll
println("HDF5_LIBRARY=", HDF5_jll.libhdf5)
for file in ("test_output_snapshots.jl", "test_selected_snapshots.jl", "test_netcdf_stream.jl", "test_async_output_lifetime.jl")
    println("RUNNING ", file); flush(stdout)
    mod = Module(gensym(:CompatCheck))
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Base.include(mod, joinpath(root,"test","core",file))
end
println("HDF5_114_COMPAT_PASSED")
