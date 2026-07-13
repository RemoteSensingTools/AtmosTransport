using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = normpath(joinpath(@__DIR__, ".."))))
Pkg.instantiate()

include(joinpath(@__DIR__, "make.jl"))
