using Test
using AtmosTransport

@testset "AtmosTransport" begin
    include("test_architectures.jl")
    include("test_grids.jl")
    include("test_fields.jl")
    include("test_advection.jl")
    include("test_diffusion.jl")
    include("test_convection.jl")
    include("test_numerics_tm5.jl")
    include("test_timestepper.jl")
    include("test_adjoint_gradient.jl")
    include("test_io.jl")
    include("test_met_integration.jl")
    include("test_regridding.jl")
    include("test_callbacks.jl")
    include("test_mass_flux_advection.jl")
    include("test_cubed_sphere_mass_flux.jl")
end
