using CUDA, AtmosTransport, Adapt, TOML, Statistics
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
const M = AtmosTransport.Models
const input = "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin"
const driver = TransportBinaryDriver(input; FT=Float32, Hp=3, validate_windows=false)
const grid = AtmosTransport.MetDrivers.driver_grid(driver)
const recipe = M.build_runtime_physics_recipe(TOML.parsefile("/tmp/atmos-main-real-input-after/tracers32.toml"), driver, Float32; halo_width=3)
close(driver)
function build(::Val{:host}, state, fluxes)
    model = TransportModel(state, fluxes, grid, recipe.advection;
        diffusion=recipe.diffusion, convection=recipe.convection)
    return Adapt.adapt(CuArray, model)
end
function build(::Val{:device}, state, fluxes)
    model = TransportModel(Adapt.adapt(CuArray, state), Adapt.adapt(CuArray, fluxes),
        grid, recipe.advection; diffusion=recipe.diffusion, convection=recipe.convection)
    return Adapt.adapt(CuArray, model)
end
function measure(mode, state, fluxes)
    GC.gc(true); CUDA.reclaim()
    stats = @timed begin
        model = build(mode, state, fluxes)
        CUDA.synchronize()
        model
    end
    @assert stats.value.state.air_mass[1] isa CuArray
    @assert stats.value.workspace.advection_ws.rm_4d_A isa CuArray
    return (;seconds=stats.time, allocated_bytes=stats.bytes, gc_seconds=stats.gctime)
end
results = Dict{String,Any}()
for nt in (6,32)
    air = ntuple(_ -> fill(1f0,96,96,66),6)
    tracers = Dict(Symbol("t",i) => air for i in 1:nt)
    state = CubedSphereState(DryBasis, grid.horizontal, air; tracers...)
    fluxes = allocate_face_fluxes(grid.horizontal,66;FT=Float32,basis=DryBasis)
    for sample in 0:5
        for mode in (iseven(sample) ? (:host,:device) : (:device,:host))
            stats = measure(Val(mode),state,fluxes)
            results["$(nt)_$(sample)_$(mode)"] = Dict(string(k)=>v for (k,v) in pairs(stats))
            println("CONSTRUCTION ",nt," ",sample," ",mode," ",stats);flush(stdout)
        end
    end
end
open("/tmp/atmos-main-construction-ab.toml","w") do io
    TOML.print(io, results)
end
