"""
    Simulation

Minimal simulation harness for the standalone `src` transport model.
"""
mutable struct Simulation{ModelT, FT, CB}
    model      :: ModelT
    Δt         :: FT
    stop_time  :: FT
    time       :: FT
    iteration  :: Int
    callbacks  :: CB
end

function Simulation(model::TransportModel;
                    Δt::Real,
                    stop_time::Real,
                    callbacks = NamedTuple())
    FT = promote_type(typeof(float(Δt)), typeof(float(stop_time)))
    return Simulation{typeof(model), FT, typeof(callbacks)}(
        model, FT(Δt), FT(stop_time), zero(FT), 0, callbacks)
end

function step!(sim::Simulation)
    step!(sim.model, sim.Δt)
    sim.time += sim.Δt
    sim.iteration += 1
    for callback in values(sim.callbacks)
        callback(sim)
    end
    return nothing
end

function run!(sim::Simulation)
    while sim.time < sim.stop_time
        step!(sim)
    end
    return sim
end

export Simulation, run!
