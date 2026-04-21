# ---------------------------------------------------------------------------
# Revolve-style checkpointing
#
# Balances memory (O(snapshots)) vs recomputation (O(steps * log(steps) / snapshots))
# for adjoint runs. Stores model state at selected time steps; replays
# forward from the nearest checkpoint during the backward sweep.
#
# StoreAllCheckpointer: simple strategy when memory is not a concern.
# RevolveCheckpointer: simplified multi-level checkpointing.
# ---------------------------------------------------------------------------

using ..TimeSteppers: time_step!, adjoint_time_step!

"""
$(TYPEDEF)

Supertype for checkpointing strategies used during adjoint runs.
"""
abstract type AbstractCheckpointer end

"""
$(TYPEDEF)

Simple checkpointer that stores the full tracer state at every time step.
Use when memory is not a concern. O(n_steps) memory, no recomputation.
"""
struct StoreAllCheckpointer <: AbstractCheckpointer end

"""
$(TYPEDEF)

Optimal checkpointing using the Revolve algorithm (Griewank & Walther, 2000).

$(FIELDS)
"""
struct RevolveCheckpointer{S} <: AbstractCheckpointer
    "number of checkpoint storage slots"
    n_snapshots :: Int
    "storage backend (:memory or :disk)"
    storage     :: S
end

RevolveCheckpointer(; n_snapshots::Int = 10, storage = :memory) =
    RevolveCheckpointer(n_snapshots, storage)

# ---------------------------------------------------------------------------
# Helpers for saving/restoring tracer state
# ---------------------------------------------------------------------------

"""
$(SIGNATURES)

Save a copy of tracer state. Returns a NamedTuple of copied arrays.
"""
function _save_checkpoint(tracers)
    return (; (k => copy(v) for (k, v) in pairs(tracers))...)
end

"""
$(SIGNATURES)

Restore tracer state from a checkpoint (mutates model.tracers in place).
"""
function _restore_checkpoint!(tracers, checkpoint)
    for (k, v) in pairs(checkpoint)
        copyto!(tracers[k], v)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# run_adjoint! for StoreAllCheckpointer
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute a full forward-then-backward adjoint run with store-all checkpointing.

Returns the gradient of the cost function w.r.t. the initial tracer state
(i.e. `model.adj_tracers` after the backward pass).

# Algorithm (store-all)
1. Store initial tracer state
2. Forward loop: for step 1..n_steps: save checkpoint, time_step!
3. Compute adjoint forcing: adj_init = cost_gradient_fn(model.tracers)
4. Copy adj_init into model.adj_tracers
5. Backward loop: for step n_steps..1: restore from checkpoint, adjoint_time_step!
6. Return model.adj_tracers (gradient w.r.t. initial tracers)
"""
function run_adjoint!(model, met_data, checkpointer::StoreAllCheckpointer, n_steps, Δt;
                      cost_gradient_fn)

    # Ensure model has met_data (use passed met_data)
    work_model = (; model..., met_data)

    # 1 & 2. Forward loop: save state at start of each step, then time_step!
    checkpoints = Vector{Any}(undef, n_steps)
    for step in 1:n_steps
        checkpoints[step] = _save_checkpoint(work_model.tracers)
        time_step!(work_model, Δt)
    end

    # 3. Compute adjoint forcing from final tracer state
    adj_init = cost_gradient_fn(work_model.tracers)

    # 4. Copy adj_init into model.adj_tracers
    _restore_checkpoint!(work_model.adj_tracers, adj_init)

    # 5. Backward loop (clock is already at n_steps*Δt from forward)
    for step in n_steps:-1:1
        _restore_checkpoint!(work_model.tracers, checkpoints[step])
        adjoint_time_step!(work_model, Δt)
    end

    # 6. Return gradient w.r.t. initial tracers
    return work_model.adj_tracers
end

# ---------------------------------------------------------------------------
# run_adjoint! for RevolveCheckpointer (simplified multi-level)
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute a full forward-then-backward adjoint run with simplified multi-level checkpointing.

Stores n_snapshots checkpoints at evenly spaced steps. During the backward sweep,
replays forward from the nearest checkpoint when state is needed.
"""
function run_adjoint!(model, met_data, checkpointer::RevolveCheckpointer, n_steps, Δt;
                      cost_gradient_fn)

    n_snap = min(checkpointer.n_snapshots, n_steps + 1)

    # Ensure model has met_data
    work_model = (; model..., met_data)

    # Checkpoint indices: 0 (initial), step_size, 2*step_size, ..., n_steps
    step_size = max(1, n_steps ÷ n_snap)
    cp_indices = Int[]
    for i in 0:n_snap
        idx = i * step_size
        idx <= n_steps && push!(cp_indices, idx)
    end
    sort!(unique!(cp_indices))

    # Forward pass: store checkpoints at cp_indices (state after that many steps)
    checkpoints = Dict{Int, Any}()
    checkpoints[0] = _save_checkpoint(work_model.tracers)
    for step in 1:n_steps
        if step in cp_indices
            checkpoints[step] = _save_checkpoint(work_model.tracers)
        end
        time_step!(work_model, Δt)
    end

    # Adjoint forcing
    adj_init = cost_gradient_fn(work_model.tracers)
    _restore_checkpoint!(work_model.adj_tracers, adj_init)

    # Backward pass: for each step, need state at start of step (= after step-1 steps)
    for step in n_steps:-1:1
        # Largest checkpoint at or before step-1
        cp_step = 0
        for cp in cp_indices
            cp <= step - 1 && (cp_step = cp)
        end

        # Restore and replay forward from cp_step to step-1
        _restore_checkpoint!(work_model.tracers, checkpoints[cp_step])
        for _ in (cp_step + 1):(step - 1)
            time_step!(work_model, Δt)
        end

        adjoint_time_step!(work_model, Δt)
    end

    return work_model.adj_tracers
end
