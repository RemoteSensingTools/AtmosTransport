# ---------------------------------------------------------------------------
# chemistry_block! — block-level composer for a tuple of chemistry operators
#
# This is the "chemistry block" called by `step!` (plan 15, OPERATOR_COMPOSITION
# §3.1). It differs from `CompositeChemistry` in kind, not in behaviour:
#
#   * `CompositeChemistry(op1, op2)` is a SINGLE operator that happens to wrap
#     others; `apply!(state, ..., ::CompositeChemistry, dt)` iterates them.
#   * `chemistry_block!(state, meteo, grid, (op1, op2), dt)` is the step-level
#     scaffolding: takes an EXTERNAL tuple of operators and applies them in
#     order. The tuple shape matches `TransportModel.chemistry` in plan 15
#     Commit 3, which carries the operators chosen at model-construction
#     time.
#
# Both are sequential composition; the choice is whether the composition is
# part of the operator value or part of the block.
# ---------------------------------------------------------------------------

"""
    chemistry_block!(state::CellState, meteo, grid, operators, dt;
                     workspace=nothing)

Apply each operator in the `operators` tuple to `state`, in order.
Returns `state` for chaining.

The `operators` collection may be:
- a `Tuple` of `AbstractChemistryOperator`s (zero or more),
- a single `AbstractChemistryOperator` (wrapped into a 1-tuple internally),
- an empty `Tuple()` (identity).

This is the step-level chemistry entry point; `apply!(::CompositeChemistry)`
is the operator-level composer. Both exist and both are supported.
"""
function chemistry_block! end

# Tuple of operators: iterate
function chemistry_block!(state, meteo, grid,
                          operators::Tuple, dt; workspace = nothing)
    for op in operators
        apply!(state, meteo, grid, op, dt; workspace = workspace)
    end
    return state
end

# Single operator: wrap into 1-tuple
function chemistry_block!(state, meteo, grid,
                          op::AbstractChemistryOperator, dt; workspace = nothing)
    apply!(state, meteo, grid, op, dt; workspace = workspace)
    return state
end
