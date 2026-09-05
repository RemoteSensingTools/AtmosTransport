# Keep global signed diagnostics separate from visualization precision. Each
# device lane returns both terms of a compensated sum; adding them together on
# the device would discard small residuals before cancellation between lanes.
@kernel function _snapshot_total_pairs!(pairs, values, nlanes, nvalues)
    lane = @index(Global, Linear)
    total = 0.0
    correction = 0.0
    @inbounds for i in lane:nlanes:nvalues
        total, correction = _compensated_add_f64(total, correction, values[i])
    end
    @inbounds pairs[1, lane] = total
    @inbounds pairs[2, lane] = correction
end

function _accumulate_host_total(total, correction, values)
    @inbounds for value in values
        total, correction = _compensated_add_f64(total, correction, value)
    end
    return total, correction
end

function _accumulate_backend_total(total, correction, values::AbstractArray)
    backend = get_backend(values)
    backend isa KA_CPU && return _accumulate_host_total(total, correction, values)
    isempty(values) && return total, correction
    if _snapshot_accumulator_type(backend) === Float64
        nlanes = min(4096, cld(length(values), 256))
        partials = similar(values, Float64, (2, nlanes))
        _snapshot_total_pairs!(backend, 256)(partials, values, nlanes, length(values);
                                             ndrange=nlanes)
        synchronize(backend)
        # Interleaved sum/correction pairs retain residuals across lanes and
        # panels. Storage is bounded independently of the number of levels.
        return _accumulate_host_total(total, correction, Array(partials))
    end
    # Metal has no Float64 arithmetic. Copy bounded slabs and preserve the CPU
    # compensated accumulation order rather than reducing signed totals in F32.
    axis = ndims(values)
    for first in 1:16:size(values, axis)
        last = min(first + 15, size(values, axis))
        slab = Array(selectdim(values, axis, first:last))
        total, correction = _accumulate_host_total(total, correction, slab)
    end
    return total, correction
end

function _accumulate_backend_total(total, correction, panels::NTuple{6})
    for panel in panels
        total, correction = _accumulate_backend_total(total, correction, panel)
    end
    return total, correction
end

function _backend_tracer_total(values)
    total, correction = _accumulate_backend_total(0.0, 0.0, values)
    result = total + correction
    isfinite(result) || throw(ArgumentError(
        "snapshot tracer total is not finite; check the captured tracer state"))
    return result
end
