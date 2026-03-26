# ---------------------------------------------------------------------------
# Mass conservation diagnostics and tracking
#
# Extracted from run_implementations.jl for modularity.
# Provides dispatched mass computation for both LatLon and CubedSphere grids.
# ---------------------------------------------------------------------------

# =====================================================================
# Mass tracking types
# =====================================================================

"""
Mutable state for tracking mass conservation across the simulation.
Updated every window; used for progress bar display and final summary.
"""
mutable struct MassDiagnostics
    initial_mass  :: Dict{Symbol, Float64}
    current_mass  :: Dict{Symbol, Float64}
    pre_adv_mass  :: Dict{Symbol, Float64}
    expected_mass :: Dict{Symbol, Float64}   # initial + cumulative emissions (no transport error)
    showvalue     :: String
    fix_value     :: String
    cfl_value     :: String
    # Mass fixer tracking: cumulative scaling factor (product of per-window corrections)
    fix_interval_scale :: Dict{Symbol, Float64}   # product of scale factors since last output write (reset at each output)
    fix_total_scale    :: Dict{Symbol, Float64}   # product of all scale factors from sim start
    fix_scale_series   :: Dict{Symbol, Vector{Float64}}  # per-output-timestep interval scale factors
end

MassDiagnostics() = MassDiagnostics(
    Dict{Symbol, Float64}(),
    Dict{Symbol, Float64}(),
    Dict{Symbol, Float64}(),
    Dict{Symbol, Float64}(),
    "", "", "",
    Dict{Symbol, Float64}(),
    Dict{Symbol, Float64}(),
    Dict{Symbol, Vector{Float64}}()
)

# =====================================================================
# Dispatched mass computation
# =====================================================================

"""
    compute_mass_totals(tracers, grid::CubedSphereGrid) → Dict{Symbol, Float64}

Sum total tracer mass (kg) across all 6 panels, skipping halos.
"""
function compute_mass_totals(cs_tracers, grid::CubedSphereGrid)
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    result = Dict{Symbol, Float64}()
    for (tname, rm_t) in pairs(cs_tracers)
        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end
        result[tname] = total
    end
    return result
end

"""
    compute_mass_totals(tracers, grid::LatitudeLongitudeGrid) → Dict{Symbol, Float64}

Sum total tracer mass (kg) for lat-lon tracers (simple 3D arrays).
"""
function compute_mass_totals(tracers, grid::LatitudeLongitudeGrid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    result = Dict{Symbol, Float64}()
    for (tname, rm) in pairs(tracers)
        rm_cpu = Array(rm)
        total = 0.0
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            total += Float64(rm_cpu[i, j, k])
        end
        result[tname] = total
    end
    return result
end

"""
    compute_mass_totals_subset(cs_tracers, grid, subset) → Dict{Symbol, Float64}

Like `compute_mass_totals` but only for tracers in `subset` with nonzero values.
Avoids GPU→CPU copies for tracers that don't need mass fixing.
"""
function compute_mass_totals_subset(cs_tracers, grid::CubedSphereGrid,
                                     subset::Dict{Symbol, Float64})
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    result = Dict{Symbol, Float64}()
    for (tname, rm_t) in pairs(cs_tracers)
        haskey(subset, tname) || continue
        subset[tname] == 0.0 && continue
        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end
        result[tname] = total
    end
    return result
end

"""
    mass_showvalue(mass_totals, expected_mass) → String

Format mass closure diagnostics: Δ = (modeled - expected) / expected.
`expected_mass` = initial + cumulative emissions, so Δ shows only transport bias.
"""
function mass_showvalue(mass_totals::Dict{Symbol, Float64},
                        expected_mass::Dict{Symbol, Float64})
    parts = String[]
    for tname in sort(collect(keys(mass_totals)))
        total = mass_totals[tname]
        if haskey(expected_mass, tname) && abs(expected_mass[tname]) > 1e-30
            rel = (total - expected_mass[tname]) / abs(expected_mass[tname]) * 100
            push!(parts, @sprintf("%s:Δ=%.4e%%", tname, rel))
        end
    end
    return join(parts, "  ")
end

"""
    apply_global_mass_fixer!(cs_tracers, grid, target_mass) → String

Scale rm globally so total tracer mass matches `target_mass` for each tracer.
Returns a compact string describing the correction magnitude (in ppm).
"""
function apply_global_mass_fixer!(cs_tracers, grid::CubedSphereGrid,
                                   target_mass::Dict{Symbol, Float64};
                                   fix_interval_scale::Union{Nothing, Dict{Symbol, Float64}}=nothing,
                                   fix_total_scale::Union{Nothing, Dict{Symbol, Float64}}=nothing,
                                   allowed_tracers::Vector{String}=String[])
    Nc, Hp, Nz = grid.Nc, grid.Hp, grid.Nz
    parts = String[]
    for (tname, rm_t) in pairs(cs_tracers)
        haskey(target_mass, tname) || continue
        # Skip tracers not in the allowed list (if list is non-empty)
        if !isempty(allowed_tracers) && !(String(tname) in allowed_tracers)
            continue
        end
        m0 = target_mass[tname]
        m0 == 0.0 && continue

        total = 0.0
        for p in 1:6
            rm_cpu = Array(rm_t[p])
            @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
                total += Float64(rm_cpu[Hp+i, Hp+j, k])
            end
        end

        total == 0.0 && continue
        scale = m0 / total
        correction_ppm = (scale - 1.0) * 1e6

        FT = eltype(rm_t[1])
        for p in 1:6
            rm_t[p] .*= FT(scale)
        end
        push!(parts, @sprintf("%s:%.1fppm", tname, correction_ppm))

        # Accumulate scaling factors (multiplicative)
        if fix_interval_scale !== nothing
            fix_interval_scale[tname] = get(fix_interval_scale, tname, 1.0) * scale
        end
        if fix_total_scale !== nothing
            fix_total_scale[tname] = get(fix_total_scale, tname, 1.0) * scale
        end
    end
    return join(parts, " ")
end

"""Snapshot the interval mass fixer scale into the time series and reset for next interval."""
function snapshot_massfixer_interval!(diag::MassDiagnostics)
    for (tname, scale) in diag.fix_interval_scale
        series = get!(diag.fix_scale_series, tname, Float64[])
        push!(series, scale)
    end
    # Reset interval accumulator to 1.0 (multiplicative identity)
    for k in keys(diag.fix_interval_scale)
        diag.fix_interval_scale[k] = 1.0
    end
end

"""
    mass_total_f64(rm_panels, Nc, Hp, Nz) → Float64

Sum total tracer mass across all 6 panels in Float64 precision.
Used for per-stage mass balance diagnostics.
"""
function mass_total_f64(rm_panels, Nc, Hp, Nz)
    total = 0.0
    for p in 1:6
        rm_cpu = Array(rm_panels[p])
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            total += Float64(rm_cpu[Hp+i, Hp+j, k])
        end
    end
    return total
end

# Global storage for mass balance diagnostics (populated during first N windows)
const MASS_DIAG = Dict{String, Vector{Float64}}()
const MASS_DIAG_WINDOWS = Ref(3)

"""Legacy wrapper: print mass conservation via @info."""
function log_mass_conservation(cs_tracers, grid::CubedSphereGrid, window::Int,
                                label::String;
                                initial_mass::Union{Nothing, Dict{Symbol, Float64}}=nothing,
                                expected_mass::Union{Nothing, Dict{Symbol, Float64}}=nothing)
    totals = compute_mass_totals(cs_tracers, grid)
    ref = something(expected_mass, initial_mass, nothing)
    for tname in sort(collect(keys(totals)))
        total = totals[tname]
        if ref !== nothing && haskey(ref, tname)
            m0 = ref[tname]
            rel = m0 == 0.0 ? 0.0 : (total - m0) / abs(m0) * 100
            @info @sprintf("  MASS %s win=%d [%s]: %.6e kg (Δ=%.4e%%)",
                           tname, window, label, total, rel)
        else
            @info @sprintf("  MASS %s win=%d [%s]: %.6e kg",
                           tname, window, label, total)
        end
    end
end

# =====================================================================
# Convenience: update MassDiagnostics in-place
# =====================================================================

"""Snapshot current mass into `diag.pre_adv_mass` (called after emissions, before advection).
Also updates `expected_mass` by adding emissions: expected += (pre_adv - current_before_emissions).
"""
function snapshot_pre_advection!(diag::MassDiagnostics, tracers, grid)
    diag.pre_adv_mass = compute_mass_totals(tracers, grid)
    # Accumulate emissions into expected mass:
    # emissions_this_window = pre_adv_mass - current_mass (current_mass still has prev window's value)
    if !isempty(diag.expected_mass)
        for (tname, m_pre) in diag.pre_adv_mass
            emissions = m_pre - get(diag.current_mass, tname, 0.0)
            diag.expected_mass[tname] = get(diag.expected_mass, tname, 0.0) + emissions
        end
    end
end

"""Update `diag.current_mass` and `diag.showvalue` (called after all physics)."""
function update_mass_diagnostics!(diag::MassDiagnostics, tracers, grid)
    diag.current_mass = compute_mass_totals(tracers, grid)
    if !isempty(diag.expected_mass)
        diag.showvalue = mass_showvalue(diag.current_mass, diag.expected_mass)
    end
end

"""Record initial mass (called once at IC finalization)."""
function record_initial_mass!(diag::MassDiagnostics, tracers, grid)
    diag.initial_mass = compute_mass_totals(tracers, grid)
    diag.current_mass = copy(diag.initial_mass)
    diag.expected_mass = copy(diag.initial_mass)
end

