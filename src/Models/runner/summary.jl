function _diffusion_label(op::ImplicitVerticalDiffusion)
    coupling = uses_diffusive_surface_flux_boundary(op) ? ", surface_flux=before_full_solve" :
               ", surface_flux=midpoint_split"
    return string(nameof(typeof(op)), coupling)
end

function _schedule_label(driver)
    schedule = steps_per_window_schedule(driver)
    if isempty(schedule)
        return "n/a"
    end
    lo, hi = extrema(schedule)
    if lo == hi
        return string(first(schedule))
    end
    return @sprintf("%d..%d, max=%d", lo, hi, steps_per_window(driver))
end

function _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                  backend, FT, recipe, driver, tracers,
                                  binary_count, snapshot_file)
    scheme = _cyan(_advection_label(recipe.advection))
    return (
        @sprintf("%s", _bold(String(topology))),
        @sprintf("|-- grid:      %s, levels=%d, Hp=%d",
                 mesh_label, levels, halo_width),
        @sprintf("|-- numerics:  scheme=%s, FT=%s, backend=%s",
                 scheme, FT, backend),
        @sprintf("|-- physics:   diffusion=%s, convection=%s",
                 _diffusion_label(recipe.diffusion),
                 nameof(typeof(recipe.convection))),
        @sprintf("|-- schedule:  window_dt=%.0fs, steps/window=%s, binaries=%d",
                 Float64(window_dt(driver)), _schedule_label(driver),
                 binary_count),
        @sprintf("|-- tracers:   %s", join(String.(tracers), ", ")),
        @sprintf("`-- output:    %s", snapshot_file),
    )
end

function _log_runtime_summary(; topology, mesh_label, levels, halo_width,
                                backend, FT, recipe, driver, tracers,
                                binary_count, snapshot_file)
    lines = _physics_summary_lines(; topology, mesh_label, levels, halo_width,
                                   backend, FT, recipe, driver, tracers,
                                   binary_count, snapshot_file)
    @info "Driven runtime\n" * join(lines, "\n")
end

