# ---------------------------------------------------------------------------
# TOML-based run configuration
# ---------------------------------------------------------------------------

using TOML

"""
    load_configuration(path::String)

Load a TOML configuration file and return a Dict.

Standard sections:
- `[grid]`      — grid type, resolution, vertical levels
- `[met_data]`  — met data source, path, variables
- `[tracers]`   — tracer names and initial conditions
- `[physics]`   — advection, convection, diffusion scheme choices
- `[time]`      — start/end time, time step
- `[output]`    — output fields, schedule, filename
- `[adjoint]`   — checkpointing, cost function (optional)
"""
function load_configuration(path::String)
    return TOML.parsefile(path)
end

export load_configuration
