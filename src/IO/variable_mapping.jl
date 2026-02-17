# ---------------------------------------------------------------------------
# Canonical variable names and unit conversion
#
# Physics code uses canonical names (:u_wind, :temperature, etc.).
# Each met data type has its own native names and units.
#
# Canonical variables are now defined in config/canonical_variables.toml.
# This module loads them at runtime and provides accessor functions.
# ---------------------------------------------------------------------------

using TOML

"""
    canonical_variables()

Return the list of canonical meteorological variable names used by the model,
loaded from `config/canonical_variables.toml`.
"""
function canonical_variables()
    config = load_canonical_config(joinpath(default_config_dir(), "canonical_variables.toml"))
    return Tuple(keys(config))
end

"""
    canonical_units()

Return the expected SI units for each canonical variable,
loaded from `config/canonical_variables.toml`.
"""
function canonical_units()
    config = load_canonical_config(joinpath(default_config_dir(), "canonical_variables.toml"))
    return Dict(name => get(info, "units", "") for (name, info) in config)
end

"""
    canonical_required()

Return the set of required canonical variable names.
"""
function canonical_required()
    config = load_canonical_config(joinpath(default_config_dir(), "canonical_variables.toml"))
    return Set(name for (name, info) in config if get(info, "required", false))
end

"""
    canonical_dimensions()

Return a dict mapping canonical variable names to their dimensionality (2 or 3).
"""
function canonical_dimensions()
    config = load_canonical_config(joinpath(default_config_dir(), "canonical_variables.toml"))
    return Dict(name => get(info, "dimensions", 3) for (name, info) in config)
end

export canonical_variables, canonical_units, canonical_required, canonical_dimensions
