# ---------------------------------------------------------------------------
# Physical constants from defaults.toml (single source of truth)
# ---------------------------------------------------------------------------
const _defaults = TOML.parsefile(joinpath(PREPROCESS_SPECTRAL_V4_REPO_ROOT, "config", "defaults.toml"))
const R_EARTH = Float64(_defaults["planet"]["radius"])
const GRAV    = Float64(_defaults["planet"]["gravity"])
