#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Generate the full 5-day Catrine campaign config set:
#   - 6 LL spectral preprocessing configs (3 grids × F32+F64), TM5 enabled
#   - 36 run configs (4 grids × F32+F64 × 3 op-stacks × {CPU, GPU})
#     Coarse grids (LL 72×37, CS C48): both CPU + GPU.
#     Fine   grids (LL 144×73, CS C180): GPU only.
#
# Outputs land under config/preprocessing/catrine5d/ and
# config/runs/catrine5d/ to keep the campaign artifacts separable
# from the historical 3-day configs.
#
# Usage:  julia --project=. scripts/preprocessing/generate_campaign5d_configs.jl
# ---------------------------------------------------------------------------

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", ".."))
const PREP_DIR   = joinpath(REPO_ROOT, "config", "preprocessing", "catrine5d")
const RUNS_DIR   = joinpath(REPO_ROOT, "config", "runs", "catrine5d")
const PHYSICS_BIN_DIR = "~/data/AtmosTransport/met/era5/0.5x0.5/physics_bin"
const GRIDFED_FILE    = "~/data/AtmosTransport/catrine/Emissions/gridfed/GCP-GridFEDv2024.0_2021.short.nc"
const START_DATE      = "2021-12-01"
const END_DATE        = "2021-12-05"
# Snapshot every 6 h over 5 days -> 21 snapshots (0..120).
const SNAPSHOT_HOURS  = collect(0:6:120)

mkpath(PREP_DIR)
mkpath(RUNS_DIR)

# ----------- LL spectral preprocessing configs --------------------------------

# Two LL preprocessing flavors:
#   * `ll720_dec2021_<ft>_tm5.toml`  — TM5 convection enabled.  Native
#     720×361 only; output is the source for the CS regrid.
#   * `<grid>_dec2021_<ft>.toml`     — LL 72/144 with TM5 disabled
#     (no `[tm5_convection]` block).  Used directly by the LL runs;
#     advdiffconv configs aren't emitted for LL targets.
const LL_GRIDS = [
    (name = "ll72",  Nx = 72,  Ny = 37,  out = "ll72x37_advresln",  with_tm5 = false),
    (name = "ll144", Nx = 144, Ny = 73,  out = "ll144x73_advresln", with_tm5 = false),
    (name = "ll720", Nx = 720, Ny = 361, out = "ll720x361_v4",      with_tm5 = true),
]

function ll_prep_toml(g, ft_str::String)
    ft_tag = ft_str == "Float32" ? "f32" : "f64"
    out_suffix = g.with_tm5 ? "$(ft_tag)_tm5" : "$ft_tag"
    cfg_suffix = g.with_tm5 ? "$(ft_tag)_tm5" : "$ft_tag"
    tm5_block = g.with_tm5 ? """

[tm5_convection]
enable          = true
physics_bin_dir = "$(PHYSICS_BIN_DIR)"
""" : ""
    return """
    # 5-day Catrine campaign — $(g.name) ERA5 spectral → daily transport binary
    # ($(ft_str)$(g.with_tm5 ? ", TM5 convection enabled" : ", advection-only")).
    #
    # Usage:
    #   for d in 01 02 03 04 05; do
    #     julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl \\
    #         config/preprocessing/catrine5d/$(g.name)_dec2021_$(cfg_suffix).toml \\
    #         --day 2021-12-\$d
    #   done

    [input]
    spectral_dir = "~/data/AtmosTransport/met/era5/0.5x0.5/spectral_hourly"
    thermo_dir   = "~/data/AtmosTransport/met/era5/0.5x0.5/physics"
    coefficients = "config/era5_L137_coefficients.toml"

    [output]
    directory  = "~/data/AtmosTransport/met/era5/$(g.out)/transport_binary_v2_tropo34_dec2021_$(out_suffix)"
    mass_basis = "dry"
    include_qv = false

    [grid]
    type                   = "latlon"
    nlon                   = $(g.Nx)
    nlat                   = $(g.Ny)
    level_top              = 1
    level_bot              = 137
    echlevs                = "ml137_tropo34"
    merge_min_thickness_Pa = 1000.0

    [numerics]
    float_type   = "$ft_str"
    dt           = 900.0
    met_interval = 3600.0

    [mass_fix]
    enable                = true
    target_ps_dry_pa      = 98726.0
    qv_global_climatology = 0.00247
$(tm5_block)"""
end

# Wipe stale LL preprocessing configs from the earlier path-2 generation so
# the directory matches the path-1 scope.
for stale in readdir(PREP_DIR; join = true)
    occursin("ll", basename(stale)) && rm(stale)
end

for g in LL_GRIDS, ft_str in ("Float32", "Float64")
    ft_tag = ft_str == "Float32" ? "f32" : "f64"
    cfg_suffix = g.with_tm5 ? "$(ft_tag)_tm5" : "$ft_tag"
    path = joinpath(PREP_DIR, "$(g.name)_dec2021_$(cfg_suffix).toml")
    write(path, ll_prep_toml(g, ft_str))
    println("wrote $path")
end

# ----------- Run configs ------------------------------------------------------

# LL 72/144 binaries are built WITHOUT TM5 sections — the spectral
# preprocessor's TM5 path is hard-coded to native physics-BIN shape
# (720×361) and porting LL→LL regrid would require a new
# `reconstruct_ll_fluxes!` helper. CS C48/C180 binaries DO carry TM5
# (regridded LL 720 → CS via the existing path).  Consequence:
# advdiffconv configs are emitted only for CS targets.
function binary_folder(grid::Symbol, ft_tag::String)
    if grid === :ll72
        "~/data/AtmosTransport/met/era5/ll72x37_advresln/transport_binary_v2_tropo34_dec2021_$ft_tag"
    elseif grid === :ll144
        "~/data/AtmosTransport/met/era5/ll144x73_advresln/transport_binary_v2_tropo34_dec2021_$ft_tag"
    elseif grid === :c48
        "~/data/AtmosTransport/met/era5/cs_c48/transport_binary_v2_tropo34_dec2021_$(ft_tag)_tm5"
    elseif grid === :c180
        "~/data/AtmosTransport/met/era5/cs_c180/transport_binary_v2_tropo34_dec2021_$(ft_tag)_tm5"
    else
        error("unknown grid $grid")
    end
end

function grid_block(grid::Symbol)
    if grid === :ll72
        # The runner detects LL via [input.folder] header sniff; no grid block needed.
        ""
    elseif grid === :ll144
        ""
    elseif grid === :c48
        ""
    elseif grid === :c180
        ""
    end
end

function operator_blocks(op::Symbol)
    blocks = String[]
    if op === :advdiff || op === :advdiffconv
        push!(blocks, """
        [diffusion]
        kind  = "constant"
        value = 1.0
        """)
    end
    if op === :advdiffconv
        push!(blocks, """
        [convection]
        kind = "tm5"
        """)
    end
    return join(blocks, "\n")
end

function tracers_block()
    return """
    [tracers.co2_natural]
    [tracers.co2_natural.init]
    kind = "catrine_co2"

    [tracers.co2_fossil]
    [tracers.co2_fossil.init]
    kind = "uniform"
    background = 0.0

    # GridFED monthly-mean fossil-CO2 emissions in kg/m²/s. December 2021 → time_index = 12.
    [tracers.co2_fossil.surface_flux]
    kind       = "gridfed_fossil_co2"
    file       = "$(GRIDFED_FILE)"
    time_index = 12
    """
end

function run_toml(grid::Symbol, ft_str::String, op::Symbol, hw::Symbol)
    ft_tag = ft_str == "Float32" ? "f32" : "f64"
    use_gpu = hw === :gpu
    folder = binary_folder(grid, ft_tag)
    out_root = "~/data/AtmosTransport/output/catrine5d"
    snapshot_basename = "$(grid)_$(ft_tag)_$(op)_$(hw).nc"
    snapshot_path = joinpath(out_root, String(grid), snapshot_basename)
    return """
    # 5-day Catrine campaign — $(grid) $(ft_str) $(op) $(uppercase(string(hw)))
    #
    # Runner:
    #   julia -t8 --project=. scripts/run_transport.jl \\
    #         config/runs/catrine5d/$(grid)_$(ft_tag)_$(op)_$(hw).toml

    [input]
    folder     = "$folder"
    start_date = "$START_DATE"
    end_date   = "$END_DATE"

    [architecture]
    use_gpu = $use_gpu

    [numerics]
    float_type = "$ft_str"

    [run]
    scheme = "slopes"

    $(tracers_block())
    $(operator_blocks(op))
    [output]
    snapshot_hours = $(SNAPSHOT_HOURS)
    snapshot_file = "$snapshot_path"
    """
end

# Coarse: CPU + GPU. Fine: GPU only.
const COARSE = (:ll72, :c48)
const FINE   = (:ll144, :c180)
const PRECS  = ("Float32", "Float64")
const OPS    = (:advonly, :advdiff, :advdiffconv)

count_runs = 0
const LL_GRIDS_RUNTIME = (:ll72, :ll144)  # no TM5 carry-through → no advdiffconv
for grid in (COARSE..., FINE...), ft_str in PRECS, op in OPS
    # Skip advdiffconv on LL grids — their binaries don't carry TM5
    # sections (path-1 scope decision; would need an LL→LL regrid path).
    op === :advdiffconv && grid in LL_GRIDS_RUNTIME && continue
    hws = grid in COARSE ? (:cpu, :gpu) : (:gpu,)
    for hw in hws
        ft_tag = ft_str == "Float32" ? "f32" : "f64"
        path = joinpath(RUNS_DIR, "$(grid)_$(ft_tag)_$(op)_$(hw).toml")
        write(path, run_toml(grid, ft_str, op, hw))
        global count_runs += 1
    end
end

println("\ngenerated $(length(LL_GRIDS) * 2) preprocessing configs in $PREP_DIR")
println("generated $count_runs run configs in $RUNS_DIR")
