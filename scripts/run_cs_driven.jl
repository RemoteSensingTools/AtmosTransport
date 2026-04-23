#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Cubed-sphere transport runner — DrivenSimulation path with physics.
#
# Unlike `run_cs_transport.jl` (advection-only, panel-native low level), this
# runner uses `CubedSphereTransportDriver` + `CubedSphereState` +
# `TransportModel` + `DrivenSimulation`, exposing operator composition via
# TOML:
#
#   [advection]  ← scheme  (upwind | slopes | ppm + ppm_order)
#   [diffusion]  ← kind = none|constant  (extendable to profile/precomputed)
#   [convection] ← kind = none|tm5|cmfmc
#
# Usage:
#   julia --project=. scripts/run_cs_driven.jl <config.toml>
# ---------------------------------------------------------------------------

using Printf
using TOML
using NCDatasets
using Adapt

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
# All operator types, field types, state types, drivers, and simulation
# entry points below are re-exported by `AtmosTransport` at top level.
# Explicit module-qualified imports would be redundant; `using .AtmosTransport`
# brings them all in, verified via the src/AtmosTransport.jl export list.

# ---------------------------------------------------------------------------
# GPU helpers (parallel to run_cs_transport.jl)
# ---------------------------------------------------------------------------

_use_gpu(cfg) = Bool(get(get(cfg, "architecture", Dict{String,Any}()), "use_gpu", false))

function ensure_gpu!(cfg)
    _use_gpu(cfg) || return false
    isdefined(Main, :CUDA) || Core.eval(Main, :(using CUDA))
    CUDA = Core.eval(Main, :CUDA)
    Base.invokelatest(getproperty(CUDA, :functional)) ||
        error("CUDA runtime not functional")
    Base.invokelatest(getproperty(CUDA, :allowscalar), false)
    return true
end

cfg_float_type(cfg) = let s = get(get(cfg, "numerics", Dict()), "float_type", "Float64")
    s == "Float32" ? Float32 : Float64
end

cfg_architecture(cfg) = _use_gpu(cfg) ? (ensure_gpu!(cfg); GPU()) : CPU()

# ---------------------------------------------------------------------------
# Operator construction from TOML
# ---------------------------------------------------------------------------

"""
Build an advection scheme from the `[advection]` (or legacy `[run]`) section.
"""
function build_scheme(cfg)
    adv = get(cfg, "advection", get(cfg, "run", Dict{String,Any}()))
    name = Symbol(lowercase(String(get(adv, "scheme", "upwind"))))
    if name === :slopes
        return SlopesScheme()
    elseif name === :ppm
        # PPMScheme in the current src path uses a MonotoneLimiter by default.
        # The `ppm_order` knob applies only to the Lin-Rood cross-term PPM
        # variant (`strang_split_linrood_ppm!` in run_cs_transport.jl); it is
        # not relevant to DrivenSimulation's standard strang_split path.
        haskey(adv, "ppm_order") && @warn(
            "ppm_order is ignored by run_cs_driven.jl (Lin-Rood only); " *
            "using default monotone-limited PPMScheme")
        return PPMScheme()
    elseif name === :upwind
        return UpwindScheme()
    else
        error("Unknown advection scheme: $name (supported: upwind | slopes | ppm)")
    end
end

"""
Build a diffusion operator from the `[diffusion]` section. Supported kinds:
- `none`       → `NoDiffusion()` (default when section absent)
- `constant`   → `ImplicitVerticalDiffusion` with a uniform `value` [m²/s]
                  Kz wrapped as `CubedSphereField(ntuple(_ -> ConstantField, 6))`.

Profile/precomputed/derived Kz fields are a straightforward extension but
require per-column or per-cell data that doesn't live in a simple scalar
TOML entry; add dispatch here when that schema settles.
"""
function build_diffusion(cfg, ::Type{FT}) where FT
    section = get(cfg, "diffusion", nothing)
    section === nothing && return NoDiffusion()
    kind = Symbol(lowercase(String(get(section, "kind", "none"))))
    if kind === :none
        return NoDiffusion()
    elseif kind === :constant
        value = FT(get(section, "value", 1.0))
        kz = CubedSphereField(ntuple(_ -> ConstantField{FT, 3}(value), 6))
        return ImplicitVerticalDiffusion(; kz_field=kz)
    else
        error("Unknown diffusion kind: $kind (supported: none | constant). " *
              "Profile/precomputed/derived Kz need their own TOML schema.")
    end
end

"""
Build a convection operator from the `[convection]` section. Supported kinds:
- `none`  → `NoConvection()` (default when section absent)
- `tm5`   → `TM5Convection()` — reads entu/detu/entd/detd from the binary
           (binary must be preprocessed with `[tm5_convection] enable=true`)
- `cmfmc` → `CMFMCConvection()` — reads CMFMC + DTRAIN (GEOS-style binaries)

Both operator types are parameterless singletons; the per-window mass-flux
data is supplied by the driver via the transport window payload.
"""
function build_convection(cfg, reader)
    section = get(cfg, "convection", nothing)
    section === nothing && return NoConvection()
    kind = Symbol(lowercase(String(get(section, "kind", "none"))))
    if kind === :none
        return NoConvection()
    elseif kind === :tm5
        # Header flag check would live here once we expose `has_tm5_convection`
        # on the CS reader (plan 24 Commit 4 emitted sections but the reader
        # boolean accessor is grouped with has_cmfmc). For now: allow and let
        # the first window load fail loudly if sections are missing.
        return TM5Convection()
    elseif kind === :cmfmc
        has_cmfmc(reader) ||
            error("[convection] kind = \"cmfmc\" requires a binary with CMFMC section " *
                  "(this one lacks it). Use a GEOS-style preprocessed binary or switch " *
                  "to kind = \"tm5\" / \"none\".")
        return CMFMCConvection()
    else
        error("Unknown convection kind: $kind (supported: none | tm5 | cmfmc)")
    end
end

# ---------------------------------------------------------------------------
# Tracer initial conditions
# ---------------------------------------------------------------------------

const CATRINE_BACKGROUND = 4.11e-4

"""
Build a per-tracer initial-condition panel tuple matching `air_mass` shape.

Supported kinds (minimal): `uniform` (with `background` scalar), `catrine_co2`
(flat 411 ppm placeholder — a file-based loader mirroring the LL path lives
in `run_transport_binary.jl:build_initial_mixing_ratio` but hasn't been
ported to CS panels yet). Extend here when needed.
"""
function build_tracer_panels(init_cfg, air_mass, ::Type{FT}) where FT
    kind = Symbol(lowercase(String(get(init_cfg, "kind", "uniform"))))
    bg = if kind === :catrine_co2
        FT(CATRINE_BACKGROUND)
    elseif kind === :uniform
        FT(get(init_cfg, "background", 0.0))
    else
        error("Unsupported tracer init kind for CS runner: $kind (supported: uniform | catrine_co2)")
    end
    return ntuple(p -> air_mass[p] .* bg, 6)
end

# ---------------------------------------------------------------------------
# NetCDF snapshot export (reuses the run_cs_transport.jl convention)
# ---------------------------------------------------------------------------

"""
Strip halo padding and return the interior `(Nc, Nc, Nz)` view as a CPU Array.
"""
_interior(panel, Hp, Nc) = Array(panel[Hp+1:Hp+Nc, Hp+1:Hp+Nc, :])

function write_snapshots(nc_path, snap_data, snap_m, snapshot_hours, mesh, Nz)
    Nc = mesh.Nc
    ntime = length(snap_data)
    isfile(nc_path) && rm(nc_path)
    mkpath(dirname(nc_path))
    ds = NCDataset(nc_path, "c")
    defDim(ds, "Xdim", Nc); defDim(ds, "Ydim", Nc); defDim(ds, "nf", 6)
    defDim(ds, "lev", Nz); defDim(ds, "time", ntime)
    ds.attrib["Conventions"] = "CF"
    ds.attrib["Source"] = "AtmosTransport.jl run_cs_driven"
    ds.attrib["Nc"] = Nc

    defVar(ds, "time", Float64, ("time",),
           attrib=Dict("units"=>"hours since t0"))[:] = Float64.(snapshot_hours[1:ntime])
    defVar(ds, "nf", Int32, ("nf",))[:] = Int32.(1:6)

    m_v = defVar(ds, "air_mass", Float64, ("Xdim", "Ydim", "nf", "lev", "time"))
    for t in 1:ntime, p in 1:6
        m_v[:, :, p, :, t] = snap_m[t][p]
    end
    for name in keys(snap_data[1])
        v = defVar(ds, String(name), Float64, ("Xdim", "Ydim", "nf", "lev", "time"))
        for t in 1:ntime, p in 1:6
            rm = snap_data[t][name][p]
            # Mixing ratio VMR = rm / m, clamped.
            m  = snap_m[t][p]
            v[:, :, p, :, t] = rm ./ max.(m, eps(Float64))
        end
    end
    close(ds)
    println("Saved snapshots: $nc_path (C$Nc × 6 panels × $Nz levels × $ntime times)")
end

# ---------------------------------------------------------------------------
# Main driven runner
# ---------------------------------------------------------------------------

function run_cs_driven(cfg)
    FT   = cfg_float_type(cfg)
    arch = cfg_architecture(cfg)

    binary_paths = [expanduser(String(p)) for p in cfg["input"]["binary_paths"]]
    isempty(binary_paths) && error("[input].binary_paths is empty")

    run_cfg = get(cfg, "run", Dict{String,Any}())
    Hp = Int(get(run_cfg, "Hp", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)

    output_cfg = get(cfg, "output", Dict{String,Any}())
    snapshot_hours = Float64.(get(output_cfg, "snapshot_hours", Float64[0, 24, 48]))
    snapshot_file  = expanduser(String(get(output_cfg, "snapshot_file", "cs_driven_snapshot.nc")))

    tracers_cfg = get(cfg, "tracers", Dict{String,Any}())
    isempty(tracers_cfg) && error("[tracers] must define at least one tracer")
    tracer_init = Dict(Symbol(n) => get(c, "init", Dict("kind"=>"uniform","background"=>0.0))
                       for (n, c) in tracers_cfg)

    scheme = build_scheme(cfg)

    # --- First driver + model setup (reuses air_mass from window 1) --------
    driver1 = CubedSphereTransportDriver(first(binary_paths); FT=FT, arch=arch, Hp=Hp)
    grid    = driver_grid(driver1)
    mesh    = grid.horizontal
    window1 = load_transport_window(driver1, 1)
    air_mass = window1.air_mass
    Nz = size(air_mass[1], 3)

    tracer_kwargs = Dict{Symbol, NTuple{6, typeof(air_mass[1])}}()
    for (name, init_cfg) in tracer_init
        tracer_kwargs[name] = build_tracer_panels(init_cfg, air_mass, FT)
    end

    state   = CubedSphereState(DryBasis, mesh, air_mass; tracer_kwargs...)
    fluxes  = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)

    diffusion  = build_diffusion(cfg, FT)
    convection = build_convection(cfg, driver1.reader)

    model = TransportModel(state, fluxes, grid, scheme;
                           diffusion  = diffusion,
                           convection = convection)

    println("="^60)
    @printf("CS driven runner  C%d × %d levels  Hp=%d  %s  FT=%s  %s\n",
            mesh.Nc, Nz, Hp, typeof(scheme).name.name, FT,
            _use_gpu(cfg) ? "GPU" : "CPU")
    @printf("Physics: advection=%s  diffusion=%s  convection=%s\n",
            typeof(scheme).name.name, typeof(diffusion).name.name,
            typeof(convection).name.name)
    @printf("Tracers: %s   Binaries: %d → %s\n",
            join(String.(keys(tracer_init)), ", "), length(binary_paths), snapshot_file)
    println("="^60)

    # --- Snapshot storage ---
    snap_data = Dict{Symbol, NTuple{6, Array{FT,3}}}[]
    snap_m    = NTuple{6, Array{FT,3}}[]
    snap_taken = Float64[]

    function capture!(hour_total)
        cur_m = ntuple(p -> _interior(state.air_mass[p], Hp, mesh.Nc), 6)
        cur_tracers = Dict(n => ntuple(p -> _interior(get_tracer(state, n)[p], Hp, mesh.Nc), 6)
                           for n in keys(tracer_init))
        push!(snap_m, cur_m)
        push!(snap_data, cur_tracers)
        push!(snap_taken, hour_total)
    end

    snap_idx = 1
    total_hour = 0.0
    if snap_idx <= length(snapshot_hours) && abs(snapshot_hours[snap_idx]) < 0.5
        capture!(0.0); @printf("  Snapshot %d at t=%.1fh\n", snap_idx, 0.0)
        snap_idx += 1
    end

    t0 = time()
    drivers = [driver1; [CubedSphereTransportDriver(expanduser(p); FT=FT, arch=arch, Hp=Hp)
                         for p in binary_paths[2:end]]]

    for driver in drivers
        stop_window = stop_window_override === nothing ?
                       total_windows(driver) : min(Int(stop_window_override), total_windows(driver))
        window_hours = window_dt(driver) / 3600.0

        # Rebuild sim around each day's driver — state + physics carry over;
        # plan-39 Commit G removed the window-boundary air_mass reset, so the
        # cross-day handoff is continuity-consistent.
        if driver !== driver1
            fluxes_d = allocate_face_fluxes(mesh, Nz; FT=FT, basis=DryBasis)
            model = TransportModel(state, fluxes_d, grid, scheme;
                                    diffusion  = diffusion,
                                    convection = convection)
        end
        sim = DrivenSimulation(model, driver; start_window=1, stop_window=stop_window)

        while sim.iteration < sim.final_iteration
            step!(sim)
            if sim.iteration % sim.steps_per_window == 0
                total_hour += window_hours
                while snap_idx <= length(snapshot_hours) &&
                      abs(total_hour - snapshot_hours[snap_idx]) < 0.5
                    capture!(total_hour)
                    @printf("  Snapshot %d at t=%.1fh\n", snap_idx, total_hour)
                    snap_idx += 1
                end
            end
        end
        close(driver)
    end

    @printf("\nDone: %.1fs  (%d snapshots, final t=%.1fh)\n",
             time() - t0, length(snap_data), total_hour)

    # --- Diagnostics ---
    for name in keys(tracer_init)
        total = total_mass(state, name)
        @printf("  %s total mass: %.6e kg\n", name, total)
    end

    write_snapshots(snapshot_file, snap_data, snap_m, snap_taken, mesh, Nz)
    return nothing
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    isempty(ARGS) && error("Usage: julia --project=. scripts/run_cs_driven.jl <config.toml>")
    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)
    run_cs_driven(cfg)
end

# Guarded so the script can be `include`d without auto-running (tests /
# interactive construction of builders below).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
