#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# scripts/run_transport.jl — unified driven transport runner (plan 40 Commit 6c).
#
# ONE CLI. Reads a TOML config, opens the first transport binary,
# dispatches on its `grid_type` header field (`:latlon`,
# `:reduced_gaussian`, `:cubed_sphere`), and runs the loop. All
# topology-specific logic (IC pipeline, surface flux, snapshot output,
# GPU residency check, capability validation) lives in
# `src/Models/DrivenRunner.jl` and dispatches on mesh type via
# multiple dispatch.
#
# The TOML `[input]` block accepts either shape:
#   [input]
#   binary_paths = [ "a.bin", "b.bin" ]
# OR
#   [input]
#   folder       = "~/data/.../"
#   start_date   = "YYYY-MM-DD"
#   end_date     = "YYYY-MM-DD"
#   file_pattern = "<prefix>{YYYYMMDD}<suffix>"   # optional
#
# Usage:
#   julia --project=. scripts/run_transport.jl <config.toml>
# ---------------------------------------------------------------------------

using Logging
using TOML

# Default to 2 Julia threads. Per the GPU-profiling work on 2026-05-28, going
# from --threads=1 → --threads=2 cut the 3-day C180 full-physics wall by 34 %
# (130 s → 86 s): NetCDF/PCIe write overlaps GPU compute on the background
# thread, and host-side launch dispatch pipelines across threads. Users who
# want a different count can set JULIA_NUM_THREADS or pass --threads N
# explicitly; setting ATMOSTR_NO_AUTO_THREADS=1 disables the re-exec.
if Threads.nthreads() == 1 &&
   get(ENV, "JULIA_NUM_THREADS", "") == "" &&
   get(ENV, "ATMOSTR_NO_AUTO_THREADS", "") != "1"
    @info "Re-executing with --threads=2 (set ATMOSTR_NO_AUTO_THREADS=1 to disable)"
    julia = Base.julia_cmd()
    base = filter(a -> !startswith(String(a), "--threads"), julia.exec)
    new_cmd = Cmd([base; "--threads=2"; PROGRAM_FILE; ARGS...])
    run(new_cmd; wait = true)
    exit(0)
end

# Preload the GPU backend BEFORE `AtmosTransport` gets included so the
# whole stack compiles in a single world age. Doing the load dynamically
# later (from `_ensure_gpu_runtime!`) means every CuArray method
# (`size`, `getindex`, `Adapt.adapt_storage(CuArray, …)`) arrives in a
# newer world than the function bodies that call it, and Julia refuses
# to dispatch — `method too new to be called from this world context`.
# Inspecting the config here is ~1 ms and avoids the whole problem.
if !isempty(ARGS)
    _cfg_path = expanduser(ARGS[1])
    if isfile(_cfg_path)
        _cfg = try TOML.parsefile(_cfg_path) catch; nothing end
        if _cfg !== nothing
            _arch = get(_cfg, "architecture", Dict{String, Any}())
            _use_gpu = Bool(get(_arch, "use_gpu", false))
            _backend = lowercase(String(get(_arch, "backend",
                                            _use_gpu ? "auto" : "cpu")))
            _backend = replace(_backend, '-' => '_', ' ' => '_')
            _gpu_requested = _use_gpu ||
                             _backend in ("auto", "gpu", "cuda", "nvidia",
                                          "metal", "apple", "apple_metal")
            if _gpu_requested
                if _backend in ("metal", "apple", "apple_metal") ||
                   (_backend in ("auto", "gpu") && Sys.isapple())
                    @info "Preloading Metal (GPU backend)"
                    using Metal
                elseif _backend in ("cuda", "nvidia") ||
                       _backend in ("auto", "gpu")
                    @info "Preloading CUDA (GPU backend)"
                    using CUDA
                end
            end
        end
    end
end

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport

function _quickstart_configs()
    dir = joinpath(@__DIR__, "..", "config", "runs", "quickstart")
    isdir(dir) || return String[]
    return sort([joinpath("config", "runs", "quickstart", f)
                 for f in readdir(dir) if endswith(f, ".toml")])
end

function _print_help(io::IO = stdout)
    println(io, "Usage:")
    println(io, "  julia --project=. scripts/run_transport.jl <config.toml>")
    println(io)
    println(io, "Canonical runtime entry point for AtmosTransport run TOMLs.")
    println(io, "Configs may use ~/..., \$ATMOSTRANSPORT_DATA_ROOT/..., or")
    println(io, "\$ATMOSTRANSPORT_DATA_ROOT_quickstart/... paths.")
    println(io)
    println(io, "Quickstart configs:")
    for cfg in _quickstart_configs()
        println(io, "  ", cfg)
    end
    println(io)
    println(io, "Example:")
    println(io, "  bash scripts/download_quickstart_data.sh ll")
    println(io, "  julia --project=. scripts/run_transport.jl config/runs/quickstart/ll72x37_advonly.toml")
    return nothing
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    if any(arg -> arg in ("-h", "--help"), ARGS)
        _print_help(stdout)
        return nothing
    end
    if isempty(ARGS)
        _print_help(stderr)
        error("missing required <config.toml>")
    end
    cfg_path = expanduser(ARGS[1])
    isfile(cfg_path) || error("Config not found: $cfg_path")
    cfg = TOML.parsefile(cfg_path)
    return _run_with_optional_profiling(cfg)
end

# GPU profiling brackets, gated on ATMOSTR_PROFILE_MODE.
#   "full"   — wrap the entire run in CUDA.@profile; CUPTI activity summary
#              printed at end. Honors nsys if launched under it.
#   "window" — spawn an async timer that calls CUDA.Profile.start() after
#              ATMOSTR_PROFILE_WARMUP_SEC and CUDA.Profile.stop() +
#              process exit after ATMOSTR_PROFILE_DUR_SEC. Used with
#              `nsys profile -c cudaProfilerApi --capture-range-end=stop`.
#   ""       — unchanged.
function _run_with_optional_profiling(cfg)
    mode = lowercase(get(ENV, "ATMOSTR_PROFILE_MODE", ""))
    if mode == "" || !isdefined(Main, :CUDA)
        return run_driven_simulation(cfg)
    end
    CUDA = getfield(Main, :CUDA)
    if mode == "full"
        @info "ATMOSTR_PROFILE_MODE=full → CUDA.@profile wrap of full run"
        return CUDA.@profile run_driven_simulation(cfg)
    elseif mode == "window"
        warmup = parse(Float64, get(ENV, "ATMOSTR_PROFILE_WARMUP_SEC", "120"))
        dur    = parse(Float64, get(ENV, "ATMOSTR_PROFILE_DUR_SEC", "60"))
        @info "ATMOSTR_PROFILE_MODE=window → start after $(warmup)s, stop after +$(dur)s, then exit"
        Threads.@spawn begin
            sleep(warmup)
            @info "Profile window: CUDA.Profile.start()"
            CUDA.Profile.start()
            sleep(dur)
            @info "Profile window: CUDA.Profile.stop() and process exit"
            CUDA.Profile.stop()
            exit(0)
        end
        return run_driven_simulation(cfg)
    else
        @warn "Unknown ATMOSTR_PROFILE_MODE=$(mode); running without profile bracket"
        return run_driven_simulation(cfg)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
