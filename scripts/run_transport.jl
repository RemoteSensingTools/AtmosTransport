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
    base = filter(a -> !startswith(String(a), "--threads") &&
                       !startswith(String(a), "--project"), julia.exec)
    # `Base.julia_cmd()` does NOT carry `--project`, so the re-exec would land in
    # the default environment and fail to find the project deps (e.g.
    # KernelAbstractions). Propagate the active project explicitly.
    proj = Base.active_project()
    proj_flag = proj === nothing ? String[] : ["--project=$(dirname(proj))"]
    new_cmd = Cmd([base; proj_flag; "--threads=2"; PROGRAM_FILE; ARGS...])
    run(new_cmd; wait = true)
    exit(0)
end

# Preload the GPU backend BEFORE `AtmosTransport` is imported so the
# CLI stack compiles in a single world age. The library entry point also has a
# one-time world-age trampoline for callers that load a backend on demand, but
# resolving it here keeps CLI startup and optional profiling straightforward.
# Inspecting the config costs about 1 ms.
if !isempty(ARGS)
    _cfg_path = expanduser(ARGS[1])
    if isfile(_cfg_path)
        _cfg = try TOML.parsefile(_cfg_path) catch; nothing end
        if _cfg !== nothing
            _arch = get(_cfg, "architecture", Dict{String, Any}())
            _use_gpu_raw = get(_arch, "use_gpu", false)
            _use_gpu_raw isa Bool || throw(ArgumentError(
                "[architecture].use_gpu must be true or false; got $(repr(_use_gpu_raw))"))
            _use_gpu = _use_gpu_raw
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

using AtmosTransport

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

function _parse_cli(args::Vector{String})
    if args == ["-h"] || args == ["--help"]
        return nothing
    end
    isempty(args) && throw(ArgumentError("missing required <config.toml>"))
    length(args) == 1 || throw(ArgumentError(
        "expected exactly one <config.toml>; got $(length(args)) arguments"))
    startswith(args[1], "-") && throw(ArgumentError("unknown option: $(args[1])"))
    cfg_path = expanduser(args[1])
    isfile(cfg_path) || throw(ArgumentError("Config not found: $cfg_path"))
    return cfg_path
end

function main()
    global_logger(ConsoleLogger(stderr, Logging.Info; show_limited = false))
    cfg_path = _parse_cli(ARGS)
    if cfg_path === nothing
        _print_help(stdout)
        return nothing
    end
    cfg = TOML.parsefile(cfg_path)
    return _run_with_optional_profiling(cfg)
end

# GPU profiling brackets, gated on ATMOSTR_PROFILE_MODE.
#   "full"   — enable CUDA profiling for the entire run.
#   "window" — spawn an async timer that calls CUDA.Profile.start() after
#              ATMOSTR_PROFILE_WARMUP_SEC and CUDA.Profile.stop() +
#              process exit after ATMOSTR_PROFILE_DUR_SEC. Used with
#              `nsys profile -c cudaProfilerApi --capture-range-end=stop`.
#   ""       — unchanged.
const _CUDA_PKGID = Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

_loaded_cuda_module() = get(Base.loaded_modules, _CUDA_PKGID, nothing)

function _profile_full_run(CUDA, cfg)
    run = () -> run_driven_simulation(cfg)
    if CUDA.Profile.detect_cupti()
        @info "CUDA session is already externally profiled; using external collection"
        return CUDA.Profile.profile_externally(run)
    end
    return CUDA.Profile.profile_internally(run)
end

function _run_with_optional_profiling(cfg)
    mode = lowercase(get(ENV, "ATMOSTR_PROFILE_MODE", ""))
    CUDA = _loaded_cuda_module()
    if mode == "" || CUDA === nothing
        return run_driven_simulation(cfg)
    end
    if mode == "full"
        @info "ATMOSTR_PROFILE_MODE=full → CUDA activity profile of full run"
        return _profile_full_run(CUDA, cfg)
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
