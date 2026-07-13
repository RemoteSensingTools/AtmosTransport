module AtmosTransportBenchmarks

using Adapt
using ArgParse
using Dates
using JSON3
using KernelAbstractions
using Printf
using Serialization
using Statistics

using AtmosTransport
using AtmosTransport: AtmosGrid, CubedSphereMesh, CubedSphereState,
    DryBasis, HybridSigmaPressure, NoAdvection, UpwindScheme, SlopesScheme,
    PPMScheme, LinRoodPPMScheme, MonotoneLimiter, NoDiffusion,
    ImplicitVerticalDiffusion, ConstantField, NoConvection, TM5Convection,
    ConvectionForcing, TransportModel, step!, fill_panel_halos!
using AtmosTransport.Architectures: CPUBackend, CUDAGPUBackend, MetalGPUBackend,
    backend_name, backend_label, backend_device_name
using AtmosTransport.Adjoints: CSColumnMeanObjective, cs_surface_emission_footprint
using AtmosTransport.Grids: GEOSNativePanelConvention
using AtmosTransport.Operators.Advection: NoLimiter
using AtmosTransport.State: CubedSphereFaceFluxState, DryMassFluxBasis
using AtmosTransport.State.Fields: CubedSphereField
using AtmosTransport.SectionTimer

export main, run_benchmarks

const DEFAULT_PHASES = (:io, :advection, :diffusion, :convection, :adjoint_forward, :adjoint_reverse)

struct BenchmarkCase
    backend::Symbol
    float_type::DataType
    grid_nc::Int
    levels::Int
    tracers::Int
    operator::Symbol
    scheme::Symbol
    steps::Int
    warmup_steps::Int
    repeats::Int
    group::String
end

_split_csv(s) = String.(split(String(s), ","; keepempty = false))
_parse_int_list(s) = parse.(Int, replace.(_split_csv(s), r"^[Cc]" => ""))

function _parse_float_type(s::AbstractString)
    key = lowercase(String(s))
    key in ("f32", "float32") && return Float32
    key in ("f64", "float64") && return Float64
    throw(ArgumentError("unsupported float type $(s); use Float32 or Float64"))
end

function _canonical_backend(s::AbstractString)
    key = lowercase(replace(String(s), '-' => '_'))
    key in ("cpu", "processor") && return :cpu
    key in ("cuda", "gpu_cuda") && return :cuda
    key in ("metal", "gpu_metal") && return :metal
    throw(ArgumentError("unsupported backend $(s); use cpu, cuda, or metal"))
end

function _adapter(backend::Symbol)
    backend === :cpu && return Array
    if backend === :cuda
        isdefined(Main, :CUDA) || error("CUDA must be loaded before AtmosTransport for CUDA benchmarks")
        return getproperty(Main.CUDA, :CuArray)
    elseif backend === :metal
        isdefined(Main, :Metal) || error("Metal must be loaded before AtmosTransport for Metal benchmarks")
        return getproperty(Main.Metal, :MtlArray)
    end
    error("unsupported backend $(backend)")
end

function _sync(backend::Symbol)
    backend === :cpu && return nothing
    if backend === :cuda
        return Base.invokelatest(getproperty(Main.CUDA, :synchronize))
    elseif backend === :metal
        if isdefined(Main.Metal, :synchronize)
            return Base.invokelatest(getproperty(Main.Metal, :synchronize))
        end
        return KernelAbstractions.synchronize(getproperty(Main.Metal, :MetalBackend)())
    end
    return nothing
end

function _runtime_backend(backend::Symbol)
    backend === :cpu && return CPUBackend()
    backend === :cuda && return CUDAGPUBackend()
    backend === :metal && return MetalGPUBackend()
    error("unsupported backend $(backend)")
end

function _cpu_vendor()
    model = try
        isempty(Sys.cpu_info()) ? "" : Sys.cpu_info()[1].model
    catch
        ""
    end
    occursin(r"AMD|EPYC|Ryzen"i, model) && return "AMD"
    occursin(r"Intel|Xeon|Core"i, model) && return "Intel"
    occursin(r"Apple|M[0-9]"i, model) && return "Apple"
    return "CPU"
end

function _backend_class(backend::Symbol)
    backend === :cpu && return "CPU-$(_cpu_vendor())"
    backend === :cuda && return "GPU-CUDA"
    backend === :metal && return "GPU-Metal"
    error("unsupported backend $(backend)")
end

function _device_name(backend::Symbol)
    rb = _runtime_backend(backend)
    try
        return backend_device_name(rb)
    catch err
        return string(backend_name(rb), " device unavailable: ", sprint(showerror, err))
    end
end

function _scheme(sym::Symbol)
    sym === :upwind && return UpwindScheme()
    sym === :slopes && return SlopesScheme(MonotoneLimiter())
    sym === :ppm && return PPMScheme(MonotoneLimiter())
    sym === :linrood5 && return LinRoodPPMScheme(5)
    sym === :linrood7 && return LinRoodPPMScheme(7)
    throw(ArgumentError("unsupported scheme $(sym)"))
end

function _haloed_panel(::Type{FT}, Nc::Int, Hp::Int, Nz::Int) where {FT}
    return zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
end

function _fill_mass!(panels_m, mesh::CubedSphereMesh, ::Type{FT}) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    @inbounds for p in 1:6
        m = panels_m[p]
        for k in axes(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(size(m, 3) - 1), one(FT))
            m[ii, jj, k] = FT(1e9) * (FT(1) + FT(0.05) * sinpi(FT(2) * x) +
                           FT(0.04) * cospi(FT(2) * y) + FT(0.02) * z +
                           FT(0.001) * p)
        end
    end
    fill_panel_halos!(panels_m, mesh; dir = 0)
    return panels_m
end

function _make_tracer(panels_m, mesh::CubedSphereMesh, ::Type{FT}, tracer_idx::Int) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    panels_rm = ntuple(_ -> similar(panels_m[1]), 6)
    @inbounds for p in 1:6
        m = panels_m[p]
        rm = panels_rm[p]
        fill!(rm, zero(FT))
        for k in axes(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            x = FT(i - 1) / max(FT(Nc - 1), one(FT))
            y = FT(j - 1) / max(FT(Nc - 1), one(FT))
            z = FT(k - 1) / max(FT(size(m, 3) - 1), one(FT))
            q = FT(1e-6) * FT(tracer_idx) *
                (FT(1) + FT(0.15) * sinpi(FT(2) * x + FT(0.1) * p) +
                 FT(0.10) * cospi(FT(2) * y) + FT(0.05) * z)
            rm[ii, jj, k] = m[ii, jj, k] * q
        end
    end
    fill_panel_halos!(panels_rm, mesh; dir = 0)
    return panels_rm
end

function _fill_fluxes!(am, bm, cm, panels_m, mesh::CubedSphereMesh,
                       ::Type{FT}; cfl::Real = 0.35) where {FT}
    Nc, Hp = mesh.Nc, mesh.Hp
    cf = FT(cfl)
    @inbounds for p in 1:6
        m = panels_m[p]
        ax = am[p]
        by = bm[p]
        cz = cm[p]
        fill!(ax, zero(FT))
        fill!(by, zero(FT))
        fill!(cz, zero(FT))
        for k in axes(m, 3), j in 1:Nc, i in 1:(Nc + 1)
            ii_l = Hp + max(i - 1, 1)
            ii_r = Hp + min(i, Nc)
            jj = Hp + j
            donor = min(m[ii_l, jj, k], m[ii_r, jj, k])
            ax[Hp + i, jj, k] = cf * FT(0.08) * donor *
                                sin(FT(2pi) * FT(j + p) / FT(Nc + 6))
        end
        for k in axes(m, 3), j in 1:(Nc + 1), i in 1:Nc
            ii = Hp + i
            jj_l = Hp + max(j - 1, 1)
            jj_r = Hp + min(j, Nc)
            donor = min(m[ii, jj_l, k], m[ii, jj_r, k])
            by[ii, Hp + j, k] = cf * FT(0.06) * donor *
                                cos(FT(2pi) * FT(i + 2p) / FT(Nc + 12))
        end
        for k in 2:size(m, 3), j in 1:Nc, i in 1:Nc
            ii = Hp + i
            jj = Hp + j
            donor = min(m[ii, jj, k - 1], m[ii, jj, k])
            cz[ii, jj, k] = cf * FT(0.03) * donor *
                            sin(FT(2pi) * FT(i + j + p) / FT(2Nc + 6))
        end
    end
    return nothing
end

function _vertical_grid(::Type{FT}, Nz::Int) where {FT}
    A_ifc = zeros(FT, Nz + 1)
    B_ifc = FT.(range(FT(0), FT(1); length = Nz + 1))
    return HybridSigmaPressure(A_ifc, B_ifc)
end

function _build_model(case::BenchmarkCase)
    FT = case.float_type
    FT === Float64 && case.backend === :metal &&
        throw(ArgumentError("Metal benchmark runs require Float32"))

    Nc, Nz, Nt = case.grid_nc, case.levels, case.tracers
    Hp = 3
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp,
                           convention = GEOSNativePanelConvention())
    grid = AtmosGrid(mesh, _vertical_grid(FT, Nz), AtmosTransport.CPU(); FT = FT)

    panels_m_cpu = ntuple(_ -> _haloed_panel(FT, Nc, Hp, Nz), 6)
    _fill_mass!(panels_m_cpu, mesh, FT)
    tracers_cpu = ntuple(t -> _make_tracer(panels_m_cpu, mesh, FT, t), Nt)

    N = Nc + 2Hp
    am_cpu = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
    bm_cpu = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
    cm_cpu = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    _fill_fluxes!(am_cpu, bm_cpu, cm_cpu, panels_m_cpu, mesh, FT)

    adapter = _adapter(case.backend)
    panels_m = Adapt.adapt(adapter, panels_m_cpu)
    tracer_names = ntuple(t -> Symbol("tr", t), Nt)
    tracer_values = ntuple(t -> Adapt.adapt(adapter, tracers_cpu[t]), Nt)
    state = CubedSphereState(DryBasis, mesh, panels_m; NamedTuple{tracer_names}(tracer_values)...)
    fluxes = CubedSphereFaceFluxState{DryMassFluxBasis}(
        Adapt.adapt(adapter, am_cpu),
        Adapt.adapt(adapter, bm_cpu),
        Adapt.adapt(adapter, cm_cpu))

    advection = case.operator === :diffusion || case.operator === :convection ? NoAdvection() : _scheme(case.scheme)
    diffusion = case.operator in (:diffusion, :full) ?
        ImplicitVerticalDiffusion(; kz_field = CubedSphereField(ConstantField{FT, 3}(FT(1.0)))) :
        NoDiffusion()
    # use_collab_lu=true is the production-representative path. The
    # collab-LU kernel parallelises the inner tridiagonal solve across the
    # GPU's wide warps; with use_collab_lu=false the per-column solve runs
    # serial inside each warp, making convection ~5x slower on GPU and
    # distorting the convection:advection ratio relative to CPU. See
    # `tm5_convection_perf_findings_2026_05_22.md` for the measurement.
    convection = case.operator in (:convection, :full) ?
        TM5Convection(; use_collab_lu = true) :
        NoConvection()
    convection_forcing = _convection_forcing(case, mesh, adapter)

    model = TransportModel(state, fluxes, grid, advection;
                           diffusion = diffusion,
                           convection = convection,
                           convection_forcing = convection_forcing)
    # The model constructor allocates topology metrics from the CPU mesh.
    # Adapt the composed model once so workspace internals such as CS
    # convection cell areas live on the requested backend too.
    model = case.backend === :cpu ? model : Adapt.adapt(adapter, model)
    if !(diffusion isa NoDiffusion)
        dz = model.workspace.diffusion_ws.layer_thickness
        if dz isa Tuple
            foreach(panel -> fill!(panel, FT(100.0)), dz)
        else
            fill!(dz, FT(100.0))
        end
    end
    return model
end

function _build_cs_payload(::Type{FT}, Nc::Int, Nz::Int, Nt::Int, backend::Symbol) where {FT}
    Hp = 3
    mesh = CubedSphereMesh(; FT = FT, Nc = Nc, Hp = Hp,
                           convention = GEOSNativePanelConvention())
    panels_m_cpu = ntuple(_ -> _haloed_panel(FT, Nc, Hp, Nz), 6)
    _fill_mass!(panels_m_cpu, mesh, FT)
    tracers_cpu = ntuple(t -> _make_tracer(panels_m_cpu, mesh, FT, t), Nt)
    panels_rm_cpu = tracers_cpu[1]

    N = Nc + 2Hp
    am_cpu = ntuple(_ -> zeros(FT, N + 1, N, Nz), 6)
    bm_cpu = ntuple(_ -> zeros(FT, N, N + 1, Nz), 6)
    cm_cpu = ntuple(_ -> zeros(FT, N, N, Nz + 1), 6)
    _fill_fluxes!(am_cpu, bm_cpu, cm_cpu, panels_m_cpu, mesh, FT)

    adapter = _adapter(backend)
    return (mesh = mesh,
            panels_m = Adapt.adapt(adapter, panels_m_cpu),
            panels_rm = Adapt.adapt(adapter, panels_rm_cpu),
            panels_am = Adapt.adapt(adapter, am_cpu),
            panels_bm = Adapt.adapt(adapter, bm_cpu),
            panels_cm = Adapt.adapt(adapter, cm_cpu))
end

_hostify(x::AbstractArray) = Array(x)
_hostify(x::Tuple) = map(_hostify, x)
_hostify(x::NamedTuple) = map(_hostify, x)
_hostify(x) = x

function _write_panel_payload(path::AbstractString, payload)
    open(path, "w") do io
        serialize(io, _hostify(payload))
    end
    return filesize(path)
end

function _convection_forcing(case::BenchmarkCase, mesh, adapter)
    case.operator in (:convection, :full) || return ConvectionForcing()
    FT, Nc, Nz = case.float_type, case.grid_nc, case.levels
    entu = ntuple(_ -> begin
        a = zeros(FT, Nc, Nc, Nz)
        lo = max(2, Nz ÷ 3)
        hi = max(lo, min(Nz - 1, 2Nz ÷ 3))
        a[:, :, lo:hi] .= FT(2e-5)
        a
    end, 6)
    detu = ntuple(_ -> begin
        a = zeros(FT, Nc, Nc, Nz)
        lo = max(2, Nz ÷ 3)
        hi = max(lo, min(Nz - 1, 2Nz ÷ 3))
        a[:, :, lo:hi] .= FT(2e-5)
        a
    end, 6)
    entd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    detd = ntuple(_ -> zeros(FT, Nc, Nc, Nz), 6)
    return Adapt.adapt(adapter, ConvectionForcing(nothing, nothing, (; entu, detu, entd, detd)))
end

function _phase_summary_from_csv(path)
    phases = Dict{String, Float64}()
    isfile(path) || return phases
    for (line_no, line) in enumerate(eachline(path))
        line_no == 1 && continue
        cols = split(line, ",")
        length(cols) >= 3 || continue
        phases[String(cols[1])] = parse(Float64, cols[3])
    end
    return phases
end

function _time_io_case(case::BenchmarkCase)
    payload = _build_cs_payload(case.float_type, case.grid_nc, case.levels,
                                case.tracers, case.backend)
    times = Float64[]
    bytes = 0
    tmpdir = mktempdir(; prefix = "at_bench_io_")
    try
        for _ in 1:case.warmup_steps
            path = joinpath(tmpdir, "warmup.bin")
            _sync(case.backend)
            _write_panel_payload(path, payload)
            open(deserialize, path)
            rm(path; force = true)
        end
        for r in 1:case.repeats
            path = joinpath(tmpdir, "payload_$(r).bin")
            _sync(case.backend)
            t0 = time_ns()
            bytes = _write_panel_payload(path, payload)
            open(deserialize, path)
            _sync(case.backend)
            push!(times, (time_ns() - t0) / 1e9)
            rm(path; force = true)
        end
    finally
        rm(tmpdir; force = true, recursive = true)
    end
    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    return _result_dict(case, median_s, mad_s, Dict("io" => sum(times)),
                        bytes / median_s;
                        metric_name = "bytes_per_second")
end

function _time_adjoint_case(case::BenchmarkCase)
    payload = _build_cs_payload(case.float_type, case.grid_nc, case.levels,
                                case.tracers, case.backend)
    nsteps = max(case.steps, 1)
    panels_am_steps = [payload.panels_am for _ in 1:nsteps]
    panels_bm_steps = [payload.panels_bm for _ in 1:nsteps]
    panels_cm_steps = [payload.panels_cm for _ in 1:nsteps]
    obj_i = max(1, min(2, case.grid_nc))
    objective = CSColumnMeanObjective(1, obj_i, obj_i)
    scheme = PPMScheme(NoLimiter())
    dt = case.float_type(600)

    for _ in 1:case.warmup_steps
        cs_surface_emission_footprint(payload.panels_rm, payload.panels_m,
                                      panels_am_steps, panels_bm_steps,
                                      panels_cm_steps, payload.mesh, objective;
                                      scheme = scheme, dt = dt, tape_storage = :device)
    end

    times = Float64[]
    for _ in 1:case.repeats
        _sync(case.backend)
        t0 = time_ns()
        result = cs_surface_emission_footprint(payload.panels_rm, payload.panels_m,
                                               panels_am_steps, panels_bm_steps,
                                               panels_cm_steps, payload.mesh, objective;
                                               scheme = scheme, dt = dt,
                                               tape_storage = :device)
        _sync(case.backend)
        push!(times, (time_ns() - t0) / 1e9)
        result === nothing && error("adjoint benchmark returned no result")
    end
    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    return _result_dict(case, median_s / nsteps, mad_s / nsteps,
                        Dict("adjoint_forward_reverse" => sum(times) / case.repeats),
                        1 / median_s;
                        metric_name = "adjoint_runs_per_second")
end

function _time_case(case::BenchmarkCase)
    case.operator === :io && return _time_io_case(case)
    case.operator === :adjoint && return _time_adjoint_case(case)

    model = _build_model(case)
    dt = case.float_type(600)
    for _ in 1:case.warmup_steps
        step!(model, dt)
    end
    _sync(case.backend)

    SectionTimer.enable!()
    times = Float64[]
    for _ in 1:case.repeats
        _sync(case.backend)
        t0 = time_ns()
        for _ in 1:case.steps
            step!(model, dt)
        end
        _sync(case.backend)
        push!(times, (time_ns() - t0) / 1e9 / case.steps)
    end
    SectionTimer.disable!()
    csv_path = tempname() * ".csv"
    SectionTimer.write_csv(csv_path)
    phase_totals = _phase_summary_from_csv(csv_path)
    rm(csv_path; force = true)

    median_s = median(times)
    mad_s = median(abs.(times .- median_s))
    cells = 6 * case.grid_nc^2 * case.levels
    cell_tracer_updates = cells * case.tracers / median_s

    return _result_dict(case, median_s, mad_s, phase_totals, cell_tracer_updates;
                        metric_name = "cell_tracer_updates_per_second")
end

function _result_dict(case::BenchmarkCase, median_s::Real, mad_s::Real,
                      phase_totals, throughput::Real; metric_name::AbstractString)
    cells = 6 * case.grid_nc^2 * case.levels
    return Dict(
        "name" => _case_name(case),
        "group" => case.group,
        "backend_class" => _backend_class(case.backend),
        "backend" => String(case.backend),
        "device_name" => _device_name(case.backend),
        "float_type" => string(case.float_type),
        "grid" => "C$(case.grid_nc)",
        "levels" => case.levels,
        "tracers" => case.tracers,
        "operator" => String(case.operator),
        "scheme" => String(case.scheme),
        "steps_per_repeat" => case.steps,
        "repeats" => case.repeats,
        "warmup_steps" => case.warmup_steps,
        "time_per_step_seconds" => median_s,
        "mad_seconds" => mad_s,
        "steps_per_second" => 1 / median_s,
        metric_name => throughput,
        "cell_tracer_updates_per_second" =>
            metric_name == "cell_tracer_updates_per_second" ?
            throughput : cells * case.tracers / median_s,
        "phase_times_seconds" => phase_totals,
        "metadata" => _metadata(case),
    )
end

function _metadata(case::BenchmarkCase)
    return Dict(
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "julia_version" => string(VERSION),
        "commit" => _git_commit(),
        "backend_label" => try
            backend_label(_runtime_backend(case.backend))
        catch
            _backend_class(case.backend)
        end,
    )
end

function _git_commit()
    try
        chomp(read(`git rev-parse HEAD`, String))
    catch
        ""
    end
end

function _case_name(case::BenchmarkCase)
    return join((case.group,
                 "C$(case.grid_nc) L$(case.levels) $(case.operator) $(case.tracers)tr",
                 _backend_class(case.backend),
                 string(case.float_type)), " / ")
end

function _dashboard_records(results)
    records = Any[]
    for r in results
        extra = Dict(
            "device" => r["device_name"],
            "updates_per_second" => r["cell_tracer_updates_per_second"],
        )
        haskey(r, "bytes_per_second") &&
            (extra["bytes_per_second"] = r["bytes_per_second"])
        haskey(r, "adjoint_runs_per_second") &&
            (extra["adjoint_runs_per_second"] = r["adjoint_runs_per_second"])
        push!(records, Dict(
            "name" => r["name"],
            "unit" => "s/step",
            "value" => r["time_per_step_seconds"],
            "extra" => extra,
        ))
        for (phase, total_s) in r["phase_times_seconds"]
            phase in ("window_advance", "forcing_refresh", "chemistry") && continue
            per_step = total_s / max(1, r["steps_per_repeat"] * r["repeats"])
            per_step <= 0 && continue
            push!(records, Dict(
                "name" => join(("Operator Breakdown",
                                "$(r["grid"]) $(r["float_type"]) $(r["backend_class"])",
                                "$(r["operator"])",
                                phase), " / "),
                "unit" => "s/step",
                "value" => per_step,
                "extra" => Dict("source_case" => r["name"]),
            ))
        end
    end
    return records
end

function _write_outputs(raw_path::AbstractString, results)
    mkpath(dirname(abspath(raw_path)))
    open(raw_path, "w") do io
        JSON3.pretty(io, results)
        println(io)
    end
    dashboard_path = replace(raw_path, r"\.json$" => ".github-action-benchmark.json")
    dashboard_path == raw_path && (dashboard_path = raw_path * ".github-action-benchmark.json")
    open(dashboard_path, "w") do io
        JSON3.pretty(io, _dashboard_records(results))
        println(io)
    end
    return raw_path, dashboard_path
end

function _settings()
    s = ArgParseSettings(description = "AtmosTransport benchmark runner")
    @add_arg_table! s begin
        "--backend"
            help = "Backend: cpu, cuda, or metal."
            default = "cpu"
        "--float-type"
            help = "Comma-separated Float32/Float64 list."
            default = "Float32"
        "--grid"
            help = "Comma-separated CS grid sizes, e.g. C24,C48,C90."
            default = "C24"
        "--levels"
            help = "Comma-separated vertical level counts."
            default = "32"
        "--tracers"
            help = "Comma-separated tracer counts."
            default = "1"
        "--operator"
            help = "Comma-separated operators: advection,diffusion,convection,full,io,adjoint."
            default = "advection"
        "--scheme"
            help = "Advection scheme: upwind, slopes, ppm, linrood5, linrood7."
            default = "ppm"
        "--steps"
            help = "Steps per timed repeat."
            arg_type = Int
            default = 5
        "--warmup-steps"
            help = "Warmup steps before timing."
            arg_type = Int
            default = 2
        "--repeats"
            help = "Timed repeats."
            arg_type = Int
            default = 5
        "--group"
            help = "Dashboard group name."
            default = "Synthetic CS Sweep"
        "--output"
            help = "Raw JSON output path. A github-action-benchmark JSON is written next to it."
            default = "benchmark_results.json"
    end
    return s
end

function _cases(args)
    parsed = parse_args(args, _settings())
    backend = _canonical_backend(parsed["backend"])
    fts = _parse_float_type.(_split_csv(parsed["float-type"]))
    grids = _parse_int_list(parsed["grid"])
    levels = _parse_int_list(parsed["levels"])
    tracers = _parse_int_list(parsed["tracers"])
    operators = Symbol.(lowercase.(_split_csv(parsed["operator"])))
    schemes = Symbol.(lowercase.(_split_csv(parsed["scheme"])))
    cases = BenchmarkCase[]
    for FT in fts, Nc in grids, Nz in levels, Nt in tracers, op in operators, sch in schemes
        op in (:advection, :diffusion, :convection, :full, :io, :adjoint) ||
            throw(ArgumentError("unsupported operator $(op)"))
        push!(cases, BenchmarkCase(backend, FT, Nc, Nz, Nt, op, sch,
                                   parsed["steps"], parsed["warmup-steps"],
                                   parsed["repeats"], parsed["group"]))
    end
    return cases, parsed["output"]
end

function run_benchmarks(args = ARGS)
    cases, output = _cases(args)
    results = Any[]
    for case in cases
        @info "Running benchmark" backend=case.backend float_type=case.float_type grid=case.grid_nc levels=case.levels tracers=case.tracers operator=case.operator
        push!(results, _time_case(case))
        r = results[end]
        @printf("%-70s %10.6f s/step  %.3e cell-tracer/s\n",
                r["name"], r["time_per_step_seconds"],
                r["cell_tracer_updates_per_second"])
    end
    raw, dashboard = _write_outputs(output, results)
    @info "Wrote benchmark results" raw dashboard
    return results
end

main(args = ARGS) = run_benchmarks(args)

end
