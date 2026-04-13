# ===========================================================================
# Download pipeline — top-level entry point
#
# Mirrors the Preprocessing pattern:
#   parse config → build source/protocol → iterate dates → execute tasks
# ===========================================================================

# Dates and Printf are available from the parent module scope (Downloads.jl)

"""
    download_data!(cfg::Dict{String,Any}; start_date=nothing, end_date=nothing,
                   dry_run=false, verify_only=false)

Main entry point. Parses TOML config and executes downloads.

- `dry_run`: print what would be downloaded without executing
- `verify_only`: check existing files against expected sizes
"""
function download_data!(cfg::Dict{String, Any};
                        start_date::Union{Date, Nothing}=nothing,
                        end_date::Union{Date, Nothing}=nothing,
                        dry_run::Bool=false,
                        verify_only::Bool=false)
    config = parse_download_config(cfg)

    # Override dates from CLI if provided
    sched = if !isnothing(start_date) || !isnothing(end_date)
        ScheduleConfig(
            something(start_date, config.schedule.start_date),
            something(end_date, config.schedule.end_date),
            config.schedule.chunk,
            config.schedule.max_concurrent,
            config.schedule.max_retries,
            config.schedule.retry_wait_seconds,
            config.schedule.skip_existing,
        )
    else
        config.schedule
    end

    _print_banner(config, sched, dry_run, verify_only)

    # Group dates by chunk strategy
    date_groups = _group_dates(sched)

    out_dir = canonical_output_dir(config.output)
    mkpath(out_dir)

    n_total = length(date_groups)
    n_ok = 0
    n_skip = 0
    n_fail = 0
    total_bytes = 0

    for (i, group) in enumerate(date_groups)
        label = _group_label(group, sched.chunk)
        @info "[$i/$n_total] $label"

        tasks = build_tasks(config.source, config.protocol, group,
                            config.output, config.requests)

        if dry_run
            _print_dry_run(tasks)
            n_ok += length(tasks)
            continue
        end

        if verify_only
            for task in tasks
                if isfile(task.dest_path)
                    sz = filesize(task.dest_path)
                    @info "  ✓ $(basename(task.dest_path)) ($(sz ÷ 1_000_000) MB)"
                    n_ok += 1
                    total_bytes += sz
                else
                    @warn "  ✗ MISSING: $(task.dest_path)"
                    n_fail += 1
                end
            end
            continue
        end

        for task in tasks
            if sched.skip_existing && isfile(task.dest_path) &&
               filesize(task.dest_path) > 0
                @info "  Skip (exists): $(basename(task.dest_path))"
                n_skip += 1
                total_bytes += filesize(task.dest_path)
                continue
            end

            success = execute!(task, config.protocol;
                               max_retries=sched.max_retries,
                               retry_wait=sched.retry_wait_seconds)
            if success
                n_ok += 1
                total_bytes += isfile(task.dest_path) ? filesize(task.dest_path) : 0
            else
                n_fail += 1
            end
        end
    end

    _print_summary(n_ok, n_skip, n_fail, total_bytes, dry_run, verify_only)
end

# ---------------------------------------------------------------------------
# Date grouping by chunk strategy
# ---------------------------------------------------------------------------

"""
    _group_dates(sched::ScheduleConfig) -> Vector{Vector{Date}}

Group the date range by chunk strategy.
"""
function _group_dates(sched::ScheduleConfig)
    all_dates = collect(sched.start_date:Day(1):sched.end_date)

    if sched.chunk == :monthly
        # Group by (year, month) — dates are sorted, so consecutive grouping works
        groups = Vector{Date}[]
        for d in all_dates
            if isempty(groups) || (year(d), month(d)) != (year(groups[end][1]), month(groups[end][1]))
                push!(groups, Date[])
            end
            push!(groups[end], d)
        end
        return groups
    elseif sched.chunk in (:daily, :per_file)
        return [[d] for d in all_dates]
    else
        error("Unknown chunk strategy: $(sched.chunk). Use: monthly, daily, per_file")
    end
end

function _group_label(group::Vector{Date}, chunk::Symbol)
    if chunk == :monthly
        return Dates.format(group[1], "yyyy-mm") * " ($(length(group)) days)"
    else
        return Dates.format(group[1], "yyyy-mm-dd")
    end
end

# ---------------------------------------------------------------------------
# Task building — dispatches on source type
# ---------------------------------------------------------------------------

"""
    build_tasks(source, protocol, dates, output, requests) -> Vector{DownloadTask}

Build download tasks for a date group. Dispatches on source type.
"""
function build_tasks(source::AbstractDownloadSource,
                     protocol::AbstractDownloadProtocol,
                     dates::Vector{Date},
                     output::OutputConfig,
                     requests::Vector)
    error("build_tasks not implemented for $(typeof(source)) × $(typeof(protocol)). " *
          "Load the appropriate source module.")
end

# ---------------------------------------------------------------------------
# Task execution — dispatches on protocol type
# ---------------------------------------------------------------------------

"""
    execute!(task::DownloadTask, protocol; max_retries=3, retry_wait=30) -> Bool

Execute a single download task. Dispatches on protocol type.
"""
function execute!(task::DownloadTask, protocol::AbstractDownloadProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    error("execute! not implemented for $(typeof(protocol)). " *
          "Load the appropriate protocol module.")
end

# ---------------------------------------------------------------------------
# Retry scaffold (shared by S3, CDS, MARS)
# ---------------------------------------------------------------------------

"""
    _with_retries(f, label, dest; max_retries, retry_wait) -> Bool

Execute `f()` up to `max_retries` times. `f()` should return `true` on
success. On failure, cleans up `dest` and waits before retrying.
"""
function _with_retries(f::Function, label::String, dest::String;
                       max_retries::Int, retry_wait::Int)
    for attempt in 1:max_retries
        try
            f(attempt) && return true
        catch e
            @warn "  Attempt $attempt failed: $e"
            isfile(dest) && rm(dest; force=true)
            attempt < max_retries && sleep(retry_wait)
        end
    end
    @error "  Failed after $max_retries attempts: $label"
    return false
end

# OPeNDAP protocol execution (Phase 4 — requires NCDatasets remote access)
function execute!(task::DownloadTask, proto::OPeNDAPProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    # TODO: implement OPeNDAP subset download via NCDatasets remote read
    error("OPeNDAP download not yet implemented. " *
          "Use the legacy script for MERRA-2: scripts/downloads/download_test_data.jl")
end

# HTTP protocol execution
function execute!(task::DownloadTask, ::HTTPProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    mkpath(dirname(task.dest_path))
    return verified_download(task.source_url, task.dest_path;
                             max_retries=max_retries)
end

# S3 protocol execution
function execute!(task::DownloadTask, proto::S3Protocol;
                  max_retries::Int=3, retry_wait::Int=30)
    mkpath(dirname(task.dest_path))
    s3_url = "s3://$(proto.bucket)/$(task.source_url)"
    _with_retries(basename(task.dest_path), task.dest_path;
                  max_retries, retry_wait) do attempt
        cmd = proto.no_sign_request ?
              `aws s3 cp --no-sign-request $s3_url $(task.dest_path)` :
              `aws s3 cp $s3_url $(task.dest_path)`
        @info "  Downloading $(basename(task.dest_path)) (attempt $attempt)..."
        run(cmd)
        isfile(task.dest_path) && filesize(task.dest_path) > 0
    end
end

# CDS protocol execution
function execute!(task::DownloadTask, proto::CDSProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    mkpath(dirname(task.dest_path))
    dataset = get(task.request, "dataset", "reanalysis-era5-complete")
    request = Dict{String,Any}(k => v for (k, v) in task.request if k != "dataset")
    _with_retries(task.name, task.dest_path; max_retries, retry_wait) do attempt
        script = build_cds_retrieve_script(dataset, request, task.dest_path)
        @info "  CDS retrieve: $(task.name) (attempt $attempt)..."
        run_python(script, proto.python_env; label=task.name)
        isfile(task.dest_path) && filesize(task.dest_path) > 0
    end
end

# MARS protocol execution (with CDS fallback)
function execute!(task::DownloadTask, proto::MARSProtocol;
                  max_retries::Int=3, retry_wait::Int=30)
    mkpath(dirname(task.dest_path))
    request = Dict{String,Any}(k => v for (k, v) in task.request if k != "dataset")
    success = _with_retries(task.name, task.dest_path; max_retries, retry_wait) do attempt
        script = build_mars_retrieve_script(request, task.dest_path)
        @info "  MARS retrieve: $(task.name) (attempt $attempt)..."
        run_python(script, proto.python_env; label=task.name)
        isfile(task.dest_path) && filesize(task.dest_path) > 0
    end
    if !success && proto.fallback_to_cds
        @warn "  MARS failed, falling back to CDS for $(task.name)"
        return execute!(task, CDSProtocol(proto.python_env);
                        max_retries=max_retries, retry_wait=retry_wait)
    end
    return success
end

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

function _print_banner(config::DownloadConfig, sched::ScheduleConfig,
                       dry_run::Bool, verify_only::Bool)
    mode = dry_run ? " [DRY RUN]" : verify_only ? " [VERIFY]" : ""
    out_dir = canonical_output_dir(config.output)
    n_days = Dates.value(sched.end_date - sched.start_date) + 1

    println("=" ^ 70)
    println("AtmosTransport Download$mode")
    println("=" ^ 70)
    println("  Source:   $(source_name(config.source))")
    println("  Period:   $(sched.start_date) to $(sched.end_date) ($n_days days)")
    println("  Chunk:    $(sched.chunk)")
    println("  Output:   $out_dir")
    if !isempty(config.requests)
        println("  Requests: $(length(config.requests))")
        for req in config.requests
            name = get(req, "name", "unnamed")
            desc = get(req, "description", "")
            println("    - $name: $desc")
        end
    end
    println("=" ^ 70)
end

function _print_dry_run(tasks::Vector{DownloadTask})
    for task in tasks
        est = if task.estimated_size_mb >= 1000
            " (~$(@sprintf("%.0f", task.estimated_size_mb / 1000)) GB)"
        elseif task.estimated_size_mb > 0
            " (~$(@sprintf("%.0f", task.estimated_size_mb)) MB)"
        else
            ""
        end
        println("  → $(task.dest_path)$est")
    end
end

function _print_summary(n_ok, n_skip, n_fail, total_bytes, dry_run, verify_only)
    println()
    println("=" ^ 70)
    action = dry_run ? "Would download" : verify_only ? "Verified" : "Downloaded"
    println("$action: $n_ok")
    !dry_run && !verify_only && n_skip > 0 && println("Skipped (existing): $n_skip")
    n_fail > 0 && println("Failed: $n_fail")
    total_bytes > 0 && println("Total size: $(@sprintf("%.1f", total_bytes / 1e9)) GB")
    println("=" ^ 70)
end
