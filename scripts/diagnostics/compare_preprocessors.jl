#!/usr/bin/env julia
# Compare Plan 41 legacy and unified preprocessor paths on one TOML config.

using Dates
using JSON3
using Logging
using Printf
using TOML

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Preprocessing

const DEFAULT_IGNORED_HEADER_KEYS = Set{Symbol}([:creation_time])
const COMPARE_CHUNK_BYTES = 64 * 1024 * 1024

function _usage()
    return """
    Usage:
      julia --project=. scripts/diagnostics/compare_preprocessors.jl CONFIG.toml --day YYYY-MM-DD [options]

    Options:
      --start YYYY-MM-DD       Start date for native-source ranges.
      --end YYYY-MM-DD         End date for native-source ranges.
      --workdir DIR            Directory for legacy/unified outputs.
      --keep                   Keep the generated comparison directory.
      --ignore-header-key KEY  Ignore an additional top-level JSON header key.
      --warn-only-positivity   Set [numerics].require_substep_positivity=false
                               for both runs.

    The script runs the config twice with separate output directories:
    legacy  -> [preprocessor].unified = false
    unified -> [preprocessor].unified = true

    It reports exact byte equality when possible. If bytes differ, it parses
    the fixed JSON header, ignores volatile header keys such as creation_time,
    and requires both normalized header equality and payload byte equality.
    """
end

function _parse_args(args)
    isempty(args) && error(_usage())
    cfg_path = abspath(expanduser(args[1]))
    isfile(cfg_path) || error("Config not found: $cfg_path")

    day = nothing
    start_date = nothing
    end_date = nothing
    workdir = nothing
    keep = false
    warn_only_positivity = false
    ignored = copy(DEFAULT_IGNORED_HEADER_KEYS)

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--day" && i + 1 <= length(args)
            day = args[i + 1]
            i += 2
        elseif arg == "--start" && i + 1 <= length(args)
            start_date = args[i + 1]
            i += 2
        elseif arg == "--end" && i + 1 <= length(args)
            end_date = args[i + 1]
            i += 2
        elseif arg == "--workdir" && i + 1 <= length(args)
            workdir = abspath(expanduser(args[i + 1]))
            i += 2
        elseif arg == "--ignore-header-key" && i + 1 <= length(args)
            push!(ignored, Symbol(args[i + 1]))
            i += 2
        elseif arg == "--keep"
            keep = true
            i += 1
        elseif arg == "--warn-only-positivity"
            warn_only_positivity = true
            i += 1
        elseif arg == "--help" || arg == "-h"
            println(_usage())
            exit(0)
        else
            error("Unknown or incomplete argument `$arg`.\n$(_usage())")
        end
    end

    day === nothing && (start_date === nothing || end_date === nothing) &&
        error("Provide either `--day YYYY-MM-DD` or `--start/--end`.")

    return (cfg_path = cfg_path,
            day = day,
            start_date = start_date,
            end_date = end_date,
            workdir = workdir,
            keep = keep,
            warn_only_positivity = warn_only_positivity,
            ignored = ignored)
end

function _deepcopy_cfg(x)
    if x isa AbstractDict
        return Dict{String, Any}(String(k) => _deepcopy_cfg(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [_deepcopy_cfg(v) for v in x]
    else
        return x
    end
end

function _ensure_table!(cfg::Dict{String, Any}, key::String)
    table = get!(cfg, key) do
        Dict{String, Any}()
    end
    table isa Dict || error("`[$key]` must be a TOML table.")
    return table
end

function _prepared_cfg(base_cfg,
                       out_dir::AbstractString,
                       unified::Bool,
                       warn_only_positivity::Bool)
    cfg = _deepcopy_cfg(base_cfg)
    output = _ensure_table!(cfg, "output")
    output["directory"] = String(out_dir)
    preprocessor = _ensure_table!(cfg, "preprocessor")
    preprocessor["unified"] = unified
    if warn_only_positivity
        numerics = _ensure_table!(cfg, "numerics")
        numerics["require_substep_positivity"] = false
    end
    return cfg
end

function _run_preprocessor!(cfg, parsed)
    return process_day(cfg;
                       day_override = parsed.day,
                       start_date = parsed.start_date,
                       end_date = parsed.end_date)
end

function _collect_files(root::AbstractString)
    isdir(root) || error("Output directory not created: $root")
    rels = String[]
    for (dir, _dirs, files) in walkdir(root)
        for file in files
            path = joinpath(dir, file)
            isfile(path) || continue
            push!(rels, relpath(path, root))
        end
    end
    sort!(rels)
    isempty(rels) && error("No output files found under $root")
    return rels
end

function _json_to_julia(x)
    if x isa JSON3.Object
        return Dict{Symbol, Any}(Symbol(k) => _json_to_julia(v)
                                 for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return [_json_to_julia(v) for v in x]
    else
        return x
    end
end

function _read_header(path::AbstractString)
    prefix = UInt8[]
    result = nothing
    open(path, "r") do io
        while !eof(io) && result === nothing
            append!(prefix, read(io, min(COMPARE_CHUNK_BYTES, 1024 * 1024)))
            nul = findfirst(==(0x00), prefix)
            if nul !== nothing
                header = _json_to_julia(JSON3.read(String(prefix[1:nul - 1])))
                header_bytes = Int(get(header, :header_bytes,
                                       AtmosTransport.Preprocessing.HEADER_SIZE))
                header_bytes <= filesize(path) ||
                    error("Header declares header_bytes=$header_bytes, " *
                          "but file has only $(filesize(path)) bytes: $path")
                result = (header, header_bytes)
            end
            length(prefix) <= 4 * 1024 * 1024 ||
                error("Could not find NUL-padded JSON header boundary in first 4 MiB: $path")
        end
    end
    result === nothing &&
        error("Could not find NUL-padded JSON header boundary in empty/truncated file: $path")
    return result
end

function _files_equal(a::AbstractString,
                      b::AbstractString;
                      offset_a::Integer = 0,
                      offset_b::Integer = 0,
                      chunk_bytes::Integer = COMPARE_CHUNK_BYTES)
    size_a = filesize(a) - Int(offset_a)
    size_b = filesize(b) - Int(offset_b)
    (size_a >= 0 && size_b >= 0) || return false
    size_a == size_b || return false
    equal = true
    open(a, "r") do io_a
        open(b, "r") do io_b
            seek(io_a, Int(offset_a))
            seek(io_b, Int(offset_b))
            remaining = size_a
            while remaining > 0
                n = min(Int(chunk_bytes), remaining)
                if read(io_a, n) != read(io_b, n)
                    equal = false
                    break
                end
                remaining -= n
            end
        end
    end
    return equal
end

function _read_header_and_payload(bytes::Vector{UInt8})
    nul = findfirst(==(0x00), bytes)
    nul === nothing && error("Could not find NUL-padded JSON header boundary.")
    header = _json_to_julia(JSON3.read(String(bytes[1:nul - 1])))
    header_bytes = Int(get(header, :header_bytes,
                           AtmosTransport.Preprocessing.HEADER_SIZE))
    header_bytes <= length(bytes) ||
        error("Header declares header_bytes=$header_bytes, but file has only $(length(bytes)) bytes.")
    return header, @view(bytes[header_bytes + 1:end])
end

function _normalize_header!(header::Dict{Symbol, Any}, ignored::Set{Symbol})
    for key in ignored
        delete!(header, key)
    end
    return header
end

function _changed_header_keys(a::Dict{Symbol, Any}, b::Dict{Symbol, Any})
    keys_all = collect(union(keys(a), keys(b)))
    sort!(keys_all; by=String)
    return [k for k in keys_all if get(a, k, nothing) != get(b, k, nothing)]
end

function _compare_file_pair(legacy_path::AbstractString,
                            unified_path::AbstractString,
                            ignored::Set{Symbol})
    legacy_size = filesize(legacy_path)
    unified_size = filesize(unified_path)
    if legacy_size == unified_size && _files_equal(legacy_path, unified_path)
        return (status = :exact,
                legacy_size = legacy_size,
                unified_size = unified_size,
                changed_header_keys = Symbol[])
    end

    legacy_header, legacy_header_bytes = _read_header(legacy_path)
    unified_header, unified_header_bytes = _read_header(unified_path)
    _normalize_header!(legacy_header, ignored)
    _normalize_header!(unified_header, ignored)

    headers_equal = legacy_header == unified_header
    payload_equal = _files_equal(legacy_path, unified_path;
                                 offset_a = legacy_header_bytes,
                                 offset_b = unified_header_bytes)
    status = headers_equal && payload_equal ? :header_normalized :
             payload_equal ? :header_mismatch :
             :payload_mismatch
    return (status = status,
            legacy_size = legacy_size,
            unified_size = unified_size,
            changed_header_keys = headers_equal ? Symbol[] :
                _changed_header_keys(legacy_header, unified_header))
end

function _compare_outputs(legacy_dir::AbstractString,
                          unified_dir::AbstractString,
                          ignored::Set{Symbol})
    legacy_files = _collect_files(legacy_dir)
    unified_files = _collect_files(unified_dir)
    legacy_files == unified_files || error(
        "Output file sets differ.\nlegacy:  $(legacy_files)\nunified: $(unified_files)")

    results = Pair{String, Any}[]
    ok = true
    for rel in legacy_files
        result = _compare_file_pair(joinpath(legacy_dir, rel),
                                    joinpath(unified_dir, rel),
                                    ignored)
        push!(results, rel => result)
        ok &= result.status in (:exact, :header_normalized)
    end
    return ok, results
end

function main(args = ARGS)
    parsed = _parse_args(args)
    base_cfg = TOML.parsefile(parsed.cfg_path)
    workdir = parsed.workdir === nothing ?
        mktempdir(; prefix = "preprocessor_compare_") :
        parsed.workdir
    legacy_dir = joinpath(workdir, "legacy")
    unified_dir = joinpath(workdir, "unified")

    generated_workdir = parsed.workdir === nothing
    keep = parsed.keep || !generated_workdir
    base_logger = ConsoleLogger(stderr, Logging.Info; show_limited = false)
    global_logger(AtmosTransport.Preprocessing._FlushingLogger(base_logger))

    try
        rm(legacy_dir; recursive = true, force = true)
        rm(unified_dir; recursive = true, force = true)
        mkpath(legacy_dir)
        mkpath(unified_dir)

        @info "Running legacy preprocessor" config = parsed.cfg_path output = legacy_dir
        _run_preprocessor!(_prepared_cfg(base_cfg, legacy_dir, false,
                                         parsed.warn_only_positivity), parsed)

        @info "Running unified preprocessor" config = parsed.cfg_path output = unified_dir
        _run_preprocessor!(_prepared_cfg(base_cfg, unified_dir, true,
                                         parsed.warn_only_positivity), parsed)

        ok, results = _compare_outputs(legacy_dir, unified_dir, parsed.ignored)
        println()
        println("Preprocessor comparison: ", ok ? "PASS" : "FAIL")
        println("workdir: ", workdir)
        for (rel, result) in results
            changed = isempty(result.changed_header_keys) ? "" :
                " changed_header_keys=$(join(String.(result.changed_header_keys), ","))"
            println(@sprintf("  %-36s %-18s legacy=%d unified=%d%s",
                             rel, String(result.status),
                             result.legacy_size, result.unified_size, changed))
        end
        ok || exit(1)
    finally
        if !keep
            rm(workdir; recursive = true, force = true)
        else
            @info "Kept comparison outputs" workdir
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
