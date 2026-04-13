# Transport-binary v2 preprocessing interface.
#
# Audience:
# - users call `build_transport_binary_v2_target`, `target_summary`, and
#   `run_transport_binary_v2_preprocessor`
# - contributors add new targets by subtyping
#   `AbstractTransportBinaryV2Target` and implementing the hook methods below
#
# Stability: this is the stable in-repo API for transport-binary v2
# preprocessors. The `Val(...)` builders remain the internal dispatch layer.

"""
    AbstractTransportBinaryV2Target

Stable repo-level target interface for transport-binary v2 preprocessors.

Users normally construct a target with `build_transport_binary_v2_target(...)`
and run it with `run_transport_binary_v2_preprocessor(target)`.

Contributors extend the interface by subtyping this type and implementing:
- `target_input_path`
- `target_output_path`
- `target_float_type`
- `prepare_transport_binary_v2_target`
- `collect_transport_binary_v2_windows`
- `build_transport_binary_v2_header`
- `write_transport_binary_v2_output`

`target_summary` is optional but recommended for user-facing logging.
"""
abstract type AbstractTransportBinaryV2Target end

"""
    parse_flag_args(argv) -> Dict{String,String}

Minimal `--key value` parser shared by transport-binary preprocessors.
"""
function parse_flag_args(argv::AbstractVector{<:AbstractString})
    args = Dict{String, String}()
    i = 1
    while i <= length(argv)
        if startswith(argv[i], "--")
            key = argv[i][3:end]
            val = i < length(argv) ? argv[i + 1] : ""
            args[key] = val
            i += 2
        else
            i += 1
        end
    end
    return args
end

@inline function _require_flag(args::AbstractDict{<:AbstractString,<:AbstractString},
                               key::AbstractString)
    value = get(args, key, "")
    isempty(value) && error("--$key <value> required")
    return value
end

"""
    target_summary(target) -> String

Return a short user-facing summary for logging, help text, and docs examples.
"""
target_summary(target::AbstractTransportBinaryV2Target) = string(typeof(target))

target_input_path(target::AbstractTransportBinaryV2Target) =
    error("target_input_path not implemented for $(typeof(target))")

target_output_path(target::AbstractTransportBinaryV2Target) =
    error("target_output_path not implemented for $(typeof(target))")

target_float_type(::AbstractTransportBinaryV2Target) =
    error("target_float_type not implemented")

"""
    prepare_transport_binary_v2_target(target, reader) -> ctx

Contributor hook executed once after the input transport binary is opened.
Return any reusable context needed by later hooks.
"""
prepare_transport_binary_v2_target(target::AbstractTransportBinaryV2Target, reader) =
    error("prepare_transport_binary_v2_target not implemented for $(typeof(target))")

"""
    collect_transport_binary_v2_windows(target, ctx, reader) -> windows

Contributor hook that materializes the output windows for a target.
"""
collect_transport_binary_v2_windows(target::AbstractTransportBinaryV2Target, ctx, reader) =
    error("collect_transport_binary_v2_windows not implemented for $(typeof(target))")

"""
    build_transport_binary_v2_header(target, ctx, reader, windows) -> header

Contributor hook that builds the output header dictionary for a target.
"""
build_transport_binary_v2_header(target::AbstractTransportBinaryV2Target, ctx, reader, windows) =
    error("build_transport_binary_v2_header not implemented for $(typeof(target))")

"""
    write_transport_binary_v2_output(target, ctx, reader, header, windows) -> bytes_written

Contributor hook that writes the final artifact and returns the byte count.
"""
write_transport_binary_v2_output(target::AbstractTransportBinaryV2Target, ctx, reader, header, windows) =
    error("write_transport_binary_v2_output not implemented for $(typeof(target))")

"""
    build_transport_binary_v2_target(kind::Symbol, argv; FT=Float64) -> target

Stable repo entrypoint for transport-binary v2 preprocessors.

`argv` uses the same `["--key", "value", ...]` shape as the CLI wrappers.

Example:
```julia
target = build_transport_binary_v2_target(
    :cubed_sphere_conservative,
    ["--input", "latlon.bin", "--output", "cs.bin", "--Nc", "90"],
)
```
"""
build_transport_binary_v2_target(::Val{kind},
                                 argv::AbstractVector{<:AbstractString};
                                 FT::Type{T} = Float64) where {kind, T <: AbstractFloat} =
    error("Unknown transport-binary v2 target kind: :$kind")

"""
    build_transport_binary_v2_target(kind::Symbol, argv; FT=Float64) -> target

Stable symbol-based front door for transport-binary v2 preprocessors.
"""
function build_transport_binary_v2_target(kind::Symbol,
                                          argv::AbstractVector{<:AbstractString};
                                          FT::Type{T} = Float64) where T <: AbstractFloat
    return build_transport_binary_v2_target(Val(kind), argv; FT=FT)
end

build_transport_binary_v2_target(kind::AbstractString,
                                 argv::AbstractVector{<:AbstractString};
                                 FT::Type{T} = Float64) where T <: AbstractFloat =
    build_transport_binary_v2_target(Symbol(kind), argv; FT=FT)

"""
    run_transport_binary_v2_preprocessor(target) -> NamedTuple

Generic driver for transport-binary preprocessors.

This opens the input binary with `TransportBinaryReader`, runs the
target-specific hooks, and returns `(path, bytes_written, nwindow)`.
"""
function run_transport_binary_v2_preprocessor(target::AbstractTransportBinaryV2Target)
    input_path = target_input_path(target)
    isfile(input_path) || error("Input file not found: $input_path")

    FT = target_float_type(target)
    reader = TransportBinaryReader(input_path; FT=FT)
    try
        ctx = prepare_transport_binary_v2_target(target, reader)
        windows = collect_transport_binary_v2_windows(target, ctx, reader)
        header = build_transport_binary_v2_header(target, ctx, reader, windows)
        bytes_written = write_transport_binary_v2_output(target, ctx, reader, header, windows)
        return (path = target_output_path(target),
                bytes_written = bytes_written,
                nwindow = length(windows))
    finally
        close(reader.io)
    end
end
