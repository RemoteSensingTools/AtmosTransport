# Transport-binary format: topology-generic reader/writer/inspector.
#
# This file was a single 2658-line monolith mixing header schema, contracts,
# reader, payload sections, writers, the cubed-sphere streaming path, and
# inspection. It is now a thin includer over concept-sized files in
# `transport_binary/`. The split is a PURE CODE MOVE (no behaviour change); the
# parts share this module's namespace and the `using`s below.
#
# Include order is dependency-driven (types must be defined before the methods
# that dispatch on them at include time):
#   header  →  contract  →  reader  →  payload_sections  →  writer
#           →  streaming_writer  →  cubed_sphere  →  inspect
# Function bodies may still call across files freely (Julia dispatch is
# late-bound); only struct/const definitions are order-sensitive.

using Mmap
using JSON3
using Printf: @printf
using Base.Threads
using ..Architectures: CPU
import ..State: mass_basis

include("transport_binary/header.jl")
include("transport_binary/contract.jl")
include("transport_binary/reader.jl")
include("transport_binary/payload_sections.jl")
include("transport_binary/writer.jl")
include("transport_binary/streaming_writer.jl")
include("transport_binary/cubed_sphere.jl")
include("transport_binary/inspect.jl")

export TransportBinaryHeader, TransportBinaryReader
export grid_type, horizontal_topology, load_grid, load_qv_window!, load_qv_pair_window!, load_flux_delta_window!
export has_qv, has_qv_endpoints, has_flux_delta, has_surface, write_transport_binary
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
