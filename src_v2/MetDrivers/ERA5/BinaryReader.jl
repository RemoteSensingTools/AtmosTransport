# ---------------------------------------------------------------------------
# ERA5BinaryReader — mmap-based reader for lat-lon mass-flux binary files
#
# Reads preprocessed ERA5 mass-flux binaries (v1–v4 format) into v2 types.
# The binary contains moist-basis fluxes; this reader produces
# StructuredFaceFluxState{MoistMassFluxBasis} to be converted downstream.
#
# Format on disk:
#   [JSON header padded to header_bytes] [window₁ Float32 data] [window₂] …
#
# Per window (v4, default fields):
#   m | am | bm | cm | ps [| qv | cmfmc | sfc×4 | tm5conv×4 | T | dam | dbm | dm | dcm]
# ---------------------------------------------------------------------------

using Mmap
using JSON3

const _HEADER_SIZE_V1 = 4096
const _HEADER_SIZE_V2 = 16384

"""
    ERA5BinaryHeader

Parsed metadata from the binary file's JSON header.  Stored as a plain struct
rather than kept as a raw JSON object for type stability.
"""
struct ERA5BinaryHeader
    version         :: Int
    header_bytes    :: Int
    Nx              :: Int
    Ny              :: Int
    Nz              :: Int
    Nt              :: Int
    n_m             :: Int
    n_am            :: Int
    n_bm            :: Int
    n_cm            :: Int
    n_ps            :: Int
    dt_seconds      :: Float64
    half_dt_seconds :: Float64
    steps_per_met   :: Int
    level_top       :: Int
    level_bot       :: Int
    has_qv          :: Bool
    n_qv            :: Int
    has_cmfmc       :: Bool
    n_cmfmc         :: Int
    has_surface     :: Bool
    n_sfc           :: Int
    has_tm5conv     :: Bool
    n_tm5conv       :: Int
    has_temperature :: Bool
    n_temperature   :: Int
    has_flux_delta  :: Bool
    n_dam           :: Int
    n_dbm           :: Int
    n_dm            :: Int
    n_dcm           :: Int
    A_ifc           :: Vector{Float64}
    B_ifc           :: Vector{Float64}
    lons_f64        :: Vector{Float64}
    lats_f64        :: Vector{Float64}
    elems_per_window :: Int
end

"""
    ERA5BinaryReader{FT}

Mmap-based reader for ERA5 lat-lon mass-flux binary files (v1–v4).

Produces `StructuredFaceFluxState{MoistMassFluxBasis}` from `load_window!`,
since ERA5 preprocessed fluxes are on a moist (total-air) basis.

# Type parameter
- `FT` — float type for output arrays (Float32 or Float64)

# Usage
```julia
reader = ERA5BinaryReader("/path/to/binary.bin"; FT=Float64)
m, ps, fluxes = load_window!(reader, 1)
# fluxes :: StructuredFaceFluxState{MoistMassFluxBasis}
close(reader)
```
"""
struct ERA5BinaryReader{FT}
    data    :: Vector{Float32}
    io      :: IOStream
    header  :: ERA5BinaryHeader
    path    :: String
end

Nx(r::ERA5BinaryReader) = r.header.Nx
Ny(r::ERA5BinaryReader) = r.header.Ny
Nz(r::ERA5BinaryReader) = r.header.Nz
window_count(r::ERA5BinaryReader) = r.header.Nt
has_qv(r::ERA5BinaryReader) = r.header.has_qv
has_flux_delta(r::ERA5BinaryReader) = r.header.has_flux_delta
A_ifc(r::ERA5BinaryReader) = r.header.A_ifc
B_ifc(r::ERA5BinaryReader) = r.header.B_ifc

function _parse_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    version = Int(get(hdr, :version, 1))
    header_bytes = version >= 2 ? Int(get(hdr, :header_bytes, _HEADER_SIZE_V2)) : _HEADER_SIZE_V1

    nx = Int(hdr.Nx); ny = Int(hdr.Ny); nz = Int(hdr.Nz); nt = Int(hdr.Nt)
    n_m  = Int(hdr.n_m)
    n_am = Int(hdr.n_am)
    n_bm = Int(hdr.n_bm)
    n_cm = Int(hdr.n_cm)
    n_ps = Int(hdr.n_ps)

    _has_qv      = version >= 2 && Bool(get(hdr, :include_qv, false))
    _has_cmfmc   = version >= 2 && Bool(get(hdr, :include_cmfmc, false))
    _has_surface = version >= 2 && Bool(get(hdr, :include_surface, false))
    _n_qv    = _has_qv      ? Int(get(hdr, :n_qv, 0))   : 0
    _n_cmfmc = _has_cmfmc   ? Int(get(hdr, :n_cmfmc, 0)) : 0
    _n_sfc   = _has_surface ? Int(get(hdr, :n_pblh, 0))  : 0

    _has_tm5conv     = version >= 3 && Bool(get(hdr, :include_tm5conv, false))
    _has_temperature = version >= 3 && Bool(get(hdr, :include_temperature, false))
    _n_tm5conv     = _has_tm5conv     ? Int(get(hdr, :n_entu, 0))        : 0
    _n_temperature = _has_temperature ? Int(get(hdr, :n_temperature, 0)) : 0

    _has_flux_delta = version >= 4 && Bool(get(hdr, :include_flux_delta, false))
    _n_dam = _has_flux_delta ? Int(get(hdr, :n_dam, 0)) : 0
    _n_dbm = _has_flux_delta ? Int(get(hdr, :n_dbm, 0)) : 0
    _n_dm  = _has_flux_delta ? Int(get(hdr, :n_dm, 0))  : 0
    _n_dcm = _has_flux_delta ? Int(get(hdr, :n_dcm, 0)) : 0

    elems = n_m + n_am + n_bm + n_cm + n_ps +
            _n_qv + _n_cmfmc +
            4 * _n_tm5conv + 4 * _n_sfc +
            _n_temperature +
            _n_dam + _n_dbm + _n_dm + _n_dcm

    _A_ifc = version >= 2 && haskey(hdr, :A_ifc) ? Float64.(collect(hdr.A_ifc)) : Float64[]
    _B_ifc = version >= 2 && haskey(hdr, :B_ifc) ? Float64.(collect(hdr.B_ifc)) : Float64[]

    lons = Float64.(collect(hdr.lons))
    lats = Float64.(collect(hdr.lats))

    ERA5BinaryHeader(
        version, header_bytes,
        nx, ny, nz, nt,
        n_m, n_am, n_bm, n_cm, n_ps,
        Float64(hdr.dt_seconds), Float64(hdr.half_dt_seconds),
        Int(hdr.steps_per_met_window), Int(hdr.level_top), Int(hdr.level_bot),
        _has_qv, _n_qv,
        _has_cmfmc, _n_cmfmc,
        _has_surface, _n_sfc,
        _has_tm5conv, _n_tm5conv,
        _has_temperature, _n_temperature,
        _has_flux_delta, _n_dam, _n_dbm, _n_dm, _n_dcm,
        _A_ifc, _B_ifc, lons, lats, elems)
end

function ERA5BinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")

    read_sz = min(_HEADER_SIZE_V2, filesize(bin_path))
    raw = read(io, read_sz)
    header = _parse_header(raw)

    total_elems = header.elems_per_window * header.Nt
    seek(io, header.header_bytes)
    data = Mmap.mmap(io, Vector{Float32}, total_elems, header.header_bytes)

    return ERA5BinaryReader{FT}(data, io, header, bin_path)
end

Base.close(r::ERA5BinaryReader) = close(r.io)

"""Byte offset (in Float32 elements) to the start of window `win`."""
@inline _window_offset(r::ERA5BinaryReader, win::Int) =
    (win - 1) * r.header.elems_per_window

"""Core fields offset (past m + am + bm + cm + ps)."""
@inline _core_offset(h::ERA5BinaryHeader) = h.n_m + h.n_am + h.n_bm + h.n_cm + h.n_ps

"""
    load_window!(reader::ERA5BinaryReader{FT}, win;
                 m  = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                 ps = Array{FT}(undef, reader.header.Nx, reader.header.Ny),
                 am = Array{FT}(undef, reader.header.Nx+1, reader.header.Ny, reader.header.Nz),
                 bm = Array{FT}(undef, reader.header.Nx, reader.header.Ny+1, reader.header.Nz),
                 cm = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz+1)
                ) -> (m, ps, StructuredFaceFluxState{MoistMassFluxBasis})

Load core met fields for window `win` into pre-allocated (or freshly allocated)
arrays.  Returns `(m, ps, fluxes)` where `fluxes` is tagged with
`MoistMassFluxBasis`.

Allocating on every call is convenient for testing; pass pre-allocated buffers
in production to avoid GC pressure.
"""
function load_window!(reader::ERA5BinaryReader{FT}, win::Int;
                      m  = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                      ps = Array{FT}(undef, reader.header.Nx, reader.header.Ny),
                      am = Array{FT}(undef, reader.header.Nx + 1, reader.header.Ny, reader.header.Nz),
                      bm = Array{FT}(undef, reader.header.Nx, reader.header.Ny + 1, reader.header.Nz),
                      cm = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz + 1)
                     ) where FT
    h = reader.header
    o = _window_offset(reader, win)
    copyto!(m,  1, reader.data, o + 1, h.n_m);  o += h.n_m
    copyto!(am, 1, reader.data, o + 1, h.n_am); o += h.n_am
    copyto!(bm, 1, reader.data, o + 1, h.n_bm); o += h.n_bm
    copyto!(cm, 1, reader.data, o + 1, h.n_cm); o += h.n_cm
    copyto!(ps, 1, reader.data, o + 1, h.n_ps)

    fluxes = StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm)
    return (m, ps, fluxes)
end

"""
    load_qv_window!(reader::ERA5BinaryReader{FT}, win;
                    qv = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz)
                   ) -> Union{Array{FT,3}, Nothing}

Load specific humidity from a v2+ binary.  Returns `nothing` if the binary
does not contain QV.
"""
function load_qv_window!(reader::ERA5BinaryReader{FT}, win::Int;
                          qv = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz)
                         ) where FT
    reader.header.has_qv || return nothing
    o = _window_offset(reader, win) + _core_offset(reader.header)
    copyto!(qv, 1, reader.data, o + 1, reader.header.n_qv)
    return qv
end

"""
    load_flux_delta_window!(reader::ERA5BinaryReader{FT}, win;
                            dam = ..., dbm = ..., dm = ..., dcm = ...)
        -> Union{NamedTuple, Nothing}

Load v4 flux deltas for substep interpolation.  Returns `nothing` if the
binary is pre-v4.
"""
function load_flux_delta_window!(reader::ERA5BinaryReader{FT}, win::Int;
                                  dam = Array{FT}(undef, reader.header.Nx + 1, reader.header.Ny, reader.header.Nz),
                                  dbm = Array{FT}(undef, reader.header.Nx, reader.header.Ny + 1, reader.header.Nz),
                                  dm  = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                                  dcm = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz + 1)
                                 ) where FT
    h = reader.header
    h.has_flux_delta || return nothing

    o = _window_offset(reader, win) +
        _core_offset(h) + h.n_qv + h.n_cmfmc +
        4 * h.n_tm5conv + 4 * h.n_sfc + h.n_temperature

    copyto!(dam, 1, reader.data, o + 1, h.n_dam); o += h.n_dam
    copyto!(dbm, 1, reader.data, o + 1, h.n_dbm); o += h.n_dbm
    copyto!(dm,  1, reader.data, o + 1, h.n_dm);  o += h.n_dm
    if h.n_dcm > 0
        copyto!(dcm, 1, reader.data, o + 1, h.n_dcm)
    end

    return (; dam, dbm, dm, dcm)
end

export ERA5BinaryReader, ERA5BinaryHeader
export load_window!, load_qv_window!, load_flux_delta_window!
export window_count, has_qv, has_flux_delta, A_ifc, B_ifc
