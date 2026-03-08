# ---------------------------------------------------------------------------
# Binary met-data readers — mmap-based, zero-copy
#
# Two concrete readers:
#   MassFluxBinaryReader — lat-lon mass-flux shards (from preprocess_mass_fluxes.jl)
#   CSBinaryReader       — cubed-sphere panel binary (from preprocess_geosfp_cs.jl)
#
# Both use a fixed-size JSON header followed by a contiguous float array.
# ---------------------------------------------------------------------------

using Mmap
using JSON3

"""
$(TYPEDEF)

Supertype for mmap-based binary met-data readers.

# Interface
    load_window!(cpu_bufs..., reader, window_index)
    Base.close(reader)
    window_count(reader) → Int
"""
abstract type AbstractBinaryReader end

"""Number of met windows stored in this binary file."""
window_count(r::AbstractBinaryReader) = r.Nt

const BINARY_HEADER_SIZE = 4096
const CS_HEADER_SIZE     = 8192
const EDGAR_HEADER_SIZE  = 4096

# =====================================================================
# Lat-lon mass-flux binary reader
# =====================================================================

"""
$(TYPEDEF)

Mmap-based reader for lat-lon pre-computed mass-flux binary files.
File layout: [4096-byte JSON header | window₁ data | window₂ data | …]

Each window contains (in order): m, am, bm, cm, ps — as flat Float32/Float64.

$(FIELDS)
"""
struct MassFluxBinaryReader{FT} <: AbstractBinaryReader
    "mmap'd flat vector over entire data region"
    data   :: Vector{FT}
    "underlying IOStream (must stay open while mmap is live)"
    io     :: IOStream
    Nx     :: Int
    Ny     :: Int
    Nz     :: Int
    Nt     :: Int
    n_m    :: Int
    n_am   :: Int
    n_bm   :: Int
    n_cm   :: Int
    n_ps   :: Int
    elems_per_window :: Int
    lons   :: Vector{FT}
    lats   :: Vector{FT}
    dt_seconds      :: FT
    half_dt_seconds :: FT
    steps_per_met   :: Int
    level_top       :: Int
    level_bot       :: Int
end

function MassFluxBinaryReader(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, BINARY_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), BINARY_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nx = Int(hdr.Nx); Ny = Int(hdr.Ny); Nz = Int(hdr.Nz); Nt = Int(hdr.Nt)
    n_m  = Int(hdr.n_m)
    n_am = Int(hdr.n_am)
    n_bm = Int(hdr.n_bm)
    n_cm = Int(hdr.n_cm)
    n_ps = Int(hdr.n_ps)
    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps
    total_elems = elems_per_window * Nt

    seek(io, BINARY_HEADER_SIZE)
    data = Mmap.mmap(io, Vector{FT}, total_elems, BINARY_HEADER_SIZE)

    lons = FT.(collect(hdr.lons))
    lats = FT.(collect(hdr.lats))

    MassFluxBinaryReader{FT}(
        data, io, Nx, Ny, Nz, Nt,
        n_m, n_am, n_bm, n_cm, n_ps, elems_per_window,
        lons, lats,
        FT(hdr.dt_seconds), FT(hdr.half_dt_seconds),
        Int(hdr.steps_per_met_window), Int(hdr.level_top), Int(hdr.level_bot))
end

"""
    load_window!(m, am, bm, cm, ps, reader::MassFluxBinaryReader, win)

Copy met fields for window `win` from the mmap'd binary into pre-allocated
CPU arrays. Zero-copy from the mmap region via `copyto!`.
"""
function load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu,
                      reader::MassFluxBinaryReader, win::Int)
    off = (win - 1) * reader.elems_per_window
    o = off
    copyto!(m_cpu,  1, reader.data, o + 1, reader.n_m);  o += reader.n_m
    copyto!(am_cpu, 1, reader.data, o + 1, reader.n_am); o += reader.n_am
    copyto!(bm_cpu, 1, reader.data, o + 1, reader.n_bm); o += reader.n_bm
    copyto!(cm_cpu, 1, reader.data, o + 1, reader.n_cm); o += reader.n_cm
    copyto!(ps_cpu, 1, reader.data, o + 1, reader.n_ps)
    return nothing
end

Base.close(r::MassFluxBinaryReader) = close(r.io)

# =====================================================================
# Cubed-sphere panel binary reader
# =====================================================================

"""
$(TYPEDEF)

Mmap-based reader for cubed-sphere preprocessed binary files.
File layout: [8192-byte JSON header | window₁ data | window₂ data | …]

Each window contains (in order): delp panels × 6, am panels × 6, bm panels × 6.

$(FIELDS)
"""
struct CSBinaryReader{FT} <: AbstractBinaryReader
    "mmap'd flat vector over entire data region (always Float32 on disk)"
    data   :: Vector{Float32}
    "underlying IOStream"
    io     :: IOStream
    "cells per panel edge"
    Nc     :: Int
    "number of vertical levels"
    Nz     :: Int
    "halo width"
    Hp     :: Int
    "number of met windows"
    Nt     :: Int
    "elements per panel for delp (haloed)"
    n_delp_panel :: Int
    "elements per panel for am (staggered x)"
    n_am_panel   :: Int
    "elements per panel for bm (staggered y)"
    n_bm_panel   :: Int
    "total elements per window"
    elems_per_window :: Int
end

function CSBinaryReader(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, CS_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), CS_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc); Nz = Int(hdr.Nz); Hp = Int(hdr.Hp); Nt = Int(hdr.Nt)
    n_delp = Int(hdr.n_delp_panel)
    n_am   = Int(hdr.n_am_panel)
    n_bm   = Int(hdr.n_bm_panel)
    elems  = Int(hdr.elems_per_window)
    total  = elems * Nt

    seek(io, CS_HEADER_SIZE)
    # Binary files are always Float32 on disk — mmap as Float32, convert at load time
    data = Mmap.mmap(io, Vector{Float32}, (total,), CS_HEADER_SIZE; grow=false)

    CSBinaryReader{FT}(data, io, Nc, Nz, Hp, Nt, n_delp, n_am, n_bm, elems)
end

"""
    load_cs_window!(delp_cpu, am_cpu, bm_cpu, reader::CSBinaryReader, win)

Copy cubed-sphere met fields for window `win` from the mmap'd binary into
pre-allocated CPU panel tuples (NTuple{6}).
"""
function load_cs_window!(delp_cpu::NTuple{6}, am_cpu::NTuple{6}, bm_cpu::NTuple{6},
                          reader::CSBinaryReader, win::Int)
    off = (win - 1) * reader.elems_per_window
    o = off
    for p in 1:6
        n = reader.n_delp_panel
        copyto!(vec(delp_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_am_panel
        copyto!(vec(am_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_bm_panel
        copyto!(vec(bm_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    return nothing
end

Base.close(r::CSBinaryReader) = close(r.io)

# =====================================================================
# EDGAR cubed-sphere binary reader
# =====================================================================

"""
    load_edgar_cs_binary(bin_path, FT) → NTuple{6, Matrix{FT}}

Read a preprocessed EDGAR emission file on cubed-sphere panels.
File layout: [4096-byte JSON header | panel₁ data | … | panel₆ data]
"""
function load_edgar_cs_binary(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")
    hdr_bytes = read(io, EDGAR_HEADER_SIZE)
    json_end = something(findfirst(==(0x00), hdr_bytes), EDGAR_HEADER_SIZE + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    Nc = Int(hdr.Nc)

    # Binary files are always Float32 on disk; convert to FT
    buf = Array{Float32}(undef, Nc, Nc)
    flux_panels = ntuple(6) do _
        read!(io, buf)
        FT === Float32 ? copy(buf) : FT.(buf)
    end
    close(io)
    return flux_panels
end
