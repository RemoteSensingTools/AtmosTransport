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
    "v2+: includes CX/CY Courant numbers (same staggering as am/bm)"
    has_courant :: Bool
    "v3: includes precomputed XFX/YFX area fluxes (same staggering as am/bm)"
    has_area_flux :: Bool
    "v4: includes QV at start/end of each window (Nc×Nc×Nz per panel, no halo)"
    has_qv :: Bool
    "v4: includes PS at start/end of each window (Nc×Nc per panel)"
    has_ps :: Bool
    "elements per panel for QV (Nc×Nc×Nz, no halo)"
    n_qv_panel :: Int
    "elements per panel for PS (Nc×Nc)"
    n_ps_panel :: Int
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

    # v2+: detect CX/CY Courant numbers; v3: detect XFX/YFX area fluxes
    version = Int(get(hdr, :version, 1))
    has_courant = version >= 2 && Bool(get(hdr, :include_courant, false))
    has_area_flux = version >= 3 && Bool(get(hdr, :include_area_flux, false))

    # v4: embedded QV and PS at window boundaries
    has_qv = version >= 4 && Bool(get(hdr, :include_qv, false))
    has_ps = version >= 4 && Bool(get(hdr, :include_ps, false))
    n_qv_panel = has_qv ? Int(get(hdr, :n_qv_panel, 0)) : 0
    n_ps_panel = has_ps ? Int(get(hdr, :n_ps_panel, 0)) : 0

    elems = Int(hdr.elems_per_window)
    total = elems * Nt

    seek(io, CS_HEADER_SIZE)
    data = Mmap.mmap(io, Vector{Float32}, (total,), CS_HEADER_SIZE; grow=false)

    CSBinaryReader{FT}(data, io, Nc, Nz, Hp, Nt, n_delp, n_am, n_bm, elems,
                        has_courant, has_area_flux, has_qv, has_ps,
                        n_qv_panel, n_ps_panel)
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

"""
    load_cs_cx_cy_window!(cx_cpu, cy_cpu, reader::CSBinaryReader, win)

Load CX/CY Courant numbers from a v2 binary file. Returns `false` if
the file doesn't contain Courant numbers (v1 format).
"""
function load_cs_cx_cy_window!(cx_cpu::NTuple{6}, cy_cpu::NTuple{6},
                                reader::CSBinaryReader, win::Int)
    reader.has_courant || return false
    off = (win - 1) * reader.elems_per_window
    # CX/CY follow after DELP(6) + AM(6) + BM(6) panels
    o = off + 6 * (reader.n_delp_panel + reader.n_am_panel + reader.n_bm_panel)
    for p in 1:6
        n = reader.n_am_panel   # CX same shape as AM
        copyto!(vec(cx_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_bm_panel   # CY same shape as BM
        copyto!(vec(cy_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    return true
end

"""
    load_cs_xfx_yfx_window!(xfx_cpu, yfx_cpu, reader::CSBinaryReader, win)

Load precomputed area fluxes from a v3 binary file. Returns `false` if
the file doesn't contain area fluxes (v1/v2 format).
"""
function load_cs_xfx_yfx_window!(xfx_cpu::NTuple{6}, yfx_cpu::NTuple{6},
                                   reader::CSBinaryReader, win::Int)
    reader.has_area_flux || return false
    off = (win - 1) * reader.elems_per_window
    # XFX/YFX follow after DELP(6) + AM(6) + BM(6) + CX(6) + CY(6)
    o = off + 6 * (reader.n_delp_panel + 2*reader.n_am_panel + 2*reader.n_bm_panel)
    for p in 1:6
        n = reader.n_am_panel   # XFX same shape as AM
        copyto!(vec(xfx_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    for p in 1:6
        n = reader.n_bm_panel   # YFX same shape as BM
        copyto!(vec(yfx_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    return true
end

"""
    load_cs_qv_ps_window!(qv_start, qv_end, ps_start, ps_end, reader, win)

Load embedded QV and PS at window boundaries from a v4 binary file.
QV panels are (Nc, Nc, Nz) without halos; PS panels are (Nc, Nc).
Returns `false` if the file doesn't contain QV/PS (v1-v3 format).
"""
function load_cs_qv_ps_window!(qv_start_cpu::NTuple{6}, qv_end_cpu::NTuple{6},
                                ps_start_cpu::NTuple{6}, ps_end_cpu::NTuple{6},
                                reader::CSBinaryReader, win::Int)
    (reader.has_qv && reader.has_ps) || return false
    off = (win - 1) * reader.elems_per_window
    # Skip past v3 fields: DELP(6) + AM(6) + BM(6) + CX(6) + CY(6) + XFX(6) + YFX(6)
    o = off + 6 * (reader.n_delp_panel + 3*reader.n_am_panel + 3*reader.n_bm_panel)
    # QV_start
    for p in 1:6
        n = reader.n_qv_panel
        copyto!(vec(qv_start_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    # QV_end
    for p in 1:6
        n = reader.n_qv_panel
        copyto!(vec(qv_end_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    # PS_start
    for p in 1:6
        n = reader.n_ps_panel
        copyto!(vec(ps_start_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    # PS_end
    for p in 1:6
        n = reader.n_ps_panel
        copyto!(vec(ps_end_cpu[p]), 1, reader.data, o + 1, n)
        o += n
    end
    return true
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
