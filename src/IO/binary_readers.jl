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

"""
    _check_binary_provenance(bin_path, hdr) → nothing

Compare a binary file's recorded provenance (script_path, script_mtime,
git_commit, git_dirty, creation_time) against the current source tree.
Warns loudly if any mismatch is detected — designed to catch the silent
"runtime running against a stale binary written by an obsolete pipeline"
failure mode that wasted hours on 2026-04-06.

Provenance fields are optional (only present in binaries written by
preprocess_spectral_v4_binary.jl 2026-04-06 or later). Older binaries
trigger a single "no provenance" info line — not an error.

Set ENV["ATMOSTR_NO_STALE_CHECK"]="1" to disable.
"""
function _check_binary_provenance(bin_path::String, hdr)
    get(ENV, "ATMOSTR_NO_STALE_CHECK", "") == "1" && return nothing

    has_prov = haskey(hdr, :script_path) && haskey(hdr, :script_mtime_unix)
    if !has_prov
        @info "BinaryProvenance: $(basename(bin_path)) has NO provenance fields. " *
              "Cannot verify staleness — was likely written by an old preprocessor. " *
              "Regenerate with preprocess_spectral_v4_binary.jl to enable checks."
        return nothing
    end

    script_path = String(hdr.script_path)
    script_mtime_recorded = Float64(hdr.script_mtime_unix)
    git_commit_recorded = haskey(hdr, :git_commit) ? String(hdr.git_commit) : "unknown"
    git_dirty_recorded = haskey(hdr, :git_dirty) ? Bool(hdr.git_dirty) : false
    creation_time = haskey(hdr, :creation_time) ? String(hdr.creation_time) : "unknown"

    issues = String[]

    # Check 1: does the script still exist?
    if !isfile(script_path)
        push!(issues, "preprocessor script no longer exists at $script_path")
    else
        script_mtime_now = mtime(script_path)
        if script_mtime_now > script_mtime_recorded + 1.0  # 1s tolerance for fs precision
            push!(issues, "preprocessor script has been modified since binary was written " *
                  "(recorded mtime $(round(Int, script_mtime_recorded)), current $(round(Int, script_mtime_now)))")
        end
    end

    # Check 2: is the git commit current?
    git_commit_now = try
        readchomp(`git -C $(dirname(@__FILE__)) rev-parse HEAD`)
    catch
        "unknown"
    end
    if git_commit_now != "unknown" && git_commit_recorded != "unknown" &&
       git_commit_now != git_commit_recorded
        push!(issues, "binary written at git $git_commit_recorded, current HEAD is $git_commit_now")
    end

    # Check 3: was the source tree dirty when the binary was written?
    git_dirty_recorded && push!(issues, "binary was written from a DIRTY working tree (uncommitted changes)")

    if !isempty(issues)
        @warn """
        ╔══════════════════════════════════════════════════════════════════╗
        ║ STALE BINARY WARNING — $(basename(bin_path))
        ╠══════════════════════════════════════════════════════════════════╣
        Binary creation: $creation_time
        Recorded script: $script_path
        Recorded commit: $git_commit_recorded$(git_dirty_recorded ? " (DIRTY)" : "")

        Issues:
        $(join("  • " .* issues, "\n        "))

        The binary may not reflect the current preprocessor source. If you
        want to be sure your run is using the LATEST preprocessor logic,
        regenerate the binary. To suppress this warning set
        ENV["ATMOSTR_NO_STALE_CHECK"]="1".
        ╚══════════════════════════════════════════════════════════════════╝
        """
    else
        @info "BinaryProvenance: $(basename(bin_path)) — OK " *
              "(written $creation_time, git $(git_commit_recorded[1:min(7, length(git_commit_recorded))]))"
    end
    return nothing
end

# =====================================================================
# Lat-lon mass-flux binary reader
# =====================================================================

"""
$(TYPEDEF)

Mmap-based reader for lat-lon pre-computed mass-flux binary files.

Supports three format versions:
- **v1**: [4096-byte header | m|am|bm|cm|ps per window] — core mass fluxes only
- **v2**: [16384-byte header | m|am|bm|cm|ps|qv|cmfmc|sfc per window] — self-describing
  with embedded QV, CMFMC, surface fields, A/B coefficients, and merge provenance.
- **v3**: [16384-byte header | v2 layout + tm5conv + temperature per window] — adds
  TM5 convection fields (entu/detu/entd/detd) and model-level temperature.

The version is auto-detected from the header's `version` field (default 1).

$(FIELDS)
"""
struct MassFluxBinaryReader{FT} <: AbstractBinaryReader
    "mmap'd flat vector over entire data region (always Float32 on disk, promoted to FT on read)"
    data   :: Vector{Float32}
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
    # v2 fields (all false/empty for v1)
    "header size in bytes (4096 for v1, 16384 for v2)"
    header_size     :: Int
    "v2: has embedded specific humidity"
    has_qv          :: Bool
    "v2: has embedded convective mass flux"
    has_cmfmc       :: Bool
    "v2: has embedded surface fields (pblh, t2m, ustar, hflux)"
    has_surface     :: Bool
    "v2: elements per window for QV (Nx*Ny*Nz, 0 if absent)"
    n_qv            :: Int
    "v2: elements per window for CMFMC (Nx*Ny*(Nz+1), 0 if absent)"
    n_cmfmc         :: Int
    "v2: elements per window for each surface field (Nx*Ny, 0 if absent)"
    n_sfc           :: Int
    "v2: interface A coefficients (Nz+1 values), empty for v1"
    A_ifc           :: Vector{Float64}
    "v2: interface B coefficients (Nz+1 values), empty for v1"
    B_ifc           :: Vector{Float64}
    # v3 fields (all false/0 for v1/v2)
    "v3: has TM5 convection fields (entu, detu, entd, detd)"
    has_tm5conv     :: Bool
    "v3: has model-level temperature"
    has_temperature :: Bool
    "v3: elements per window for each TM5 conv field (Nx*Ny*Nz, 0 if absent)"
    n_tm5conv       :: Int
    "v3: elements per window for temperature (Nx*Ny*Nz, 0 if absent)"
    n_temperature   :: Int
    # v4 fields: flux deltas for TM5-style temporal interpolation
    "v4: has flux deltas (dam, dbm, dm) for per-substep interpolation"
    has_flux_delta  :: Bool
    "v4: elements for dam = am_next - am_curr ((Nx+1)*Ny*Nz)"
    n_dam           :: Int
    "v4: elements for dbm = bm_next - bm_curr (Nx*(Ny+1)*Nz)"
    n_dbm           :: Int
    "v4: elements for dm = m_next - m_curr (Nx*Ny*Nz)"
    n_dm            :: Int
    "v4: elements for dcm = cm_next - cm_curr (Nx*Ny*(Nz+1))"
    n_dcm           :: Int
end

function MassFluxBinaryReader(bin_path::String, ::Type{FT}) where FT
    io = open(bin_path, "r")

    # Read the maximum possible header to detect version
    max_hdr = 16384
    file_sz = filesize(bin_path)
    read_sz = min(max_hdr, file_sz)
    hdr_bytes = read(io, read_sz)
    json_end = something(findfirst(==(0x00), hdr_bytes), read_sz + 1) - 1
    hdr = JSON3.read(String(hdr_bytes[1:json_end]))

    version = Int(get(hdr, :version, 1))
    hdr_size = version >= 2 ? Int(get(hdr, :header_bytes, 16384)) : BINARY_HEADER_SIZE

    Nx = Int(hdr.Nx); Ny = Int(hdr.Ny); Nz = Int(hdr.Nz); Nt = Int(hdr.Nt)
    n_m  = Int(hdr.n_m)
    n_am = Int(hdr.n_am)
    n_bm = Int(hdr.n_bm)
    n_cm = Int(hdr.n_cm)
    n_ps = Int(hdr.n_ps)

    # v2 optional field counts
    has_qv      = version >= 2 && Bool(get(hdr, :include_qv, false))
    has_cmfmc   = version >= 2 && Bool(get(hdr, :include_cmfmc, false))
    has_surface = version >= 2 && Bool(get(hdr, :include_surface, false))
    n_qv    = has_qv      ? Int(get(hdr, :n_qv, 0))    : 0
    n_cmfmc = has_cmfmc   ? Int(get(hdr, :n_cmfmc, 0))  : 0
    n_sfc   = has_surface  ? Int(get(hdr, :n_pblh, 0))  : 0  # each sfc field same size

    # v3 optional field counts
    has_tm5conv     = version >= 3 && Bool(get(hdr, :include_tm5conv, false))
    has_temperature = version >= 3 && Bool(get(hdr, :include_temperature, false))
    n_tm5conv     = has_tm5conv     ? Int(get(hdr, :n_entu, 0))        : 0
    n_temperature = has_temperature ? Int(get(hdr, :n_temperature, 0)) : 0

    # v4: flux deltas for TM5-style temporal interpolation
    has_flux_delta = version >= 4 && Bool(get(hdr, :include_flux_delta, false))
    n_dam = has_flux_delta ? Int(get(hdr, :n_dam, 0)) : 0
    n_dbm = has_flux_delta ? Int(get(hdr, :n_dbm, 0)) : 0
    n_dm  = has_flux_delta ? Int(get(hdr, :n_dm, 0))  : 0
    n_dcm = has_flux_delta ? Int(get(hdr, :n_dcm, 0)) : 0

    elems_per_window = n_m + n_am + n_bm + n_cm + n_ps +
                       n_qv + n_cmfmc +
                       4 * n_tm5conv +    # entu + detu + entd + detd
                       4 * n_sfc +        # pblh + t2m + ustar + hflux
                       n_temperature +
                       n_dam + n_dbm + n_dm + n_dcm
    total_elems = elems_per_window * Nt

    # Always mmap as Float32 (disk format). Promotion to FT happens in load_window!
    # via copyto!, which handles Float32→Float64 conversion automatically.
    seek(io, hdr_size)
    data = Mmap.mmap(io, Vector{Float32}, total_elems, hdr_size)

    lons = FT.(collect(hdr.lons))
    lats = FT.(collect(hdr.lats))

    # v2: embedded vertical coordinate
    A_ifc = version >= 2 && haskey(hdr, :A_ifc) ? Float64.(collect(hdr.A_ifc)) : Float64[]
    B_ifc = version >= 2 && haskey(hdr, :B_ifc) ? Float64.(collect(hdr.B_ifc)) : Float64[]

    if version >= 2
        @info "MassFluxBinaryReader v$(version): $(Nx)x$(Ny)x$(Nz), $(Nt) windows, " *
              "QV=$(has_qv) CMFMC=$(has_cmfmc) Sfc=$(has_surface)" *
              (version >= 3 ? " TM5=$(has_tm5conv) T=$(has_temperature)" : "") *
              (version >= 4 ? " FluxDelta=$(has_flux_delta)" : "")
    end

    # --- Provenance staleness check (added 2026-04-06) ---
    # If the binary was written by a preprocessor older than the current
    # source tree, warn loudly. This catches the silent-stale failure mode
    # we hit on 2026-04-06 where the active binary was made by the now-obsolete
    # convert_merged_massflux_to_binary.jl pipeline (broken cm) but the runtime
    # was reading it as if fresh. Fields are optional — only checked if present.
    _check_binary_provenance(bin_path, hdr)

    MassFluxBinaryReader{FT}(
        data, io, Nx, Ny, Nz, Nt,
        n_m, n_am, n_bm, n_cm, n_ps, elems_per_window,
        lons, lats,
        FT(hdr.dt_seconds), FT(hdr.half_dt_seconds),
        Int(hdr.steps_per_met_window), Int(hdr.level_top), Int(hdr.level_bot),
        hdr_size, has_qv, has_cmfmc, has_surface,
        n_qv, n_cmfmc, n_sfc, A_ifc, B_ifc,
        has_tm5conv, has_temperature, n_tm5conv, n_temperature,
        has_flux_delta, n_dam, n_dbm, n_dm, n_dcm)
end

"""
    load_window!(m, am, bm, cm, ps, reader::MassFluxBinaryReader, win)

Copy core met fields for window `win` from the mmap'd binary into pre-allocated
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

"""Offset past core fields (m+am+bm+cm+ps) within a window."""
_core_offset(r::MassFluxBinaryReader) = r.n_m + r.n_am + r.n_bm + r.n_cm + r.n_ps

"""
    load_qv_window!(qv_cpu, reader::MassFluxBinaryReader, win) → Bool

Load QV from a v2 binary. Returns `false` if QV is not embedded.
"""
function load_qv_window!(qv_cpu, reader::MassFluxBinaryReader, win::Int)
    reader.has_qv || return false
    off = (win - 1) * reader.elems_per_window + _core_offset(reader)
    copyto!(qv_cpu, 1, reader.data, off + 1, reader.n_qv)
    return true
end

"""
    load_cmfmc_window!(cmfmc_cpu, reader::MassFluxBinaryReader, win) → Bool

Load convective mass flux from a v2 binary. Returns `false` if not embedded.
"""
function load_cmfmc_window!(cmfmc_cpu, reader::MassFluxBinaryReader, win::Int)
    reader.has_cmfmc || return false
    off = (win - 1) * reader.elems_per_window + _core_offset(reader) + reader.n_qv
    copyto!(cmfmc_cpu, 1, reader.data, off + 1, reader.n_cmfmc)
    return true
end

"""
    load_tm5conv_window!(entu, detu, entd, detd, reader::MassFluxBinaryReader, win) → Bool

Load TM5 convection fields from a v3 binary. Returns `false` if not embedded.
"""
function load_tm5conv_window!(entu_cpu, detu_cpu, entd_cpu, detd_cpu,
                               reader::MassFluxBinaryReader, win::Int)
    reader.has_tm5conv || return false
    off = (win - 1) * reader.elems_per_window +
          _core_offset(reader) + reader.n_qv + reader.n_cmfmc
    n = reader.n_tm5conv
    copyto!(entu_cpu, 1, reader.data, off + 1,       n)
    copyto!(detu_cpu, 1, reader.data, off + n + 1,    n)
    copyto!(entd_cpu, 1, reader.data, off + 2*n + 1,  n)
    copyto!(detd_cpu, 1, reader.data, off + 3*n + 1,  n)
    return true
end

"""Offset past core + qv + cmfmc + tm5conv fields within a window."""
_pre_surface_offset(r::MassFluxBinaryReader) =
    _core_offset(r) + r.n_qv + r.n_cmfmc + 4 * r.n_tm5conv

"""
    load_surface_window!(pblh, t2m, ustar, hflux, reader::MassFluxBinaryReader, win) → Bool

Load surface fields from a v2/v3 binary. Returns `false` if not embedded.
"""
function load_surface_window!(pblh_cpu, t2m_cpu, ustar_cpu, hflux_cpu,
                               reader::MassFluxBinaryReader, win::Int)
    reader.has_surface || return false
    off = (win - 1) * reader.elems_per_window + _pre_surface_offset(reader)
    n = reader.n_sfc
    copyto!(pblh_cpu,  1, reader.data, off + 1,       n)
    copyto!(t2m_cpu,   1, reader.data, off + n + 1,    n)
    copyto!(ustar_cpu, 1, reader.data, off + 2*n + 1,  n)
    copyto!(hflux_cpu, 1, reader.data, off + 3*n + 1,  n)
    return true
end

"""
    load_temperature_window!(t_cpu, reader::MassFluxBinaryReader, win) → Bool

Load model-level temperature from a v3 binary. Returns `false` if not embedded.
"""
function load_temperature_window!(t_cpu, reader::MassFluxBinaryReader, win::Int)
    reader.has_temperature || return false
    off = (win - 1) * reader.elems_per_window +
          _pre_surface_offset(reader) + 4 * reader.n_sfc
    copyto!(t_cpu, 1, reader.data, off + 1, reader.n_temperature)
    return true
end

"""Offset past all pre-delta fields within a window."""
_pre_delta_offset(r::MassFluxBinaryReader) =
    _pre_surface_offset(r) + 4 * r.n_sfc + r.n_temperature

"""
    load_flux_delta_window!(dam, dbm, dm, reader, win) �� Bool

Load v4 flux deltas (dam, dbm, dm) for TM5-style temporal interpolation.
Returns `false` if not a v4 binary.
"""
function load_flux_delta_window!(dam_cpu, dbm_cpu, dm_cpu, dcm_cpu,
                                  reader::MassFluxBinaryReader, win::Int)
    reader.has_flux_delta || return false
    off = (win - 1) * reader.elems_per_window + _pre_delta_offset(reader)
    o = off
    copyto!(dam_cpu, 1, reader.data, o + 1, reader.n_dam); o += reader.n_dam
    copyto!(dbm_cpu, 1, reader.data, o + 1, reader.n_dbm); o += reader.n_dbm
    copyto!(dm_cpu,  1, reader.data, o + 1, reader.n_dm);  o += reader.n_dm
    if reader.n_dcm > 0 && length(dcm_cpu) > 0
        copyto!(dcm_cpu, 1, reader.data, o + 1, reader.n_dcm)
    end
    return true
end

# Backward-compatible 3-argument version (no dcm)
function load_flux_delta_window!(dam_cpu, dbm_cpu, dm_cpu,
                                  reader::MassFluxBinaryReader, win::Int)
    load_flux_delta_window!(dam_cpu, dbm_cpu, dm_cpu, Float32[], reader, win)
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
