# ---------------------------------------------------------------------------
# ERA5BinaryReader — mmap-based reader for lat-lon mass-flux binary files
#
# Reads preprocessed ERA5 mass-flux binaries (v1–v5 format) into v2 types.
#
# The format stores a padded JSON header followed by contiguous per-window
# payload blocks. See docs/BINARY_FORMAT_V5.md for the full specification.
#
# ## Payload tiers
#
# Tier 1 (core advection):   m, am, bm, cm, ps
# Tier 1b (optional):        qv
# Tier 2 (physics):          cmfmc, surface, tm5conv, temperature
# Tier 1c (temporal interp): dam, dbm, dm, dcm
#
# The advection path only requires Tier 1.  Everything else is optional and
# the reader returns `nothing` from the corresponding loaders when absent.
# ---------------------------------------------------------------------------

using Mmap
using JSON3

const _HEADER_SIZE_V1 = 4096
const _HEADER_SIZE_V2 = 16384

"""
    ERA5BinaryHeader

Parsed metadata from the binary file's JSON header.

Fields are grouped by tier to make the advection-vs-physics separation
explicit. An advection-only consumer reads only the Tier 1 fields.

# Tier 1 — core advection (always present)

- `version`, `header_bytes` — format metadata
- `Nx`, `Ny`, `Nz`, `Nt` — grid dimensions and window count
- `n_m`, `n_am`, `n_bm`, `n_cm`, `n_ps` — element counts per window
- `dt_seconds`, `half_dt_seconds` — time metadata
- `steps_per_met` — advection substeps per window
- `level_top`, `level_bot` — native vertical extent
- `A_ifc`, `B_ifc` — hybrid sigma-pressure interface coefficients
- `lons_f64`, `lats_f64` — cell-center coordinates [degrees]
- `mass_basis` — `:moist` or `:dry`, determines basis tag on loaded fluxes

# On-disk precision

- `on_disk_float_type` — `:Float32` or `:Float64`, from `float_type` header key
- `float_bytes` — 4 or 8, bytes per on-disk element

# Tier 1b — optional humidity

- `has_qv`, `n_qv` — specific humidity block

# Tier 1c — optional flux deltas (substep temporal interpolation)

- `has_flux_delta`, `n_dam`, `n_dbm`, `n_dm`, `n_dcm`

# Tier 2 — physics extensions (NOT part of advection contract)

- `has_cmfmc`, `n_cmfmc` — convective mass flux
- `has_surface`, `n_sfc` — surface fields (pblh, t2m, ustar, hflux)
- `has_tm5conv`, `n_tm5conv` — TM5 convection (entu, detu, entd, detd)
- `has_temperature`, `n_temperature` — model-level temperature

# Payload

- `elems_per_window` — total elements per window (sum of all present blocks)
"""
struct ERA5BinaryHeader
    # ---- format metadata ----
    version         :: Int
    header_bytes    :: Int
    on_disk_float_type :: Symbol   # :Float32 or :Float64
    float_bytes     :: Int         # 4 or 8

    # ---- Tier 1: core advection (required) ----
    Nx              :: Int
    Ny              :: Int
    Nz              :: Int
    Nt              :: Int
    n_m             :: Int         # Nx × Ny × Nz
    n_am            :: Int         # (Nx+1) × Ny × Nz
    n_bm            :: Int         # Nx × (Ny+1) × Nz
    n_cm            :: Int         # Nx × Ny × (Nz+1)
    n_ps            :: Int         # Nx × Ny
    dt_seconds      :: Float64
    half_dt_seconds :: Float64
    steps_per_met   :: Int
    level_top       :: Int
    level_bot       :: Int
    A_ifc           :: Vector{Float64}
    B_ifc           :: Vector{Float64}
    lons_f64        :: Vector{Float64}
    lats_f64        :: Vector{Float64}
    mass_basis      :: Symbol      # :moist or :dry

    # ---- Tier 1b: optional humidity ----
    has_qv          :: Bool
    n_qv            :: Int

    # ---- Tier 2: physics extensions ----
    has_cmfmc       :: Bool
    n_cmfmc         :: Int
    has_surface     :: Bool
    n_sfc           :: Int
    has_tm5conv     :: Bool
    n_tm5conv       :: Int
    has_temperature :: Bool
    n_temperature   :: Int

    # ---- Tier 1c: optional flux deltas ----
    has_flux_delta  :: Bool
    n_dam           :: Int
    n_dbm           :: Int
    n_dm            :: Int
    n_dcm           :: Int

    # ---- derived ----
    elems_per_window :: Int
end

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

"""
    ERA5BinaryReader{FT, DiskFT}

Mmap-based reader for ERA5 lat-lon mass-flux binary files (v1–v5).

# Type parameters
- `FT`     — working float type for output arrays (Float32 or Float64)
- `DiskFT` — on-disk float type, read from header (Float32 or Float64)

# Physical semantics of loaded fields

**`m`** (from `load_window!`) is the **total (moist) air mass per grid cell** [kg]
when `mass_basis = :moist` (the default for all v1–v4 files), defined as:

    m[i,j,k] = dp[k](i,j) × A[j] / g

where `dp[k]` is the total pressure thickness of layer k (from hybrid A/B
coefficients and surface pressure), `A[j]` is cell area, and `g` is gravity.
This is total air mass (dry + water vapor).

To obtain dry air mass: `m_dry = m × (1 - qv)`.

**`am`, `bm`, `cm`** are mass fluxes scaled to `half_dt_seconds` [kg per half-step].

# Usage
```julia
reader = ERA5BinaryReader("/path/to/binary.bin"; FT=Float64)
m, ps, fluxes = load_window!(reader, 1)
# fluxes :: StructuredFaceFluxState{MoistMassFluxBasis}  if mass_basis=:moist
# fluxes :: StructuredFaceFluxState{DryMassFluxBasis}    if mass_basis=:dry
close(reader)
```
"""
struct ERA5BinaryReader{FT, DiskFT}
    data    :: Vector{DiskFT}
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
has_cmfmc(r::ERA5BinaryReader) = r.header.has_cmfmc
has_surface(r::ERA5BinaryReader) = r.header.has_surface
has_tm5conv(r::ERA5BinaryReader) = r.header.has_tm5conv
has_temperature(r::ERA5BinaryReader) = r.header.has_temperature
mass_basis(r::ERA5BinaryReader) = r.header.mass_basis
A_ifc(r::ERA5BinaryReader) = r.header.A_ifc
B_ifc(r::ERA5BinaryReader) = r.header.B_ifc

# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

function _parse_on_disk_float_type(hdr)
    ft_str = string(get(hdr, :float_type, "Float32"))
    if ft_str == "Float64"
        return :Float64, 8
    else
        return :Float32, 4
    end
end

function _parse_mass_basis(hdr, version)
    basis_str = string(get(hdr, :mass_basis, "moist"))
    return basis_str == "dry" ? :dry : :moist
end

function _parse_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    version = Int(get(hdr, :version, 1))
    header_bytes = version >= 2 ? Int(get(hdr, :header_bytes, _HEADER_SIZE_V2)) : _HEADER_SIZE_V1

    disk_ft, fb = _parse_on_disk_float_type(hdr)

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

    _mass_basis = _parse_mass_basis(hdr, version)

    ERA5BinaryHeader(
        version, header_bytes, disk_ft, fb,
        nx, ny, nz, nt,
        n_m, n_am, n_bm, n_cm, n_ps,
        Float64(hdr.dt_seconds), Float64(hdr.half_dt_seconds),
        Int(hdr.steps_per_met_window), Int(hdr.level_top), Int(hdr.level_bot),
        _A_ifc, _B_ifc, lons, lats,
        _mass_basis,
        _has_qv, _n_qv,
        _has_cmfmc, _n_cmfmc,
        _has_surface, _n_sfc,
        _has_tm5conv, _n_tm5conv,
        _has_temperature, _n_temperature,
        _has_flux_delta, _n_dam, _n_dbm, _n_dm, _n_dcm,
        elems)
end

function _disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function ERA5BinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")

    read_sz = min(_HEADER_SIZE_V2, filesize(bin_path))
    raw = read(io, read_sz)
    header = _parse_header(raw)

    DiskFT = _disk_float_type(header.on_disk_float_type)

    total_elems = header.elems_per_window * header.Nt
    seek(io, header.header_bytes)
    data = Mmap.mmap(io, Vector{DiskFT}, total_elems, header.header_bytes)

    return ERA5BinaryReader{FT, DiskFT}(data, io, header, bin_path)
end

Base.close(r::ERA5BinaryReader) = close(r.io)

# ---------------------------------------------------------------------------
# Offset helpers
# ---------------------------------------------------------------------------

"""Offset (in elements) to the start of window `win`."""
@inline _window_offset(r::ERA5BinaryReader, win::Int) =
    (win - 1) * r.header.elems_per_window

"""Core fields offset past m + am + bm + cm + ps."""
@inline _core_offset(h::ERA5BinaryHeader) = h.n_m + h.n_am + h.n_bm + h.n_cm + h.n_ps

"""Offset past Tier 1 + Tier 1b + Tier 2 (to the start of flux deltas)."""
@inline _flux_delta_offset(h::ERA5BinaryHeader) =
    _core_offset(h) + h.n_qv + h.n_cmfmc +
    4 * h.n_tm5conv + 4 * h.n_sfc + h.n_temperature

# ---------------------------------------------------------------------------
# Tier 1: core advection loader
# ---------------------------------------------------------------------------

"""
    load_window!(reader, win; m=..., ps=..., am=..., bm=..., cm=...)
        -> (m, ps, StructuredFaceFluxState{<:AbstractMassFluxBasis})

Load core advection fields for window `win`.

Returns `(m, ps, fluxes)` where `fluxes` carries the basis tag matching
the file's `mass_basis` header key:
- `StructuredFaceFluxState{MoistMassFluxBasis}` when `mass_basis = :moist`
- `StructuredFaceFluxState{DryMassFluxBasis}` when `mass_basis = :dry`

**`m`** is **total (moist) air mass per cell [kg]** when `mass_basis = :moist`.
See the spec in `docs/BINARY_FORMAT_V5.md` for the definition.

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

    fluxes = _make_fluxes(h.mass_basis, am, bm, cm)
    return (m, ps, fluxes)
end

_make_fluxes(::Val{:moist}, am, bm, cm) = StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm)
_make_fluxes(::Val{:dry},   am, bm, cm) = StructuredFaceFluxState{DryMassFluxBasis}(am, bm, cm)
_make_fluxes(basis::Symbol, am, bm, cm) = _make_fluxes(Val(basis), am, bm, cm)

# ---------------------------------------------------------------------------
# Tier 1b: specific humidity
# ---------------------------------------------------------------------------

"""
    load_qv_window!(reader, win; qv=...) -> Union{Array, Nothing}

Load specific humidity for window `win`.  Returns `nothing` if the binary
does not contain QV (`include_qv = false`).
"""
function load_qv_window!(reader::ERA5BinaryReader{FT}, win::Int;
                          qv = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz)
                         ) where FT
    reader.header.has_qv || return nothing
    o = _window_offset(reader, win) + _core_offset(reader.header)
    copyto!(qv, 1, reader.data, o + 1, reader.header.n_qv)
    return qv
end

# ---------------------------------------------------------------------------
# Tier 1c: flux deltas (substep interpolation)
# ---------------------------------------------------------------------------

"""
    load_flux_delta_window!(reader, win; dam=..., dbm=..., dm=..., dcm=...)
        -> Union{NamedTuple, Nothing}

Load v4+ flux deltas for substep interpolation.  Returns `nothing` if the
binary lacks flux delta blocks.

Flux deltas are NOT part of the core advection contract.  They improve
temporal interpolation accuracy but are not required.
"""
function load_flux_delta_window!(reader::ERA5BinaryReader{FT}, win::Int;
                                  dam = Array{FT}(undef, reader.header.Nx + 1, reader.header.Ny, reader.header.Nz),
                                  dbm = Array{FT}(undef, reader.header.Nx, reader.header.Ny + 1, reader.header.Nz),
                                  dm  = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                                  dcm = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz + 1)
                                 ) where FT
    h = reader.header
    h.has_flux_delta || return nothing

    o = _window_offset(reader, win) + _flux_delta_offset(h)

    copyto!(dam, 1, reader.data, o + 1, h.n_dam); o += h.n_dam
    copyto!(dbm, 1, reader.data, o + 1, h.n_dbm); o += h.n_dbm
    copyto!(dm,  1, reader.data, o + 1, h.n_dm);  o += h.n_dm
    if h.n_dcm > 0
        copyto!(dcm, 1, reader.data, o + 1, h.n_dcm)
    end

    return (; dam, dbm, dm, dcm)
end

# ---------------------------------------------------------------------------
# Tier 2: physics extension loaders
#
# These are outside the advection contract.  They return `nothing` when the
# corresponding block is absent.  An advection-only consumer never calls them.
# ---------------------------------------------------------------------------

"""
    load_cmfmc_window!(reader, win; cmfmc=...)
        -> Union{Array, Nothing}

Load convective mass flux for window `win` (Tier 2).
Shape: `(Nx, Ny, Nz+1)`.
"""
function load_cmfmc_window!(reader::ERA5BinaryReader{FT}, win::Int;
                              cmfmc = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz + 1)
                             ) where FT
    h = reader.header
    h.has_cmfmc || return nothing
    o = _window_offset(reader, win) + _core_offset(h) + h.n_qv
    copyto!(cmfmc, 1, reader.data, o + 1, h.n_cmfmc)
    return cmfmc
end

"""
    load_surface_window!(reader, win; pblh=..., t2m=..., ustar=..., hflux=...)
        -> Union{NamedTuple, Nothing}

Load surface diagnostic fields for window `win` (Tier 2).
Each field has shape `(Nx, Ny)`.
"""
function load_surface_window!(reader::ERA5BinaryReader{FT}, win::Int;
                                pblh  = Array{FT}(undef, reader.header.Nx, reader.header.Ny),
                                t2m   = Array{FT}(undef, reader.header.Nx, reader.header.Ny),
                                ustar = Array{FT}(undef, reader.header.Nx, reader.header.Ny),
                                hflux = Array{FT}(undef, reader.header.Nx, reader.header.Ny)
                               ) where FT
    h = reader.header
    h.has_surface || return nothing
    o = _window_offset(reader, win) + _core_offset(h) + h.n_qv + h.n_cmfmc +
        4 * h.n_tm5conv   # surface comes after tm5conv in v2/v3 layout
    copyto!(pblh,  1, reader.data, o + 1, h.n_sfc); o += h.n_sfc
    copyto!(t2m,   1, reader.data, o + 1, h.n_sfc); o += h.n_sfc
    copyto!(ustar, 1, reader.data, o + 1, h.n_sfc); o += h.n_sfc
    copyto!(hflux, 1, reader.data, o + 1, h.n_sfc)
    return (; pblh, t2m, ustar, hflux)
end

"""
    load_tm5conv_window!(reader, win; entu=..., detu=..., entd=..., detd=...)
        -> Union{NamedTuple, Nothing}

Load TM5 convection fields for window `win` (Tier 2).
Each field has shape `(Nx, Ny, Nz)`.
"""
function load_tm5conv_window!(reader::ERA5BinaryReader{FT}, win::Int;
                                entu = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                                detu = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                                entd = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz),
                                detd = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz)
                               ) where FT
    h = reader.header
    h.has_tm5conv || return nothing
    o = _window_offset(reader, win) + _core_offset(h) + h.n_qv + h.n_cmfmc
    copyto!(entu, 1, reader.data, o + 1, h.n_tm5conv); o += h.n_tm5conv
    copyto!(detu, 1, reader.data, o + 1, h.n_tm5conv); o += h.n_tm5conv
    copyto!(entd, 1, reader.data, o + 1, h.n_tm5conv); o += h.n_tm5conv
    copyto!(detd, 1, reader.data, o + 1, h.n_tm5conv)
    return (; entu, detu, entd, detd)
end

"""
    load_temperature_window!(reader, win; temperature=...)
        -> Union{Array, Nothing}

Load model-level temperature for window `win` (Tier 2).
Shape: `(Nx, Ny, Nz)`.
"""
function load_temperature_window!(reader::ERA5BinaryReader{FT}, win::Int;
                                    temperature = Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.Nz)
                                   ) where FT
    h = reader.header
    h.has_temperature || return nothing
    o = _window_offset(reader, win) + _core_offset(h) + h.n_qv + h.n_cmfmc +
        4 * h.n_tm5conv + 4 * h.n_sfc
    copyto!(temperature, 1, reader.data, o + 1, h.n_temperature)
    return temperature
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export ERA5BinaryReader, ERA5BinaryHeader
export load_window!, load_qv_window!, load_flux_delta_window!
export load_cmfmc_window!, load_surface_window!, load_tm5conv_window!
export load_temperature_window!
export window_count, has_qv, has_flux_delta, has_cmfmc
export has_surface, has_tm5conv, has_temperature
export mass_basis, A_ifc, B_ifc
