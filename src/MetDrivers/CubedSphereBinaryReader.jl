# ---------------------------------------------------------------------------
# CubedSphereBinaryReader — reads CS transport binaries with 6-panel layout
#
# The standard TransportBinaryReader assumes (Nx, Ny, Nz) structured layout.
# CS binaries store 6 panels packed sequentially: [panel1][panel2]...[panel6]
# for each payload section. This reader loads all 6 panels into NTuples.
#
# Binary format: same MFLX header as LatLon/RG, but with:
#   grid_type = "cubed_sphere"
#   horizontal_topology = "StructuredDirectional"
#   Nc (cells per panel edge), npanel = 6
#   Data packed: per section, [panel 1 ... panel 6] contiguous
# ---------------------------------------------------------------------------

using JSON3
using Mmap

"""
    CubedSphereBinaryHeader

Parsed header for a cubed-sphere transport binary.
"""
struct CubedSphereBinaryHeader
    Nc               :: Int
    npanel           :: Int
    nlevel           :: Int
    nwindow          :: Int
    header_bytes     :: Int
    float_bytes      :: Int
    dt_met_seconds   :: Float64
    steps_per_window :: Int
    mass_basis       :: Symbol
    panel_convention :: Symbol   # :gnomonic or :geos_native
    cs_definition    :: Symbol
    coordinate_law   :: Symbol
    center_law       :: Symbol
    longitude_offset_deg :: Float64
    A_ifc            :: Vector{Float64}
    B_ifc            :: Vector{Float64}
    payload_sections :: Vector{Symbol}
    elems_per_window :: Int
    raw_header       :: Dict{String, Any}
end

"""
    CubedSphereBinaryReader{FT}

Reader for cubed-sphere transport binaries. Data is memory-mapped for
zero-copy access to per-window payloads.
"""
struct CubedSphereBinaryReader{FT}
    data    :: Vector{FT}
    io      :: IOStream
    header  :: CubedSphereBinaryHeader
    path    :: String
end

function _cs_on_disk_float_type(path::AbstractString)
    open(path, "r") do io
        raw = read(io, min(filesize(path), 262144))
        json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
        hdr = JSON3.read(String(raw[1:json_end]))
        float_bytes = Int(get(hdr, :float_bytes, 4))
        return float_bytes == 8 ? Float64 : Float32
    end
end

# Plan 40 Commit 5: bridge for `inspect_binary` in TransportBinary.jl.
# Lives here because the CS reader type is defined in this file but the
# inspector lives in TransportBinary.jl (which is loaded first).  Use the
# on-disk float type; otherwise inspecting a large Float32 CS binary eagerly
# converts the entire mmap to Float64.
_open_cubed_sphere_binary_reader(path::AbstractString) =
    CubedSphereBinaryReader(String(path); FT = _cs_on_disk_float_type(path))

function CubedSphereBinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float64)
    io = open(bin_path, "r")

    # Read header
    magic_bytes = read(io, 4)
    seek(io, 0)

    # Detect header size from JSON
    raw = read(io, min(filesize(bin_path), 262144))
    json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
    hdr = JSON3.read(String(raw[1:json_end]))

    header_bytes = Int(get(hdr, :header_bytes, 16384))
    float_bytes = Int(get(hdr, :float_bytes, 8))
    Nc = Int(get(hdr, :Nc, get(hdr, :Nx, 0)))
    npanel = Int(get(hdr, :npanel, 6))
    nlevel = Int(hdr.nlevel)
    nwindow = Int(hdr.nwindow)
    dt_met = Float64(hdr.dt_met_seconds)
    steps_per_window = Int(hdr.steps_per_window)
    mass_basis = Symbol(lowercase(String(get(hdr, :mass_basis, "moist"))))
    panel_convention_str = lowercase(String(get(hdr, :panel_convention, "gnomonic")))
    panel_convention = if panel_convention_str in ("gnomonic", "gnomic")
        :gnomonic
    elseif panel_convention_str in ("geos_native", "geosnative", "geos-native")
        :geos_native
    else
        @warn "Unknown panel_convention '$panel_convention_str' in binary header, defaulting to gnomonic"
        :gnomonic
    end
    default_geometry = panel_convention === :geos_native ?
        (; cs_definition = :gmao_equal_distance,
           coordinate_law = :gmao_equal_distance_gnomonic,
           center_law = :four_corner_normalized,
           longitude_offset_deg = -10.0) :
        (; cs_definition = :equiangular_gnomonic,
           coordinate_law = :equiangular_gnomonic,
           center_law = :angular_midpoint,
           longitude_offset_deg = 0.0)
    cs_definition = Symbol(replace(lowercase(String(get(hdr, :cs_definition,
        String(default_geometry.cs_definition)))), '-' => '_', ' ' => '_'))
    coordinate_law = Symbol(replace(lowercase(String(get(hdr, :cs_coordinate_law,
        get(hdr, :coordinate_law, String(default_geometry.coordinate_law))))), '-' => '_', ' ' => '_'))
    center_law = Symbol(replace(lowercase(String(get(hdr, :cs_center_law,
        get(hdr, :center_law, String(default_geometry.center_law))))), '-' => '_', ' ' => '_'))
    longitude_offset_deg = Float64(get(hdr, :longitude_offset_deg,
                                       default_geometry.longitude_offset_deg))
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))

    sections_raw = collect(hdr.payload_sections)
    payload_sections = Symbol.(lowercase.(String.(sections_raw)))
    elems_per_window = Int(hdr.elems_per_window)

    cs_header = CubedSphereBinaryHeader(
        Nc, npanel, nlevel, nwindow, header_bytes, float_bytes,
        dt_met, steps_per_window, mass_basis, panel_convention,
        cs_definition, coordinate_law, center_law, longitude_offset_deg,
        A_ifc, B_ifc,
        payload_sections, elems_per_window,
        Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr))
    )

    # Memory-map the data region
    DiskFT = float_bytes == 4 ? Float32 : Float64
    data_bytes = filesize(bin_path) - header_bytes
    n_elems = data_bytes ÷ sizeof(DiskFT)
    seek(io, 0)
    raw_data = Mmap.mmap(io, Vector{DiskFT}, n_elems, header_bytes)

    # Convert to requested FT if needed
    data = FT === DiskFT ? raw_data : FT.(raw_data)

    return CubedSphereBinaryReader{FT}(data, io, cs_header, bin_path)
end

function Base.summary(r::CubedSphereBinaryReader{FT}) where FT
    h = r.header
    disk_ft = h.float_bytes == 8 ? Float64 : Float32
    return string(
        "CubedSphereBinaryReader{", FT, "<-", disk_ft, "}(",
        basename(r.path), ", C", h.Nc, " x ", h.nlevel, ", ",
        h.nwindow, " windows)"
    )
end

function Base.show(io::IO, r::CubedSphereBinaryReader)
    h = r.header
    disk_ft = h.float_bytes == 8 ? Float64 : Float32
    print(io, summary(r), "\n",
          "├── path:          ", r.path, "\n",
          "├── geometry:      C", h.Nc, ", panels=", h.npanel,
          ", convention=", h.panel_convention, ", definition=", h.cs_definition, "\n",
          "├── storage:       ", disk_ft, " on disk, load as ", eltype(r.data), "\n",
          "├── basis:         ", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=", h.steps_per_window, "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "└── windows:       ", h.nwindow)
end

function _cs_coordinate_law_from_symbol(sym::Symbol)
    sym in (:equiangular, :equiangular_gnomonic, :legacy) &&
        return EquiangularGnomonic()
    sym in (:gmao, :geos, :gmao_equal_distance,
            :gmao_equal_distance_gnomonic, :geos_equal_distance_gnomonic) &&
        return GMAOEqualDistanceGnomonic()
    @warn "Unrecognised CS coordinate law :$sym, defaulting to equiangular"
    return EquiangularGnomonic()
end

function _cs_center_law_from_symbol(sym::Symbol)
    sym in (:angular_midpoint, :midpoint, :legacy) &&
        return AngularMidpointCenter()
    sym in (:four_corner_normalized, :cell_center2, :geos_cell_center2) &&
        return FourCornerNormalizedCenter()
    @warn "Unrecognised CS center law :$sym, defaulting to angular_midpoint"
    return AngularMidpointCenter()
end

# ---------------------------------------------------------------------------
# Section element counts
# ---------------------------------------------------------------------------

function _cs_section_elements(h::CubedSphereBinaryHeader, section::Symbol)
    Nc, Nz, np = h.Nc, h.nlevel, h.npanel
    if section === :m
        return np * Nc * Nc * Nz
    elseif section === :am
        return np * (Nc + 1) * Nc * Nz
    elseif section === :bm
        return np * Nc * (Nc + 1) * Nz
    elseif section === :cm
        return np * Nc * Nc * (Nz + 1)
    elseif section === :ps
        return np * Nc * Nc
    elseif _is_pbl_surface_payload_section(section)
        return np * Nc * Nc
    elseif section === :cmfmc
        return np * Nc * Nc * (Nz + 1)
    elseif section === :dtrain
        return np * Nc * Nc * Nz
    # TM5 convection (plan 23 Commit 3) — four layer-center fields.
    elseif section in (:entu, :detu, :entd, :detd)
        return np * Nc * Nc * Nz
    elseif section in (:qv, :qv_start, :qv_end, :dm)
        return np * Nc * Nc * Nz
    elseif section in (:dam,)
        return np * (Nc + 1) * Nc * Nz
    elseif section in (:dbm,)
        return np * Nc * (Nc + 1) * Nz
    elseif section in (:dcm,)
        return np * Nc * Nc * (Nz + 1)
    else
        error("Unknown CS binary section: $section")
    end
end

# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------

"""
    load_cs_window(reader, win) -> NamedTuple

Load window `win` from a cubed-sphere transport binary. Returns NTuples of
per-panel arrays plus optional `cmfmc` / `dtrain` payloads when they are
present in the binary.
"""
function load_cs_window(reader::CubedSphereBinaryReader{FT}, win::Int) where FT
    h = reader.header
    Nc, Nz, np = h.Nc, h.nlevel, h.npanel

    # Compute window offset in elements
    win_offset = (win - 1) * h.elems_per_window

    panels_m  = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np)
    panels_ps = ntuple(_ -> Array{FT}(undef, Nc, Nc), np)
    panels_am = ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), np)
    panels_bm = ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), np)
    panels_cm = ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz + 1), np)
    surface_present = all(s in h.payload_sections for s in _PBL_SURFACE_PAYLOAD_SECTIONS)
    surface_partial = any(s in h.payload_sections for s in _PBL_SURFACE_PAYLOAD_SECTIONS) && !surface_present
    if surface_partial
        legacy_hflux = :hflux in h.payload_sections && !(:pbl_hflux in h.payload_sections)
        msg = "CS binary has a partial PBL surface payload; expected all of pblh, ustar, pbl_hflux, t2m"
        legacy_hflux && (msg *= "\n  This binary appears to be pre-2026-05-01 (commit 66bbce3): the on-disk PBL sensible-heat section is `:hflux` rather than the renamed `:pbl_hflux`. Regenerate via scripts/preprocessing/preprocess_transport_binary.jl + regrid_ll_transport_binary_to_cs.jl.")
        throw(ArgumentError(msg))
    end
    panels_pblh  = surface_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc), np) : nothing
    panels_ustar = surface_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc), np) : nothing
    panels_hflux = surface_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc), np) : nothing
    panels_t2m   = surface_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc), np) : nothing
    panels_cmfmc = :cmfmc in h.payload_sections ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz + 1), np) : nothing
    panels_dtrain = :dtrain in h.payload_sections ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np) : nothing

    # TM5 convection fields — all four must be present together or
    # all four absent. The runtime `_validate_convection_window!`
    # rejects a partial payload, so this block trusts the header.
    tm5_present = all(s in h.payload_sections for s in (:entu, :detu, :entd, :detd))
    panels_entu = tm5_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np) : nothing
    panels_detu = tm5_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np) : nothing
    panels_entd = tm5_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np) : nothing
    panels_detd = tm5_present ? ntuple(_ -> Array{FT}(undef, Nc, Nc, Nz), np) : nothing

    o = win_offset
    for section in h.payload_sections
        if section === :m
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_m[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :am
            for p in 1:np
                n = (Nc + 1) * Nc * Nz
                copyto!(panels_am[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :bm
            for p in 1:np
                n = Nc * (Nc + 1) * Nz
                copyto!(panels_bm[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :cm
            for p in 1:np
                n = Nc * Nc * (Nz + 1)
                copyto!(panels_cm[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :ps
            for p in 1:np
                n = Nc * Nc
                copyto!(panels_ps[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :pblh
            for p in 1:np
                n = Nc * Nc
                copyto!(panels_pblh[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :ustar
            for p in 1:np
                n = Nc * Nc
                copyto!(panels_ustar[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :pbl_hflux
            for p in 1:np
                n = Nc * Nc
                copyto!(panels_hflux[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :t2m
            for p in 1:np
                n = Nc * Nc
                copyto!(panels_t2m[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :cmfmc
            for p in 1:np
                n = Nc * Nc * (Nz + 1)
                copyto!(panels_cmfmc[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :dtrain
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_dtrain[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :entu
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_entu[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :detu
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_detu[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :entd
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_entd[p], 1, reader.data, o + 1, n)
                o += n
            end
        elseif section === :detd
            for p in 1:np
                n = Nc * Nc * Nz
                copyto!(panels_detd[p], 1, reader.data, o + 1, n)
                o += n
            end
        else
            # Skip unknown sections
            n = _cs_section_elements(h, section)
            o += n
        end
    end

    # TM5 fields are returned as a NamedTuple when present, to match
    # the runtime `ConvectionForcing.tm5_fields` contract. Absent
    # means the binary doesn't carry TM5 convection data.
    tm5_fields = tm5_present ?
        (entu = panels_entu, detu = panels_detu,
         entd = panels_entd, detd = panels_detd) :
        nothing
    surface = surface_present ?
        PBLSurfaceForcing(panels_pblh, panels_ustar, panels_hflux, panels_t2m) :
        nothing

    return (
        m = panels_m,
        ps = panels_ps,
        am = panels_am,
        bm = panels_bm,
        cm = panels_cm,
        surface = surface,
        cmfmc = panels_cmfmc,
        dtrain = panels_dtrain,
        tm5_fields = tm5_fields,
    )
end

function load_flux_delta_window!(reader::CubedSphereBinaryReader{FT}, win::Int;
                                 dm = nothing) where FT
    h = reader.header
    :dm in h.payload_sections || return nothing

    dm = isnothing(dm) ? ntuple(_ -> Array{FT}(undef, h.Nc, h.Nc, h.nlevel), h.npanel) : dm
    o = (win - 1) * h.elems_per_window

    for section in h.payload_sections
        if section === :dm
            for p in 1:h.npanel
                n = h.Nc * h.Nc * h.nlevel
                copyto!(dm[p], 1, reader.data, o + 1, n)
                o += n
            end
            return (; dm)
        end
        o += _cs_section_elements(h, section)
    end

    return nothing
end

"""
    load_surface_window!(reader::CubedSphereBinaryReader, win) -> PBLSurfaceForcing | nothing

Load the raw PBL surface payload for one CS window. This is a convenience
wrapper over `load_cs_window`; callers that already need the advection fields
should use `load_cs_window(reader, win).surface` to avoid a second payload read.
"""
load_surface_window!(reader::CubedSphereBinaryReader, win::Int; kwargs...) =
    load_cs_window(reader, win).surface

"""
    cs_window_count(reader) -> Int

Number of time windows in the binary.
"""
cs_window_count(reader::CubedSphereBinaryReader) = reader.header.nwindow

"""
    mesh_convention(reader::CubedSphereBinaryReader) -> AbstractCubedSpherePanelConvention

Return the panel-numbering convention declared in the binary header.

Returns `GnomonicPanelConvention()` for ERA5-CS binaries and
`GEOSNativePanelConvention()` for GEOS-FP/IT binaries tagged with
`panel_convention="geos_native"`. Callers should pass the result directly to
`CubedSphereMesh(; convention=mesh_convention(reader))` to guarantee that the
halo exchange uses the correct edge-to-edge connectivity table.
"""
function mesh_convention(reader::CubedSphereBinaryReader)
    conv = reader.header.panel_convention
    if conv === :gnomonic
        return GnomonicPanelConvention()
    elseif conv === :geos_native
        return GEOSNativePanelConvention()
    else
        @warn "Unrecognised panel_convention :$conv in binary, defaulting to GnomonicPanelConvention"
        return GnomonicPanelConvention()
    end
end

"""
    mesh_definition(reader::CubedSphereBinaryReader) -> CubedSphereDefinition

Return the full cubed-sphere geometry definition declared in the binary
header. New binaries record `cs_coordinate_law`, `cs_center_law`,
`panel_convention`, and `longitude_offset_deg`; older binaries are upgraded by
convention (`geos_native` -> GMAO equal-distance, `gnomonic` -> legacy
equiangular).
"""
function mesh_definition(reader::CubedSphereBinaryReader)
    h = reader.header
    return CubedSphereDefinition(_cs_coordinate_law_from_symbol(h.coordinate_law),
                                 _cs_center_law_from_symbol(h.center_law),
                                 mesh_convention(reader);
                                 longitude_offset_deg = h.longitude_offset_deg,
                                 tag = h.cs_definition)
end

export CubedSphereBinaryReader, CubedSphereBinaryHeader
export load_cs_window, load_surface_window!, cs_window_count, mesh_convention, mesh_definition
