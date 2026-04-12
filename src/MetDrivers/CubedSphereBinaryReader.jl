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
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))

    sections_raw = collect(hdr.payload_sections)
    payload_sections = Symbol.(lowercase.(String.(sections_raw)))
    elems_per_window = Int(hdr.elems_per_window)

    cs_header = CubedSphereBinaryHeader(
        Nc, npanel, nlevel, nwindow, header_bytes, float_bytes,
        dt_met, steps_per_window, mass_basis, A_ifc, B_ifc,
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
    load_cs_window(reader, win) -> (panels_m, panels_ps, panels_am, panels_bm, panels_cm)

Load window `win` from a cubed-sphere transport binary. Returns NTuples of
per-panel arrays.
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
        else
            # Skip unknown sections
            n = _cs_section_elements(h, section)
            o += n
        end
    end

    return panels_m, panels_ps, panels_am, panels_bm, panels_cm
end

"""
    cs_window_count(reader) -> Int

Number of time windows in the binary.
"""
cs_window_count(reader::CubedSphereBinaryReader) = reader.header.nwindow

export CubedSphereBinaryReader, CubedSphereBinaryHeader
export load_cs_window, cs_window_count
