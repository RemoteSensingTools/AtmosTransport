using Mmap
using JSON3
using ..Architectures: CPU
import ..State: mass_basis

"""
    TransportBinaryHeader

Metadata for a topology-generic preprocessed transport binary.
"""
struct TransportBinaryHeader
    format_version       :: Int
    header_bytes         :: Int
    on_disk_float_type   :: Symbol
    float_bytes          :: Int
    grid_type            :: Symbol
    horizontal_topology  :: Symbol
    ncell                :: Int
    nface_h              :: Int
    nlevel               :: Int
    nwindow              :: Int
    dt_met_seconds       :: Float64
    half_dt_seconds      :: Float64
    steps_per_window     :: Int
    A_ifc                :: Vector{Float64}
    B_ifc                :: Vector{Float64}
    mass_basis           :: Symbol
    payload_sections     :: Vector{Symbol}
    include_qv           :: Bool
    include_qv_endpoints :: Bool
    n_qv                 :: Int
    n_qv_start           :: Int
    n_qv_end             :: Int
    n_geometry_elems     :: Int
    elems_per_window     :: Int
    Nx                   :: Int
    Ny                   :: Int
    lons_f64             :: Vector{Float64}
    lats_f64             :: Vector{Float64}
    nlat                 :: Int
    ring_latitudes_f64   :: Vector{Float64}
    nlon_per_ring        :: Vector{Int}
end

"""
    TransportBinaryReader{FT, DiskFT}

Reader for topology-generic preprocessed transport binaries.

Currently implemented:
- `grid_type = :reduced_gaussian`
- `horizontal_topology = :faceindexed`
"""
struct TransportBinaryReader{FT, DiskFT}
    data   :: Vector{DiskFT}
    io     :: IOStream
    header :: TransportBinaryHeader
    path   :: String
end

window_count(r::TransportBinaryReader) = r.header.nwindow
mass_basis(r::TransportBinaryReader) = r.header.mass_basis
grid_type(r::TransportBinaryReader) = r.header.grid_type
horizontal_topology(r::TransportBinaryReader) = r.header.horizontal_topology
A_ifc(r::TransportBinaryReader) = r.header.A_ifc
B_ifc(r::TransportBinaryReader) = r.header.B_ifc
has_qv(r::TransportBinaryReader) = r.header.include_qv || r.header.include_qv_endpoints
has_qv_endpoints(r::TransportBinaryReader) = r.header.include_qv_endpoints

function _transport_disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function _transport_parse_on_disk_float_type(hdr)
    ft_str = string(get(hdr, :float_type, "Float32"))
    if ft_str == "Float64"
        return :Float64, 8
    else
        return :Float32, 4
    end
end

function _transport_parse_mass_basis(hdr)
    basis_str = lowercase(string(get(hdr, :mass_basis, "moist")))
    return basis_str == "dry" ? :dry : :moist
end

_transport_parse_grid_type(hdr) = Symbol(lowercase(string(hdr.grid_type)))
_transport_parse_topology(hdr) = Symbol(lowercase(string(hdr.horizontal_topology)))

function _transport_parse_sections(hdr)
    sections = haskey(hdr, :payload_sections) ? collect(hdr.payload_sections) : ["m", "hflux", "cm", "ps"]
    return Symbol.(lowercase.(String.(sections)))
end

function _transport_header_axis(hdr, n::Int, key1::Symbol, key2::Symbol)
    if haskey(hdr, key1)
        return Float64.(collect(getproperty(hdr, key1)))
    elseif haskey(hdr, key2)
        return Float64.(collect(getproperty(hdr, key2)))
    elseif n == 0
        return Float64[]
    else
        error("Transport binary header missing $(key1) / $(key2)")
    end
end

function _parse_transport_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    haskey(hdr, :format_version) ||
        error("TransportBinaryReader requires the topology-generic binary family header (`format_version` missing)")

    format_version = Int(hdr.format_version)
    header_bytes = Int(get(hdr, :header_bytes, 16384))
    disk_ft, float_bytes = _transport_parse_on_disk_float_type(hdr)
    grid_type = _transport_parse_grid_type(hdr)
    topology = _transport_parse_topology(hdr)
    ncell = Int(hdr.ncell)
    nface_h = Int(hdr.nface_h)
    nlevel = Int(hdr.nlevel)
    nwindow = Int(hdr.nwindow)
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))
    payload_sections = _transport_parse_sections(hdr)
    include_qv = Bool(get(hdr, :include_qv, false))
    include_qv_endpoints = Bool(get(hdr, :include_qv_endpoints, false))
    n_qv = Int(get(hdr, :n_qv, include_qv ? ncell * nlevel : 0))
    n_qv_start = Int(get(hdr, :n_qv_start, include_qv_endpoints ? ncell * nlevel : 0))
    n_qv_end = Int(get(hdr, :n_qv_end, include_qv_endpoints ? ncell * nlevel : 0))
    n_geometry_elems = Int(get(hdr, :n_geometry_elems, 0))
    elems_per_window = Int(hdr.elems_per_window)

    Nx = haskey(hdr, :Nx) ? Int(hdr.Nx) : 0
    Ny = haskey(hdr, :Ny) ? Int(hdr.Ny) : 0
    lons_f64 = _transport_header_axis(hdr, Nx, :lons, :lon_centers)
    lats_f64 = _transport_header_axis(hdr, Ny, :lats, :lat_centers)

    nlon_per_ring = haskey(hdr, :nlon_per_ring) ? Int.(collect(hdr.nlon_per_ring)) : Int[]
    ring_latitudes = if haskey(hdr, :latitudes)
        Float64.(collect(hdr.latitudes))
    elseif haskey(hdr, :ring_latitudes)
        Float64.(collect(hdr.ring_latitudes))
    else
        Float64[]
    end
    nlat = haskey(hdr, :nlat) ? Int(hdr.nlat) : length(ring_latitudes)

    return TransportBinaryHeader(
        format_version,
        header_bytes,
        disk_ft,
        float_bytes,
        grid_type,
        topology,
        ncell,
        nface_h,
        nlevel,
        nwindow,
        Float64(hdr.dt_met_seconds),
        Float64(hdr.half_dt_seconds),
        Int(hdr.steps_per_window),
        A_ifc,
        B_ifc,
        _transport_parse_mass_basis(hdr),
        payload_sections,
        include_qv,
        include_qv_endpoints,
        n_qv,
        n_qv_start,
        n_qv_end,
        n_geometry_elems,
        elems_per_window,
        Nx,
        Ny,
        lons_f64,
        lats_f64,
        nlat,
        ring_latitudes,
        nlon_per_ring,
    )
end

function TransportBinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")
    read_sz = min(16384, filesize(bin_path))
    raw = read(io, read_sz)
    header = _parse_transport_header(raw)

    DiskFT = _transport_disk_float_type(header.on_disk_float_type)
    total_elems = header.n_geometry_elems + header.elems_per_window * header.nwindow
    seek(io, header.header_bytes)
    data = Mmap.mmap(io, Vector{DiskFT}, total_elems, header.header_bytes)

    return TransportBinaryReader{FT, DiskFT}(data, io, header, bin_path)
end

Base.close(r::TransportBinaryReader) = close(r.io)

@inline _transport_window_offset(r::TransportBinaryReader, win::Int) =
    r.header.n_geometry_elems + (win - 1) * r.header.elems_per_window

function _transport_section_elements(h::TransportBinaryHeader, section::Symbol)
    if section === :m
        return h.ncell * h.nlevel
    elseif section === :hflux
        return h.nface_h * h.nlevel
    elseif section === :cm
        return h.ncell * (h.nlevel + 1)
    elseif section === :ps
        return h.ncell
    elseif section === :qv
        return h.n_qv
    elseif section === :qv_start
        return h.n_qv_start
    elseif section === :qv_end
        return h.n_qv_end
    else
        error("Unsupported payload section: $section")
    end
end

function _transport_make_fluxes(::Val{:dry}, hflux, cm)
    return FaceIndexedFluxState{DryMassFluxBasis}(hflux, cm)
end

function _transport_make_fluxes(::Val{:moist}, hflux, cm)
    return FaceIndexedFluxState{MoistMassFluxBasis}(hflux, cm)
end

_transport_make_fluxes(basis::Symbol, hflux, cm) = _transport_make_fluxes(Val(basis), hflux, cm)


_transport_payload_sections(window) = begin
    sections = Symbol[:m, :hflux, :cm, :ps]
    haskey(window, :qv) && push!(sections, :qv)
    haskey(window, :qv_start) && push!(sections, :qv_start)
    haskey(window, :qv_end) && push!(sections, :qv_end)
    return sections
end

_transport_basis_symbol(sym::Symbol) = lowercase(String(sym)) == "dry" ? :dry : :moist
_transport_basis_symbol(::DryBasis) = :dry
_transport_basis_symbol(::MoistBasis) = :moist

_transport_window_mass(window) =
    haskey(window, :m) ? window.m :
    haskey(window, :state) ? window.state.air_mass :
    error("transport-binary window is missing `m` or `state`")

_transport_window_hflux(window) =
    haskey(window, :hflux) ? window.hflux :
    haskey(window, :fluxes) ? window.fluxes.horizontal_flux :
    error("transport-binary window is missing `hflux` or `fluxes`")

_transport_window_cm(window) =
    haskey(window, :cm) ? window.cm :
    haskey(window, :fluxes) ? window.fluxes.cm :
    error("transport-binary window is missing `cm` or `fluxes`")

function _transport_validate_reduced_window(window, ncell::Int, nface_h::Int, nlevel::Int, basis_sym::Symbol)
    m = _transport_window_mass(window)
    hflux = _transport_window_hflux(window)
    cm = _transport_window_cm(window)
    ps = window.ps

    size(m) == (ncell, nlevel) ||
        throw(DimensionMismatch("window m has size $(size(m)), expected ($(ncell), $(nlevel))"))
    size(hflux) == (nface_h, nlevel) ||
        throw(DimensionMismatch("window hflux has size $(size(hflux)), expected ($(nface_h), $(nlevel))"))
    size(cm) == (ncell, nlevel + 1) ||
        throw(DimensionMismatch("window cm has size $(size(cm)), expected ($(ncell), $(nlevel + 1))"))
    size(ps) == (ncell,) ||
        throw(DimensionMismatch("window ps has size $(size(ps)), expected ($(ncell),)"))

    if haskey(window, :qv)
        size(window.qv) == (ncell, nlevel) ||
            throw(DimensionMismatch("window qv has size $(size(window.qv)), expected ($(ncell), $(nlevel))"))
    end
    if haskey(window, :qv_start)
        size(window.qv_start) == (ncell, nlevel) ||
            throw(DimensionMismatch("window qv_start has size $(size(window.qv_start)), expected ($(ncell), $(nlevel))"))
    end
    if haskey(window, :qv_end)
        size(window.qv_end) == (ncell, nlevel) ||
            throw(DimensionMismatch("window qv_end has size $(size(window.qv_end)), expected ($(ncell), $(nlevel))"))
    end

    if haskey(window, :state)
        _transport_basis_symbol(mass_basis(window.state)) == basis_sym ||
            throw(ArgumentError("window state basis does not match requested transport binary basis $(basis_sym)"))
    end
    if haskey(window, :fluxes)
        _transport_basis_symbol(mass_basis(window.fluxes)) == basis_sym ||
            throw(ArgumentError("window flux basis does not match requested transport binary basis $(basis_sym)"))
    end

    return nothing
end

function write_transport_binary(path::AbstractString,
                                grid::AtmosGrid{<:ReducedGaussianMesh},
                                windows::AbstractVector;
                                FT::Type{<:AbstractFloat} = floattype(grid),
                                header_bytes::Int = 16384,
                                dt_met_seconds::Real = 3600.0,
                                half_dt_seconds::Real = dt_met_seconds / 2,
                                steps_per_window::Integer = 2,
                                mass_basis::Symbol = :moist)
    isempty(windows) && throw(ArgumentError("write_transport_binary requires at least one window"))

    mesh = grid.horizontal
    vc = grid.vertical
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(first(windows))

    for window in windows
        _transport_payload_sections(window) == payload_sections ||
            throw(ArgumentError("all transport-binary windows must carry the same payload sections"))
        _transport_validate_reduced_window(window, ncell, nface_h, nlevel, basis_sym)
    end

    n_qv = (:qv in payload_sections) ? ncell * nlevel : 0
    n_qv_start = (:qv_start in payload_sections) ? ncell * nlevel : 0
    n_qv_end = (:qv_end in payload_sections) ? ncell * nlevel : 0

    elems_per_window = ncell * nlevel + nface_h * nlevel + ncell * (nlevel + 1) + ncell + n_qv + n_qv_start + n_qv_end

    header = Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 1,
        "header_bytes" => header_bytes,
        "float_type" => string(FT),
        "float_bytes" => sizeof(FT),
        "grid_type" => "reduced_gaussian",
        "horizontal_topology" => "FaceIndexed",
        "ncell" => ncell,
        "nface_h" => nface_h,
        "nlevel" => nlevel,
        "nwindow" => length(windows),
        "vertical_coordinate_type" => "hybrid_sigma_pressure",
        "A_ifc" => Float64.(vc.A),
        "B_ifc" => Float64.(vc.B),
        "dt_met_seconds" => Float64(dt_met_seconds),
        "half_dt_seconds" => Float64(half_dt_seconds),
        "steps_per_window" => Int(steps_per_window),
        "mass_basis" => String(basis_sym),
        "payload_sections" => String.(payload_sections),
        "include_qv" => :qv in payload_sections,
        "include_qv_endpoints" => (:qv_start in payload_sections) || (:qv_end in payload_sections),
        "n_qv" => n_qv,
        "n_qv_start" => n_qv_start,
        "n_qv_end" => n_qv_end,
        "n_geometry_elems" => 0,
        "elems_per_window" => elems_per_window,
        "nlat" => nrings(mesh),
        "latitudes" => Float64.(mesh.latitudes),
        "nlon_per_ring" => mesh.nlon_per_ring,
    )

    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("transport binary header exceeds header_bytes=$(header_bytes)")

    open(path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        for window in windows
            m = _transport_window_mass(window)
            hflux = _transport_window_hflux(window)
            cm = _transport_window_cm(window)
            ps = window.ps
            for section in payload_sections
                field = if section === :m
                    m
                elseif section === :hflux
                    hflux
                elseif section === :cm
                    cm
                elseif section === :ps
                    ps
                elseif section === :qv
                    window.qv
                elseif section === :qv_start
                    window.qv_start
                elseif section === :qv_end
                    window.qv_end
                else
                    error("unsupported transport-binary section $(section)")
                end
                write(io, vec(FT.(field)))
            end
        end
    end

    return path
end

function load_grid(reader::TransportBinaryReader;
                   FT::Type{<:AbstractFloat} = Float64,
                   arch = CPU())
    h = reader.header

    if h.grid_type === :reduced_gaussian && h.horizontal_topology === :faceindexed
        mesh = ReducedGaussianMesh(h.ring_latitudes_f64, h.nlon_per_ring; FT=FT)
        vc = HybridSigmaPressure(FT.(h.A_ifc), FT.(h.B_ifc))
        return AtmosGrid(mesh, vc, arch; FT=FT)
    else
        throw(ArgumentError("Unsupported transport binary grid/topology combination: $(h.grid_type) / $(h.horizontal_topology)"))
    end
end

function load_window!(reader::TransportBinaryReader{FT}, win::Int;
                      m = Array{FT}(undef, reader.header.ncell, reader.header.nlevel),
                      ps = Array{FT}(undef, reader.header.ncell),
                      hflux = Array{FT}(undef, reader.header.nface_h, reader.header.nlevel),
                      cm = Array{FT}(undef, reader.header.ncell, reader.header.nlevel + 1)
                     ) where FT
    h = reader.header
    h.grid_type === :reduced_gaussian || throw(ArgumentError("TransportBinaryReader currently loads only reduced_gaussian payloads"))
    h.horizontal_topology === :faceindexed || throw(ArgumentError("TransportBinaryReader currently loads only FaceIndexed payloads"))

    o = _transport_window_offset(reader, win)
    saw_m = saw_hflux = saw_cm = saw_ps = false

    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :m
            copyto!(m, 1, reader.data, o + 1, n)
            saw_m = true
        elseif section === :hflux
            copyto!(hflux, 1, reader.data, o + 1, n)
            saw_hflux = true
        elseif section === :cm
            copyto!(cm, 1, reader.data, o + 1, n)
            saw_cm = true
        elseif section === :ps
            copyto!(ps, 1, reader.data, o + 1, n)
            saw_ps = true
        end
        o += n
    end

    saw_m || error("Transport binary payload is missing required section `m`")
    saw_hflux || error("Transport binary payload is missing required section `hflux`")
    saw_cm || error("Transport binary payload is missing required section `cm`")
    saw_ps || error("Transport binary payload is missing required section `ps`")

    fluxes = _transport_make_fluxes(h.mass_basis, hflux, cm)
    return m, ps, fluxes
end

function load_qv_window!(reader::TransportBinaryReader{FT}, win::Int;
                         qv = Array{FT}(undef, reader.header.ncell, reader.header.nlevel)) where FT
    h = reader.header
    h.include_qv || return nothing

    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :qv
            copyto!(qv, 1, reader.data, o + 1, n)
            return qv
        end
        o += n
    end

    return nothing
end

function load_qv_pair_window!(reader::TransportBinaryReader{FT}, win::Int;
                              qv_start = Array{FT}(undef, reader.header.ncell, reader.header.nlevel),
                              qv_end = Array{FT}(undef, reader.header.ncell, reader.header.nlevel)) where FT
    h = reader.header
    h.include_qv_endpoints || return nothing

    o = _transport_window_offset(reader, win)
    found_start = false
    found_end = false
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :qv_start
            copyto!(qv_start, 1, reader.data, o + 1, n)
            found_start = true
        elseif section === :qv_end
            copyto!(qv_end, 1, reader.data, o + 1, n)
            found_end = true
        end
        o += n
    end

    return found_start && found_end ? (; qv_start, qv_end) : nothing
end

export TransportBinaryHeader, TransportBinaryReader
export grid_type, horizontal_topology, load_grid, load_qv_pair_window!
export has_qv_endpoints, write_transport_binary
