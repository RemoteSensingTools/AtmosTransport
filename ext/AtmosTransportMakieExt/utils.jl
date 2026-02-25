# Shared helpers for the Makie visualization extension

"""Resolve a domain keyword to (lon_range, lat_range) tuples."""
function _resolve_domain(domain)
    if domain isa Symbol
        haskey(PREDEFINED_DOMAINS, domain) ||
            error("Unknown domain :$domain. Available: $(join(keys(PREDEFINED_DOMAINS), ", "))")
        d = PREDEFINED_DOMAINS[domain]
        return d.lon_range, d.lat_range
    elseif domain isa NamedTuple
        return domain.lon_range, domain.lat_range
    elseif domain === nothing
        return (-180.0, 180.0), (-90.0, 90.0)
    else
        error("domain must be a Symbol, NamedTuple{(:lon_range,:lat_range)}, or nothing")
    end
end

"""Build a PROJ string from a projection symbol or pass through a string."""
function _projection_string(proj::Symbol)
    proj === :Robinson     && return "+proj=robin"
    proj === :Orthographic && return "+proj=ortho"
    proj === :PlateCarree  && return "+proj=eqc"
    proj === :Mollweide    && return "+proj=moll"
    proj === :Mercator     && return "+proj=merc"
    error("Unknown projection :$proj. Pass a PROJ string directly, e.g. \"+proj=robin\".")
end

_projection_string(proj::AbstractString) = String(proj)

"""
Shift longitude from 0..360 to -180..180, sort both axes ascending,
and reorder the data matrix accordingly. Returns `(lon, lat, data)`.
"""
function _normalize_lonlat(lon_raw::AbstractVector, lat_raw::AbstractVector,
                            data::AbstractMatrix)
    lon_shifted = [l > 180 ? l - 360.0 : l for l in lon_raw]
    lon_order   = sortperm(lon_shifted)
    lon         = lon_shifted[lon_order]

    lat_order = lat_raw[1] > lat_raw[end] ? (length(lat_raw):-1:1) : (1:length(lat_raw))
    lat = lat_raw[collect(lat_order)]

    data_reordered = data[lon_order, collect(lat_order)]
    return lon, lat, data_reordered
end
