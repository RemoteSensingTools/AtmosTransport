"""
    column_mean_mixing_ratio(air_mass, tracer_mass)

Compute the air-mass-weighted vertical mean mixing ratio for one snapshot field.

The returned shape is topology-native horizontal storage:

- LL: `(lon, lat)`
- RG: `(cell,)`
- CS: `NTuple{6, Matrix{Float64}}`
"""
function column_mean_mixing_ratio(air_mass::AbstractArray, tracer_mass::AbstractArray)
    ndims(air_mass) in (2, 3) ||
        throw(ArgumentError("column_mean_mixing_ratio expects 2D or 3D arrays, got ndims=$(ndims(air_mass))"))
    size(air_mass) == size(tracer_mass) ||
        throw(DimensionMismatch("air_mass size $(size(air_mass)) and tracer_mass size $(size(tracer_mass)) differ"))

    if ndims(air_mass) == 3
        Nx, Ny, Nz = size(air_mass)
        out = zeros(Float64, Nx, Ny)
        @inbounds for j in 1:Ny, i in 1:Nx
            num = 0.0
            den = 0.0
            for k in 1:Nz
                m = Float64(air_mass[i, j, k])
                num += Float64(tracer_mass[i, j, k])
                den += m
            end
            out[i, j] = den > 0 ? num / den : NaN
        end
        return out
    else
        Nc, Nz = size(air_mass)
        out = zeros(Float64, Nc)
        @inbounds for c in 1:Nc
            num = 0.0
            den = 0.0
            for k in 1:Nz
                m = Float64(air_mass[c, k])
                num += Float64(tracer_mass[c, k])
                den += m
            end
            out[c] = den > 0 ? num / den : NaN
        end
        return out
    end
end

function column_mean_mixing_ratio(air_mass::NTuple{6, <:AbstractArray},
                                  tracer_mass::NTuple{6, <:AbstractArray})
    return ntuple(p -> column_mean_mixing_ratio(air_mass[p], tracer_mass[p]), 6)
end

"""
    mixing_ratio_field(air_mass, tracer_mass)

Return per-layer VMR from tracer mass divided by the stored air mass. Tiny or
non-positive mass cells are written as `NaN` rather than silently clamped.
"""
function mixing_ratio_field(air_mass::AbstractArray, tracer_mass::AbstractArray)
    size(air_mass) == size(tracer_mass) ||
        throw(DimensionMismatch("air_mass size $(size(air_mass)) and tracer_mass size $(size(tracer_mass)) differ"))
    out = Array{Float64}(undef, size(air_mass))
    @inbounds for idx in eachindex(out, air_mass, tracer_mass)
        m = Float64(air_mass[idx])
        out[idx] = m > 0 ? Float64(tracer_mass[idx]) / m : NaN
    end
    return out
end

mixing_ratio_field(air_mass::NTuple{6, <:AbstractArray},
                   tracer_mass::NTuple{6, <:AbstractArray}) =
    ntuple(p -> mixing_ratio_field(air_mass[p], tracer_mass[p]), 6)

"""
    layer_mass_per_area(field, mesh)

Convert per-cell layer mass `[kg cell⁻¹]` to mass per horizontal area
`[kg m⁻²]` without changing topology-native storage.
"""
function layer_mass_per_area(field::AbstractArray, mesh::LatLonMesh)
    Nx, Ny, Nz = size(field)
    out = Array{Float64}(undef, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        out[i, j, k] = Float64(field[i, j, k]) / Float64(cell_area(mesh, i, j))
    end
    return out
end

function layer_mass_per_area(field::AbstractArray, mesh::ReducedGaussianMesh)
    Nc, Nz = size(field)
    out = Array{Float64}(undef, Nc, Nz)
    @inbounds for k in 1:Nz, c in 1:Nc
        out[c, k] = Float64(field[c, k]) / Float64(cell_area(mesh, c))
    end
    return out
end

function layer_mass_per_area(field::NTuple{6, <:AbstractArray}, mesh::CubedSphereMesh)
    return ntuple(p -> begin
        Nc, _, Nz = size(field[p])
        out = Array{Float64}(undef, Nc, Nc, Nz)
        @inbounds for k in 1:Nz, j in 1:Nc, i in 1:Nc
            out[i, j, k] = Float64(field[p][i, j, k]) / Float64(cell_area(mesh, i, j))
        end
        out
    end, 6)
end

"""
    column_mass_per_area(field, mesh)

Vertically integrate per-cell mass and divide by horizontal cell area.
"""
function column_mass_per_area(field, mesh)
    layer = layer_mass_per_area(field, mesh)
    return _column_sum(layer)
end

function _column_sum(values::AbstractArray)
    if ndims(values) == 3
        Nx, Ny, Nz = size(values)
        out = zeros(Float64, Nx, Ny)
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            out[i, j] += Float64(values[i, j, k])
        end
        return out
    elseif ndims(values) == 2
        Nc, Nz = size(values)
        out = zeros(Float64, Nc)
        @inbounds for k in 1:Nz, c in 1:Nc
            out[c] += Float64(values[c, k])
        end
        return out
    else
        throw(ArgumentError("_column_sum expects 2D or 3D arrays, got ndims=$(ndims(values))"))
    end
end

_column_sum(values::NTuple{6, <:AbstractArray}) = ntuple(p -> _column_sum(values[p]), 6)
