abstract type AbstractTargetGeometry end

"""
    LatLonTargetGeometry

Structured target geometry used by the current spectral preprocessor. Stores
the `LatLonMesh` plus cached coordinate and metric arrays needed by the
spectral-synthesis and mass-flux routines.
"""
struct LatLonTargetGeometry{FT, M <: LatLonMesh{FT}} <: AbstractTargetGeometry
    mesh    :: M
    lons    :: Vector{FT}
    lats    :: Vector{FT}
    dlon    :: FT
    dlat    :: FT
    cos_lat :: Vector{FT}
    area    :: Matrix{FT}
end

"""
    ReducedGaussianTargetGeometry

Geometry descriptor for native ERA5 reduced-Gaussian grids. The target-geometry
plumbing is in place, but the spectral mass-flux preprocessing path is not yet
implemented for this target.
"""
struct ReducedGaussianTargetGeometry{FT, M <: ReducedGaussianMesh{FT}} <: AbstractTargetGeometry
    mesh                 :: M
    geometry_source_grib :: String
    gaussian_number      :: Int
    nlon_per_ring        :: Vector{Int}
    lats                 :: Vector{FT}
    lons_by_ring         :: Vector{Vector{FT}}
end

nlon(grid::LatLonTargetGeometry) = grid.mesh.Nx
nlat(grid::LatLonTargetGeometry) = grid.mesh.Ny
grid_kind(::LatLonTargetGeometry) = :latlon
grid_kind(::ReducedGaussianTargetGeometry) = :era5_native_reduced_gaussian

supports_spectral_massflux_preprocessing(::AbstractTargetGeometry) = false
supports_spectral_massflux_preprocessing(::LatLonTargetGeometry) = true

target_spectral_truncation(grid::LatLonTargetGeometry) = div(nlon(grid), 2) - 1
target_spectral_truncation(grid::ReducedGaussianTargetGeometry) =
    div(maximum(grid.nlon_per_ring), 2) - 1

target_header_metadata(grid::LatLonTargetGeometry) = Dict{String, Any}(
    "horizontal_topology" => "StructuredDirectional",
    "grid_type" => "latlon",
    "grid_convention" => "TM5",
    "lons" => Float64.(grid.lons),
    "lats" => Float64.(grid.lats),
    "longitude_interval" => Float64[first(grid.mesh.λᶠ), last(grid.mesh.λᶠ)],
    "latitude_interval" => Float64[first(grid.mesh.φᶠ), last(grid.mesh.φᶠ)],
    "lon_center_start_deg" => Float64(first(grid.lons)),
    "lon_center_step_deg" => Float64(grid.mesh.Δλ),
    "lat_center_start_deg" => Float64(first(grid.lats)),
    "lat_center_step_deg" => Float64(grid.mesh.Δφ),
)

target_header_metadata(grid::ReducedGaussianTargetGeometry) = Dict{String, Any}(
    "horizontal_topology" => "FaceIndexed",
    "grid_type" => "reduced_gaussian",
    "grid_convention" => "ERA5 native reduced Gaussian",
    "gaussian_number" => grid.gaussian_number,
    "ring_latitudes" => Float64.(grid.lats),
    "nlon_per_ring" => copy(grid.nlon_per_ring),
    "geometry_source_grib" => grid.geometry_source_grib,
)

"""
    target_summary(grid) -> String

Return a short human-readable summary of the configured target grid.
"""
function target_summary(grid::LatLonTargetGeometry)
    return "$(nlon(grid)) x $(nlat(grid)) ($(grid.mesh.Δλ) deg x $(grid.mesh.Δφ) deg, TM5 convention)"
end

function target_summary(grid::ReducedGaussianTargetGeometry)
    nlon_min = minimum(grid.nlon_per_ring)
    nlon_max = maximum(grid.nlon_per_ring)
    return "ERA5 native reduced Gaussian N$(grid.gaussian_number) ($(nrings(grid.mesh)) rings, nlon/ring=$nlon_min-$nlon_max)"
end

"""
    ensure_supported_target(grid)

Fail fast when the selected target geometry is known to the configuration layer
but does not yet have a spectral mass-flux implementation.
"""
function ensure_supported_target(grid::AbstractTargetGeometry)
    supports_spectral_massflux_preprocessing(grid) && return nothing
    error("Target grid $(target_summary(grid)) is configured, but this preprocessor currently " *
          "implements only the regular lat-lon spectral synthesis and mass-flux path. " *
          "The mesh-aware target-geometry wiring is now in place; native reduced-Gaussian " *
          "spectral synthesis and flux construction still need dedicated methods.")
end

@inline _grid_float_type(::Type{FT}) where FT <: AbstractFloat = FT

"""
    _latlon_area_matrix(mesh) -> Matrix

Expand the latitude-band area vector from `LatLonMesh` into a full `(Nx, Ny)`
matrix so later kernels can index cell area directly by `(i, j)`.
"""
function _latlon_area_matrix(mesh::LatLonMesh{FT}) where FT
    area_by_lat = cell_areas_by_latitude(mesh)
    area = Array{FT}(undef, mesh.Nx, mesh.Ny)
    for j in 1:mesh.Ny
        @views area[:, j] .= area_by_lat[j]
    end
    return area
end

"""
    build_target_geometry(::Val{:latlon}, cfg_grid, FT) -> LatLonTargetGeometry

Build the regular lat-lon target geometry used by the current v4 spectral
preprocessor.
"""
function build_target_geometry(::Val{:latlon}, cfg_grid, ::Type{FT}) where FT <: AbstractFloat
    mesh = LatLonMesh(; FT=FT,
                      size=(Int(cfg_grid["nlon"]), Int(cfg_grid["nlat"])),
                      longitude=get(cfg_grid, "longitude", (-180, 180)),
                      latitude=get(cfg_grid, "latitude", (-90, 90)),
                      radius=FT(R_EARTH))
    return LatLonTargetGeometry{FT, typeof(mesh)}(
        mesh,
        copy(mesh.λᶜ),
        copy(mesh.φᶜ),
        FT(deg2rad(mesh.Δλ)),
        FT(deg2rad(mesh.Δφ)),
        FT.(cosd.(mesh.φᶜ)),
        _latlon_area_matrix(mesh),
    )
end

build_target_geometry(::Val{:regular_latlon}, cfg_grid, ::Type{FT}) where FT <: AbstractFloat =
    build_target_geometry(Val(:latlon), cfg_grid, FT)

@inline _optional_int(cfg, key::AbstractString) = haskey(cfg, key) ? Int(cfg[key]) : nothing

"""
    build_target_geometry(::Val{:era5_native_reduced_gaussian}, cfg_grid, FT)

Build the reduced-Gaussian geometry metadata from a native ERA5 GRIB file.
This is currently intended for geometry discovery and future native-grid
preprocessing work rather than the active lat-lon synthesis path.
"""
function build_target_geometry(::Val{:era5_native_reduced_gaussian}, cfg_grid, ::Type{FT}) where FT <: AbstractFloat
    geometry_source = expanduser(String(cfg_grid["geometry_source_grib"]))
    param_id = _optional_int(cfg_grid, "geometry_param_id")
    level = _optional_int(cfg_grid, "geometry_level")
    date = _optional_int(cfg_grid, "geometry_date")
    time = _optional_int(cfg_grid, "geometry_time")

    geom = read_era5_reduced_gaussian_geometry(geometry_source; FT=FT,
                                               param_id=param_id,
                                               level=level,
                                               date=date,
                                               time=time)
    mesh = read_era5_reduced_gaussian_mesh(geometry_source; FT=FT,
                                           radius=FT(R_EARTH),
                                           param_id=param_id,
                                           level=level,
                                           date=date,
                                           time=time)

    return ReducedGaussianTargetGeometry{FT, typeof(mesh)}(
        mesh,
        geometry_source,
        geom.gaussian_number,
        copy(geom.nlon_per_ring),
        copy(geom.latitudes),
        [FT.(ring_longitudes(mesh, j)) for j in 1:nrings(mesh)],
    )
end

build_target_geometry(::Val{:reduced_gaussian}, cfg_grid, ::Type{FT}) where FT <: AbstractFloat =
    build_target_geometry(Val(:era5_native_reduced_gaussian), cfg_grid, FT)

"""
    build_target_geometry(cfg_grid, FT=Float64) -> AbstractTargetGeometry

Dispatch configuration-driven target-geometry construction from the user-facing
`grid.type` string.
"""
function build_target_geometry(cfg_grid, ::Type{FT}=Float64) where FT <: AbstractFloat
    raw_kind = lowercase(String(get(cfg_grid, "type", "latlon")))
    kind = Symbol(replace(raw_kind, '-' => '_', ' ' => '_'))
    return build_target_geometry(Val(kind), cfg_grid, FT)
end
