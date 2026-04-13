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

"""Number of longitude cells (LatLon only)."""
nlon(grid::LatLonTargetGeometry) = grid.mesh.Nx

"""Number of latitude cells (LatLon only)."""
nlat(grid::LatLonTargetGeometry) = grid.mesh.Ny

"""Return a symbol identifying the grid type (`:latlon` or `:era5_native_reduced_gaussian`)."""
grid_kind(::LatLonTargetGeometry) = :latlon
grid_kind(::ReducedGaussianTargetGeometry) = :era5_native_reduced_gaussian

"""Whether this target geometry supports spectral mass-flux preprocessing."""
supports_spectral_massflux_preprocessing(::AbstractTargetGeometry) = false
supports_spectral_massflux_preprocessing(::LatLonTargetGeometry) = true
supports_spectral_massflux_preprocessing(::ReducedGaussianTargetGeometry) = true

"""Spectral truncation T for the target grid: Nyquist = `nlon ÷ 2 − 1` to avoid aliasing."""
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
    build_target_geometry(::Val{:synthetic_reduced_gaussian}, cfg_grid, FT)

Build a reduced-Gaussian target geometry from scratch without needing a source
GRIB file. `cfg_grid["gaussian_number"]` controls the truncation N so the mesh
has `2N` Gauss-Legendre latitude rings. `cfg_grid["nlon_mode"]` selects the
longitude layout:

* `"regular"` (default): every ring has `4N` cells — the classical "regular
  reduced" Gaussian grid used e.g. in the TL159/N80 family.
* `"octahedral"`: ECMWF octahedral layout `nlon = 4k + 16` per hemisphere,
  mirrored between the two hemispheres with `k = 1` at the pole-adjacent ring.

Mesh latitudes are produced from `FastGaussQuadrature.gausslegendre(2N)`,
which returns ascending Gauss-Legendre nodes on `(-1, 1)`. `geometry_source_grib`
is left as an empty string to mark the grid as synthetic.
"""
function build_target_geometry(::Val{:synthetic_reduced_gaussian},
                               cfg_grid, ::Type{FT}) where FT <: AbstractFloat
    N = Int(cfg_grid["gaussian_number"])
    N > 0 || error("synthetic_reduced_gaussian: gaussian_number must be positive, got $N")
    mode = lowercase(String(get(cfg_grid, "nlon_mode", "regular")))

    # 2N Gauss-Legendre latitudes on (-1, 1), ascending south -> north.
    nodes, _ = gausslegendre(2 * N)
    lat_deg = FT.(asind.(nodes))

    nlon_per_ring = if mode == "regular"
        fill(4 * N, 2 * N)
    elseif mode == "octahedral"
        # ECMWF octahedral grid: per hemisphere `nlon[k] = 4k + 16`, with
        # `k = 1` at the pole-adjacent ring and `k = N` at the equator
        # (max). Ring latitudes run south pole -> north pole, so the
        # layout is: grow from 20 at the south pole to 4N+16 at the
        # southern equator ring, then shrink back to 20 at the north
        # pole. Total cells = 4N(N+9) — e.g. 3168 for N=24.
        hemi = Int[4 * k + 16 for k in 1:N]        # [20, 24, ..., 4N+16]
        vcat(hemi, reverse(hemi))                   # S-pole → equator → N-pole
    else
        error("synthetic_reduced_gaussian: unknown nlon_mode \"$mode\" " *
              "(use \"regular\" or \"octahedral\")")
    end

    mesh = ReducedGaussianMesh(lat_deg, nlon_per_ring; FT=FT, radius=FT(R_EARTH))
    lons_by_ring = [FT.(ring_longitudes(mesh, j)) for j in 1:nrings(mesh)]

    return ReducedGaussianTargetGeometry{FT, typeof(mesh)}(
        mesh,
        "",                 # synthetic: no GRIB source
        N,
        copy(nlon_per_ring),
        copy(lat_deg),
        lons_by_ring,
    )
end

"""
    CubedSphereTargetGeometry

Target geometry for spectral→CS transport binary preprocessing. The CS mesh
stores gnomonic cell areas and metric terms; the face table and Poisson scratch
are pre-built at construction time for efficient per-window balance.

The `staging_nlon × staging_nlat` fields define an internal LL grid used for
spectral synthesis before conservative regridding to CS panels.
"""
struct CubedSphereTargetGeometry{FT, M <: CubedSphereMesh{FT}} <: AbstractTargetGeometry
    mesh             :: M
    Nc               :: Int
    face_table       :: CSGlobalFaceTable
    cell_degree      :: Vector{Int}
    poisson_scratch  :: CSPoissonScratch
    cache_dir        :: String
    staging_nlon     :: Int
    staging_nlat     :: Int
end

grid_kind(::CubedSphereTargetGeometry) = :cubed_sphere
supports_spectral_massflux_preprocessing(::CubedSphereTargetGeometry) = true

function target_spectral_truncation(grid::CubedSphereTargetGeometry)
    # Nyquist on the staging LL grid
    return div(grid.staging_nlon, 2) - 1
end

function target_header_metadata(grid::CubedSphereTargetGeometry)
    return Dict{String, Any}(
        "horizontal_topology" => "StructuredDirectional",
        "grid_type" => "cubed_sphere",
        "Nc" => grid.Nc,
        "npanel" => 6,
    )
end

function target_summary(grid::CubedSphereTargetGeometry)
    nc = 6 * grid.Nc^2
    return "C$(grid.Nc) cubed sphere ($(nc) cells, staging $(grid.staging_nlon)×$(grid.staging_nlat))"
end

"""
    build_target_geometry(::Val{:cubed_sphere}, cfg_grid, FT)

Build a cubed-sphere target geometry from the `[grid]` config section.

Required keys:
- `Nc :: Int` — cells per panel edge

Optional keys:
- `regridder_cache_dir` — directory for CR.jl weight cache (default `~/.cache/AtmosTransport/cr_regridding`)
- `staging_nlon`, `staging_nlat` — override the internal LL staging grid size
  (defaults: `max(4Nc, 360)` × `max(2Nc+1, 181)`)
"""
function build_target_geometry(::Val{:cubed_sphere}, cfg_grid, ::Type{FT}) where FT <: AbstractFloat
    Nc = Int(cfg_grid["Nc"])
    Nc > 0 || error("cubed_sphere: Nc must be positive, got $Nc")

    cache_dir = expanduser(String(get(cfg_grid, "regridder_cache_dir",
                                      "~/.cache/AtmosTransport/cr_regridding")))

    staging_nlon = Int(get(cfg_grid, "staging_nlon", max(4 * Nc, 360)))
    staging_nlat = Int(get(cfg_grid, "staging_nlat", max(2 * Nc + 1, 181)))

    mesh = CubedSphereMesh(; Nc=Nc, FT=FT, radius=FT(R_EARTH),
                            convention=GnomonicPanelConvention())

    conn = default_panel_connectivity()
    ft = build_cs_global_face_table(Nc, conn)
    degree = cs_cell_face_degree(ft)
    scratch = CSPoissonScratch(ft.nc)

    return CubedSphereTargetGeometry{FT, typeof(mesh)}(
        mesh, Nc, ft, degree, scratch, cache_dir,
        staging_nlon, staging_nlat,
    )
end

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
