"""
    Visualization

Plotting and animation functions for AtmosTransport output.

Requires CairoMakie and GeoMakie to be loaded. When they are, the
`AtmosTransportMakieExt` extension automatically provides implementations
for all exported functions.

# Usage

```julia
using CairoMakie, GeoMakie
using AtmosTransport

# Plot a 2D field
fig = plot_field(data, lons, lats; projection=:Robinson, colormap=:YlOrRd)

# Plot from NetCDF output
fig = plot_output("output.nc", :co2; time_index=10, domain=:us)

# Animate from NetCDF output
animate_output("output.nc", :co2; output_path="anim.gif", domain=:global)
```
"""
module Visualization

export plot_field, plot_output, animate_output
export PREDEFINED_DOMAINS

# ---- Predefined regional domains ----

"""
Predefined geographic domains for visualization.

Available domains: `:global`, `:us`, `:europe`, `:asia`,
`:north_america`, `:south_america`, `:africa`.
"""
const PREDEFINED_DOMAINS = Dict{Symbol, NamedTuple{(:lon_range, :lat_range),
                                                    Tuple{Tuple{Float64,Float64},
                                                          Tuple{Float64,Float64}}}}(
    :global         => (lon_range=(-180.0, 180.0), lat_range=(-90.0, 90.0)),
    :us             => (lon_range=(-130.0, -60.0), lat_range=(24.0, 50.0)),
    :europe         => (lon_range=(-15.0, 35.0),   lat_range=(35.0, 72.0)),
    :asia           => (lon_range=(60.0, 150.0),   lat_range=(10.0, 55.0)),
    :north_america  => (lon_range=(-170.0, -50.0), lat_range=(15.0, 75.0)),
    :south_america  => (lon_range=(-85.0, -30.0),  lat_range=(-60.0, 15.0)),
    :africa         => (lon_range=(-20.0, 55.0),   lat_range=(-40.0, 40.0)),
)

# ---- Stub functions (implemented by AtmosTransportMakieExt) ----

const _MAKIE_NOT_LOADED = "CairoMakie and GeoMakie must be loaded: `using CairoMakie, GeoMakie`"

"""
    plot_field(data, lons, lats; kwargs...) → Figure

Plot a 2D scalar field on a map projection.

# Keyword arguments
- `projection`: `:Robinson` (default), `:Orthographic`, `:PlateCarree`, `:Mollweide`, or a PROJ string
- `domain`: `:global`, `:us`, `:europe`, `:asia`, or custom `(lon_range=(...), lat_range=(...))`
- `colormap`: any Makie colormap (default `:YlOrRd`)
- `colorrange`: `(vmin, vmax)` or `nothing` for auto
- `colorscale`: e.g. `identity`, `sqrt`, `log10`
- `coastlines`: overlay coastlines (default `true`)
- `title`: plot title string
- `size`: figure size tuple (default `(1000, 520)`)
- `colorbar_label`: label for the colorbar
- `figure`: existing Figure to plot into (default: create new)
"""
function plot_field end

"""
    plot_output(nc_path, variable; kwargs...) → Figure

Read a variable from a NetCDF output file and plot a single time step.

# Keyword arguments
- `time_index`: which time step to plot (default last)
- All keywords from `plot_field` are also accepted.
"""
function plot_output end

"""
    animate_output(nc_path, variable; kwargs...) → output_path

Create an animation (GIF or MP4) from a NetCDF output file.

# Keyword arguments
- `output_path`: path for the animation file (default: derived from nc_path)
- `framerate`: frames per second (default 10)
- `frame_step`: skip frames (default: auto for ~120 frames max)
- All keywords from `plot_field` are also accepted.
"""
function animate_output end

end # module Visualization
