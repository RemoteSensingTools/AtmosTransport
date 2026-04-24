"""
Makie extension for AtmosTransport visualization.

Load a Makie backend before plotting, for example:

```julia
using CairoMakie
using AtmosTransport
```

The core `AtmosTransport.Visualization` module remains Makie-free; this
extension adds rendering methods for topology-aware snapshot fields.
"""
module AtmosTransportMakieExt

import AtmosTransport
import AtmosTransport.Visualization:
    HorizontalField, RasterField, SnapshotDataset, SnapshotRegridCache, PlotSpec,
    fieldview, frame_indices, as_raster, robust_colorrange,
    mapplot, mapplot!, snapshot_grid, movie, movie_grid

using Makie

function _label(raster::RasterField)
    isempty(raster.units) ? String(raster.name) : "$(raster.name) [$(raster.units)]"
end

function _title(raster::RasterField; override=nothing)
    override !== nothing && return String(override)
    return "$(raster.name) $(raster.transform), t=$(round(raster.time; digits=2)) h"
end

function _colorrange(rasters, colorrange)
    colorrange === :auto && return robust_colorrange(rasters)
    return colorrange
end

function _heatmap_axis!(fig, slot, raster::RasterField;
                        colormap=:viridis,
                        colorrange=:auto,
                        title=nothing,
                        hide_ticks::Bool=true)
    ax = Makie.Axis(fig[slot...];
                    aspect=Makie.DataAspect(),
                    title=_title(raster; override=title),
                    xlabel="longitude",
                    ylabel="latitude")
    if hide_ticks
        ax.xticksvisible[] = false
        ax.yticksvisible[] = false
        ax.xticklabelsvisible[] = false
        ax.yticklabelsvisible[] = false
    end
    hm = Makie.heatmap!(ax, raster.lons, raster.lats, raster.values;
                        colormap=colormap,
                        colorrange=_colorrange([raster], colorrange))
    return ax, hm
end

"""
    mapplot(field; kwargs...) -> Figure

Render one topology-native `HorizontalField` as a regular lon-lat debug map.
CS fields are conservatively regridded through `as_raster`.
"""
function mapplot(field::HorizontalField;
                 resolution::Tuple{Int, Int}=(360, 181),
                 cache::SnapshotRegridCache=SnapshotRegridCache(),
                 colormap=:viridis,
                 colorrange=:auto,
                 title=nothing,
                 colorbar::Bool=true,
                 size=(1000, 520))
    raster = as_raster(field; resolution=resolution, cache=cache)
    fig = Makie.Figure(size=size)
    ax, hm = _heatmap_axis!(fig, (1, 1), raster;
                            colormap=colormap,
                            colorrange=colorrange,
                            title=title,
                            hide_ticks=false)
    colorbar && Makie.Colorbar(fig[1, 2], hm; label=_label(raster))
    return fig
end

"""
    mapplot!(ax, field; kwargs...)

Draw one field into an existing Makie axis and return the heatmap plot object.
"""
function mapplot!(ax, field::HorizontalField;
                  resolution::Tuple{Int, Int}=(360, 181),
                  cache::SnapshotRegridCache=SnapshotRegridCache(),
                  colormap=:viridis,
                  colorrange=:auto)
    raster = as_raster(field; resolution=resolution, cache=cache)
    return Makie.heatmap!(ax, raster.lons, raster.lats, raster.values;
                          colormap=colormap,
                          colorrange=_colorrange([raster], colorrange))
end

function _grid_size(n::Int, cols::Int)
    c = max(1, cols)
    r = cld(n, c)
    return r, c
end

"""
    snapshot_grid(snapshot, variable; times=:all, cols=4, kwargs...) -> Figure

Render a multi-panel figure containing one variable at several snapshot times.
"""
function snapshot_grid(snapshot::SnapshotDataset,
                       variable::Union{Symbol, AbstractString};
                       transform::Symbol=:column_mean,
                       times=:all,
                       level::Union{Nothing, Int}=nothing,
                       unit::Symbol=:native,
                       cols::Int=4,
                       resolution::Tuple{Int, Int}=(360, 181),
                       cache::SnapshotRegridCache=SnapshotRegridCache(),
                       colormap=:viridis,
                       colorrange=:auto,
                       size=nothing)
    indices = frame_indices(snapshot, times)
    fields = [fieldview(snapshot, variable;
                        transform=transform,
                        time=i,
                        level=level,
                        unit=unit) for i in indices]
    rasters = [as_raster(f; resolution=resolution, cache=cache) for f in fields]
    cr = _colorrange(rasters, colorrange)
    rows, columns = _grid_size(length(rasters), cols)
    fig_size = size === nothing ? (260 * columns + 90, 180 * rows + 70) : size
    fig = Makie.Figure(size=fig_size)

    hm = nothing
    for (idx, raster) in enumerate(rasters)
        row = div(idx - 1, columns) + 1
        col = mod(idx - 1, columns) + 1
        ax, h = _heatmap_axis!(fig, (row, col), raster;
                               colormap=colormap,
                               colorrange=cr,
                               title="t=$(round(raster.time; digits=2)) h")
        hm === nothing && (hm = h)
    end
    hm !== nothing && Makie.Colorbar(fig[1:rows, columns + 1], hm;
                                     label=_label(first(rasters)),
                                     height=Makie.Relative(0.9))
    return fig
end

function _spec_title(spec::PlotSpec, raster::RasterField)
    spec.title === nothing || return spec.title
    if spec.transform === :level_slice || spec.transform === :surface_slice
        lev = spec.level === nothing ? "" : " level $(spec.level)"
        return "$(spec.variable) $(spec.transform)$(lev)"
    end
    return "$(spec.variable) $(spec.transform)"
end

function _spec_rasters(snapshot::SnapshotDataset, spec::PlotSpec, indices;
                       resolution, cache)
    fields = [fieldview(snapshot, spec.variable;
                        transform=spec.transform,
                        time=i,
                        level=spec.level,
                        unit=spec.unit) for i in indices]
    return [as_raster(f; resolution=resolution, cache=cache) for f in fields]
end

"""
    movie_grid(snapshot, specs, out; times=:all, fps=8, kwargs...) -> String

Record a multi-panel GIF/MP4 from snapshot fields. Each `PlotSpec` becomes one
panel, and expensive topology geometry is cached across all frames.
"""
function movie_grid(snapshot::SnapshotDataset,
                    specs::AbstractVector{PlotSpec},
                    out::AbstractString;
                    times=:all,
                    fps::Real=8,
                    cols::Int=max(1, length(specs)),
                    resolution::Tuple{Int, Int}=(360, 181),
                    cache::SnapshotRegridCache=SnapshotRegridCache(),
                    colormap=:viridis,
                    colorrange=:auto,
                    size=nothing)
    isempty(specs) && throw(ArgumentError("movie_grid requires at least one PlotSpec"))
    indices = frame_indices(snapshot, times)
    isempty(indices) && throw(ArgumentError("movie_grid has no frames"))

    rasters_by_spec = [_spec_rasters(snapshot, spec, indices;
                                     resolution=resolution,
                                     cache=cache) for spec in specs]
    ranges = [colorrange === :auto ? robust_colorrange(rasters) : colorrange
              for rasters in rasters_by_spec]
    rows, columns = _grid_size(length(specs), cols)
    fig_size = size === nothing ? (340 * columns + 90, 220 * rows + 70) : size
    fig = Makie.Figure(size=fig_size)

    data_obs = Makie.Observable[]
    axes = Any[]
    for (idx, spec) in enumerate(specs)
        raster = rasters_by_spec[idx][1]
        row = div(idx - 1, columns) + 1
        col = mod(idx - 1, columns) + 1
        ax = Makie.Axis(fig[row, col];
                        aspect=Makie.DataAspect(),
                        title="$( _spec_title(spec, raster) ), t=$(round(raster.time; digits=2)) h")
        ax.xticksvisible[] = false
        ax.yticksvisible[] = false
        ax.xticklabelsvisible[] = false
        ax.yticklabelsvisible[] = false
        obs = Makie.Observable(raster.values)
        hm = Makie.heatmap!(ax, raster.lons, raster.lats, obs;
                            colormap=colormap,
                            colorrange=ranges[idx])
        Makie.Colorbar(fig[row, columns + 1], hm; label=_label(raster))
        push!(data_obs, obs)
        push!(axes, ax)
    end

    mkpath(dirname(String(out)))
    Makie.record(fig, String(out), eachindex(indices); framerate=round(Int, fps)) do frame
        for idx in eachindex(specs)
            raster = rasters_by_spec[idx][frame]
            data_obs[idx][] = raster.values
            axes[idx].title[] = "$(_spec_title(specs[idx], raster)), t=$(round(raster.time; digits=2)) h"
        end
    end
    return String(out)
end

"""
    movie(snapshot, variable, out; kwargs...) -> String

Record a one-panel movie for one variable.
"""
function movie(snapshot::SnapshotDataset,
               variable::Union{Symbol, AbstractString},
               out::AbstractString;
               transform::Symbol=:column_mean,
               level::Union{Nothing, Int}=nothing,
               unit::Symbol=:native,
               kwargs...)
    spec = PlotSpec(Symbol(variable); transform=transform, level=level, unit=unit)
    return movie_grid(snapshot, [spec], out; kwargs...)
end

end # module AtmosTransportMakieExt
