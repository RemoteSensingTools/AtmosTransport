#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Selected-layer snapshot output (`[output.fields] layers = "selected"`).
#
# Regression: `_ensure_selected_lev!` guarded re-use with
# `length(ds.dim["lev_selected"])` — but `ds.dim[name]` IS the length (an
# Int), so the guard threw "all selected-layer outputs must use the same
# levels" for every selected-layer variable after the first. Any file with
# >= 2 selected-layer outputs (two tracers, or air_mass + one tracer) was
# unwritable on every topology.
# ---------------------------------------------------------------------------

using Test
using NCDatasets

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.State: CellState, DryBasis
using .AtmosTransport.Grids: LatLonMesh
using .AtmosTransport.Output: OutputFieldSpec, TracerOutputFields,
    FullLayerSelection, SelectedLayerSelection

include(joinpath(@__DIR__, "..", "fixtures", "mini_models.jl"))

function _mini_ll_model(; kwargs...)
    fx = fixture_ll_model(; kwargs...)
    return fx.model, fx.grid
end

@testset "selected-layer output: >= 2 selected-layer variables in one file" begin
    model, grid = _mini_ll_model()
    frames = [capture_snapshot(model; time_hours = Float64(h)) for h in 0:1]
    levels = [1, 3]
    fields = OutputFieldSpec(nothing, levels,
                             TracerOutputFields(SelectedLayerSelection(), true, true),
                             Dict{Symbol, TracerOutputFields}(),
                             SelectedLayerSelection(),  # air also selected
                             true, true, true)
    mktempdir() do dir
        path = joinpath(dir, "sel.nc")
        # two tracers + air on the selected-layer path: the buggy guard threw
        # on the second variable.
        write_snapshot_netcdf(path, frames, grid; mass_basis = :dry,
                              fields = fields)
        NCDataset(path) do ds
            @test ds.dim["lev_selected"] == length(levels)
            @test ds["lev_selected"][:] == Float64.(levels)
            for v in ("air_mass", "co2", "sf6")
                @test haskey(ds, v)
                @test "lev_selected" in NCDatasets.dimnames(ds[v])
            end
            # the selected slices must match the full field at those levels
            full_q = Array{Float32}(undef, 6, 4, 3)
            am = frames[1].air_mass
            full_q .= frames[1].tracers[:co2] ./ am
            @test ds["co2"][:, :, :, 1] ≈ full_q[:, :, levels]
        end
    end
end

println("test_netcdf_writer_selected_levels.jl OK")
