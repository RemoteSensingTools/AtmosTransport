#!/usr/bin/env julia
# Export the saved LinRood LA footprint binary to CSV with
# Julia-computed lon/lat for plotting in Python.

using Printf

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

const BIN_PATH = joinpath(@__DIR__, "..", "..",
                          "artifacts", "linrood_la_footprint_c24_6h.bin")
const CSV_PATH = joinpath(@__DIR__, "..", "..",
                          "artifacts", "linrood_la_footprint_c24_6h.csv")

open(BIN_PATH, "r") do io
    Nc      = read(io, Int64)
    Nz      = read(io, Int64)
    nsteps  = read(io, Int64)
    la_pan  = read(io, Int64)
    la_i    = read(io, Int64)
    la_j    = read(io, Int64)
    panels = ntuple(_ -> read!(io, Array{Float64, 2}(undef, Nc, Nc)), 6)

    mesh = AT.CubedSphereMesh(Nc=Nc, Hp=3, FT=Float64)

    open(CSV_PATH, "w") do out
        println(out, "panel,i,j,lon,lat,dJdE")
        # Also save metadata as a comment.
        @printf(out, "# Nc=%d Nz=%d nsteps=%d la_panel=%d la_i=%d la_j=%d\n",
                Nc, Nz, nsteps, la_pan, la_i, la_j)
        for p in 1:6
            lons, lats = AT.Grids.panel_cell_center_lonlat(mesh, p)
            for j in 1:Nc, i in 1:Nc
                @printf(out, "%d,%d,%d,%.6f,%.6f,%.6e\n",
                        p, i, j, lons[i, j], lats[i, j], panels[p][i, j])
            end
        end
    end
    @printf("Wrote %s (%d cells)\n", CSV_PATH, 6 * Nc * Nc)
end
