#!/usr/bin/env julia
# ===========================================================================
# Pre-regrid all CATRINE emissions to the ERA5 target grid (720×361)
#
# Uses build_model_from_config to load + regrid emissions, then saves each
# TimeVaryingSurfaceFlux as a compact binary. At runtime, use emission="prebuilt"
# to load the binary directly (~100× faster than on-the-fly regrid).
#
# Output: one .bin file per tracer in <output_dir>/emissions_prebuild/
#
# Usage:
#   julia --project=. scripts/preprocessing/prebuild_era5_emissions.jl \
#       config/runs/catrine_era5_spectral_1week.toml
# ===========================================================================

using TOML
using Dates

using AtmosTransport
using AtmosTransport.IO: build_model_from_config
using AtmosTransport.Sources: TimeVaryingSurfaceFlux, flux_data, LatLonLayout,
                               AbstractSurfaceFlux

length(ARGS) >= 1 || error("Usage: julia --project=. scripts/preprocessing/prebuild_era5_emissions.jl <config.toml>")

config_path = ARGS[1]
@info "Building model from config: $config_path"

# Force CPU mode for preprocessing (no CUDA needed)
config = TOML.parsefile(config_path)
config["architecture"]["use_gpu"] = false

t0 = time()
model = build_model_from_config(config)
@info "Model built in $(round(time()-t0, digits=1))s"

# Output directory
output_cfg = get(TOML.parsefile(config_path), "output", Dict())
outdir = dirname(expanduser(get(output_cfg, "filename", "output/emissions")))
emissions_dir = joinpath(outdir, "emissions_prebuild")
mkpath(emissions_dir)

grid = model.grid
Nx, Ny = grid.Nx, grid.Ny
@info "Target grid: $(Nx)×$(Ny), output: $emissions_dir"

for src in model.sources
    if src isa TimeVaryingSurfaceFlux{LatLonLayout}
        fd = src.flux_data  # full 3D array (Nx, Ny, Nt), not current slice
        Nt = size(fd, 3)
        species = src.species
        outpath = joinpath(emissions_dir, "$(species)_$(Nx)x$(Ny).bin")

        header = Dict(
            "Nx" => Nx, "Ny" => Ny, "Nt" => Nt,
            "species" => string(species),
            "molar_mass" => Float64(src.molar_mass),
            "label" => src.label,
            "cyclic" => src.cyclic
        )
        header_toml = sprint(TOML.print, header)

        open(outpath, "w") do io
            header_bytes = Vector{UInt8}(header_toml)
            push!(header_bytes, 0x00)
            header_block = zeros(UInt8, 4096)
            copyto!(header_block, 1, header_bytes, 1, length(header_bytes))
            write(io, header_block)
            write(io, src.time_hours)
            write(io, Float32.(fd))
        end

        sz_mb = round(filesize(outpath) / 1e6, digits=1)
        @info "  Saved: $(basename(outpath)) ($sz_mb MB, $Nt timesteps, $(src.label))"
    else
        @info "  Skipping $(typeof(src)) — not TimeVaryingSurfaceFlux{LatLon}"
    end
end

@info "Done! Pre-built emissions in $emissions_dir"
