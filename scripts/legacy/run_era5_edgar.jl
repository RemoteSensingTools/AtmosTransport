#!/usr/bin/env julia
# ===========================================================================
# ERA5 + EDGAR CO2 Forward Transport (Double-Buffered)
#
# Minimal script — all logic lives in the core library.
# Configure via TOML or programmatic setup below.
#
# Usage:
#   USE_GPU=true julia --project=. scripts/run_era5_edgar.jl
#   # or with TOML:
#   julia --project=. -e '
#       using AtmosTransport
#       model = build_model_from_config("config/runs/era5_edgar.toml")
#       run!(model)
#   '
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures: GPU, CPU

const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end
using AtmosTransport.Grids: LatitudeLongitudeGrid
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients, ERA5MetDriver,
                              find_era5_files, NetCDFOutputWriter,
                              TimeIntervalSchedule
using AtmosTransport.Sources: EdgarSource, load_inventory
using AtmosTransport.Diagnostics: ColumnMeanDiagnostic
using AtmosTransport.Models: TransportModel, DoubleBuffer
import AtmosTransport.Models: run!
using AtmosTransport.Parameters: load_parameters

# --- Configuration ---
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "false"))
const FT = USE_FLOAT32 ? Float32 : Float64
const LEVEL_TOP = parse(Int, get(ENV, "LEVEL_TOP", "50"))
const LEVEL_BOT = parse(Int, get(ENV, "LEVEL_BOT", "137"))

arch = USE_GPU ? GPU() : CPU()
params = load_parameters(FT)
pp = params.planet

config = default_met_config("era5")
vc = build_vertical_coordinate(config; FT, level_range=LEVEL_TOP:LEVEL_BOT)
A_full, B_full = load_vertical_coefficients(config; FT)
A_coeff = A_full[LEVEL_TOP:LEVEL_BOT+1]
B_coeff = B_full[LEVEL_TOP:LEVEL_BOT+1]

datadirs = [expanduser(d) for d in split(get(ENV, "ERA5_DATADIRS",
    "~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607"), ":")]
files = find_era5_files(datadirs)

# Grid metadata from first file
using AtmosTransport.IO: get_era5_grid_info
lons, lats, _, Nx, Ny, Nz, _ = get_era5_grid_info(files[1], FT)
Δlon = lons[2] - lons[1]

grid = LatitudeLongitudeGrid(arch; FT, size=(Nx, Ny, Nz),
    longitude=(FT(lons[1]), FT(lons[end]) + FT(Δlon)),
    latitude=(FT(-90), FT(90)),
    vertical=vc,
    radius=pp.radius, gravity=pp.gravity,
    reference_pressure=pp.reference_surface_pressure)

met = ERA5MetDriver(; FT, files, A_coeff, B_coeff,
    met_interval=FT(21600), dt=parse(FT, get(ENV, "DT", "900")),
    level_top=LEVEL_TOP, level_bot=LEVEL_BOT)

sources = [load_inventory(EdgarSource(), grid; year=2022)]

output = NetCDFOutputWriter(
    expanduser(get(ENV, "OUTFILE", "~/data/output/era5_edgar.nc")),
    Dict(:co2 => ColumnMeanDiagnostic(:co2)),
    TimeIntervalSchedule(3600.0))

model = TransportModel(; grid, tracers=(;co2=nothing), met_data=met,
    Δt=900.0, sources, output_writers=[output], buffering=DoubleBuffer())

run!(model)
