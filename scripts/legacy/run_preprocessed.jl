#!/usr/bin/env julia
# ===========================================================================
# Forward Transport from Preprocessed Mass Fluxes + EDGAR CO2
#
# Minimal script — all logic lives in the core library.
#
# Usage:
#   MASSFLUX_DIR=~/data/metDrivers/era5/preprocessed \
#     USE_GPU=true julia --project=. scripts/run_preprocessed.jl
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures: GPU, CPU
using AtmosTransportModel.Grids: LatitudeLongitudeGrid
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients,
                              PreprocessedLatLonMetDriver,
                              find_massflux_shards, MassFluxBinaryReader,
                              NetCDFOutputWriter, TimeIntervalSchedule
using AtmosTransportModel.Sources: EdgarSource, load_inventory
using AtmosTransportModel.Diagnostics: ColumnMeanDiagnostic
using AtmosTransportModel.Models: TransportModel, SingleBuffer, run!
using AtmosTransportModel.Parameters: load_parameters

# --- Configuration ---
const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "false"))
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "false"))
const FT = USE_FLOAT32 ? Float32 : Float64

arch = USE_GPU ? GPU() : CPU()

# --- Discover mass flux files ---
ft_tag = USE_FLOAT32 ? "float32" : "float64"
mf_dir = expanduser(get(ENV, "MASSFLUX_DIR", "~/data/metDrivers/era5/preprocessed"))
files = find_massflux_shards(mf_dir, ft_tag)
isempty(files) && error("No mass-flux files found in $mf_dir")

met = PreprocessedLatLonMetDriver(; FT, files)

# --- Build grid from met driver metadata ---
params = load_parameters(FT)
pp = params.planet
config = default_met_config("era5")
vc = build_vertical_coordinate(config; FT,
    level_range=met.level_top:met.level_bot)

grid = LatitudeLongitudeGrid(arch; FT, size=(met.Nx, met.Ny, met.Nz),
    longitude=(FT(met.lons[1]), FT(met.lons[end]) + FT(met.lons[2] - met.lons[1])),
    latitude=(FT(-90), FT(90)),
    vertical=vc,
    radius=pp.radius, gravity=pp.gravity,
    reference_pressure=pp.reference_surface_pressure)

sources = [load_inventory(EdgarSource(), grid; year=2022)]

output = NetCDFOutputWriter(
    expanduser(get(ENV, "OUTFILE", "~/data/output/preprocessed_edgar.nc")),
    Dict(:co2 => ColumnMeanDiagnostic(:co2)),
    TimeIntervalSchedule(3600.0))

model = TransportModel(; grid, tracers=(;co2=nothing), met_data=met,
    Δt=met.dt, sources, output_writers=[output], buffering=SingleBuffer())

run!(model)
