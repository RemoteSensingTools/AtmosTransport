#!/usr/bin/env julia
# ===========================================================================
# GEOS-FP C720 Cubed-Sphere + EDGAR CO2 Forward Transport
#
# Minimal script — all logic lives in the core library.
# Configure via TOML or programmatic setup below.
#
# Usage:
#   julia --project=. scripts/run_geosfp_cs_edgar.jl
#   # or with TOML:
#   julia --project=. -e '
#       using AtmosTransport
#       model = build_model_from_config("config/runs/geosfp_cs_edgar.toml")
#       run!(model)
#   '
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures: GPU, CPU

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "true"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

using AtmosTransport.Grids: CubedSphereGrid
using AtmosTransport.IO: build_model_from_config, default_met_config,
                              build_vertical_coordinate,
                              GEOSFPCubedSphereMetDriver, LatLonOutputGrid,
                              NetCDFOutputWriter, TimeIntervalSchedule
using AtmosTransport.Sources: EdgarSource, load_inventory
using AtmosTransport.Diagnostics: ColumnMeanDiagnostic
using AtmosTransport.Models: TransportModel, SingleBuffer, DoubleBuffer
import AtmosTransport.Models: run!
using AtmosTransport.Parameters: load_parameters
using Dates

# --- Configuration ---
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "true"))
const FT = USE_FLOAT32 ? Float32 : Float64

arch = USE_GPU ? GPU() : CPU()
params = load_parameters(FT)
pp = params.planet

config = default_met_config("geosfp")
vc = build_vertical_coordinate(config; FT)

grid = CubedSphereGrid(arch; FT, Nc=720,
    vertical=vc,
    radius=pp.radius, gravity=pp.gravity,
    reference_pressure=pp.reference_surface_pressure)

met = GEOSFPCubedSphereMetDriver(; FT,
    preprocessed_dir=expanduser(get(ENV, "PREPROCESSED_DIR", "")),
    start_date=Date(get(ENV, "GEOSFP_START", "2024-06-01")),
    end_date=Date(get(ENV, "GEOSFP_END", "2024-06-05")),
    dt=parse(FT, get(ENV, "DT", "900")),
    met_interval=FT(3600))

sources = [load_inventory(EdgarSource(), grid; year=2022)]

output = NetCDFOutputWriter(
    expanduser(get(ENV, "OUTFILE", "~/data/output/geosfp_cs_edgar.nc")),
    Dict(:co2 => ColumnMeanDiagnostic(:co2)),
    TimeIntervalSchedule(3600.0);
    output_grid=LatLonOutputGrid(720, 361))

const BUFFERING = get(ENV, "BUFFERING", "double")
buffering = BUFFERING == "single" ? SingleBuffer() : DoubleBuffer()

model = TransportModel(; grid, tracers=(;co2=nothing), met_data=met,
    Δt=900.0, sources, output_writers=[output], buffering)

run!(model)
