#!/usr/bin/env julia
# ===========================================================================
# One-command quickstart: check data → run simulation → visualize
#
# Usage:
#   julia --project=. scripts/quickstart.jl
#
# Data setup (one-time):
#   1. Download the quickstart tarball (~1.4 GB) from:
#      https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-v1
#   2. Extract:
#      mkdir -p ~/data/AtmosTransport
#      tar -xzf quickstart_met_data.tar.gz -C ~/data/AtmosTransport/
#
# What it does:
#   1. Runs a 12-hour CO₂ transport simulation on cubed-sphere C180 (~0.5°)
#   2. Uses EDGAR v8.0 anthropogenic CO₂ emissions
#   3. Produces a lat-lon regridded visualization (PNG snapshot)
# ===========================================================================

const DATA_DIR = expanduser("~/data/AtmosTransport/quickstart_met_data")
const RELEASE_URL = "https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-v1"

# ---------------------------------------------------------------------------
# Step 1: Check that data is available
# ---------------------------------------------------------------------------

function check_data()
    if !isdir(DATA_DIR)
        println(stderr, """
        ERROR: Quickstart data not found at:
          $DATA_DIR

        Download and extract the data first:
          1. Download quickstart_met_data.tar.gz from:
             $RELEASE_URL
          2. Extract:
             mkdir -p ~/data/AtmosTransport
             tar -xzf quickstart_met_data.tar.gz -C ~/data/AtmosTransport/
        """)
        exit(1)
    end

    # Check for met binary
    bin_files = filter(f -> endswith(f, ".bin") && startswith(f, "geosfp_cs"),
                       readdir(DATA_DIR))
    if isempty(bin_files)
        println(stderr, "ERROR: No met binary files found in $DATA_DIR")
        exit(1)
    end

    @info "Quickstart data found: $DATA_DIR"
    return DATA_DIR
end

# ---------------------------------------------------------------------------
# Step 2: Run the simulation
# ---------------------------------------------------------------------------

function run_simulation(data_dir::String)
    @info "Loading AtmosTransport..."
    using AtmosTransport
    using AtmosTransport.IO: build_model_from_config
    import AtmosTransport.Models: run!
    import TOML

    config_path = joinpath(@__DIR__, "..", "config", "runs", "quickstart.toml")
    config = TOML.parsefile(config_path)

    # Override paths to use verified data directory
    config["met_data"]["preprocessed_dir"] = data_dir

    edgar_file = joinpath(data_dir, "edgar_co2_cs_c180_float32.bin")
    if isfile(edgar_file)
        config["tracers"]["co2"]["edgar_file"] = edgar_file
    end

    # Put output in current directory
    output_file = joinpath(pwd(), "quickstart_output.nc")
    config["output"]["filename"] = output_file

    @info "Building model (CPU, C180 cubed-sphere, 12 hours)..."
    model = build_model_from_config(config)

    @info "Running simulation..."
    t0 = time()
    run!(model)
    elapsed = round(time() - t0, digits=1)
    @info "Simulation complete in $(elapsed)s — output: $output_file"

    return output_file
end

# ---------------------------------------------------------------------------
# Step 3: Visualize results
# ---------------------------------------------------------------------------

function visualize(nc_file::String)
    if !isfile(nc_file)
        @warn "Output file not found: $nc_file — skipping visualization"
        return
    end

    @info "Creating visualization..."
    try
        using CairoMakie
        using NCDatasets

        NCDataset(nc_file, "r") do ds
            lon = ds["lon"][:]
            lat = ds["lat"][:]
            time_var = ds["time"][:]
            Nt = length(time_var)

            # Snapshot at final timestep
            col_mean = ds["co2_column_mean"][:, :, Nt]
            sfc      = ds["co2_surface"][:, :, Nt]

            fig = Figure(; size=(1200, 500), fontsize=13)

            ax1 = Axis(fig[1, 1]; title="Column-mean CO₂ (ppm)", xlabel="Lon", ylabel="Lat")
            hm1 = heatmap!(ax1, lon, lat, col_mean'; colormap=:YlOrRd)
            Colorbar(fig[1, 2], hm1; label="ppm")

            ax2 = Axis(fig[1, 3]; title="Surface CO₂ (ppm)", xlabel="Lon", ylabel="Lat")
            hm2 = heatmap!(ax2, lon, lat, sfc'; colormap=:YlOrRd)
            Colorbar(fig[1, 4], hm2; label="ppm")

            png_path = replace(nc_file, ".nc" => "_snapshot.png")
            save(png_path, fig; px_per_unit=2)
            @info "Snapshot saved: $png_path"
        end
    catch e
        @warn "Visualization failed (CairoMakie not installed?): $e"
        @info "Install with: using Pkg; Pkg.add(\"CairoMakie\")"
        @info "Output NetCDF can be viewed with any NetCDF viewer (e.g. Panoply, ncview)"
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    println("=" ^ 70)
    println("  AtmosTransport Quickstart")
    println("  12-hour CO₂ transport on GEOS-IT C180 cubed-sphere (~0.5°)")
    println("=" ^ 70)
    println()

    data_dir = check_data()
    nc_file  = run_simulation(data_dir)
    visualize(nc_file)

    println()
    println("=" ^ 70)
    println("  Done! Output: $nc_file")
    println()
    println("  Next steps:")
    println("    - View output with: ncview $nc_file")
    println("    - Try GPU mode: set use_gpu=true in config (requires CUDA or Metal)")
    println("    - Run a real simulation: see docs/QUICKSTART.md")
    println("=" ^ 70)
end

main()
