#!/usr/bin/env julia
# ===========================================================================
# One-command quickstart: download data → run simulation → visualize
#
# Usage:
#   julia --project=. scripts/quickstart.jl
#
# What it does:
#   1. Downloads coarsened GEOS-FP met data via Julia artifact (or OPeNDAP fallback)
#   2. Runs a 7-day CO₂ transport simulation on CPU (~2 min)
#   3. Produces a visualization (PNG snapshot + animated GIF)
#
# Data: ~100 MB, auto-downloaded on first run, cached for subsequent runs.
# No authentication, GPU, or external preprocessing required.
# ===========================================================================

using Pkg

# ---------------------------------------------------------------------------
# Step 1: Ensure met data is available
# ---------------------------------------------------------------------------

const DATA_DIR = joinpath(homedir(), "data", "metDrivers", "geosfp", "quickstart")
const ARTIFACT_NAME = "quickstart_met_data"

function ensure_data()
    # Check if artifact is configured
    artifacts_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    if isfile(artifacts_toml)
        try
            hash = Pkg.Artifacts.artifact_hash(ARTIFACT_NAME, artifacts_toml)
            if hash !== nothing
                path = Pkg.Artifacts.artifact_path(hash)
                if isdir(path) && !isempty(readdir(path))
                    @info "Using artifact data: $path"
                    return path
                end
                Pkg.Artifacts.ensure_artifact_installed(ARTIFACT_NAME, artifacts_toml)
                @info "Downloaded artifact data: $path"
                return path
            end
        catch e
            @warn "Artifact download failed, trying OPeNDAP fallback: $e"
        end
    end

    # Fallback: download via OPeNDAP (no auth required)
    if isdir(DATA_DIR) && !isempty(filter(f -> endswith(f, ".bin"), readdir(DATA_DIR)))
        @info "Using cached quickstart data: $DATA_DIR"
        return DATA_DIR
    end

    @info "Downloading quickstart data via OPeNDAP (no authentication required)..."
    @info "This may take a few minutes on first run."
    include(joinpath(@__DIR__, "build_quickstart_artifact.jl"))
    build_quickstart_data(DATA_DIR)
    return DATA_DIR
end

# ---------------------------------------------------------------------------
# Step 2: Run the simulation
# ---------------------------------------------------------------------------

function run_simulation(data_dir::String)
    # Load AtmosTransport (CPU mode — no GPU needed)
    @info "Loading AtmosTransport..."
    using AtmosTransport
    using AtmosTransport.IO: build_model_from_config
    import AtmosTransport.Models: run!
    import TOML

    # Read config and override data path
    config_path = joinpath(@__DIR__, "..", "config", "runs", "quickstart.toml")
    config = TOML.parsefile(config_path)

    # Point met data to resolved path
    config["met_data"]["directory"] = data_dir

    # Put output in current directory
    output_file = joinpath(pwd(), "quickstart_output.nc")
    config["output"]["filename"] = output_file

    @info "Building model (CPU, 144×73×72, 7 days)..."
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
    println("  7-day CO₂ transport simulation on coarsened GEOS-FP data")
    println("=" ^ 70)
    println()

    data_dir = ensure_data()
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
