#!/usr/bin/env julia
# ===========================================================================
# One-command quickstart: download data → run simulation → visualize
#
# Usage:
#   julia --project=. scripts/quickstart.jl
#
# What it does:
#   1. Downloads preprocessed GEOS-IT C180 data via Julia artifact (~700 MB)
#   2. Runs a 1-day CO₂ transport simulation on cubed-sphere C180 (~0.5°)
#   3. Produces a lat-lon regridded visualization (PNG snapshot)
#
# Data: ~700 MB compressed, auto-downloaded on first run, cached thereafter.
# No authentication, GPU, or manual preprocessing required.
# ===========================================================================

using Pkg

# ---------------------------------------------------------------------------
# Step 1: Ensure met data is available
# ---------------------------------------------------------------------------

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
            @warn "Artifact download failed: $e"
        end
    end

    @warn "Quickstart artifact not configured in Artifacts.toml."
    @info "To build the artifact locally (maintainer only):"
    @info "  julia --project=. scripts/build_quickstart_artifact.jl"
    error("No quickstart data available. See above for instructions.")
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

    # Point met data to resolved preprocessed directory
    config["met_data"]["preprocessed_dir"] = data_dir

    # Put output in current directory
    output_file = joinpath(pwd(), "quickstart_output.nc")
    config["output"]["filename"] = output_file

    @info "Building model (CPU, C180 cubed-sphere, 1 day)..."
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
    println("  1-day CO₂ transport on GEOS-IT C180 cubed-sphere (~0.5°)")
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
