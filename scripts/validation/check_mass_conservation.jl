#!/usr/bin/env julia
#
# Mass conservation check for AtmosTransport output
#
# Verifies that global column mass changes equal cumulative surface emission fluxes:
#   ΔM_column(t) ≈ M_emitted(t) = ∫₀ᵗ Σᵢ [emission_flux(i,τ) × area(i)] dτ
#
# Usage:
#   julia --project=. scripts/check_mass_conservation.jl <output_dir> [species]
#
# Examples:
#   julia --project=. scripts/check_mass_conservation.jl /temp2/catrine-runs/output fossil_co2
#   julia --project=. scripts/check_mass_conservation.jl /temp2/catrine-runs/output sf6

using NCDatasets, Dates, Printf
using CairoMakie

# =============================================================================
# Arguments
# =============================================================================

if length(ARGS) < 1
    println("Usage: julia --project=. scripts/check_mass_conservation.jl <output_dir> [species]")
    println("  output_dir: directory containing *_YYYYMMDD.nc files")
    println("  species: tracer name (default: fossil_co2)")
    exit(1)
end

output_dir = expanduser(ARGS[1])
species = length(ARGS) >= 2 ? ARGS[2] : "fossil_co2"

column_mass_var = "$(species == "fossil_co2" ? "fco2" : species)_column_mass"
emission_var    = "$(species == "fossil_co2" ? "fco2" : species)_emission"

# =============================================================================
# Collect NC files
# =============================================================================

nc_files = sort(filter(f -> endswith(f, ".nc") && occursin(r"_\d{8}\.nc$", f),
                       readdir(output_dir; join=true)))
if isempty(nc_files)
    error("No NC files matching *_YYYYMMDD.nc found in $output_dir")
end
println("Found $(length(nc_files)) NC files")

# =============================================================================
# Compute cell areas from corner coordinates (first file)
# =============================================================================

"""Compute spherical triangle area using excess formula (Girard's theorem)."""
function spherical_polygon_area(lons_deg, lats_deg; R=6.371e6)
    n = length(lons_deg)
    λ = deg2rad.(lons_deg)
    φ = deg2rad.(lats_deg)

    # Convert to Cartesian
    x = cos.(φ) .* cos.(λ)
    y = cos.(φ) .* sin.(λ)
    z = sin.(φ)

    # Sum interior angles using spherical excess
    angle_sum = 0.0
    for i in 1:n
        # vectors from center of sphere to vertices
        j = mod1(i - 1, n)
        k = mod1(i + 1, n)

        # Great circle normals for adjacent edges
        a = [x[j], y[j], z[j]]
        b = [x[i], y[i], z[i]]
        c = [x[k], y[k], z[k]]

        # Normal to plane containing edge (j,i)
        n1 = cross(a, b)
        n1_norm = sqrt(sum(n1 .^ 2))
        n1_norm < 1e-15 && continue
        n1 ./= n1_norm

        # Normal to plane containing edge (i,k)
        n2 = cross(b, c)
        n2_norm = sqrt(sum(n2 .^ 2))
        n2_norm < 1e-15 && continue
        n2 ./= n2_norm

        cos_angle = clamp(dot(n1, n2), -1.0, 1.0)
        angle_sum += π - acos(cos_angle)
    end

    excess = angle_sum - (n - 2) * π
    return R^2 * abs(excess)
end

function cross(a, b)
    return [a[2]*b[3] - a[3]*b[2],
            a[3]*b[1] - a[1]*b[3],
            a[1]*b[2] - a[2]*b[1]]
end

function dot(a, b)
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

function compute_cell_areas(nc_file)
    ds = NCDataset(nc_file)
    # corner_lons/lats: (XCdim=Nc+1, YCdim=Nc+1, nf=6) in Julia ordering
    clons = ds["corner_lons"][:, :, :]  # (Nc+1, Nc+1, 6)
    clats = ds["corner_lats"][:, :, :]
    close(ds)

    Ncp1 = size(clons, 1)
    Nc = Ncp1 - 1
    Nf = size(clons, 3)

    areas = zeros(Float64, Nc, Nc, Nf)
    for p in 1:Nf, j in 1:Nc, i in 1:Nc
        # 4 corners of cell (i,j) on panel p — counterclockwise
        lons4 = Float64[clons[i, j, p], clons[i+1, j, p],
                        clons[i+1, j+1, p], clons[i, j+1, p]]
        lats4 = Float64[clats[i, j, p], clats[i+1, j, p],
                        clats[i+1, j+1, p], clats[i, j+1, p]]
        areas[i, j, p] = spherical_polygon_area(lons4, lats4)
    end
    return areas
end

println("Computing cell areas from corner coordinates...")
cell_areas = compute_cell_areas(nc_files[1])
total_area = sum(cell_areas)
earth_area = 4π * 6.371e6^2
println(@sprintf("  Total grid area: %.6e m² (Earth: %.6e m², ratio: %.6f)",
                 total_area, earth_area, total_area / earth_area))

# =============================================================================
# Read time series of global column mass and emission flux
# =============================================================================

times = DateTime[]
global_column_mass = Float64[]
global_emission_flux = Float64[]  # kg/s at each output time

println("Reading $column_mass_var and $emission_var ...")

for nc_file in nc_files
    ds = NCDataset(nc_file)

    # Check variables exist
    if !(column_mass_var in keys(ds))
        @warn "Variable $column_mass_var not found in $(basename(nc_file)), skipping"
        close(ds)
        continue
    end

    # Time: "minutes since ..." — NCDatasets auto-decodes to DateTime
    t_vals = ds["time"][:]
    Nt = length(t_vals)

    cm = ds[column_mass_var]  # (Xdim, Ydim, nf, time) in Julia
    em = haskey(ds, emission_var) ? ds[emission_var] : nothing

    for ti in 1:Nt
        push!(times, t_vals[ti])

        # Column mass: sum over grid (kg/m² × m² = kg)
        cm_t = Float64.(cm[:, :, :, ti])  # (Nc, Nc, 6)
        push!(global_column_mass, sum(cm_t .* cell_areas))

        # Emission flux: sum over grid (kg/m²/s × m² = kg/s)
        if em !== nothing
            em_t = Float64.(em[:, :, :, ti])
            push!(global_emission_flux, sum(em_t .* cell_areas))
        else
            push!(global_emission_flux, 0.0)
        end
    end

    close(ds)
end

Nt_total = length(times)
println("  Total timesteps: $Nt_total")
println("  Time range: $(times[1]) — $(times[end])")

# =============================================================================
# Compute cumulative emitted mass via trapezoidal integration
# =============================================================================

# Time differences in seconds
dt_seconds = Float64[Dates.value(times[i+1] - times[i]) / 1000.0 for i in 1:Nt_total-1]

cumulative_emission = zeros(Float64, Nt_total)
for i in 2:Nt_total
    # Trapezoidal: ∫ = Σ (f(t_i) + f(t_{i+1}))/2 × Δt
    avg_flux = 0.5 * (global_emission_flux[i-1] + global_emission_flux[i])
    cumulative_emission[i] = cumulative_emission[i-1] + avg_flux * dt_seconds[i-1]
end

# Column mass change relative to first timestep
delta_column_mass = global_column_mass .- global_column_mass[1]

# =============================================================================
# Summary statistics
# =============================================================================

println("\n" * "="^70)
println("Mass Conservation Check — $species")
println("="^70)

final_emitted = cumulative_emission[end]
final_delta_cm = delta_column_mass[end]
ratio = final_delta_cm / final_emitted

println(@sprintf("  Initial global column mass:  %12.6e kg", global_column_mass[1]))
println(@sprintf("  Final global column mass:    %12.6e kg", global_column_mass[end]))
println(@sprintf("  ΔM_column (final - initial): %12.6e kg", final_delta_cm))
println(@sprintf("  Cumulative emission:         %12.6e kg", final_emitted))
println(@sprintf("  Ratio ΔM/emission:           %12.8f", ratio))
println(@sprintf("  Mass deficit:                %12.6e kg (%.4f%%)",
                 final_delta_cm - final_emitted,
                 100.0 * (final_delta_cm - final_emitted) / final_emitted))

# Print timestep-by-timestep table (at most 20 rows)
step = max(1, div(Nt_total, 20))
println("\n  Time                  ΔM_column [kg]    Emitted [kg]    Ratio")
println("  " * "-"^66)
for i in 1:step:Nt_total
    if cumulative_emission[i] > 0
        r = delta_column_mass[i] / cumulative_emission[i]
        println(@sprintf("  %s  %14.6e  %14.6e  %10.6f",
                         times[i], delta_column_mass[i], cumulative_emission[i], r))
    else
        println(@sprintf("  %s  %14.6e  %14.6e       N/A",
                         times[i], delta_column_mass[i], cumulative_emission[i]))
    end
end

# =============================================================================
# Plot
# =============================================================================

hours = Float64[Dates.value(t - times[1]) / (3600 * 1000) for t in times]

fig = Figure(size=(1000, 600))

ax1 = Axis(fig[1, 1];
           xlabel="Hours since start",
           ylabel="Mass [kg]",
           title="Mass Conservation: $species")

lines!(ax1, hours, delta_column_mass; label="ΔM column mass", linewidth=2)
lines!(ax1, hours, cumulative_emission; label="Cumulative emission", linewidth=2,
       linestyle=:dash)
axislegend(ax1; position=:lt)

ax2 = Axis(fig[2, 1];
           xlabel="Hours since start",
           ylabel="Ratio ΔM / Emission",
           title="Conservation ratio (ideal = 1.0)")

# Skip t=0 where emission is zero
valid = cumulative_emission .> 0
ratios = delta_column_mass[valid] ./ cumulative_emission[valid]
lines!(ax2, hours[valid], ratios; linewidth=2, color=:red)
hlines!(ax2, [1.0]; linestyle=:dash, color=:gray)

out_png = joinpath(output_dir, "mass_conservation_$(species).png")
save(out_png, fig)
println("\nPlot saved to: $out_png")
