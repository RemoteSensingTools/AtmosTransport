#!/usr/bin/env julia
# Diagnostic: check mass values before/after halo exchange on CPU
# to identify why CFL explodes at panel boundaries for C180

using Pkg
Pkg.activate(".")

using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels, cgrid_to_staggered_panels
using AtmosTransport.Grids: CubedSphereGrid, fill_panel_halos!, default_panel_connectivity
using AtmosTransport.Architectures: CPU
using AtmosTransport.Advection: compute_air_mass_panel!, max_cfl_x_cs, max_cfl_y_cs
using Printf

FT = Float32
Nc = 180
Hp = 3
Nz = 72

# Read one timestep from the GEOS-IT C180 file
filepath = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc")
@info "Reading $filepath..."
ts = read_geosfp_cs_timestep(filepath; FT, time_index=1, convert_to_kgs=true)

# Create haloed delp and staggered mass fluxes
delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)
am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

# Build grid for area/gravity
@info "Building CubedSphereGrid..."
grid = CubedSphereGrid(CPU(); FT, Nc,
    vertical=AtmosTransport.Parameters.load_parameters(FT) |> p -> AtmosTransport.IO.default_met_config("geosfp") |> mc -> AtmosTransport.IO.build_vertical_coordinate(mc; FT),
    radius=FT(6.371e6), gravity=FT(9.81),
    reference_pressure=FT(101325.0))

g = FT(9.81)

# Compute air mass for all panels (interior only)
m_panels = ntuple(6) do _
    zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)
end

@info "Computing air mass..."
for p in 1:6
    area_p = Matrix{FT}(undef, Nc, Nc)
    @inbounds for j in 1:Nc, i in 1:Nc
        area_p[i, j] = FT(grid.Aᶜ[p][i, j])
    end
    compute_air_mass_panel!(m_panels[p], delp_haloed[p], area_p, g, Nc, Nz, Hp)
end

# Print interior mass statistics per panel
println("\n==== INTERIOR MASS (before halo fill) ====")
for p in 1:6
    m_int = @view m_panels[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]
    @printf("  Panel %d: min=%.4e  max=%.4e  mean=%.4e\n",
            p, minimum(m_int), maximum(m_int), sum(m_int)/length(m_int))
end

# Print boundary mass values (first/last interior row/column)
println("\n==== BOUNDARY INTERIOR MASS (at edges, k=36 mid-level) ====")
k = 36  # mid-atmosphere level
for p in 1:6
    m = m_panels[p]
    west  = m[Hp+1, Hp+1:Hp+Nc, k]
    east  = m[Hp+Nc, Hp+1:Hp+Nc, k]
    south = m[Hp+1:Hp+Nc, Hp+1, k]
    north = m[Hp+1:Hp+Nc, Hp+Nc, k]
    @printf("  Panel %d: west=[%.3e,%.3e] east=[%.3e,%.3e] south=[%.3e,%.3e] north=[%.3e,%.3e]\n",
            p, minimum(west), maximum(west), minimum(east), maximum(east),
            minimum(south), maximum(south), minimum(north), maximum(north))
end

# Print halo mass before fill (should be all zeros)
println("\n==== HALO MASS BEFORE FILL (k=36, should be zero) ====")
for p in 1:6
    m = m_panels[p]
    # West halo: column Hp (just outside interior)
    west_halo  = m[Hp, Hp+1:Hp+Nc, k]
    east_halo  = m[Hp+Nc+1, Hp+1:Hp+Nc, k]
    south_halo = m[Hp+1:Hp+Nc, Hp, k]
    north_halo = m[Hp+1:Hp+Nc, Hp+Nc+1, k]
    @printf("  Panel %d: west=[%.3e,%.3e] east=[%.3e,%.3e] south=[%.3e,%.3e] north=[%.3e,%.3e]\n",
            p, minimum(west_halo), maximum(west_halo), minimum(east_halo), maximum(east_halo),
            minimum(south_halo), maximum(south_halo), minimum(north_halo), maximum(north_halo))
end

# Fill halos
@info "Filling panel halos..."
fill_panel_halos!(m_panels, grid)

# Print halo mass after fill
println("\n==== HALO MASS AFTER FILL (k=36) ====")
for p in 1:6
    m = m_panels[p]
    west_halo  = m[Hp, Hp+1:Hp+Nc, k]
    east_halo  = m[Hp+Nc+1, Hp+1:Hp+Nc, k]
    south_halo = m[Hp+1:Hp+Nc, Hp, k]
    north_halo = m[Hp+1:Hp+Nc, Hp+Nc+1, k]
    @printf("  Panel %d: west=[%.3e,%.3e] east=[%.3e,%.3e] south=[%.3e,%.3e] north=[%.3e,%.3e]\n",
            p, minimum(west_halo), maximum(west_halo), minimum(east_halo), maximum(east_halo),
            minimum(south_halo), maximum(south_halo), minimum(north_halo), maximum(north_halo))
end

# Compare: ratio of interior boundary mass to halo mass
println("\n==== RATIO: interior_boundary / halo (should be ~1.0) ====")
for p in 1:6
    m = m_panels[p]
    # West: interior[Hp+1,:] vs halo[Hp,:]
    int_w = m[Hp+1, Hp+1:Hp+Nc, k]
    halo_w = m[Hp, Hp+1:Hp+Nc, k]
    valid_w = halo_w .> 0
    if any(valid_w)
        ratios_w = int_w[valid_w] ./ halo_w[valid_w]
        @printf("  Panel %d west:  min_ratio=%.3f  max_ratio=%.3f  mean_ratio=%.3f (n_valid=%d/%d)\n",
                p, minimum(ratios_w), maximum(ratios_w), sum(ratios_w)/length(ratios_w),
                count(valid_w), length(halo_w))
    else
        @printf("  Panel %d west:  ALL HALO ZERO!\n", p)
    end

    # East: interior[Hp+Nc,:] vs halo[Hp+Nc+1,:]
    int_e = m[Hp+Nc, Hp+1:Hp+Nc, k]
    halo_e = m[Hp+Nc+1, Hp+1:Hp+Nc, k]
    valid_e = halo_e .> 0
    if any(valid_e)
        ratios_e = int_e[valid_e] ./ halo_e[valid_e]
        @printf("  Panel %d east:  min_ratio=%.3f  max_ratio=%.3f  mean_ratio=%.3f\n",
                p, minimum(ratios_e), maximum(ratios_e), sum(ratios_e)/length(ratios_e))
    else
        @printf("  Panel %d east:  ALL HALO ZERO!\n", p)
    end

    # South: interior[:,Hp+1] vs halo[:,Hp]
    int_s = m[Hp+1:Hp+Nc, Hp+1, k]
    halo_s = m[Hp+1:Hp+Nc, Hp, k]
    valid_s = halo_s .> 0
    if any(valid_s)
        ratios_s = int_s[valid_s] ./ halo_s[valid_s]
        @printf("  Panel %d south: min_ratio=%.3f  max_ratio=%.3f  mean_ratio=%.3f\n",
                p, minimum(ratios_s), maximum(ratios_s), sum(ratios_s)/length(ratios_s))
    else
        @printf("  Panel %d south: ALL HALO ZERO!\n", p)
    end

    # North: interior[:,Hp+Nc] vs halo[:,Hp+Nc+1]
    int_n = m[Hp+1:Hp+Nc, Hp+Nc, k]
    halo_n = m[Hp+1:Hp+Nc, Hp+Nc+1, k]
    valid_n = halo_n .> 0
    if any(valid_n)
        ratios_n = int_n[valid_n] ./ halo_n[valid_n]
        @printf("  Panel %d north: min_ratio=%.3f  max_ratio=%.3f  mean_ratio=%.3f\n",
                p, minimum(ratios_n), maximum(ratios_n), sum(ratios_n)/length(ratios_n))
    else
        @printf("  Panel %d north: ALL HALO ZERO!\n", p)
    end
end

# Compute CFL with halo-filled mass
println("\n==== CFL WITH HALO-FILLED MASS ====")
half_dt = FT(900.0 / 2)
for p in 1:6
    am_stag[p] .*= half_dt
    bm_stag[p] .*= half_dt
end

# Allocate CFL scratch arrays at exact sizes
cfl_x = zeros(FT, Nc+1, Nc, Nz)
cfl_y = zeros(FT, Nc, Nc+1, Nz)

for p in 1:6
    cx = max_cfl_x_cs(am_stag[p], m_panels[p], cfl_x, Hp)
    cy = max_cfl_y_cs(bm_stag[p], m_panels[p], cfl_y, Hp)
    @printf("  Panel %d: max_cfl_x=%.4f  max_cfl_y=%.4f\n", p, cx, cy)

    # Find WHERE the max CFL occurs in x
    max_idx = argmax(cfl_x)
    i, j, k2 = Tuple(max_idx)
    @printf("    max_cfl_x at (i=%d, j=%d, k=%d): am=%.4e  m_upwind=%.4e\n",
            i, j, k2, am_stag[p][i, j, k2],
            am_stag[p][i, j, k2] >= 0 ? m_panels[p][Hp+i-1, Hp+j, k2] : m_panels[p][Hp+i, Hp+j, k2])
end

println("\nDone.")
