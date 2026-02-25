#!/usr/bin/env julia
# GPU diagnostic: check halo exchange on GPU vs CPU for C180
# If halos differ between CPU and GPU, the bug is in the GPU kernel path.

using Pkg
Pkg.activate(".")

using CUDA
using AtmosTransport
using AtmosTransport.IO: read_geosfp_cs_timestep, to_haloed_panels, cgrid_to_staggered_panels
using AtmosTransport.Grids: CubedSphereGrid, fill_panel_halos!, allocate_cubed_sphere_field
using AtmosTransport.Architectures: CPU, GPU as GPUArch, array_type
using AtmosTransport.Advection: compute_air_mass_panel!, max_cfl_x_cs, max_cfl_y_cs,
                                 build_geometry_cache, allocate_cs_massflux_workspace
using Printf

FT = Float32
Nc = 180
Hp = 3
Nz = 72

# Read one timestep
filepath = expanduser("~/data/geosit_c180/20230601/GEOSIT.20230601.CTM_A1.C180.nc")
@info "Reading $filepath..."
ts = read_geosfp_cs_timestep(filepath; FT, time_index=1, convert_to_kgs=true)

delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp)
am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

# Build GPU grid
@info "Building GPU CubedSphereGrid..."
params = AtmosTransport.Parameters.load_parameters(FT)
pp = params.planet
mc = AtmosTransport.IO.default_met_config("geosfp")
vc = AtmosTransport.IO.build_vertical_coordinate(mc; FT)

grid_gpu = CubedSphereGrid(GPUArch(); FT, Nc,
    vertical=vc, radius=pp.radius, gravity=pp.gravity,
    reference_pressure=pp.reference_surface_pressure)

grid_cpu = CubedSphereGrid(CPU(); FT, Nc,
    vertical=vc, radius=pp.radius, gravity=pp.gravity,
    reference_pressure=pp.reference_surface_pressure)

g = FT(9.81)

# ====== CPU path ======
@info "=== CPU PATH ==="
m_cpu = ntuple(_ -> zeros(FT, Nc+2Hp, Nc+2Hp, Nz), 6)
for p in 1:6
    area_p = Matrix{FT}(undef, Nc, Nc)
    @inbounds for j in 1:Nc, i in 1:Nc
        area_p[i, j] = FT(grid_cpu.Aᶜ[p][i, j])
    end
    compute_air_mass_panel!(m_cpu[p], delp_haloed[p], area_p, g, Nc, Nz, Hp)
end
fill_panel_halos!(m_cpu, grid_cpu)

# ====== GPU path ======
@info "=== GPU PATH ==="
# Upload delp to GPU
delp_gpu = ntuple(6) do p
    CuArray(delp_haloed[p])
end

# Allocate GPU mass panels (zeros)
m_gpu = ntuple(6) do _
    CuArray(zeros(FT, Nc+2Hp, Nc+2Hp, Nz))
end

# Build geometry cache on GPU
ref_panel = m_gpu[1]
gc = build_geometry_cache(grid_gpu, ref_panel)

# Compute air mass on GPU
for p in 1:6
    compute_air_mass_panel!(m_gpu[p], delp_gpu[p], gc.area[p], gc.gravity, Nc, Nz, Hp)
end
CUDA.synchronize()

# Check GPU mass before halo fill
@info "GPU mass BEFORE halo fill:"
for p in 1:6
    m_arr = Array(m_gpu[p])
    int_max = maximum(view(m_arr, Hp+1:Hp+Nc, Hp+1:Hp+Nc, :))
    halo_w = maximum(abs, view(m_arr, Hp, Hp+1:Hp+Nc, :))
    @printf("  Panel %d: int_max=%.3e  west_halo=%.3e\n", p, int_max, halo_w)
end

# Compare CPU vs GPU interior mass
@info "CPU vs GPU interior mass comparison:"
for p in 1:6
    m_gpu_arr = Array(m_gpu[p])
    diff = maximum(abs, view(m_cpu[p], Hp+1:Hp+Nc, Hp+1:Hp+Nc, :) .- view(m_gpu_arr, Hp+1:Hp+Nc, Hp+1:Hp+Nc, :))
    rel_diff = diff / maximum(view(m_cpu[p], Hp+1:Hp+Nc, Hp+1:Hp+Nc, :))
    @printf("  Panel %d: max_abs_diff=%.3e  max_rel_diff=%.3e\n", p, diff, rel_diff)
end

# Fill halos on GPU
@info "Filling GPU halos..."
fill_panel_halos!(m_gpu, grid_gpu)
CUDA.synchronize()

# Check GPU halo values
@info "GPU halo values AFTER fill:"
for p in 1:6
    m_arr = Array(m_gpu[p])
    int_max = maximum(view(m_arr, Hp+1:Hp+Nc, Hp+1:Hp+Nc, :))
    halo_w = maximum(view(m_arr, Hp, Hp+1:Hp+Nc, :))
    halo_e = maximum(view(m_arr, Hp+Nc+1, Hp+1:Hp+Nc, :))
    halo_s = maximum(view(m_arr, Hp+1:Hp+Nc, Hp, :))
    halo_n = maximum(view(m_arr, Hp+1:Hp+Nc, Hp+Nc+1, :))
    halo_w_min = minimum(view(m_arr, Hp, Hp+1:Hp+Nc, :))
    @printf("  Panel %d: int=%.3e  W=%.3e(min=%.3e) E=%.3e S=%.3e N=%.3e\n",
            p, int_max, halo_w, halo_w_min, halo_e, halo_s, halo_n)
end

# Compare CPU vs GPU halo values
@info "CPU vs GPU HALO comparison:"
for p in 1:6
    m_gpu_arr = Array(m_gpu[p])
    for (label, slice_fn) in [("west",  m -> view(m, Hp, Hp+1:Hp+Nc, :)),
                               ("east",  m -> view(m, Hp+Nc+1, Hp+1:Hp+Nc, :)),
                               ("south", m -> view(m, Hp+1:Hp+Nc, Hp, :)),
                               ("north", m -> view(m, Hp+1:Hp+Nc, Hp+Nc+1, :))]
        cpu_halo = slice_fn(m_cpu[p])
        gpu_halo = slice_fn(m_gpu_arr)
        diff = maximum(abs, cpu_halo .- gpu_halo)
        if maximum(abs, cpu_halo) > 0
            rel = diff / maximum(abs, cpu_halo)
            @printf("  Panel %d %5s: max_diff=%.3e  rel=%.3e  cpu_max=%.3e  gpu_max=%.3e\n",
                    p, label, diff, rel, maximum(cpu_halo), maximum(gpu_halo))
        else
            @printf("  Panel %d %5s: BOTH ZERO\n", p, label)
        end
    end
end

# CFL test
@info "=== CFL TEST ==="
half_dt = FT(900.0 / 2)
am_gpu_t = ntuple(p -> CuArray(am_stag[p] .* half_dt), 6)
bm_gpu_t = ntuple(p -> CuArray(bm_stag[p] .* half_dt), 6)

cfl_x = CuArray(zeros(FT, Nc+1, Nc, Nz))
cfl_y = CuArray(zeros(FT, Nc, Nc+1, Nz))

for p in 1:6
    cx = max_cfl_x_cs(am_gpu_t[p], m_gpu[p], cfl_x, Hp)
    cy = max_cfl_y_cs(bm_gpu_t[p], m_gpu[p], cfl_y, Hp)
    @printf("  Panel %d: GPU cfl_x=%.4f  cfl_y=%.4f\n", p, cx, cy)
end

@info "Done."
