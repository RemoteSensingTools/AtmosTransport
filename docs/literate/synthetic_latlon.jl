# # Tutorial: synthetic lat-lon end-to-end
#
# This tutorial builds a tiny synthetic transport binary in memory,
# loads it with the runtime driver, runs a couple of advection steps,
# and checks mass conservation — all without any external met data.
# Everything here uses public API; the same pattern underlies
# `test/test_driven_simulation.jl`.
#
# Use this as a template when you want to:
#
# - smoke-test the runtime on a fresh install,
# - explore the data flow (binary → driver → state → simulation),
# - build a custom synthetic case for a unit test.

using AtmosTransport

# ## 1. Build a tiny synthetic binary in memory
#
# We assemble two windows of trivially flat fields on an 8 × 4 × 2
# lat-lon grid. Mass fluxes are zero (no actual transport), so air
# mass should hold constant and tracer mixing ratio should not move.

function build_demo_latlon_binary(path::AbstractString;
                                  FT::Type{<:AbstractFloat} = Float64,
                                  Nx::Int = 8, Ny::Int = 4, Nz::Int = 2,
                                  nwindow::Int = 2)
    mesh     = LatLonMesh(; FT = FT, Nx = Nx, Ny = Ny)
    vertical = HybridSigmaPressure(FT[0, 100, 300], FT[0, 0, 1])
    grid     = AtmosGrid(mesh, vertical, CPU(); FT = FT)

    windows = [
        (; m  = ones(FT, Nx, Ny, Nz),
           am = zeros(FT, Nx + 1, Ny, Nz),
           bm = zeros(FT, Nx, Ny + 1, Nz),
           cm = zeros(FT, Nx, Ny, Nz + 1),
           ps = fill(FT(95_000 + 100w), Nx, Ny),
           qv_start = fill(FT(0.01w),         Nx, Ny, Nz),
           qv_end   = fill(FT(0.01w + 0.01),  Nx, Ny, Nz),
           dam      = zeros(FT, Nx + 1, Ny, Nz),
           dbm      = zeros(FT, Nx, Ny + 1, Nz),
           dcm      = zeros(FT, Nx, Ny, Nz + 1),
           dm       = zeros(FT, Nx, Ny, Nz))
        for w in 1:nwindow
    ]

    write_transport_binary(path, grid, windows;
                           FT                   = FT,
                           dt_met_seconds       = 3600.0,
                           half_dt_seconds      = 1800.0,
                           steps_per_window     = 2,
                           mass_basis           = :dry,
                           source_flux_sampling = :window_start_endpoint,
                           flux_sampling        = :window_constant,
                           extra_header         = Dict(
                               "poisson_balance_target_scale" => 0.25,
                               "poisson_balance_target_semantics" =>
                                   "forward_window_mass_difference / (2 * steps_per_window)",
                           ))
    return grid
end

bin_path = joinpath(mktempdir(), "synthetic_latlon.bin")
grid = build_demo_latlon_binary(bin_path; FT = Float64)
@info "Wrote synthetic binary" bin_path

# ## 2. Inspect the binary
#
# `inspect_binary` is the public diagnostic; it returns a NamedTuple
# of capability flags so a runtime can decide which operators are
# eligible.

caps = AtmosTransport.inspect_binary(bin_path)
caps

# ## 3. Construct driver, state, model, simulation
#
# - `TransportBinaryDriver` memory-maps the binary.
# - `CellState` carries air mass + tracers.
# - `TransportModel` bundles state + fluxes + grid + advection scheme.
# - `DrivenSimulation` ties the model to the driver and steps it
#   forward window by window.

driver = TransportBinaryDriver(bin_path; FT = Float64, arch = CPU())

state = CellState(DryBasis,
                  ones(Float64, 8, 4, 2);
                  CO2 = fill(400e-6, 8, 4, 2))

fluxes = allocate_face_fluxes(grid.horizontal, 2; FT = Float64,
                              basis = DryBasis)

model = TransportModel(state, fluxes, grid, UpwindScheme())

sim = DrivenSimulation(model, driver;
                       start_window = 1,
                       stop_window  = 2);    # `;` suppresses the object dump

# ## 4. Step the simulation
#
# Each `step!` advances by one substep (here: 30 minutes; the binary
# declares `dt_met_seconds = 3600` and `steps_per_window = 2`).

step!(sim)
step!(sim)
step!(sim)

# ## 5. Confirm mass conservation
#
# With zero mass fluxes everywhere, air mass should be exactly 1.0
# and the dry CO₂ VMR should be exactly 4.0e-4.

using Statistics
m_min, m_max = extrema(sim.model.state.air_mass)
@info "Air mass extrema after 3 steps" m_min m_max

vmr = mixing_ratio(sim.model.state, :CO2)
vmr_min, vmr_max = extrema(vmr)
@info "CO2 dry-VMR extrema after 3 steps" vmr_min vmr_max

# Cleanup the driver (closes the memory map):
close(driver)

# ## What's next
#
# - Replace the zero-flux `windows` builder with one that diverges
#   actual mass — the runtime will then transport the CO₂ tracer.
# - Swap `UpwindScheme()` for `SlopesScheme()` or `PPMScheme()` to
#   see scheme-dependent behavior on the same forcing.
# - Add a non-trivial initial condition to `CellState`'s `CO2 = …`
#   keyword (e.g. a Gaussian blob) and watch advection move it.
# - When the downloadable quickstart bundle lands, swap the synthetic
#   binary for a real preprocessed ERA5 day from
#   `~/data/AtmosTransport_quickstart/met/era5_ll72x37_dec2021_f32/`
#   and the same code structure runs against real meteorology.
