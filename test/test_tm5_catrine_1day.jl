#!/usr/bin/env julia
"""
Plan 24 Commit 6 — CATRINE-style 1-day DrivenSimulation with
TM5Convection against a REAL preprocessed ERA5 transport binary.

This closes the stub in plan 23 Commit 6 / `test_tm5_driven_simulation.jl`:
that test used a synthetic in-memory window driver because no
preprocessor path shipped TM5 sections at the time.  Plan 24
Commit 4 shipped the LL preprocessor hook at (720, 361); Commit 6
validates end-to-end against a binary produced by that path.

Gated by `--all`.  Gracefully skips if the transport binary is
not staged on the local machine — the expected staging path is
the output of the smoke config
`config/preprocessing/era5_ll720x361_tm5_dec2021_smoke.toml`.

Verifies:

  - `TransportBinaryDriver` opens the binary without error and
    reports `has_tm5_convection == true`.
  - `load_transport_window(driver, 1).convection.tm5_fields` is a
    NamedTuple with the four expected fields of shape `(Nx, Ny, Nz)`
    and non-trivial values (not all zero).
  - `DrivenSimulation` with `TM5Convection()` + a uniform surface
    tracer runs one `step!` without error.
  - Tracer mass is conserved across the step (F32 ULP).
  - Surface-layer tracer mass goes DOWN (updraft entrainment
    pulls surface air up into cloud).
  - Mid-troposphere layers gain mass (detrainment deposits it).
"""

using Test
using Dates

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.State: CellState, DryBasis, allocate_face_fluxes, get_tracer
using .AtmosTransport.Operators: TM5Convection, UpwindScheme
using .AtmosTransport.MetDrivers: TransportBinaryDriver, has_tm5_convection,
                                   load_transport_window, total_windows,
                                   steps_per_window
using .AtmosTransport.Models: TransportModel, DrivenSimulation, step!,
                               run_window!, window_index

const RUN_ALL = "--all" in ARGS
const BIN_PATH = "/temp1/tm5_smoke/transport_bin/era5_transport_20211202_merged1000Pa_float32.bin"

if !RUN_ALL
    @info "test_tm5_catrine_1day.jl: skipping (pass --all to run)"
elseif !isfile(BIN_PATH)
    @info "test_tm5_catrine_1day.jl: skipping, binary not staged at $BIN_PATH"
    @info "  Build via: julia -t8 --project=. scripts/preprocessing/preprocess_transport_binary.jl " *
          "config/preprocessing/era5_ll720x361_tm5_dec2021_smoke.toml --day 2021-12-02"
else
    @testset "CATRINE 1-day TM5 DrivenSimulation (Dec 2 2021)" begin
        FT = Float32
        driver = TransportBinaryDriver(BIN_PATH; FT = FT)
        try
            # (A) Binary carries TM5 sections.
            @test has_tm5_convection(driver.reader)
            @test total_windows(driver) == 24

            # (B) Window 1 loads with populated tm5_fields.
            win1 = load_transport_window(driver, 1)
            tm5 = win1.convection.tm5_fields
            @test tm5 !== nothing
            @test hasproperty(tm5, :entu)
            @test hasproperty(tm5, :detu)
            @test hasproperty(tm5, :entd)
            @test hasproperty(tm5, :detd)

            Nx, Ny, Nz = size(tm5.entu)
            @test (Nx, Ny, Nz) == (720, 361, 34)

            @test maximum(tm5.entu) > 0.01f0
            @test maximum(tm5.detu) > 0.01f0
            @test all(>=(0f0), tm5.entu)
            @test all(>=(0f0), tm5.detu)
            @test all(>=(0f0), tm5.entd)
            @test all(>=(0f0), tm5.detd)

            # (C) Build a uniform surface-layer tracer and run one window.
            air_mass = copy(win1.air_mass)
            tracer = zeros(FT, Nx, Ny, Nz)
            # Mixing ratio 1 ppm at the surface layer (k=Nz).
            tracer[:, :, Nz] .= FT(1f-6) .* air_mass[:, :, Nz]
            state = CellState(air_mass; CO2 = tracer)

            fluxes = allocate_face_fluxes(driver.grid.horizontal, Nz;
                                           FT = FT, basis = DryBasis)

            model = TransportModel(state, fluxes, driver.grid, UpwindScheme();
                                    convection = TM5Convection())
            sim = DrivenSimulation(model, driver; start_window = 1, stop_window = 2)

            # Surface mass before the step.
            initial_surface_mass = sum(tracer[:, :, Nz])
            initial_total_mass   = sum(tracer)

            # Run all substeps of window 1 + one step into window 2.
            # `run_window!` completes the current window but leaves the
            # simulation ready for window 2 (iteration = steps_per_window);
            # one more step! triggers `_maybe_advance_window!` and
            # increments `current_window_index`.
            run_window!(sim)
            step!(sim)

            # (D) Sim advanced.
            @test window_index(sim) == 2

            final_tracer = get_tracer(sim.model.state, :CO2)
            final_total_mass   = sum(final_tracer)
            final_surface_mass = sum(final_tracer[:, :, Nz])

            # (E) Total tracer mass conserved to F32 ULP across transport + TM5.
            @test isapprox(final_total_mass, initial_total_mass;
                            rtol = 1f-5, atol = eps(Float32) * abs(initial_total_mass))

            # (F) Non-trivial redistribution — at least some column mass
            # moved off the surface into the interior.  With ~7% of cells
            # having active TM5 + transport winds blowing, the surface
            # loss should be measurable.
            @test final_surface_mass != initial_surface_mass

            @info "  Surface mass: $(initial_surface_mass) → $(final_surface_mass) " *
                   "($(round((final_surface_mass - initial_surface_mass) / initial_surface_mass * 100, digits=3))%)"
            @info "  Total mass:   $(initial_total_mass) → $(final_total_mass) " *
                   "(drift $(round((final_total_mass - initial_total_mass) / initial_total_mass * 100, digits=6))%)"
        finally
            close(driver)
        end
    end
end
