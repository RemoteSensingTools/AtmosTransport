"""
Plan 17 Commit 7 — Palindrome ordering study.

Question: where should surface emissions `S` sit relative to vertical
diffusion `V` inside the Strang palindrome? Plan 17 §4.3 Decision 7
recommends `X Y Z V(dt/2) S V(dt/2) Z Y X` based on TM5 reasoning;
this test quantifies the layer-surface mass ratio under four candidate
arrangements so the recommendation is grounded in a measurement the
next plan can cite.

Arrangements:

| Label | Arrangement                          | Rationale                              |
|-------|--------------------------------------|----------------------------------------|
| A     | V(dt/2) → S(dt) → V(dt/2)            | OPERATOR_COMPOSITION.md §3.2 (default) |
| B     | S(dt) → V(dt)                        | emissions before single mix            |
| C     | V(dt) → S(dt)                        | single mix then emissions              |
| D     | S(dt) (no V)                         | pathological — expect 100% surface     |

All four share the same initial state, uniform surface flux, constant
Kz, no advection, run for 24 simulated hours at dt = 1 hour.
Diagnostic: layer-surface (k = Nz) mass as a fraction of total column
mass, averaged over all (i, j) cells.

Expected ranking of surface mass fraction:
    D  >  B  ≈  C  >  A

Recorded data (this run, Float64, 4×3×10, Kz = 5 m²/s, rate = 1 kg/s,
dt = 3600 s, 24 steps) is committed to
`docs/resources/developer_notes/plan17_ordering_study.md` (relocated
there from the deleted plan-17 subdirectory during plan-21 cleanup).
"""

using Test
using AtmosTransport
using AtmosTransport: AdvectionWorkspace,
                      SurfaceFluxSource, SurfaceFluxOperator,
                      NoSurfaceFlux, ImplicitVerticalDiffusion,
                      NoDiffusion,
                      apply_surface_flux!, apply_vertical_diffusion!,
                      ConstantField

# =====================================================================
# Helpers — stand-alone 4D buffer + workspace fixture for the study.
# No DrivenSimulation needed; we compose V and S directly on a raw 4D
# tracer buffer so each arrangement is a straight-line sequence.
# =====================================================================

function _fresh_buffer(FT, Nx, Ny, Nz, Nt; dz = 50.0)
    # q_raw is (Nx, Ny, Nz, Nt). Start with zero tracer everywhere.
    q = zeros(FT, Nx, Ny, Nz, Nt)
    # Workspace sized for Nx × Ny × Nz scratch (diffusion needs w, dz).
    dummy_m = ones(FT, Nx, Ny, Nz)
    ws = AdvectionWorkspace(dummy_m; n_tracers = Nt)
    fill!(ws.dz_scratch, FT(dz))   # uniform layer thickness [m]
    return q, ws
end

"Layer-surface (k=Nz) mass as fraction of total column mass, per (i,j)."
function _surface_fraction(q::AbstractArray{FT, 4}, tracer_idx) where FT
    Nx, Ny, Nz, _ = size(q)
    acc = zero(FT)
    @inbounds for j in 1:Ny, i in 1:Nx
        surf    = q[i, j, Nz, tracer_idx]
        col_tot = zero(FT)
        for k in 1:Nz
            col_tot += q[i, j, k, tracer_idx]
        end
        acc += col_tot > zero(FT) ? (surf / col_tot) : zero(FT)
    end
    return acc / (Nx * Ny)
end

"""
    _run_arrangement(arr, dt_full, n_steps; FT, Nx, Ny, Nz, Kz_value, rate_value, dz)

Run `n_steps` of one palindrome arrangement on a fresh 4D buffer and
return the final buffer + average layer-surface mass fraction. `arr`
is one of :A, :B, :C, :D.

Arrangement A: V(dt/2) S(dt) V(dt/2)
Arrangement B: S(dt) V(dt)
Arrangement C: V(dt) S(dt)
Arrangement D: S(dt) only
"""
function _run_arrangement(arr::Symbol, dt_full::Real, n_steps::Int;
                          FT = Float64, Nx = 4, Ny = 3, Nz = 10,
                          Kz_value = 5.0, rate_value = 1.0, dz = 50.0)
    q, ws = _fresh_buffer(FT, Nx, Ny, Nz, 1; dz = dz)

    kz = ConstantField{FT, 3}(FT(Kz_value))
    dfop = ImplicitVerticalDiffusion(; kz_field = kz)
    op = SurfaceFluxOperator(
        SurfaceFluxSource(:CO2, fill(FT(rate_value), Nx, Ny)))
    names = (:CO2,)
    half_dt = FT(dt_full / 2)
    dt_FT   = FT(dt_full)

    for _ in 1:n_steps
        if arr === :A
            apply_vertical_diffusion!(q, dfop, ws, half_dt)
            apply_surface_flux!(q, op, ws, dt_FT, nothing, nothing;
                                tracer_names = names)
            apply_vertical_diffusion!(q, dfop, ws, half_dt)
        elseif arr === :B
            apply_surface_flux!(q, op, ws, dt_FT, nothing, nothing;
                                tracer_names = names)
            apply_vertical_diffusion!(q, dfop, ws, dt_FT)
        elseif arr === :C
            apply_vertical_diffusion!(q, dfop, ws, dt_FT)
            apply_surface_flux!(q, op, ws, dt_FT, nothing, nothing;
                                tracer_names = names)
        elseif arr === :D
            apply_surface_flux!(q, op, ws, dt_FT, nothing, nothing;
                                tracer_names = names)
            # No V
        else
            error("unknown arrangement $arr")
        end
    end

    frac = _surface_fraction(q, 1)
    total = sum(q)
    return (; frac, total, q)
end

@testset "Ordering study — plan 17 Commit 7" begin

    FT       = Float64
    Nx, Ny   = 4, 3
    Nz       = 10
    dt       = 3600.0                     # 1 hour
    n_steps  = 24                         # 24 h simulated
    Kz       = 5.0                        # m²/s — typical daytime PBL mean
    rate     = 1.0                        # kg/s per cell
    dz       = 50.0                       # m layer thickness

    results = Dict{Symbol, NamedTuple}()
    for arr in (:A, :B, :C, :D)
        results[arr] = _run_arrangement(arr, dt, n_steps;
                                        FT = FT, Nx = Nx, Ny = Ny, Nz = Nz,
                                        Kz_value = Kz, rate_value = rate,
                                        dz = dz)
    end

    # Sanity: all four add the same total mass (rate × dt × steps × cells)
    expected_total = rate * dt * n_steps * Nx * Ny
    for arr in (:A, :B, :C, :D)
        @test results[arr].total ≈ expected_total
    end

    fA = results[:A].frac
    fB = results[:B].frac
    fC = results[:C].frac
    fD = results[:D].frac

    @info "Ordering study layer-surface fractions" fA=fA fB=fB fC=fC fD=fD

    # Predicates per plan 17 §4.4 Commit 7 acceptance criteria.

    @testset "Config D is pathological (all mass stuck at surface)" begin
        # No V means fresh emissions never mix out of k=Nz.
        @test fD ≈ 1.0 atol = 1.0e-12
    end

    @testset "Config A mixes more effectively than D" begin
        # The two V half-steps on either side of S pull freshly-emitted
        # mass out of the surface layer.
        @test fA < fD
    end

    @testset "Config B mixes fresh emissions after add; similar to C" begin
        # B and C both feature a single V(dt) call adjacent to S.
        # They should give similar surface fractions; the exact
        # ordering depends on discrete Backward-Euler asymmetry.
        @test fB < fD
        @test fC < fD
        # A has two half-step V calls around S, so its intermediate V
        # sees half the accumulated surface mass that B's full-dt V
        # sees — A ends up with MORE surface mass than either B or C.
        # Recorded, not required by acceptance criterion; see writeup.
    end

    @testset "Recommended arrangement A is well-mixed (≪ D)" begin
        # Concrete threshold: A should not retain more than 80% of
        # the emission in the surface layer under the given Kz-dt-dz
        # choice. With Kz = 5 m²/s, dz = 50 m, dt = 3600 s:
        # D_above/below ≈ Kz / dz² = 5 / 2500 = 2e-3,
        # α = dt·D ≈ 7.2 — highly overdamped regime, substantial
        # mixing per step.
        @test fA < 0.80
    end

    # Publish: stash the numeric result for the writeup so Commit 7
    # can reference an exact number committed alongside source.
    io = IOBuffer()
    println(io, "ORDERING_STUDY_RESULTS:")
    println(io, "  Config A (V(dt/2) S V(dt/2)) surface_fraction = ", fA)
    println(io, "  Config B (S V(dt))            surface_fraction = ", fB)
    println(io, "  Config C (V(dt) S)            surface_fraction = ", fC)
    println(io, "  Config D (S no V)             surface_fraction = ", fD)
    println(io, "  Grid = $(Nx) × $(Ny) × $(Nz), Kz = $Kz m²/s, dz = $dz m,")
    println(io, "  dt = $dt s, n_steps = $n_steps, rate = $rate kg/s/cell, FT = $FT")
    print(String(take!(io)))
end
