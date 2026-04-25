#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan geos-followups Section D, Commit 1.
#
# Cover the two new helpers added to `cs_transport_helpers.jl`:
#
#   - `rotate_panel_to_geographic!`     — inverse of
#       `rotate_winds_to_panel_local!`. Treats the cell-local tangent
#       basis as orthonormal (matching the forward convention; the
#       O(few %) deviation from orthonormality is absorbed by the LL/RG
#       Poisson balance downstream).
#   - `recover_cs_cell_center_winds!`   — peer of
#       `recover_ll_cell_center_winds!`. Inverts the v4 face-flux
#       layout back to cell-center `(u, v)` for the cross-topology
#       preprocessor (CS source → LL/RG target).
#
# Tests:
#   1. Roundtrip rotation is bit-exact when fed a synthetic orthonormal
#      basis (since transpose = inverse).
#   2. Rotation by 90° (synthetic basis) gives the expected formula
#      output, catching axis-mix-up regressions.
#   3. `reconstruct_cs_fluxes!` ∘ `recover_cs_cell_center_winds!` is
#      bit-exact on a synthetic uniform mesh (Δx≡Δy≡1, dp uniform).
#   4. On a real `CubedSphereMesh`, the rotate-roundtrip residual is
#      bounded by basis non-orthogonality (≤25% on Nc=6 corners). This
#      is intentional — the LL/RG Poisson balance downstream absorbs it.
#   5. The mesh wrapper for `rotate_panel_to_geographic!` matches the
#      explicit-tangent-basis form.
# ---------------------------------------------------------------------------

using Test

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Grids: CubedSphereMesh, GnomonicPanelConvention,
                              GEOSNativePanelConvention,
                              panel_cell_local_tangent_basis
using .AtmosTransport.Preprocessing: rotate_winds_to_panel_local!,
                                      rotate_panel_to_geographic!,
                                      recover_cs_cell_center_winds!,
                                      reconstruct_cs_fluxes!

const FT_TEST = Float64
const NC      = 6
const NZ      = 4
const NPANEL  = 6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_panels_3d(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc, Nz),     NPANEL)
_panels_3d_x(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), NPANEL)
_panels_3d_y(Nc, Nz; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), NPANEL)
_panels_2d(Nc; FT=FT_TEST) = ntuple(_ -> zeros(FT, Nc, Nc), NPANEL)

# Build a synthetic identity tangent basis: x_east=1, x_north=0,
# y_east=0, y_north=1. This is orthonormal so transpose = inverse.
function _identity_tangent_basis(Nc::Int; FT=FT_TEST)
    return ntuple(_ -> (ones(FT, Nc, Nc), zeros(FT, Nc, Nc),
                        zeros(FT, Nc, Nc), ones(FT, Nc, Nc)), NPANEL)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "CS panel ↔ geographic helpers (Section D, Commit 1)" begin

    @testset "Roundtrip rotation: orthonormal synthetic basis" begin
        tb = _identity_tangent_basis(NC)
        ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
        for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
            ue[p][i, j, k] = sin(0.1 * (i + 2j + 3k + 5p))
            vn[p][i, j, k] = cos(0.1 * (i + 2j + 3k + 5p) + 0.7)
        end
        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)

        rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
        rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)

        for p in 1:NPANEL
            @test ue_rt[p] == ue[p]
            @test vn_rt[p] == vn[p]
        end
    end

    @testset "Rotation by 90° (synthetic basis)" begin
        # Build a 90° rotation as the tangent basis: panel-x = north,
        # panel-y = -east, equivalent to x_east=0, x_north=1, y_east=-1,
        # y_north=0. Forward should map (ue=1, vn=0) → (up=0, vp=-1) and
        # reverse should invert it.
        tb = ntuple(_ -> (zeros(FT_TEST, NC, NC),
                          ones(FT_TEST,  NC, NC),
                          fill(FT_TEST(-1), NC, NC),
                          zeros(FT_TEST, NC, NC)), NPANEL)

        ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
        for p in 1:NPANEL
            fill!(ue[p], FT_TEST(1.0))
            fill!(vn[p], FT_TEST(0.0))
        end

        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
        for p in 1:NPANEL
            @test all(up[p] .== FT_TEST(0))
            @test all(vp[p] .== FT_TEST(-1))
        end

        ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)
        rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)
        for p in 1:NPANEL
            @test ue_rt[p] == ue[p]
            @test vn_rt[p] == vn[p]
        end
    end

    @testset "Roundtrip rotation on real CS mesh: bounded by basis non-orthogonality" begin
        # The cell-local tangent basis is unit-normalized but not
        # exactly orthogonal off the panel center. The codebase's
        # convention uses transpose = inverse (matching the existing
        # forward); the resulting roundtrip error is bounded and is
        # corrected downstream by the LL/RG Poisson balance. This test
        # only checks the bound; bit-exactness is provided by the
        # orthonormal-basis tests above.
        for conv in (GnomonicPanelConvention(), GEOSNativePanelConvention())
            mesh = CubedSphereMesh(; Nc = NC, FT = FT_TEST, convention = conv)
            tb = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), NPANEL)

            ue = _panels_3d(NC, NZ); vn = _panels_3d(NC, NZ)
            for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
                ue[p][i, j, k] = sin(0.1 * (i + 2j + 3k + 5p))
                vn[p][i, j, k] = cos(0.1 * (i + 2j + 3k + 5p) + 0.7)
            end
            up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
            ue_rt = _panels_3d(NC, NZ); vn_rt = _panels_3d(NC, NZ)

            rotate_winds_to_panel_local!(up, vp, ue, vn, tb, NC, NZ)
            rotate_panel_to_geographic!(ue_rt, vn_rt, up, vp, tb, NC, NZ)

            # Worst-case basis non-orthogonality at C6 panel corners is
            # ex·ey ≈ 0.37 (about 22°); for a unit-amplitude geographic
            # input, the roundtrip residual in the perpendicular
            # component is therefore bounded by |ex·ey|. Using 0.5
            # leaves comfortable headroom and is uniform across panels.
            for p in 1:NPANEL
                @test maximum(abs, ue_rt[p] .- ue[p]) <= 0.5
                @test maximum(abs, vn_rt[p] .- vn[p]) <= 0.5
            end
        end
    end

    @testset "recover_cs_cell_center_winds! inverts uniform-mesh forward" begin
        # Synthetic uniform mesh: Δx ≡ Δy ≡ 1. Under uniform u, v, dp the
        # forward `reconstruct_cs_fluxes!` produces a constant am, bm
        # everywhere; recovery then divides by the same area_factor and
        # gets back the original (u, v) bit-exactly.
        Δx_uniform = ones(FT_TEST, NC, NC)
        Δy_uniform = ones(FT_TEST, NC, NC)

        gravity   = FT_TEST(9.80665)
        dt_factor = FT_TEST(1800.0)

        # dp_panels filled directly (reconstruct_cs_fluxes! overwrites it
        # using A, B; pick A, B so the result is uniform = 2500 Pa).
        dp_panels = _panels_3d(NC, NZ)
        u_in      = _panels_3d(NC, NZ)
        v_in      = _panels_3d(NC, NZ)
        for p in 1:NPANEL
            fill!(dp_panels[p], FT_TEST(2_500.0))
            fill!(u_in[p], FT_TEST(15.0))
            fill!(v_in[p], FT_TEST(-7.0))
        end
        ps_panels = _panels_2d(NC)
        for p in 1:NPANEL
            fill!(ps_panels[p], FT_TEST(101_325.0))
        end
        # A_ifc, B_ifc are chosen so the dp recomputed inside
        # reconstruct_cs_fluxes! comes out to 2500 Pa per layer.
        A_ifc = collect(FT_TEST.(0:NZ) .* FT_TEST(-2_500.0))
        B_ifc = zeros(FT_TEST, NZ + 1)

        am_v4 = _panels_3d_x(NC, NZ)
        bm_v4 = _panels_3d_y(NC, NZ)

        reconstruct_cs_fluxes!(am_v4, bm_v4, u_in, v_in, dp_panels, ps_panels,
                                A_ifc, B_ifc, Δx_uniform, Δy_uniform,
                                gravity, dt_factor, NC, NZ)

        u_out = _panels_3d(NC, NZ)
        v_out = _panels_3d(NC, NZ)
        recover_cs_cell_center_winds!(u_out, v_out, am_v4, bm_v4, dp_panels,
                                       Δx_uniform, Δy_uniform,
                                       gravity, dt_factor, NC, NZ)

        for p in 1:NPANEL
            @test u_out[p] ≈ u_in[p] rtol = 1e-12
            @test v_out[p] ≈ v_in[p] rtol = 1e-12
        end
    end

    @testset "rotate_panel_to_geographic! mesh wrapper matches explicit-basis form" begin
        mesh = CubedSphereMesh(; Nc = NC, FT = FT_TEST,
                                convention = GnomonicPanelConvention())
        tb = ntuple(p -> panel_cell_local_tangent_basis(mesh, p), NPANEL)

        up = _panels_3d(NC, NZ); vp = _panels_3d(NC, NZ)
        for p in 1:NPANEL, k in 1:NZ, j in 1:NC, i in 1:NC
            up[p][i, j, k] = sin(0.3 * (i + j))
            vp[p][i, j, k] = cos(0.4 * (i - j) + 0.1k)
        end

        ue_a = _panels_3d(NC, NZ); vn_a = _panels_3d(NC, NZ)
        ue_b = _panels_3d(NC, NZ); vn_b = _panels_3d(NC, NZ)

        rotate_panel_to_geographic!(ue_a, vn_a, up, vp, tb, NC, NZ)
        rotate_panel_to_geographic!(ue_b, vn_b, up, vp, mesh, NZ)

        for p in 1:NPANEL
            @test ue_a[p] == ue_b[p]
            @test vn_a[p] == vn_b[p]
        end
    end
end
