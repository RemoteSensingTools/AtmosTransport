# ===========================================================================
# Mass-balance audit for a cubed-sphere transport binary (deliverable 5a + 5b).
#
#  5a. BINARY air-mass closure (per-window): the integral of the fluxes equals
#      the mass change. Per column, Σ_k(div_h[k] + Δcm[k]) = Σ_k dm[k] where
#      div_h = (am[i,j]-am[i+1,j]) + (bm[i,j]-bm[i,j+1]),  Δcm = cm[k]-cm[k+1],
#      dm = (m_next - m_cur)/(2·steps). Equivalent to cm[Nz+1]=0 to roundoff when
#      div_h closes the column. Reports RMS + max |residual| / colmass over
#      interior cells (the write-time gate) and the global Σ over all cells.
#
#  5b. GLOBAL dry-mass conservation across the run window sequence: total dry air
#      mass per window + its drift (should be pinned constant by the mass_fix).
#
# Usage:
#   ~/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia --project=. \
#     scripts/diagnostics/omega_consistent_mass_balance.jl <bin1> [bin2 ...]
# ===========================================================================
using AtmosTransport
using AtmosTransport.MetDrivers
using Printf, Statistics

const MD = AtmosTransport.MetDrivers

function audit_binary(path::AbstractString)
    reader = MD.TransportBinaryReader(path)
    h = reader.header
    Nc = h.geometry.Nc; Nz = h.nlevel; np = h.geometry.npanel
    nwin = MD.window_count(reader)
    scale_sched = h.poisson_balance_target_scale_by_window
    steps_sched = h.steps_per_window_by_window
    @printf("\n=== %s ===\n", basename(path))
    @printf("  C%d  Nz=%d  panels=%d  windows=%d\n", Nc, Nz, np, nwin)

    # Per-window dry mass + continuity. dm needs window w's m and w+1's m
    # (m_next). The last window's m_next isn't in this file, so continuity is
    # checked over windows 1..nwin-1; dry mass is reported for all windows.
    masses = Float64[]
    worst_int_rms = 0.0; worst_int_max = 0.0
    worst_glob = 0.0; worst_cmtop = 0.0
    win_for_int = 0; win_for_cmtop = 0
    cur = MD.load_window!(reader, 1)
    for w in 1:nwin
        # 5b: global dry air mass of window w (state m).
        mw = 0.0
        for p in 1:np; mw += sum(Float64, cur.m[p]); end
        push!(masses, mw)
        w == nwin && break
        nxt = MD.load_window!(reader, w + 1)
        scale = w <= length(scale_sched) ? Float64(scale_sched[w]) :
                1.0 / (2 * Float64(steps_sched[w]))
        am = cur.am; bm = cur.bm; cm = cur.cm; m = cur.m; mn = nxt.m
        # colmass scale (max column mass) for relative residual.
        colmax = 0.0
        for p in 1:np, j in 1:Nc, i in 1:Nc
            c = 0.0
            @inbounds for k in 1:Nz; c += Float64(m[p][i, j, k]); end
            colmax = max(colmax, c)
        end
        # 5a interior continuity + cm[Nz+1] closure.
        ss = 0.0; n = 0; mx = 0.0; cmtop = 0.0
        gl_res = 0.0
        for p in 1:np
            amp = am[p]; bmp = bm[p]; cmp = cm[p]; mp = m[p]; mnp = mn[p]
            for j in 1:Nc, i in 1:Nc
                interior = (1 < i < Nc) && (1 < j < Nc)
                @inbounds for k in 1:Nz
                    div_h = (Float64(amp[i, j, k]) - Float64(amp[i + 1, j, k])) +
                            (Float64(bmp[i, j, k]) - Float64(bmp[i, j + 1, k]))
                    vdiv = Float64(cmp[i, j, k]) - Float64(cmp[i, j, k + 1])
                    dm = (Float64(mnp[i, j, k]) - Float64(mp[i, j, k])) * scale
                    r = (dm - div_h - vdiv) / colmax
                    gl_res += (dm - div_h - vdiv)
                    if interior
                        ss += r * r; n += 1; mx = max(mx, abs(r))
                    end
                end
                # cm[Nz+1] closure (closed-bottom boundary) relative to colmass.
                cmtop = max(cmtop, abs(Float64(cmp[i, j, Nz + 1])) / colmax)
            end
        end
        int_rms = sqrt(ss / n)
        if int_rms > worst_int_rms
            worst_int_rms = int_rms; worst_int_max = mx; win_for_int = w
        end
        worst_glob = max(worst_glob, abs(gl_res))
        if cmtop > worst_cmtop; worst_cmtop = cmtop; win_for_cmtop = w; end
        cur = nxt
    end

    @printf("\n  5a. AIR-MASS CLOSURE (per-column continuity, interior cells):\n")
    @printf("      worst window=%d  RMS|res|/colmass=%.3e  max|res|/colmass=%.3e\n",
            win_for_int, worst_int_rms, worst_int_max)
    @printf("      worst |cm[Nz+1]|/colmass (closed-bottom) win=%d  = %.3e\n",
            win_for_cmtop, worst_cmtop)
    @printf("      worst global Σ_cells(dm-div_h-Δcm) (kg/half-substep) = %.3e\n",
            worst_glob)

    @printf("\n  5b. GLOBAL DRY-MASS CONSERVATION (per-window state mass):\n")
    m0 = masses[1]
    mmin, mmax = extrema(masses)
    @printf("      window 1 dry mass = %.9e kg\n", m0)
    @printf("      min=%.9e  max=%.9e kg\n", mmin, mmax)
    @printf("      max drift vs win-1 = %.3e kg  (rel %.3e)\n",
            mmax - mmin, (mmax - mmin) / m0)
    return (mass0 = m0, drift_rel = (mmax - mmin) / m0,
            cont_rms = worst_int_rms, cmtop = worst_cmtop)
end

function main()
    isempty(ARGS) && error("Usage: omega_consistent_mass_balance.jl <bin> [bin ...]")
    for p in ARGS
        audit_binary(expanduser(p))
    end
end
main()
