#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Validate the precomputed TM5 bldiff `:kz` payload in a cubed-sphere transport
# binary: confirm the section is present and that the layer-centre eddy
# diffusivity is physically sensible (finite, non-negative, peaking in the lower
# troposphere where the boundary layer lives, ~0 in the stratosphere).
#
# Usage:
#   julia --project=. scripts/diagnostics/validate_tm5_kz_payload.jl <path.bin>
# ---------------------------------------------------------------------------

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.MetDrivers: CubedSphereBinaryReader, load_cs_window

function main(path::AbstractString)
    reader = CubedSphereBinaryReader(path)
    try
        h = reader.header
        Nz = h.nlevel
        @printf("binary: %s\n  Nc=%d  Nz=%d  npanel=%d  nwindow=%d\n",
                basename(path), h.Nc, Nz, h.npanel, h.nwindow)
        println("  payload: ", join(String.(h.payload_sections), ", "))

        :kz in h.payload_sections ||
            error("✗ binary does NOT carry a :kz section")
        println("  ✓ :kz section present")
        haskey(h.raw_header, "precomputed_kz_payload") &&
            println("  payload tag: ", h.raw_header["precomputed_kz_payload"])

        w = load_cs_window(reader, 1)
        w.kz === nothing && error("✗ window 1 has no kz panels")
        kz = w.kz

        # Aggregate per-level statistics across all panels/cells (k=1 TOA …
        # k=Nz surface).
        finite_all = true; nonneg_all = true
        level_mean = zeros(Float64, Nz); level_max = zeros(Float64, Nz)
        for p in 1:h.npanel
            finite_all &= all(isfinite, kz[p])
            nonneg_all &= all(>=(0), kz[p])
        end
        for k in 1:Nz
            acc = 0.0; mx = 0.0; n = 0
            for p in 1:h.npanel
                slab = @view kz[p][:, :, k]
                acc += sum(slab); mx = max(mx, maximum(slab)); n += length(slab)
            end
            level_mean[k] = acc / n; level_max[k] = mx
        end

        @printf("\n  all finite: %s   all ≥ 0: %s\n", finite_all, nonneg_all)
        println("\n  level   mean Kz (m²/s)    max Kz (m²/s)")
        for k in 1:Nz
            tag = k == 1 ? " TOA" : k == Nz ? " surface" : ""
            (k <= 4 || k >= Nz - 6 || k % 20 == 0) &&
                @printf("  %4d   %12.3f   %14.3f%s\n", k, level_mean[k], level_max[k], tag)
        end

        # Physical expectation: the diffusivity is concentrated in the lower
        # troposphere (the bottom quarter of the column), not the stratosphere.
        lower = mean(level_mean[Nz - Nz÷4 : Nz])
        upper = mean(level_mean[1 : Nz÷4])
        @printf("\n  mean Kz lower-trop (bottom ¼): %.3f   upper (top ¼): %.4f\n",
                lower, upper)
        @printf("  global max Kz: %.1f m²/s\n", maximum(level_max))
        if finite_all && nonneg_all && lower > 10 * max(upper, 1e-6)
            println("\n  ✓ TM5 :kz payload is physically sensible.")
        else
            println("\n  ✗ TM5 :kz payload looks off — inspect above.")
        end
    finally
        finalize(reader)
    end
    return nothing
end

isempty(ARGS) && error("usage: validate_tm5_kz_payload.jl <path.bin>")
main(expanduser(ARGS[1]))
