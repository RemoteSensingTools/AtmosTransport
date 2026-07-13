# SH cm grid-roughness vertical profile for a CS transport binary. Used to
# compare the GEOS endpoint binary (known SH-UTLS fingering, ~0.33) against the
# ERA5/spectral production fallback (expected much smoother).
#   julia --project=. scripts/diagnostics/cm_sh_roughness_profile.jl <binary.bin> [window]
using AtmosTransport
using AtmosTransport.MetDrivers: TransportBinaryReader, load_window!, load_grid
using AtmosTransport.Grids: panel_cell_center_lonlat
using AtmosTransport.Architectures: CPU
using Printf

_std(x) = (m = sum(x)/length(x); sqrt(sum(v -> (v-m)^2, x) / max(1, length(x)-1)))
# rough_sh: RMS(normalized Laplacian f-0.25·nb)/std over SH (lat<-30) interior.
function rough_sh(f_panels, mask, Nc)
    laps = Float64[]; vals = Float64[]
    for p in 1:6
        f = f_panels[p]; m = mask[p]
        for j in 2:Nc-1, i in 2:Nc-1
            m[i,j] || continue
            nb = f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1]
            (isnan(f[i,j])||isnan(nb)) && continue
            push!(laps, Float64(f[i,j]) - 0.25 * Float64(nb))
        end
        for j in 1:Nc, i in 1:Nc; m[i,j] && !isnan(f[i,j]) && push!(vals, Float64(f[i,j])); end
    end
    isempty(laps) && return NaN
    return sqrt(sum(abs2,laps)/length(laps)) / _std(vals)
end

function main()
    bin = ARGS[1]; win = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    r = TransportBinaryReader(bin; FT=Float32)
    Nc, Nz = r.header.geometry.Nc, r.header.nlevel
    @printf("%s\n  Nc=%d Nz=%d  window=%d\n", basename(bin), Nc, Nz, win)
    mesh = load_grid(r; FT=Float32, arch=CPU(), Hp=0).horizontal
    lats = ntuple(p -> Float64.(panel_cell_center_lonlat(mesh, p)[2]), 6)
    sh = ntuple(p -> lats[p] .< -30.0, 6)
    w = load_window!(r, win)
    println("  TOA-frac   iface   SH cm rough")
    mx = 0.0; mxf = 0.0
    for iface in 2:Nz
        frac = (iface-1)/Nz
        (0.35 <= frac <= 0.75) || continue   # UTLS band, TOA-first
        rg = rough_sh(ntuple(p -> w.cm[p][:,:,iface], 6), sh, Nc)
        rg > mx && (mx = rg; mxf = frac)
        iface % 4 == 0 && @printf("   %.2f      %4d    %.4f\n", frac, iface, rg)
    end
    @printf("  >> peak SH cm roughness in UTLS band = %.4f at TOA-frac %.2f\n", mx, mxf)
end
main()
