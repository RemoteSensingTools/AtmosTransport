using AtmosTransport, LinearAlgebra
const Diff = AtmosTransport.Operators.Diffusion
const path = get(ENV, "ATMOSTR_MASS_INPUT", "/home/cfranken/data/AtmosTransport/met/era5/n320_to_c90/transport_binary_v4_l66_f32_tm5_convection_1deg_3hour/era5_n320_transport_20181201_float32.bin")
driver = TransportBinaryDriver(path; FT=Float32, Hp=3, validate_windows=false)
window = load_transport_window(driver, 1)
air, dkg = map(copy, window.air_mass), map(copy, window.dkg)
close(driver)
println("PROBE air=",size(air[1])," ",eltype(air[1])," mass33=",air[1][4,4,33]," kg; Dkg=",size(dkg[1])," max=",maximum(maximum,dkg)," kg/s")

function reference(rm, m, d, dt)
    n = length(m)
    u = dt .* d[1:n-1] ./ m[1:n-1]
    v = dt .* d[1:n-1] ./ m[2:n]
    diagonal = ones(n)
    diagonal[1:n-1] .+= u
    diagonal[2:n] .+= v
    return Tridiagonal(-u, diagonal, -v) \ rm
end

# Repack the same 216 archive columns used by the scalar prototype as C6 panels.
Nc, Nz, Nt, Hp = 6, 66, 4, 1
m32 = ntuple(_ -> zeros(Float32,Nc+2,Nc+2,Nz),6)
d32 = ntuple(_ -> zeros(Float32,Nc,Nc,Nz),6)
r32 = ntuple(_ -> zeros(Float32,Nc+2,Nc+2,Nz,Nt),6)
for p in 1:6, j in 1:Nc, i in 1:Nc
    ii, jj = 1+15(i-1), 1+15(j-1)
    m32[p][i+1,j+1,:] .= air[p][ii+3,jj+3,:]
    d32[p][i,j,:] .= dkg[p][ii,jj,:]
    for (t,k) in enumerate((16,33,50,64))
        r32[p][i+1,j+1,k,t] = m32[p][i+1,j+1,k] * 4f-4
    end
end
for FT in (Float32,Float64), method in (:vmr_thomas,:conservative_mass)
    m,d,r = map(a->FT.(a),m32),map(a->FT.(a),d32),map(a->FT.(a),r32)
    op = ImplicitVerticalDiffusion(;kz_field=AtmosTransport.State.PrecomputedCSDkgField(d))
    ws = DiffusionWorkspace(m,Hp,Nt)
    if method == :vmr_thomas
        Diff._cs_scale_tracer_mass_to_vmr!(r,m,Hp)
        Diff.apply_vertical_diffusion!(r,m,op,ws,FT(360);halo_width=Hp)
        Diff._cs_scale_vmr_to_tracer_mass!(r,m,Hp)
    else
        Diff.apply_vertical_diffusion_vmr!(r,m,op,ws,FT(360);halo_width=Hp)
    end
    drifts, errors = Float64[],Float64[]
    for p in 1:6, j in 1:Nc, i in 1:Nc, t in 1:Nt
        initial = Float64.(r32[p][i+1,j+1,:,t])
        expected = reference(initial,Float64.(m[p][i+1,j+1,:]),Float64.(d[p][i,j,:]),360.)
        actual = Float64.(r[p][i+1,j+1,:,t])
        push!(drifts,(sum(actual)-sum(initial))/sum(initial))
        push!(errors,norm(actual-expected)/norm(expected))
    end
    println(FT," ",method," max_drift=",maximum(abs,drifts)," mean_drift=",sum(drifts)/length(drifts)," max_reference_error=",maximum(errors))
end
