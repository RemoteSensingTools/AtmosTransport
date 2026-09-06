using AtmosTransport
const Diff=AtmosTransport.Operators.Diffusion
for FT in (Float32,Float64)
    m=ntuple(_->fill(FT(1e10),3,3,2),6)
    d=ntuple(_->reshape(FT[100,0],1,1,2),6)
    r=ntuple(_->zeros(FT,3,3,2),6)
    for p in 1:6; r[p][2,2,1]=FT(4e6); end
    op=ImplicitVerticalDiffusion(;kz_field=AtmosTransport.State.PrecomputedCSDkgField(d))
    ws=DiffusionWorkspace(m,1,1)
    Diff.apply_vertical_diffusion_vmr!(r,m,op,ws,one(FT);halo_width=1)
    expected=4e6*1e-8/(1+2e-8)
    println(FT," weak transfer=",r[1][2,2,2]," expected=",expected)
end
