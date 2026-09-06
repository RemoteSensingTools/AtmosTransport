using Test, AtmosTransport, KernelAbstractions
const LayoutAdv = AtmosTransport.Operators.Advection

@testset "CPU PPM kernels support CUDA launch layouts and partial tiles" begin
    backend = KernelAbstractions.CPU()
    for FT in (Float32,Float64), Nc in (5,35), Nt in (1,7), axis in (:x,:y,:z)
        Hp, Nz = 3, 3
        N = Nc + 2Hp
        m = FT[1000+5i+3j+k for i in 1:N,j in 1:N,k in 1:Nz]
        q = [t%3==0 ? zero(FT) : FT((-1)^t * t * 1e-4) for t in 1:Nt]
        rm = cat((m .* value for value in q)...; dims=4)
        original_m, original_rm = copy(m), copy(rm)
        dims = axis == :x ? (N+1,N,Nz) : axis == :y ? (N,N+1,Nz) : (N,N,Nz+1)
        flux = FT[10sin(i/3+j/4+k) for i in 1:dims[1],j in 1:dims[2],k in 1:dims[3]]
        axis == :z && (flux[:,:,1] .= 0; flux[:,:,end] .= 0)
        kernel = getproperty(LayoutAdv, Symbol("_cs_",axis,"sweep_mt_kernel!"))
        size_arg = Int32(axis == :z ? Nz : Nc)
        scheme = PPMScheme()
        reference_rm, reference_m = fill(FT(-999),size(rm)), fill(FT(-999),size(m))
        kernel(backend,256)(reference_rm,rm,reference_m,m,flux,scheme,
            size_arg,Int32(Hp),Int32(Nt),FT(0.75);ndrange=(Nc,Nc,Nz))
        synchronize(backend)
        for tile in ((32,2),32)
            actual_rm, actual_m = fill(FT(-999),size(rm)), fill(FT(-999),size(m))
            kernel(backend,tile)(actual_rm,rm,actual_m,m,flux,scheme,
                size_arg,Int32(Hp),Int32(Nt),FT(0.75);ndrange=(Nc,Nc,Nz))
            synchronize(backend)
            @test actual_rm == reference_rm
            @test actual_m == reference_m
            @test m == original_m && rm == original_rm
            @test all(isfinite,actual_rm)
            @test all(==(FT(-999)),actual_rm[[1,end],:,:,:])
            @test all(==(FT(-999)),actual_rm[:,[1,end],:,:])
            inside = Hp+1:Hp+Nc
            for t in 1:Nt
                # Independent physical invariant: a constant signed VMR must
                # follow carrier mass even when the flux diverges locally.
                actual_q = actual_rm[inside,inside,:,t] ./ actual_m[inside,inside,:]
                @test maximum(abs,actual_q .- q[t]) <= 8eps(FT)*abs(q[t])
            end
        end
    end
end
