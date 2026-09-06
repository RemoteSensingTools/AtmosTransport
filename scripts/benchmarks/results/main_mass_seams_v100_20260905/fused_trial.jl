using CUDA, AtmosTransport, Random, Test
CUDA.allowscalar(false)
occursin("V100", CUDA.name(CUDA.device())) || error("V100 required")
const Adv = AtmosTransport.Operators.Advection
@eval Adv begin
    @kernel function _trial_share_lr_all_edges!(fx_in, fx_out, fy_in, fy_out, neighbors, Nc)
        s, k, contact = @index(Global, NTuple)
        p = (contact - 1) ÷ 4 + 1
        edge = (contact - 1) % 4 + 1
        neighbor = neighbors[p][edge]
        q = neighbor.panel
        if p < q
            other_edge = 1
            for e in 1:4
                if neighbors[q][e].panel == p
                    other_edge = e
                end
            end
            t = neighbor.orientation == 0 ? s : Nc + 1 - s
            i, j = _lr_edge_face_index(edge, s, Nc)
            u, v = _lr_edge_face_index(other_edge, t, Nc)
            a_in, a_out = edge in (EDGE_EAST, EDGE_WEST) ? (fx_in[p],fx_out[p]) : (fy_in[p],fy_out[p])
            b_in, b_out = other_edge in (EDGE_EAST, EDGE_WEST) ? (fx_in[q],fx_out[q]) : (fy_in[q],fy_out[q])
            half = eltype(a_in)(0.5)
            @inbounds begin
                common = half * (half * (a_in[i,j,k]+a_out[i,j,k]) + half * (b_in[u,v,k]+b_out[u,v,k]))
                a_in[i,j,k] = common; a_out[i,j,k] = common
                b_in[u,v,k] = common; b_out[u,v,k] = common
            end
        end
    end
end
include(joinpath(pwd(),"test/core/test_linrood_seams.jl"))
function trial!(fields,mesh)
    Adv._trial_share_lr_all_edges!(CUDA.CUDABackend(),64)(fields...,mesh.connectivity.neighbors,mesh.Nc;ndrange=(mesh.Nc,size(fields[1][1],3),24))
    CUDA.synchronize()
end
@testset "Fused seam kernel versus separate edges" begin
    for FT in (Float32,Float64),Nc in (5,35,90),convention in (SeamGrids.GnomonicPanelConvention(),SeamGrids.GEOSNativePanelConvention())
        mesh=CubedSphereMesh(;FT,Nc,Hp=3,convention)
        orig=seam_face_fixture(FT,Nc,66,MersenneTwister(1291))
        fields=map(a->map(CuArray,a),orig)
        trial!(fields,mesh)
        @test map(a->map(Array,a),fields) == independent_seam_mean(orig,mesh)
        if Nc==90 && FT==Float32 && convention isa SeamGrids.GEOSNativePanelConvention
            for method in ("separate","fused")
                f = method=="separate" ? ()->Adv._share_lr_seam_faces!(fields...,mesh) : ()->trial!(fields,mesh)
                f(); t=@elapsed for _ in 1:1000;f();end
                println("SEAM_TIME ",method," ",t/1000)
            end
        end
    end
end
