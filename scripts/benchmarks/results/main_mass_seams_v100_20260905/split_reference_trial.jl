include(joinpath(@__DIR__, "split_grouped_trial.jl"))
function run_original(n)
    mesh,m,rm=make_structured_cs_state(;Nc=8,Hp=3,Nz=2)
    am,bm,cm=make_mirrored_cs_horizontal_fluxes(mesh,2)
    initial=total_interior(rm,8,3,2)
    ws=Adv.CSAdvectionWorkspace(mesh,2)
    for _ in 1:n
        Adv.strang_split_cs!(rm,m,am,bm,cm,mesh,PPMScheme(),ws;flux_scale=100. / n,subcycle_count=1)
    end
    r=vcat([vec(rm[p][4:11,4:11,:]) for p in 1:6]...)
    return r,initial
end
reference,_=run_original(512)
println("LIMIT_DIFF ",norm(reference-rref)/norm(reference))
for n in (1,2,4,8,16,32,64,128)
    r,total=run_original(n)
    g,_,_=run_grouped(n)
    println("ORIGINAL ",n," drift=",(sum(r)-total)/total," reference_error=",norm(r-reference)/norm(reference)," grouped_error_same_reference=",norm(g-reference)/norm(reference))
end
