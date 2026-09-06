using AtmosTransport, LinearAlgebra, TOML
const Adv = AtmosTransport.Operators.Advection
const G = AtmosTransport.Grids
# Replay the committed pre-fix scalar palindrome using the unchanged panel
# kernels. This avoids mixing environments while preserving its exact staging.
old = read(`git show bf9ae4cb:src/Operators/Advection/CubedSphereStrang.jl`, String)
start = findfirst("function strang_split_cs!(", old).start
stop = findnext("@inline function _check_cs_packed_workspace", old, start).start
body = replace(old[start:prevind(old, stop)], "function strang_split_cs!(" => "function strang_split_cs_baseline!("; count=1)
Core.eval(Adv, Meta.parseall(body))
const pole = normalize([0.5, -0.3, 0.8])
const center = [1.0, 0.0, 0.0]
rotate(v, angle) = cos(angle)*v + sin(angle)*cross(pole,v) + (1-cos(angle))*dot(pole,v)*pole
q_exact(point, angle) = 0.1 + exp(20*(dot(point, rotate(center,angle))-1))
function rotation_problem(Nc, convention)
    mesh = CubedSphereMesh(; Nc, Hp=3, FT=Float64, radius=1.0, convention)
    def = G.cs_definition(mesh)
    N, Nz = Nc+6, 1
    m,rm = ntuple(_ -> zeros(N,N,Nz),6), ntuple(_ -> zeros(N,N,Nz),6)
    am,bm,cm = ntuple(_->zeros(N+1,N,Nz),6),ntuple(_->zeros(N,N+1,Nz),6),ntuple(_->zeros(N,N,Nz+1),6)
    dt = 2pi/(16Nc)
    for p in 1:6
        a = collect(G._corner_xyz(def,Nc,1,1,p,Float64))
        b = collect(G._corner_xyz(def,Nc,2,1,p,Float64))
        c = collect(G._corner_xyz(def,Nc,1,2,p,Float64))
        orientation = sign(dot(cross(b-a,c-a),a))
        psi = [dot(pole,collect(G._corner_xyz(def,Nc,i,j,p,Float64))) for i in 1:Nc+1,j in 1:Nc+1]
        for j in 1:Nc,i in 1:Nc
            point = collect(G._cell_center_xyz(def,Nc,i,j,p,Float64))
            m[p][i+3,j+3,1] = mesh.cell_areas[i,j]
            rm[p][i+3,j+3,1] = m[p][i+3,j+3,1]*q_exact(point,0.)
        end
        for j in 1:Nc,i in 1:Nc+1
            am[p][i+3,j+3,1] = dt/2 * orientation * (psi[i,j+1]-psi[i,j])
        end
        for j in 1:Nc+1,i in 1:Nc
            bm[p][i+3,j+3,1] = -dt/2 * orientation * (psi[i+1,j]-psi[i,j])
        end
    end
    x=map(a->copy(a[4:Nc+4,4:Nc+3,:]),am)
    y=map(a->copy(a[4:Nc+3,4:Nc+4,:]),bm)
    AtmosTransport.Preprocessing.sync_all_cs_boundary_mirrors!(x,y,mesh.connectivity,Nc,Nz)
    for p in 1:6
        am[p][4:Nc+4,4:Nc+3,:] .= x[p]
        bm[p][4:Nc+3,4:Nc+4,:] .= y[p]
    end
    Adv.fill_panel_halos!(m,mesh); Adv.fill_panel_halos!(rm,mesh)
    divergence=maximum(abs(am[p][i+1,j,1]-am[p][i,j,1]+bm[p][i,j+1,1]-bm[p][i,j,1]) for p in 1:6,j in 4:Nc+3,i in 4:Nc+3)
    println("PROBE Nc=",Nc," convention=",typeof(convention)," mass_shape=",size(m[1])," FT=",eltype(m[1])," first_mass_kg=",m[1][4,4,1]," max_step_divergence_kg=",divergence)
    return mesh,m,rm,am,bm,cm,dt
end
rows=Dict[]
for Nc in (8,16,32), convention in (G.GnomonicPanelConvention(),G.GEOSNativePanelConvention())
    mesh,m0,r0,am,bm,cm,dt=rotation_problem(Nc,convention)
    initial=sum(sum(a[4:Nc+3,4:Nc+3,:]) for a in r0)
    for method in ("before","after")
        m,r=map(copy,m0),map(copy,r0)
        ws=Adv.CSAdvectionWorkspace(mesh,m[1])
        step! = method=="before" ? Adv.strang_split_cs_baseline! : Adv.strang_split_cs!
        for step in 1:16Nc
            step!(r,m,am,bm,cm,mesh,PPMScheme(),ws;subcycle_count=1)
            if step in (4Nc,16Nc)
                e2=0.; norm2=0.; total=0.; minq=Inf; maxq=-Inf
                for p in 1:6,j in 1:Nc,i in 1:Nc
                    point=collect(G._cell_center_xyz(G.cs_definition(mesh),Nc,i,j,p,Float64))
                    q=q_exact(point,step*dt); actual=r[p][i+3,j+3,1]/m[p][i+3,j+3,1]
                    e2 += mesh.cell_areas[i,j]*(actual-q)^2
                    norm2 += mesh.cell_areas[i,j]*(q-0.1)^2
                    total += r[p][i+3,j+3,1]
                    minq=min(minq,actual); maxq=max(maxq,actual)
                end
                row=Dict("Nc"=>Nc,"convention"=>string(typeof(convention)),"method"=>method,"rotation_fraction"=>step/(16Nc),"relative_l2"=>sqrt(e2/norm2),"relative_mass_drift"=>(total-initial)/initial,"min_q"=>minq,"max_q"=>maxq)
                push!(rows,row);println("ROTATION ",row);flush(stdout)
            end
        end
    end
end
open("/tmp/atmos-cs-solid-rotation.toml","w") do io; TOML.print(io,Dict("rows"=>rows));end
