using Test, Adapt
using AtmosTransport
const C = AtmosTransport.Operators.Convection

@testset "Runtime factories defer only requested collaborative scratch" begin
    G, S = AtmosTransport.Grids, AtmosTransport.State
    FT, nz = Float32, 8
    vc = G.HybridSigmaPressure(zeros(FT,nz+1), collect(range(0f0,1f0; length=nz+1)))
    meshes = (G.LatLonMesh(; Nx=4, Ny=3, FT),
              G.ReducedGaussianMesh(FT[-0.9,0,0.9], [4,4,4]; FT),
              G.CubedSphereMesh(; Nc=2, Hp=1, FT))
    for mesh in meshes
        grid = G.AtmosGrid(mesh, vc, AtmosTransport.CPU(); FT)
        state = if mesh isa G.LatLonMesh
            S.CellState(zeros(FT,4,3,nz); CO2=zeros(FT,4,3,nz))
        elseif mesh isa G.ReducedGaussianMesh
            S.CellState(zeros(FT,12,nz); CO2=zeros(FT,12,nz))
        else
            S.CubedSphereState(S.DryBasis, mesh,
                ntuple(_ -> zeros(FT,4,4,nz),6);
                CO2=ntuple(_ -> zeros(FT,4,4,nz),6))
        end
        for make_operator in (C.TM5Convection, C.CMFMCMatrixConvection), deferred in (false,true)
            op = make_operator(use_collab_lu=deferred)
            ws = AtmosTransport.Models._convection_workspace_for(op, state, grid)
            tm5 = ws isa C.CMFMCMatrixWorkspace ? ws.tm5_workspace : ws
            @test tm5.defer_scratch == deferred
            @test isempty(tm5.conv1) == deferred
            @test tm5.scratch_columns > 0
            @test tm5.cell_metrics !== nothing
        end
    end
end

@testset "Deferred TM5 scratch retains budget and allocates once" begin
    for FT in (Float32, Float64)
        mass = zeros(FT, 4, 3, 8)
        ws = C.TM5Workspace(mass; tile_columns=5, cell_metrics=ones(FT, 3),
                            defer_scratch=true)
        @test ws.scratch_columns == 5
        @test size(ws.conv1) == (8, 8, 0)
        @test all(isempty, (ws.conv1, ws.pivots, ws.cloud_dims, ws.amu_scratch, ws.amd_scratch))
        @test ws.f_scratch === ws.conv1
        @test ws.cell_metrics == ones(FT, 3)
        @test C._ensure_tm5_scratch!(ws) === ws
        @test size(ws.conv1) == (8, 8, 5)
        @test size(ws.pivots) == (8, 5)
        @test size(ws.cloud_dims) == (3, 5)
        @test size(ws.amu_scratch) == size(ws.amd_scratch) == (9, 5)
        @test ws.f_scratch === ws.conv1
        buffers = (ws.conv1, ws.pivots, ws.cloud_dims, ws.amu_scratch, ws.amd_scratch)
        C._ensure_tm5_scratch!(ws)
        @test buffers === (ws.conv1, ws.pivots, ws.cloud_dims, ws.amu_scratch, ws.amd_scratch)

        # Adaptation retains the allocation policy and budget, not stale scratch.
        adapted = Adapt.adapt(Array, ws)
        @test adapted.defer_scratch
        @test adapted.scratch_columns == 5
        @test isempty(adapted.conv1)
        @test adapted.f_scratch === adapted.conv1
        C._ensure_tm5_scratch!(adapted)
        @test size(adapted.conv1) == (8, 8, 5)
        @test adapted.conv1 !== ws.conv1
    end
    @test_throws ArgumentError C.TM5Workspace(zeros(Float32,2,3,8);
        tile_columns=0, defer_scratch=true)
    @test_throws ArgumentError C.TM5Workspace(zeros(Float32,2,3,8);
        tile_columns=3, tile_workspace_gib=1, defer_scratch=true)
end

@testset "Deferred adaptation preserves persistent convection data" begin
    mass = zeros(Float32, 2, 3, 8)
    ws = C.CMFMCMatrixWorkspace(mass; tile_columns=5, defer_scratch=true,
                               cache_columns=6, cell_metrics=ones(Float32,3))
    ws.derived_entu .= 0.03f0
    ws.derived_detu .= 0.02f0
    ws.derived_valid[] = true
    ws.tm5_workspace.cache_A .= 2f0
    ws.tm5_workspace.cache_pivots .= 1
    ws.tm5_workspace.cache_valid[] = true
    C._ensure_tm5_scratch!(ws.tm5_workspace)
    adapted = Adapt.adapt(Array, ws)
    @test isempty(adapted.tm5_workspace.conv1)
    @test adapted.tm5_workspace.scratch_columns == 5
    @test adapted.derived_entu == ws.derived_entu
    @test adapted.derived_detu == ws.derived_detu
    @test adapted.derived_valid[]
    @test all(iszero, adapted.zero_entd)
    @test all(iszero, adapted.zero_detd)
    @test adapted.tm5_workspace.cache_A == ws.tm5_workspace.cache_A
    @test adapted.tm5_workspace.cache_pivots == ws.tm5_workspace.cache_pivots
    @test adapted.tm5_workspace.cache_valid[]
    C.invalidate_cmfmc_matrix_cache!(adapted)
    @test ws.derived_valid[] # Sentinel ownership is independent after adaptation.
    @test ws.tm5_workspace.cache_valid[]
end

@testset "C180 L85 collaborative workspace omits legacy matrix payload" begin
    # One modest air-mass panel supplies dimensions; do not allocate a GiB tile.
    mass = zeros(Float32, 180, 180, 85)
    ws = C.TM5Workspace(mass; tile_workspace_gib=1, defer_scratch=true)
    @test ws.scratch_columns == 180^2
    @test sum(sizeof, (ws.conv1, ws.pivots, ws.cloud_dims,
                       ws.amu_scratch, ws.amd_scratch)) == 0
    # The deferred payload, excluding metrics and optional persistent caches.
    per_column = sizeof(Float32)*(85^2+2*86) + sizeof(Int)*(85+3)
    @test per_column * ws.scratch_columns > 0.9 * 2.0^30
end
