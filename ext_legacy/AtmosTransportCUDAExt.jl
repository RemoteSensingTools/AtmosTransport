"""
CUDA extension for AtmosTransport.

Loaded automatically when `using CUDA` is called alongside AtmosTransport.
Provides GPU array types, KernelAbstractions device, and multi-GPU panel-split
support for the GPU architecture.
"""
module AtmosTransportCUDAExt

import AtmosTransport
using AtmosTransport.Architectures: GPU, PanelGPUMap, PanelTuple, PerGPUWorkspace,
                                     panels_on_gpu, n_gpus
using CUDA: CUDA, CuArray, CuDevice, CUDABackend

# --- Single-GPU methods ---
AtmosTransport.Architectures.array_type(::GPU) = CuArray
AtmosTransport.Architectures.device(::GPU)     = CUDABackend()

# --- Multi-GPU: synchronize all GPUs ---
function AtmosTransport.Architectures.sync_all_gpus(m::PanelGPUMap)
    for g in 0:(m.n_gpus - 1)
        CUDA.device!(g)
        CUDA.synchronize()
    end
end

# --- Multi-GPU: enable P2P access between GPUs ---
function AtmosTransport.Architectures.enable_p2p!(m::PanelGPUMap)
    for g1 in 0:(m.n_gpus - 1), g2 in 0:(m.n_gpus - 1)
        g1 == g2 && continue
        d1, d2 = CuDevice(g1), CuDevice(g2)
        if CUDA.can_access_peer(d1, d2)
            CUDA.device!(g1)
            try
                CUDA.enable_peer_access(d2)
                @info "P2P enabled: GPU $g1 → GPU $g2"
            catch
                # Already enabled or not supported — ignore
            end
        else
            @warn "P2P not available: GPU $g1 → GPU $g2 (will use CPU staging)"
        end
    end
    CUDA.device!(0)
end

# --- Multi-GPU: allocate NTuple{6} with panels on correct GPU ---
function AtmosTransport.Architectures.allocate_ntuple_panels(m::PanelGPUMap, f::Function)
    ntuple(6) do p
        CUDA.device!(m.panel_to_gpu[p])
        f(p)
    end
end

# --- Multi-GPU: PerGPUWorkspace construction ---
function AtmosTransport.Architectures.PerGPUWorkspace(m::PanelGPUMap, f::Function)
    ws = [begin CUDA.device!(g); f(g) end for g in 0:(m.n_gpus - 1)]
    PerGPUWorkspace(ws, m)
end

# --- Multi-GPU: Strategy E — sequential device switch, async kernel queuing ---
# Launch all panels per GPU sequentially. CUDA kernels queue asynchronously,
# so GPU 0's kernels continue executing while we switch to GPU 1 and launch
# its kernels. Final sync waits for both GPUs. No threading needed.

# No-sync variant: launch only, caller handles barriers
function AtmosTransport.Architectures._foreach_gpu_nosync(f::F, m::PanelGPUMap) where F
    for g in 0:(m.n_gpus - 1)
        CUDA.device!(g)
        f(g, m.groups[g + 1])
    end
    nothing
end

function AtmosTransport.Architectures.foreach_gpu_batch_nosync(f::F, m::PanelGPUMap) where F
    AtmosTransport.Architectures._foreach_gpu_nosync(f, m)
end

# Full variant: launch + sync barrier
function AtmosTransport.Architectures.foreach_gpu_batch(f::F, m::PanelGPUMap) where F
    AtmosTransport.Architectures._foreach_gpu_nosync(f, m)
    # Barrier: wait for all GPUs to finish
    for g in 0:(m.n_gpus - 1)
        CUDA.device!(g)
        CUDA.synchronize()
    end
    nothing
end

# --- Multi-GPU: PanelTuple map_panels! (Strategy E) ---
function _dispatch_per_gpu(f::F, m::PanelGPUMap) where F
    for g in 0:(n_gpus(m) - 1)
        CUDA.device!(g)
        for p in panels_on_gpu(m, g)
            f(p)
        end
    end
    for g in 0:(n_gpus(m) - 1)
        CUDA.device!(g)
        CUDA.synchronize()
    end
    nothing
end

function AtmosTransport.Architectures.map_panels!(f::F, pt::PanelTuple{N,T,PanelGPUMap}) where {F,N,T}
    _dispatch_per_gpu(pt.panel_map) do p
        f(p, pt[p])
    end
end

function AtmosTransport.Architectures.map_panels!(f::F, pt1::PanelTuple{N,T,PanelGPUMap},
                                                    pt2::PanelTuple) where {F,N,T}
    _dispatch_per_gpu(pt1.panel_map) do p
        f(p, pt1[p], pt2[p])
    end
end

function AtmosTransport.Architectures.map_panels!(f::F, pt1::PanelTuple{N,T,PanelGPUMap},
                                                    pt2::PanelTuple, pt3::PanelTuple) where {F,N,T}
    _dispatch_per_gpu(pt1.panel_map) do p
        f(p, pt1[p], pt2[p], pt3[p])
    end
end

function AtmosTransport.Architectures.sync_panels!(pt::PanelTuple{N,T,PanelGPUMap}) where {N,T}
    for g in 0:(n_gpus(pt.panel_map) - 1)
        CUDA.device!(g)
        CUDA.synchronize()
    end
    nothing
end

end # module AtmosTransportCUDAExt
