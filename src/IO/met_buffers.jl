# ---------------------------------------------------------------------------
# Met-field buffer types for GPU/CPU staging and double-buffering
#
# Abstract hierarchy:
#   AbstractMetBuffer{FT}         — GPU-resident met fields
#   AbstractCPUStagingBuffer{FT}  — CPU-side staging for H→D transfers
#
# Concrete types dispatch on grid type:
#   LatLonMetBuffer / LatLonCPUBuffer       — for LatitudeLongitudeGrid
#   CubedSphereMetBuffer / CubedSphereCPUBuffer — for CubedSphereGrid
# ---------------------------------------------------------------------------

using ..Architectures: AbstractArchitecture, array_type
using ..Advection: MassFluxWorkspace, allocate_massflux_workspace

"""
$(TYPEDEF)

Supertype for GPU-resident met-field buffers.
"""
abstract type AbstractMetBuffer{FT} end

"""
$(TYPEDEF)

Supertype for CPU-side staging buffers used in host→device transfers.
"""
abstract type AbstractCPUStagingBuffer{FT} end

# =====================================================================
# Lat-lon met buffers
# =====================================================================

"""
$(TYPEDEF)

GPU-resident met-field buffers for lat-lon grids.

$(FIELDS)
"""
struct LatLonMetBuffer{FT, A3 <: AbstractArray{FT,3}, A2 <: AbstractArray{FT,2}} <: AbstractMetBuffer{FT}
    "reference air mass (Nx, Ny, Nz) — preserved across sub-steps"
    m_ref :: A3
    "working air mass (Nx, Ny, Nz) — modified during advection"
    m_dev :: A3
    "x mass flux (Nx+1, Ny, Nz)"
    am    :: A3
    "y mass flux (Nx, Ny+1, Nz)"
    bm    :: A3
    "z mass flux (Nx, Ny, Nz+1)"
    cm    :: A3
    "surface pressure (Nx, Ny)"
    ps    :: A2
    "pressure thickness (Nx, Ny, Nz)"
    Δp    :: A3
    "staggered u-wind (Nx+1, Ny, Nz)"
    u     :: A3
    "staggered v-wind (Nx, Ny+1, Nz)"
    v     :: A3
    "pre-allocated advection workspace"
    ws    :: MassFluxWorkspace
end

function LatLonMetBuffer(arch::AbstractArchitecture, ::Type{FT}, Nx, Ny, Nz;
                         cluster_sizes_cpu::Union{Nothing,Vector{Int32}} = nothing) where FT
    AT = array_type(arch)
    m_ref = AT(zeros(FT, Nx, Ny, Nz))
    m_dev = AT(zeros(FT, Nx, Ny, Nz))
    am    = AT(zeros(FT, Nx + 1, Ny, Nz))
    bm    = AT(zeros(FT, Nx, Ny + 1, Nz))
    cm    = AT(zeros(FT, Nx, Ny, Nz + 1))
    ps    = AT(zeros(FT, Nx, Ny))
    Δp    = AT(zeros(FT, Nx, Ny, Nz))
    u     = AT(zeros(FT, Nx + 1, Ny, Nz))
    v     = AT(zeros(FT, Nx, Ny + 1, Nz))
    ws    = allocate_massflux_workspace(m_dev, am, bm, cm;
                                         cluster_sizes_cpu = cluster_sizes_cpu)
    LatLonMetBuffer(m_ref, m_dev, am, bm, cm, ps, Δp, u, v, ws)
end

"""
$(TYPEDEF)

CPU-side staging buffer for lat-lon met data (H→D transfer source).

$(FIELDS)
"""
struct LatLonCPUBuffer{FT} <: AbstractCPUStagingBuffer{FT}
    m  :: Array{FT, 3}
    am :: Array{FT, 3}
    bm :: Array{FT, 3}
    cm :: Array{FT, 3}
    ps :: Array{FT, 2}
end

function LatLonCPUBuffer(::Type{FT}, Nx, Ny, Nz) where FT
    LatLonCPUBuffer{FT}(
        Array{FT}(undef, Nx, Ny, Nz),
        Array{FT}(undef, Nx + 1, Ny, Nz),
        Array{FT}(undef, Nx, Ny + 1, Nz),
        Array{FT}(undef, Nx, Ny, Nz + 1),
        Array{FT}(undef, Nx, Ny))
end

"""
    upload!(gpu_buf::LatLonMetBuffer, cpu_buf::LatLonCPUBuffer)

Copy CPU staging buffer contents to GPU met buffer.
"""
function upload!(buf::LatLonMetBuffer, cpu::LatLonCPUBuffer)
    copyto!(buf.m_ref, cpu.m)
    copyto!(buf.m_dev, cpu.m)
    copyto!(buf.am, cpu.am)
    copyto!(buf.bm, cpu.bm)
    copyto!(buf.cm, cpu.cm)
    copyto!(buf.ps, cpu.ps)
    return nothing
end

# =====================================================================
# Cubed-sphere met buffers
# =====================================================================

"""
$(TYPEDEF)

GPU-resident met-field buffers for cubed-sphere grids.

$(FIELDS)
"""
struct CubedSphereMetBuffer{FT, A3 <: AbstractArray{FT,3}} <: AbstractMetBuffer{FT}
    "haloed pressure thickness panels (Nc+2Hp, Nc+2Hp, Nz) × 6"
    delp :: NTuple{6, A3}
    "x mass flux panels (Nc+1, Nc, Nz) × 6"
    am   :: NTuple{6, A3}
    "y mass flux panels (Nc, Nc+1, Nz) × 6"
    bm   :: NTuple{6, A3}
    "z mass flux panels (Nc, Nc, Nz+1) × 6"
    cm   :: NTuple{6, A3}
    "x Courant number panels (Nc+1, Nc, Nz) × 6 — for GCHP-faithful transport"
    cx   :: Union{Nothing, NTuple{6, A3}}
    "y Courant number panels (Nc, Nc+1, Nz) × 6 — for GCHP-faithful transport"
    cy   :: Union{Nothing, NTuple{6, A3}}
    "x area flux panels (Nc+1, Nc, Nz) × 6 — precomputed with exact sin_sg"
    xfx  :: Union{Nothing, NTuple{6, A3}}
    "y area flux panels (Nc, Nc+1, Nz) × 6 — precomputed with exact sin_sg"
    yfx  :: Union{Nothing, NTuple{6, A3}}
end

function CubedSphereMetBuffer(arch::AbstractArchitecture, ::Type{FT},
                               Nc, Nz, Hp; use_gchp::Bool=false) where FT
    AT = array_type(arch)
    delp = ntuple(_ -> AT(zeros(FT, Nc + 2Hp, Nc + 2Hp, Nz)), 6)
    am   = ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6)
    bm   = ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6)
    cm   = ntuple(_ -> AT(zeros(FT, Nc, Nc, Nz + 1)), 6)
    cx   = use_gchp ? ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6) : nothing
    cy   = use_gchp ? ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6) : nothing
    xfx  = use_gchp ? ntuple(_ -> AT(zeros(FT, Nc + 1, Nc, Nz)), 6) : nothing
    yfx  = use_gchp ? ntuple(_ -> AT(zeros(FT, Nc, Nc + 1, Nz)), 6) : nothing
    CubedSphereMetBuffer(delp, am, bm, cm, cx, cy, xfx, yfx)
end

"""
$(TYPEDEF)

CPU-side staging buffer for cubed-sphere met data.

$(FIELDS)
"""
struct CubedSphereCPUBuffer{FT} <: AbstractCPUStagingBuffer{FT}
    delp :: NTuple{6, Array{FT, 3}}
    am   :: NTuple{6, Array{FT, 3}}
    bm   :: NTuple{6, Array{FT, 3}}
    cx   :: Union{Nothing, NTuple{6, Array{FT, 3}}}
    cy   :: Union{Nothing, NTuple{6, Array{FT, 3}}}
    xfx  :: Union{Nothing, NTuple{6, Array{FT, 3}}}
    yfx  :: Union{Nothing, NTuple{6, Array{FT, 3}}}
end

function CubedSphereCPUBuffer(::Type{FT}, Nc, Nz, Hp; use_gchp::Bool=false) where FT
    delp = ntuple(_ -> Array{FT}(undef, Nc + 2Hp, Nc + 2Hp, Nz), 6)
    am   = ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), 6)
    bm   = ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), 6)
    cx   = use_gchp ? ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), 6) : nothing
    cy   = use_gchp ? ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), 6) : nothing
    xfx  = use_gchp ? ntuple(_ -> Array{FT}(undef, Nc + 1, Nc, Nz), 6) : nothing
    yfx  = use_gchp ? ntuple(_ -> Array{FT}(undef, Nc, Nc + 1, Nz), 6) : nothing
    CubedSphereCPUBuffer{FT}(delp, am, bm, cx, cy, xfx, yfx)
end

"""
    upload!(gpu_buf::CubedSphereMetBuffer, cpu_buf::CubedSphereCPUBuffer)

Copy CPU staging buffer contents to GPU cubed-sphere met buffer.
"""
function upload!(buf::CubedSphereMetBuffer, cpu::CubedSphereCPUBuffer)
    for p in 1:6
        copyto!(buf.delp[p], cpu.delp[p])
        copyto!(buf.am[p], cpu.am[p])
        copyto!(buf.bm[p], cpu.bm[p])
    end
    # Upload Courant numbers + area fluxes if available (GCHP-faithful transport)
    if buf.cx !== nothing && cpu.cx !== nothing
        for p in 1:6
            copyto!(buf.cx[p], cpu.cx[p])
            copyto!(buf.cy[p], cpu.cy[p])
        end
    end
    if buf.xfx !== nothing && cpu.xfx !== nothing
        for p in 1:6
            copyto!(buf.xfx[p], cpu.xfx[p])
            copyto!(buf.yfx[p], cpu.yfx[p])
        end
    end
    return nothing
end
