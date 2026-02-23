# ---------------------------------------------------------------------------
# Wind processing utilities for raw met data
# ---------------------------------------------------------------------------

using NCDatasets

"""
    stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)

Interpolate cell-centered wind fields to staggered (face) positions.
u_stag has shape (Nx+1, Ny, Nz), v_stag has shape (Nx, Ny+1, Nz).
"""
function stagger_winds!(u_stag, v_stag, u_cc, v_cc, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u_stag[Nx + 1, :, :] .= u_stag[1, :, :]

    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    v_stag[:, 1, :] .= 0
    v_stag[:, Ny + 1, :] .= 0
    return nothing
end

"""
    load_era5_timestep(filepath, tidx, FT)

Read one ERA5 model-level timestep. Flips latitude from N→S to S→N.
Returns `(u, v, surface_pressure)`.
"""
function load_era5_timestep(filepath::String, tidx::Int, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        u    = FT.(ds["u"][:, :, :, tidx])[:, end:-1:1, :]
        v    = FT.(ds["v"][:, :, :, tidx])[:, end:-1:1, :]
        lnsp = FT.(ds["lnsp"][:, :, tidx])[:, end:-1:1]
        return u, v, exp.(lnsp)
    finally
        close(ds)
    end
end

"""
    get_era5_grid_info(filepath, FT)

Read ERA5 grid metadata: lons, lats (flipped S→N), level indices, and time count.
Returns `(lons, lats, levs, Nx, Ny, Nz, Nt)`.
"""
function get_era5_grid_info(filepath::String, ::Type{FT}) where {FT}
    ds = NCDataset(filepath)
    try
        lons = FT.(ds["longitude"][:])
        lats = FT.(ds["latitude"][:])
        levs = ds["model_level"][:]
        Nt   = length(ds["valid_time"][:])
        return lons, reverse(lats), levs, length(lons), length(lats), length(levs), Nt
    finally
        close(ds)
    end
end
