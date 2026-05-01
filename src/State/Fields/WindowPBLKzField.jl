"""
    WindowPBLKzField(host_cache; params = PBLPhysicsParameters{FT}())

Panel-native Kz cache for cubed-sphere window-driven PBL diffusion.

`host_cache` is an `NTuple{6, Array{FT,3}}` with interior panel shape
`(Nc, Nc, Nz)`. The runtime refreshes it from the active transport window's
raw PBL surface fields and dry air mass whenever the met window advances.
The diffusion kernels read the wrapped `PreComputedKzField`s.
"""
struct WindowPBLKzField{FT, F <: AbstractTimeVaryingField{FT, 3}, H,
                        P <: PBLPhysicsParameters{FT}} <: AbstractCubedSphereField{FT}
    panels     :: NTuple{6, F}
    host_cache :: H
    params     :: P
end

function WindowPBLKzField(host_cache::NTuple{6, Array{FT, 3}};
                          params = PBLPhysicsParameters{FT}()) where FT
    params isa PBLPhysicsParameters{FT} ||
        throw(ArgumentError("params must be a PBLPhysicsParameters{$FT}; got $(typeof(params))"))
    panels = ntuple(p -> PreComputedKzField(host_cache[p]), 6)
    return WindowPBLKzField{FT, typeof(panels[1]), typeof(host_cache), typeof(params)}(
        panels, host_cache, params)
end

@inline panel_field(f::WindowPBLKzField, p::Integer) = f.panels[Int(p)]
update_field!(f::WindowPBLKzField, ::Real) = f

function Adapt.adapt_structure(to, f::WindowPBLKzField)
    panels = Adapt.adapt(to, f.panels)
    return WindowPBLKzField{_window_pbl_eltype(f), typeof(panels[1]),
                            typeof(f.host_cache), typeof(f.params)}(
        panels, f.host_cache, f.params)
end

@inline _window_pbl_eltype(::WindowPBLKzField{FT}) where FT = FT

_host_array(a::Array) = a
_host_array(a) = Array(a)

@inline function _surface_panel(surface, name::Symbol, p::Int)
    return _host_array(getfield(surface, name)[p])
end

"""
    refresh_pbl_kz_cache!(field, surface, air_mass, cell_areas; halo_width)

Recompute `field` from a window's raw PBL surface forcing and dry air mass.
`air_mass` may be halo-padded; `halo_width` selects the interior. The computed
host cache is copied back into the field panels, which may be CPU or device
arrays.
"""
function refresh_pbl_kz_cache!(field::WindowPBLKzField{FT},
                               surface,
                               air_mass::NTuple{6},
                               cell_areas::AbstractMatrix;
                               halo_width::Integer = 0) where FT
    surface === nothing &&
        throw(ArgumentError("[diffusion] kind=\"pbl\" requires pblh/ustar/hflux/t2m surface fields in the transport window"))
    Hp = Int(halo_width)
    areas = FT.(_host_array(cell_areas))
    p = field.params
    R_dry = p.cp_dry / FT(3.5)

    @inbounds for panel in 1:6
        cache = field.host_cache[panel]
        mhost = _host_array(air_mass[panel])
        pblh  = _surface_panel(surface, :pblh,  panel)
        ustar = _surface_panel(surface, :ustar, panel)
        hflux = _surface_panel(surface, :hflux, panel)
        t2m   = _surface_panel(surface, :t2m,   panel)
        Nc, Ny, Nz = size(cache)

        for j in 1:Ny, i in 1:Nc
            h_pbl = max(FT(pblh[i, j]),  FT(100))
            us    = max(FT(ustar[i, j]), FT(0.01))
            H_sfc = FT(hflux[i, j])
            T_sfc = max(FT(t2m[i, j]),   FT(200))

            L_ob, H_kin = _obukhov_length(H_sfc, us, T_sfc, p)
            Pr_inv = _prandtl_inverse(h_pbl, us, H_kin, T_sfc, L_ob, p)
            R_T_over_g = R_dry * T_sfc / p.gravity

            z_col = zero(FT)
            p_top = zero(FT)
            for k in 1:Nz
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
                p_bot = p_top + delp_k
                p_mid = max((p_top + p_bot) / FT(2), FT(1))
                z_col += delp_k * R_T_over_g / p_mid
                p_top = p_bot
            end

            z_above = z_col
            p_top = zero(FT)
            for k in 1:Nz
                delp_k = FT(mhost[i + Hp, j + Hp, k]) * p.gravity / areas[i, j]
                p_bot = p_top + delp_k
                p_mid = max((p_top + p_bot) / FT(2), FT(1))
                dz_k = delp_k * R_T_over_g / p_mid
                z_center = z_above - dz_k / FT(2)
                cache[i, j, k] = _beljaars_viterbo_kz(z_center, h_pbl, us,
                                                      L_ob, Pr_inv, p)
                z_above -= dz_k
                p_top = p_bot
            end
        end

        copyto!(field.panels[panel].data, cache)
    end
    return field
end

export WindowPBLKzField, refresh_pbl_kz_cache!
