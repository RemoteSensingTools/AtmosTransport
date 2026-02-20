# ---------------------------------------------------------------------------
# CarbonTracker CT-NRT reader — load biosphere, ocean, and fire CO2 fluxes
#
# CT-NRT provides global surface fluxes at 1x1 deg, 3-hourly, in NetCDF.
# Flux components: bio_flux_opt, ocn_flux_opt, fire_flux_imp, fossil_flux_imp
# Units: mol/m²/s (positive = emission to atmosphere)
#
# Download: https://gml.noaa.gov/aftp/products/carbontracker/co2/CT-NRT.v2025-1/
# ---------------------------------------------------------------------------

const M_C = 12.011e-3  # kg/mol carbon

"""
$(TYPEDSIGNATURES)

Load CarbonTracker CT-NRT flux components from NetCDF files in `datadir`.
Files are expected to be named `CT-NRT.v*_flux1x1_YYYY-MM-DD.nc` (daily,
containing 8 × 3-hourly time steps).

Returns a `CompositeEmission` containing `TimeVaryingEmission` for biosphere,
ocean, and fire components, plus a static `GriddedEmission` for fossil fuels
(or `nothing` if `include_fossil=false` since EDGAR may already cover this).

Arguments:
- `datadir`: directory containing CT-NRT flux NetCDF files
- `target_grid`: model grid for regridding (if CT-NRT is already at 1x1
  matching the model, regridding is a fast identity operation)
- `year`: year to load (e.g., 2024)
- `include_fossil`: whether to include CT-NRT fossil fuel component
  (default: false, since EDGAR is typically used separately)
"""
function load_carbontracker_fluxes(datadir::String,
                                   target_grid::LatitudeLongitudeGrid{FT};
                                   year::Int = 2024,
                                   include_fossil::Bool = false) where FT
    # Discover flux files for the requested year
    all_files = sort(filter(readdir(datadir; join=true)) do f
        bn = basename(f)
        endswith(bn, ".nc") && contains(bn, "flux1x1") && contains(bn, string(year))
    end; by=basename)

    isempty(all_files) && error("No CT-NRT flux1x1 files for year $year in $datadir")
    @info "CarbonTracker: found $(length(all_files)) daily files for $year"

    # Read grid from first file to determine CT-NRT dimensions
    ds0 = NCDataset(all_files[1])
    lon_ct = ds0["longitude"][:]
    lat_ct = ds0["latitude"][:]
    Nlon_ct = length(lon_ct)
    Nlat_ct = length(lat_ct)
    close(ds0)

    Nx_m, Ny_m = target_grid.Nx, target_grid.Ny
    needs_regrid = (Nlon_ct != Nx_m || Nlat_ct != Ny_m)

    # Accumulate all time steps
    bio_all  = Vector{Matrix{FT}}()
    ocn_all  = Vector{Matrix{FT}}()
    fire_all = Vector{Matrix{FT}}()
    fos_all  = Vector{Matrix{FT}}()
    time_hours = Float64[]

    hour_offset = 0.0

    for filepath in all_files
        ds = NCDataset(filepath)

        nt = haskey(ds.dim, "time") ? ds.dim["time"] : 1

        for ti in 1:nt
            # CT-NRT fluxes are in mol/m²/s — convert to kg(CO2)/m²/s
            # mol CO2/m²/s × 44.01e-3 kg/mol = kg CO2/m²/s
            bio_raw  = _ct_read_var(ds, "bio_flux_opt", ti, FT) .* FT(M_CO2)
            ocn_raw  = _ct_read_var(ds, "ocn_flux_opt", ti, FT) .* FT(M_CO2)
            fire_raw = _ct_read_var(ds, "fire_flux_imp", ti, FT) .* FT(M_CO2)

            if needs_regrid
                bio_rg  = _simple_regrid(bio_raw, lon_ct, lat_ct, target_grid, FT)
                ocn_rg  = _simple_regrid(ocn_raw, lon_ct, lat_ct, target_grid, FT)
                fire_rg = _simple_regrid(fire_raw, lon_ct, lat_ct, target_grid, FT)
            else
                bio_rg  = _orient_to_model(bio_raw, lat_ct, FT)
                ocn_rg  = _orient_to_model(ocn_raw, lat_ct, FT)
                fire_rg = _orient_to_model(fire_raw, lat_ct, FT)
            end

            push!(bio_all,  bio_rg)
            push!(ocn_all,  ocn_rg)
            push!(fire_all, fire_rg)

            if include_fossil
                fos_raw = _ct_read_var(ds, "fossil_flux_imp", ti, FT) .* FT(M_CO2)
                fos_rg = needs_regrid ?
                    _simple_regrid(fos_raw, lon_ct, lat_ct, target_grid, FT) :
                    _orient_to_model(fos_raw, lat_ct, FT)
                push!(fos_all, fos_rg)
            end

            push!(time_hours, hour_offset)
            hour_offset += 3.0  # 3-hourly data
        end

        close(ds)
    end

    Nt = length(time_hours)
    @info "CarbonTracker: loaded $Nt time steps ($(Nt*3) hours = $(Nt*3/24) days)"

    bio_stack  = _stack_matrices(bio_all, Nx_m, Ny_m, FT)
    ocn_stack  = _stack_matrices(ocn_all, Nx_m, Ny_m, FT)
    fire_stack = _stack_matrices(fire_all, Nx_m, Ny_m, FT)

    bio_src  = TimeVaryingEmission(bio_stack, time_hours, :co2;
                                    label="CT-NRT biosphere NEE")
    ocn_src  = TimeVaryingEmission(ocn_stack, time_hours, :co2;
                                    label="CT-NRT ocean flux")
    fire_src = TimeVaryingEmission(fire_stack, time_hours, :co2;
                                    label="CT-NRT fire emissions")

    components = if include_fossil
        fos_stack = _stack_matrices(fos_all, Nx_m, Ny_m, FT)
        fos_src = TimeVaryingEmission(fos_stack, time_hours, :co2;
                                       label="CT-NRT fossil fuel")
        (bio_src, ocn_src, fire_src, fos_src)
    else
        (bio_src, ocn_src, fire_src)
    end

    return CompositeEmission(components, "CarbonTracker CT-NRT $year")
end

# --- Internal helpers ---

function _ct_read_var(ds, varname::String, ti::Int, ::Type{FT}) where FT
    if haskey(ds, varname)
        raw = ds[varname]
        if ndims(raw) == 3
            return FT.(nomissing(raw[:, :, ti], 0.0f0))
        else
            return FT.(nomissing(raw[:, :], 0.0f0))
        end
    end
    @warn "Variable $varname not found in CT-NRT file; returning zeros"
    Nlon = ds.dim["longitude"]
    Nlat = ds.dim["latitude"]
    return zeros(FT, Nlon, Nlat)
end

"""
Ensure flux array is oriented S→N latitude to match model grid convention.
"""
function _orient_to_model(flux::Matrix{FT}, lat_src, ::Type{FT}) where FT
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        return flux[:, end:-1:1]
    end
    return copy(flux)
end

"""
Nearest-neighbor regridding for 1x1→model grid when grids don't match exactly.
For matched 1x1 grids this is fast identity. For mismatched grids, uses
nearest-neighbor interpolation (conservative regridding done elsewhere for
high-res sources like EDGAR/GFAS).
"""
function _simple_regrid(flux_src::Matrix{FT}, lon_src, lat_src,
                        grid::LatitudeLongitudeGrid{FT}, ::Type{FT}) where FT
    Nx_m, Ny_m = grid.Nx, grid.Ny
    flux_out = zeros(FT, Nx_m, Ny_m)

    # Ensure S→N
    lat_sorted = lat_src[1] > lat_src[end] ? reverse(lat_src) : lat_src
    flux_sorted = lat_src[1] > lat_src[end] ? flux_src[:, end:-1:1] : flux_src

    # Handle -180:180 vs 0:360
    lon_use = Float64.(lon_src)
    if minimum(lon_use) < 0
        n = length(lon_use)
        split = findfirst(>=(0), lon_use)
        if split !== nothing
            idx = vcat(split:n, 1:split-1)
            lon_use = mod.(lon_use[idx], 360.0)
            flux_sorted = flux_sorted[idx, :]
        end
    end

    λᶜ = grid.λᶜ_cpu
    φᶜ = grid.φᶜ_cpu

    for jm in 1:Ny_m, im in 1:Nx_m
        js = _nearest_idx(φᶜ[jm], lat_sorted)
        is = _nearest_idx(λᶜ[im], lon_use)
        flux_out[im, jm] = flux_sorted[is, js]
    end

    return flux_out
end

function _nearest_idx(val, arr)
    _, idx = findmin(abs.(arr .- val))
    return idx
end

function _stack_matrices(mats::Vector{Matrix{FT}}, Nx, Ny, ::Type{FT}) where FT
    Nt = length(mats)
    stack = Array{FT, 3}(undef, Nx, Ny, Nt)
    for t in 1:Nt
        stack[:, :, t] .= mats[t]
    end
    return stack
end
