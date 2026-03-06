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

Returns a `CombinedFlux` containing `TimeVaryingSurfaceFlux` for biosphere,
ocean, and fire components, plus a static `SurfaceFlux` for fossil fuels
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
                bio_rg  = nearest_neighbor_regrid(bio_raw, lon_ct, lat_ct, target_grid)
                ocn_rg  = nearest_neighbor_regrid(ocn_raw, lon_ct, lat_ct, target_grid)
                fire_rg = nearest_neighbor_regrid(fire_raw, lon_ct, lat_ct, target_grid)
            else
                bio_rg  = ensure_south_to_north(bio_raw, lat_ct)[1]
                ocn_rg  = ensure_south_to_north(ocn_raw, lat_ct)[1]
                fire_rg = ensure_south_to_north(fire_raw, lat_ct)[1]
            end

            push!(bio_all,  bio_rg)
            push!(ocn_all,  ocn_rg)
            push!(fire_all, fire_rg)

            if include_fossil
                fos_raw = _ct_read_var(ds, "fossil_flux_imp", ti, FT) .* FT(M_CO2)
                fos_rg = needs_regrid ?
                    nearest_neighbor_regrid(fos_raw, lon_ct, lat_ct, target_grid) :
                    ensure_south_to_north(fos_raw, lat_ct)[1]
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

    bio_src  = TimeVaryingSurfaceFlux(bio_stack, time_hours, :co2;
                                    label="CT-NRT biosphere NEE")
    ocn_src  = TimeVaryingSurfaceFlux(ocn_stack, time_hours, :co2;
                                    label="CT-NRT ocean flux")
    fire_src = TimeVaryingSurfaceFlux(fire_stack, time_hours, :co2;
                                    label="CT-NRT fire emissions")

    components = if include_fossil
        fos_stack = _stack_matrices(fos_all, Nx_m, Ny_m, FT)
        fos_src = TimeVaryingSurfaceFlux(fos_stack, time_hours, :co2;
                                       label="CT-NRT fossil fuel")
        (bio_src, ocn_src, fire_src, fos_src)
    else
        (bio_src, ocn_src, fire_src)
    end

    return CombinedFlux(components, "CarbonTracker CT-NRT $year")
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


function _stack_matrices(mats::Vector{Matrix{FT}}, Nx, Ny, ::Type{FT}) where FT
    Nt = length(mats)
    stack = Array{FT, 3}(undef, Nx, Ny, Nt)
    for t in 1:Nt
        stack[:, :, t] .= mats[t]
    end
    return stack
end
