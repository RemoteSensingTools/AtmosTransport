# ---------------------------------------------------------------------------
# Generate synthetic test meteorological data for GEOS-FP format
#
# Creates small NetCDF files simulating GEOS-FP structure for integration tests.
# Run via: include("generate_test_met_data.jl") from the test/ directory,
# or it will be invoked by test_met_integration.jl.
# ---------------------------------------------------------------------------

using NCDatasets

function generate_test_met_data(; force::Bool = false)
    Nx, Ny, Nz = 8, 4, 5
    FILL = -9999.0
    data_dir = joinpath(@__DIR__, "data", "geosfp_test")
    mkpath(data_dir)

    asm_path = joinpath(data_dir, "inst3_3d_asm_Nv.nc")
    trb_path = joinpath(data_dir, "tavg3_3d_trb_Ne.nc")

    if !force && isfile(asm_path) && isfile(trb_path)
        return asm_path, trb_path
    end

    # --- File 1: inst3_3d_asm_Nv.nc (3D assimilated state on native levels) ---
    # Use netcdf3_classic for maximum compatibility (avoids HDF5 issues)
    NCDataset(asm_path, "c"; format = :netcdf3_classic) do ds
        # Dimensions
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nz)
        defDim(ds, "time", 1)

        # Coordinate variables
        lon = defVar(ds, "lon", Float64, ("lon",))
        lon.attrib["long_name"] = "longitude"
        lon[:] = range(-180.0, 180.0 - 45.0; length = Nx)

        lat = defVar(ds, "lat", Float64, ("lat",))
        lat.attrib["long_name"] = "latitude"
        lat[:] = range(-90.0, 90.0; length = Ny)

        lev = defVar(ds, "lev", Float64, ("lev",))
        lev.attrib["long_name"] = "vertical level"
        lev[:] = collect(1:Nz)

        time = defVar(ds, "time", Float64, ("time",))
        time.attrib["long_name"] = "time"
        time[:] = [0.0]

        # u: zonal wind [m/s], 5.0 + sinusoidal pattern
        u = defVar(ds, "u", Float64, ("lon", "lat", "lev", "time");
                   fillvalue = FILL)
        u.attrib["long_name"] = "zonal wind"
        u.attrib["units"] = "m/s"
        u_data = zeros(Nx, Ny, Nz, 1)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            u_data[i, j, k, 1] = 5.0 + 2.0 * sin(2π * (i - 1) / Nx) * cos(π * (j - 1) / Ny)
        end
        u[:, :, :, :] = u_data

        # v: meridional wind [m/s], -2.0 + small perturbation
        v = defVar(ds, "v", Float64, ("lon", "lat", "lev", "time");
                   fillvalue = FILL)
        v.attrib["long_name"] = "meridional wind"
        v.attrib["units"] = "m/s"
        v_data = zeros(Nx, Ny, Nz, 1)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            v_data[i, j, k, 1] = -2.0 + 0.5 * cos(2π * (i - 1) / Nx) * sin(π * (j - 1) / Ny)
        end
        v[:, :, :, :] = v_data

        # omega: vertical velocity [Pa/s], small values
        omega = defVar(ds, "omega", Float64, ("lon", "lat", "lev", "time");
                      fillvalue = FILL)
        omega.attrib["long_name"] = "vertical velocity (pressure)"
        omega.attrib["units"] = "Pa/s"
        omega_data = zeros(Nx, Ny, Nz, 1)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            omega_data[i, j, k, 1] = -0.01 * (1.0 + 0.1 * (k - 1))
        end
        omega[:, :, :, :] = omega_data

        # t: temperature [K], 280-220 decreasing with level
        t = defVar(ds, "t", Float64, ("lon", "lat", "lev", "time");
                   fillvalue = FILL)
        t.attrib["long_name"] = "temperature"
        t.attrib["units"] = "K"
        t_data = zeros(Nx, Ny, Nz, 1)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            t_data[i, j, k, 1] = 280.0 - 12.0 * k
        end
        t[:, :, :, :] = t_data

        # qv: specific humidity [kg/kg], small values
        qv = defVar(ds, "qv", Float64, ("lon", "lat", "lev", "time");
                    fillvalue = FILL)
        qv.attrib["long_name"] = "specific humidity"
        qv.attrib["units"] = "kg/kg"
        qv_data = zeros(Nx, Ny, Nz, 1)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            qv_data[i, j, k, 1] = 1e-4 * (1.0 - 0.5 * (k - 1) / Nz)
        end
        qv[:, :, :, :] = qv_data

        # ps: surface pressure [Pa]
        ps = defVar(ds, "ps", Float64, ("lon", "lat", "time"); fillvalue = FILL)
        ps.attrib["long_name"] = "surface pressure"
        ps.attrib["units"] = "Pa"
        ps[:, :, :] = fill(101325.0, Nx, Ny, 1)

        # delp: pressure thickness [Pa], realistic values
        delp = defVar(ds, "delp", Float64, ("lon", "lat", "lev", "time");
                    fillvalue = FILL)
        delp.attrib["long_name"] = "pressure thickness"
        delp.attrib["units"] = "Pa"
        delp_data = zeros(Nx, Ny, Nz, 1)
        base_thickness = 101325.0 / Nz
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            delp_data[i, j, k, 1] = base_thickness * (1.0 + 0.05 * (k - 1))
        end
        delp[:, :, :, :] = delp_data
    end

    # --- File 2: tavg3_3d_trb_Ne.nc (turbulence at layer edges) ---
    Nlev_edge = Nz + 1  # 6 levels for edges

    NCDataset(trb_path, "c"; format = :netcdf3_classic) do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "lev", Nlev_edge)
        defDim(ds, "time", 1)

        lon = defVar(ds, "lon", Float64, ("lon",))
        lon[:] = range(-180.0, 180.0 - 45.0; length = Nx)

        lat = defVar(ds, "lat", Float64, ("lat",))
        lat[:] = range(-90.0, 90.0; length = Ny)

        lev = defVar(ds, "lev", Float64, ("lev",))
        lev[:] = collect(1:Nlev_edge)

        time = defVar(ds, "time", Float64, ("time",))
        time[:] = [0.0]

        # kh: diffusivity [m²/s], exponential profile (larger near surface)
        kh = defVar(ds, "kh", Float64, ("lon", "lat", "lev", "time");
                   fillvalue = FILL)
        kh.attrib["long_name"] = "vertical diffusivity"
        kh.attrib["units"] = "m2/s"
        kh_data = zeros(Nx, Ny, Nlev_edge, 1)
        for k in 1:Nlev_edge, j in 1:Ny, i in 1:Nx
            kh_data[i, j, k, 1] = 10.0 * exp(-(Nlev_edge - k) / 2.0)
        end
        kh[:, :, :, :] = kh_data

        # ple: edge pressure [Pa]
        ple = defVar(ds, "ple", Float64, ("lon", "lat", "lev", "time");
                    fillvalue = FILL)
        ple.attrib["long_name"] = "pressure at layer edges"
        ple.attrib["units"] = "Pa"
        ple_data = zeros(Nx, Ny, Nlev_edge, 1)
        for k in 1:Nlev_edge, j in 1:Ny, i in 1:Nx
            ple_data[i, j, k, 1] = 101325.0 * (1.0 - (k - 1) / Nlev_edge)
        end
        ple[:, :, :, :] = ple_data
    end

    return asm_path, trb_path
end

# When file is loaded (include or script), ensure test data exists
generate_test_met_data()
