#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Export a cubed-sphere TRANSPORT MET binary (m, am, bm, cm, ps, dm — the flux
# fields, transport binary v4) to a self-describing NetCDF so external tools that
# cannot parse the custom .bin can read the staggered mass fluxes directly.
#
#   julia --project=. scripts/postprocess/cs_metbinary_to_netcdf.jl <in.bin> <out.nc>
#
# Streams window-by-window (a full day at C180/L72 would OOM if assembled whole).
# Layout mirrors the native GEOS-IT files: CDL dims (time, lev, nf, Ydim, Xdim);
# am on Xdim_stag(=Nc+1), bm on Ydim_stag, cm on lev_edge(=Nz+1). lons/lats per
# cell are written so the consumer needs no grid library.
# ---------------------------------------------------------------------------
using AtmosTransport
using AtmosTransport: CubedSphereBinaryReader, load_cs_window,
                      load_flux_delta_window!, cs_window_count
using AtmosTransport.Preprocessing: build_target_geometry, panel_cell_center_lonlat
using NCDatasets, Printf

function main()
    length(ARGS) >= 2 || error("usage: cs_metbinary_to_netcdf.jl <in.bin> <out.nc>")
    inpath, outpath = ARGS[1], ARGS[2]
    isfile(inpath) || error("input not found: $inpath")
    abspath(inpath) == abspath(outpath) && error("output collides with input")

    FT = Float32
    reader = CubedSphereBinaryReader(inpath; FT = FT)
    h = reader.header
    Nc, Nz, np, nwin = h.Nc, h.nlevel, h.npanel, cs_window_count(reader)
    has_dm = :dm in h.payload_sections
    @printf("in=%s  Nc=%d Nz=%d panels=%d windows=%d  dm=%s\n",
            basename(inpath), Nc, Nz, np, nwin, has_dm)

    # geometry for lons/lats (C180 GEOS-native GMAO cube)
    gridcfg = Dict{String,Any}("type"=>"cubed_sphere", "Nc"=>Nc,
        "panel_convention"=>"geos_native", "definition"=>"gmao",
        "regridder_cache_dir"=>expanduser("~/.cache/AtmosTransport/cr_regridding"))
    grid = build_target_geometry(gridcfg, Float64)
    lons = Array{Float32}(undef, Nc, Nc, np); lats = similar(lons)
    for p in 1:np
        lo, la = panel_cell_center_lonlat(grid.mesh, p)
        lons[:, :, p] .= Float32.(lo); lats[:, :, p] .= Float32.(la)
    end

    isfile(outpath) && rm(outpath)
    ds = NCDataset(outpath, "c")
    defDim(ds, "Xdim", Nc);      defDim(ds, "Ydim", Nc);   defDim(ds, "nf", np)
    defDim(ds, "lev", Nz);       defDim(ds, "lev_edge", Nz + 1)
    defDim(ds, "Xdim_stag", Nc + 1); defDim(ds, "Ydim_stag", Nc + 1)
    defDim(ds, "time", nwin)
    ds.attrib["title"] = "CS transport met binary export (dry mass fluxes)"
    ds.attrib["source_binary"] = basename(inpath)
    ds.attrib["convention"] = "k=1 is model TOP; am/bm = x/y face dry mass flux; cm = layer-edge vertical dry mass flux (cm[1]=cm[Nz+1]=0)"
    ds.attrib["units_flux"] = "kg per window"; ds.attrib["units_mass"] = "kg"

    vlon = defVar(ds, "lons", Float32, ("Xdim","Ydim","nf")); vlon[:,:,:] = lons
    vlat = defVar(ds, "lats", Float32, ("Xdim","Ydim","nf")); vlat[:,:,:] = lats
    vm  = defVar(ds, "m",  Float32, ("Xdim","Ydim","nf","lev","time"))
    vps = defVar(ds, "ps", Float32, ("Xdim","Ydim","nf","time"))
    vam = defVar(ds, "am", Float32, ("Xdim_stag","Ydim","nf","lev","time"))
    vbm = defVar(ds, "bm", Float32, ("Xdim","Ydim_stag","nf","lev","time"))
    vcm = defVar(ds, "cm", Float32, ("Xdim","Ydim","nf","lev_edge","time"))
    vdm = has_dm ? defVar(ds, "dm", Float32, ("Xdim","Ydim","nf","lev","time")) : nothing

    for w in 1:nwin
        win = load_cs_window(reader, w)
        for p in 1:np
            vm[:, :, p, :, w]  = win.m[p]
            vps[:, :, p, w]    = win.ps[p]
            vam[:, :, p, :, w] = win.am[p]
            vbm[:, :, p, :, w] = win.bm[p]
            vcm[:, :, p, :, w] = win.cm[p]
        end
        if has_dm
            dmw = load_flux_delta_window!(reader, w)
            for p in 1:np
                vdm[:, :, p, :, w] = dmw.dm[p]
            end
        end
        w % 6 == 0 && @printf("  window %d/%d written\n", w, nwin)
    end
    close(ds)
    @printf("wrote %s  (%.2f GB)\n", outpath, filesize(outpath)/2^30)
end
main()
