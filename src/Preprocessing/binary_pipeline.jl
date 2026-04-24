"""
    write_array!(io, arr) -> Int

Write a dense Julia array to the binary stream without intermediate copies and
return the number of bytes written.
"""
function write_array!(io::IO, arr::Array{FT}) where FT
    nb = sizeof(arr)
    GC.@preserve arr begin
        written = unsafe_write(io, pointer(arr), nb)
    end
    written == nb || error("Short write: expected $nb bytes, got $written")
    return written
end

const HEADER_SIZE = 16384

@inline function exact_steps_per_window(dt_met::Real, dt::Real)
    ratio = Float64(dt_met) / Float64(dt)
    steps = round(Int, ratio)
    isapprox(ratio, steps; atol=1e-12, rtol=0.0) ||
        error("met_interval / dt must be an integer for the current transport-binary contract; got dt_met=$(dt_met), dt=$(dt), ratio=$(ratio)")
    steps >= 1 || error("steps_per_window must be >= 1; got $(steps)")
    return steps
end

# The Poisson balance target is the forward-window mass difference divided
# by `2 × steps_per_window`. The factor of 2 comes from the Strang splitting:
# each full time step applies horizontal fluxes TWICE (once in the forward
# sweep X-Y-Z and once in the reverse Z-Y-X), so the per-application
# mass tendency is half the total tendency. This matches TM5 r1112's
# `poisson_balance_target_scale` in grid_type_ll.F90.
@inline poisson_balance_target_scale(steps_per_window::Integer, ::Type{FT}=Float64) where FT =
    FT(inv(2 * max(Int(steps_per_window), 1)))

"""
    script_provenance(; caller_file=nothing) -> NamedTuple

Collect reproducibility metadata. `caller_file` is the path of the CLI script
that invoked the preprocessing (set by the entry point, not the library).
"""
function script_provenance(; caller_file::Union{String, Nothing}=nothing)
    src_dir = @__DIR__
    script_path = caller_file !== nothing ? abspath(caller_file) : src_dir
    script_mtime = isfile(script_path) ? mtime(script_path) : 0.0
    git_commit = try
        readchomp(`git -C $(src_dir) rev-parse HEAD`)
    catch
        "unknown"
    end
    git_dirty = try
        !isempty(readchomp(`git -C $(src_dir) status --porcelain`))
    catch
        false
    end

    return (
        script_path = script_path,
        script_mtime = script_mtime,
        git_commit = git_commit,
        git_dirty = git_dirty,
        creation_time = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    )
end

"""
    build_v4_header(date, grid, vertical, settings, FT, counts, sizes, provenance)

Construct the fixed-size JSON header for one v4 output file, including payload
layout, vertical metadata, humidity timing semantics, and mass-fix provenance.
"""
function build_v4_header(date::Date,
                         grid::LatLonTargetGeometry,
                         vertical,
                         settings,
                         FT::Type,
                         counts,
                         sizes,
                         provenance)
    payload_sections = String["m", "am", "bm", "cm", "ps"]
    settings.include_qv && append!(payload_sections, ["qv_start", "qv_end"])
    append!(payload_sections, ["dam", "dbm", "dcm", "dm"])
    # Plan 24 Commit 4: TM5 sections follow the deltas, matching the
    # ordering in _transport_push_optional_sections! (plan 23 Commit 3).
    settings.tm5_convection_enable && append!(payload_sections, ["entu", "detu", "entd", "detd"])
    var_names = copy(payload_sections)

    ncell = sizes.Nx * sizes.Ny
    nface_h = (sizes.Nx + 1) * sizes.Ny + sizes.Nx * (sizes.Ny + 1)

    # Plan 39 Commit B: declare the self-describing transport-binary
    # contract explicitly. LL uses the memo-37 canonical window_constant
    # path (tracer drift = 0 for Upwind on uniform IC over 2 days).
    contract = canonical_window_constant_contract(
        steps_per_window   = sizes.steps_per_met,
        humidity_sampling  = settings.include_qv ? :window_endpoints : :none,
        source_flux_sampling = :window_start_endpoint,
        include_flux_delta = true,
    )

    header = Dict{String, Any}(
        "magic" => "MFLX", "version" => 4, "format_version" => 1, "header_bytes" => HEADER_SIZE,
        "grid_type" => "latlon", "horizontal_topology" => "StructuredDirectional",
        "ncell" => ncell, "nface_h" => nface_h, "nlevel" => sizes.Nz, "nwindow" => sizes.Nt,
        "Nx" => sizes.Nx, "Ny" => sizes.Ny, "Nz" => sizes.Nz,
        "Nz_native" => sizes.Nz_native, "Nt" => sizes.Nt,
        "float_type" => string(FT), "float_bytes" => sizeof(FT),
        "mass_basis" => String(settings.mass_basis),
        "payload_sections" => payload_sections,
        "elems_per_window" => counts.elems_per_window,
        # Plan 39 Commit B: the 6 semantic contract fields. Before this,
        # the LL daily writer emitted only the 2 Poisson fields; the
        # runtime parser silently defaulted the missing ones to the
        # pre-memo-37 :window_start_endpoint path, causing the plan-24
        # Commit-4 blow-up. Now declared explicitly.
        "source_flux_sampling"            => String(contract.source_flux_sampling),
        "air_mass_sampling"               => String(contract.air_mass_sampling),
        "flux_sampling"                   => String(contract.flux_sampling),
        "flux_kind"                       => String(contract.flux_kind),
        "delta_semantics"                 => String(contract.delta_semantics),
        "humidity_sampling"               => String(contract.humidity_sampling),
        "window_bytes" => counts.bytes_per_window,
        "n_m" => counts.n_m, "n_am" => counts.n_am, "n_bm" => counts.n_bm,
        "n_cm" => counts.n_cm, "n_ps" => counts.n_ps, "n_qv" => counts.n_qv,
        "n_qv_start" => counts.n_qv_start, "n_qv_end" => counts.n_qv_end,
        "n_cmfmc" => 0,
        "n_entu" => counts.n_entu, "n_detu" => counts.n_detu,
        "n_entd" => counts.n_entd, "n_detd" => counts.n_detd,
        "n_pblh" => 0, "n_t2m" => 0, "n_ustar" => 0, "n_hflux" => 0,
        "n_temperature" => 0,
        "n_dam" => counts.n_dam, "n_dbm" => counts.n_dbm, "n_dcm" => counts.n_dcm, "n_dm" => counts.n_dm,
        "include_flux_delta" => true,
        "include_qv" => false, "include_qv_endpoints" => settings.include_qv,
        "include_cmfmc" => false,
        "include_tm5conv" => settings.tm5_convection_enable, "include_surface" => false,
        "include_temperature" => false,
        "dt_met_seconds" => settings.met_interval,
        "dt_seconds" => settings.dt,
        "half_dt_seconds" => settings.half_dt,
        "steps_per_window" => sizes.steps_per_met,
        "steps_per_met_window" => sizes.steps_per_met,
        "level_top" => first(vertical.level_range), "level_bot" => last(vertical.level_range),
        "A_ifc" => Float64.(vertical.merged_vc.A), "B_ifc" => Float64.(vertical.merged_vc.B),
        "merge_map" => vertical.merge_map, "merge_min_thickness_Pa" => settings.min_dp,
        "var_names" => var_names,
        "date" => Dates.format(date, "yyyy-mm-dd"),
        "spectral_half_dt_seconds" => settings.half_dt,
        # Poisson fields driven by the same TransportBinaryContract above — single source of truth.
        "poisson_balance_target_scale"     => contract.poisson_balance_target_scale,
        "poisson_balance_target_semantics" => contract.poisson_balance_target_semantics,
        "script_path" => provenance.script_path,
        "script_mtime_unix" => provenance.script_mtime,
        "git_commit" => provenance.git_commit,
        "git_dirty" => provenance.git_dirty,
        "creation_time" => provenance.creation_time,
        "mass_fix_enabled" => settings.mass_fix_enable,
        "mass_fix_target_ps_dry_pa" => settings.target_ps_dry_pa,
        "mass_fix_qv_global_climatology" => settings.qv_global_climatology,
        "mass_fix_qv_mode" => settings.include_qv || settings.mass_basis == :dry ?
            "native_hourly_qv" : "global_qv_climatology",
        "ps_offsets_pa_per_window" => zeros(Float64, sizes.Nt),
        "ps_offsets_next_day_hour0_pa" => 0.0,
        "qv_source_type" => isempty(settings.thermo_dir) ? "none" : "era5_thermo_netcdf",
        "qv_source_directory" => settings.thermo_dir,
        "qv_variable_name" => settings.include_qv || settings.mass_basis == :dry ? "q" : "",
        "qv_units" => settings.include_qv || settings.mass_basis == :dry ? "kg kg-1" : "",
        "qv_time_alignment" => settings.include_qv || settings.mass_basis == :dry ?
            "qv_start uses same-day thermo time index i; qv_end uses time index i+1, with next-day 00 UTC for the final window" : "",
        "qv_next_day_alignment" => settings.include_qv || settings.mass_basis == :dry ?
            "final-window qv_end uses next-day thermo file time index 1 (00 UTC) when available" : "",
        "qv_latitude_handling" => settings.include_qv || settings.mass_basis == :dry ?
            "flip latitude to south-to-north if thermo file is stored north-to-south" : "",
    )

    merge!(header, target_header_metadata(grid))
    return header
end

"""
    resolve_qv_requirements(date, settings) -> (; needs_qv, thermo_path)

Determine whether the current run needs humidity data and resolve the matching
daily thermo file path.

Humidity is required whenever `output.include_qv=true` or `mass_basis=:dry`.
The time convention is:
- window `i` uses the same day's thermo time index `i`
- the next-day delta path uses the next day's time index `1`
"""
function resolve_qv_requirements(date::Date, settings)
    needs_qv_for_dry = settings.mass_basis == :dry
    needs_qv = settings.include_qv || needs_qv_for_dry
    needs_qv && isempty(settings.thermo_dir) &&
        error("mass_basis=:dry requires [input].thermo_dir")

    thermo_path = if needs_qv
        path = joinpath(settings.thermo_dir, "era5_thermo_ml_$(Dates.format(date, "yyyymmdd")).nc")
        isfile(path) || error("Thermo file not found: $path")
        path
    else
        ""
    end

    if needs_qv
        @info "  Humidity source: ERA5 thermo NetCDF `q`"
        @info "    Window i -> same-day thermo time index i (00..23 UTC)"
        @info "    Next-day delta -> next-day thermo time index 1 (00 UTC)"
    end

    return (; needs_qv, thermo_path)
end

"""
    window_element_counts(grid, Nz; include_qv=false) -> NamedTuple

Return the per-window element counts for every payload section in the output
binary.
"""
function window_element_counts(grid::LatLonTargetGeometry, Nz::Int;
                                include_qv::Bool=false,
                                tm5_convection::Bool=false)
    Nx = nlon(grid)
    Ny = nlat(grid)

    n_m = Int64(Nx) * Ny * Nz
    n_am = Int64(Nx + 1) * Ny * Nz
    n_bm = Int64(Nx) * (Ny + 1) * Nz
    n_cm = Int64(Nx) * Ny * (Nz + 1)
    n_ps = Int64(Nx) * Ny
    n_qv = Int64(0)
    n_qv_start = include_qv ? n_m : Int64(0)
    n_qv_end = include_qv ? n_m : Int64(0)
    n_tm5 = tm5_convection ? n_m : Int64(0)  # each of entu/detu/entd/detd is (Nx, Ny, Nz)

    return (
        n_m = n_m,
        n_am = n_am,
        n_bm = n_bm,
        n_cm = n_cm,
        n_ps = n_ps,
        n_qv = n_qv,
        n_qv_start = n_qv_start,
        n_qv_end = n_qv_end,
        n_dam = n_am,
        n_dbm = n_bm,
        n_dcm = n_cm,
        n_dm = n_m,
        n_entu = n_tm5,
        n_detu = n_tm5,
        n_entd = n_tm5,
        n_detd = n_tm5,
        elems_per_window = n_m + n_am + n_bm + n_cm + n_ps + n_qv_start + n_qv_end + n_am + n_bm + n_cm + n_m + 4 * n_tm5,
    )
end

"""
    window_byte_sizes(counts, FT, Nt) -> (; bytes_per_window, total_bytes)

Translate payload element counts into byte counts for one window and the full
daily file.
"""
function window_byte_sizes(counts, ::Type{FT}, Nt::Int) where FT
    bytes_per_window = counts.elems_per_window * sizeof(FT)
    total_bytes = Int64(HEADER_SIZE) + Int64(bytes_per_window) * Nt
    return (; bytes_per_window, total_bytes)
end

"""
    output_binary_path(date, out_dir, min_dp, FT) -> String

Build the canonical output filename for one daily v4 binary.
"""
function output_binary_path(date::Date, out_dir::String, min_dp::Float64, ::Type{FT}) where FT
    dp_tag = @sprintf("merged%dPa", round(Int, min_dp))
    ft_tag = FT == Float64 ? "float64" : "float32"
    date_str = Dates.format(date, "yyyymmdd")
    return joinpath(out_dir, "era5_transport_$(date_str)_$(dp_tag)_$(ft_tag).bin")
end

"""
    existing_complete_output(bin_path, total_bytes) -> Bool

Return `true` when a file already exists and matches the expected final size.
"""
existing_complete_output(bin_path::String, total_bytes::Integer) =
    isfile(bin_path) && filesize(bin_path) == total_bytes

"""
    SpectralTransformWorkspace

Scratch buffers and native-level output arrays for one spectral-to-gridpoint
hour transform.
"""
struct SpectralTransformWorkspace
    u_cc       :: Array{Float64, 3}
    v_cc       :: Array{Float64, 3}
    u_stag     :: Array{Float64, 3}
    v_stag     :: Array{Float64, 3}
    dp         :: Array{Float64, 3}
    m_arr      :: Array{Float64, 3}
    am_arr     :: Array{Float64, 3}
    bm_arr     :: Array{Float64, 3}
    cm_arr     :: Array{Float64, 3}
    sp         :: Matrix{Float64}
    P_buf      :: Matrix{Float64}
    fft_buf    :: Vector{ComplexF64}
    field_2d   :: Matrix{Float64}
    P_buf_t    :: Vector{Matrix{Float64}}
    fft_buf_t  :: Vector{Vector{ComplexF64}}
    fft_out_t  :: Vector{Vector{ComplexF64}}
    u_spec_t   :: Vector{Matrix{ComplexF64}}
    v_spec_t   :: Vector{Matrix{ComplexF64}}
    field_2d_t :: Vector{Matrix{Float64}}
    bfft_plans :: Vector{FFTW.cFFTWPlan{ComplexF64, 1, false, 1, Tuple{Int64}}}
end

"""
    MergeWorkspace

Scratch storage used while collapsing native fields onto the merged transport
levels and while forming forward differences for `dam`, `dbm`, `dcm`, and `dm`.
"""
struct MergeWorkspace{FT}
    m_merged       :: Array{FT, 3}
    am_merged      :: Array{FT, 3}
    bm_merged      :: Array{FT, 3}
    cm_merged      :: Array{FT, 3}
    m_native       :: Array{FT, 3}
    am_native      :: Array{FT, 3}
    bm_native      :: Array{FT, 3}
    dam_merged     :: Array{FT, 3}
    dbm_merged     :: Array{FT, 3}
    dcm_merged     :: Array{FT, 3}
    dm_merged      :: Array{FT, 3}
    m_next_merged  :: Array{FT, 3}
    am_next_merged :: Array{FT, 3}
    bm_next_merged :: Array{FT, 3}
end

"""
    WindowStorage

In-memory storage for all windows of the current day before the final binary
write.
"""
struct WindowStorage{FT}
    all_m        :: Vector{Array{FT, 3}}
    all_am       :: Vector{Array{FT, 3}}
    all_bm       :: Vector{Array{FT, 3}}
    all_cm       :: Vector{Array{FT, 3}}
    all_ps       :: Vector{Array{FT, 2}}
    all_qv_start :: Vector{Array{FT, 3}}   # empty vector when include_qv=false
    all_qv_end   :: Vector{Array{FT, 3}}   # empty vector when include_qv=false
    # Plan 24 Commit 4: four TM5 convection sections on the merged
    # transport grid, one per window.  Empty vectors when
    # tm5_convection=false.  Shape per window: (Nx, Ny, Nz).
    all_entu     :: Vector{Array{FT, 3}}
    all_detu     :: Vector{Array{FT, 3}}
    all_entd     :: Vector{Array{FT, 3}}
    all_detd     :: Vector{Array{FT, 3}}
end

abstract type AbstractQVWorkspace{FT} end

"""
    NoQVWorkspace

Placeholder humidity workspace for moist-only runs.
"""
struct NoQVWorkspace{FT} <: AbstractQVWorkspace{FT} end

"""
    NativeQVWorkspace

Humidity workspace backed by daily ERA5 thermo NetCDF files.

When the thermo grid resolution differs from the target grid (e.g. thermo is
720×361 but target is 96×48), `qv_thermo` holds the raw read and
`_interpolate_qv_to_target!` bilinear-interpolates into `qv_native`.
When resolutions match, `qv_thermo` is empty and the read goes directly
into `qv_native`.
"""
struct NativeQVWorkspace{FT} <: AbstractQVWorkspace{FT}
    thermo_path   :: String
    qv_native     :: Array{Float64, 3}   # (Nx_target, Ny_target, Nz_native)
    qv_native_ft  :: Array{FT, 3}
    qv_merged     :: Array{FT, 3}
    include_qv    :: Bool
    qv_thermo     :: Array{Float64, 3}   # (Nx_thermo, Ny_thermo, Nz_native) or empty
    needs_interp  :: Bool
end

Base.summary(ws::SpectralTransformWorkspace) =
    string("SpectralTransformWorkspace(", size(ws.sp, 1), "×", size(ws.sp, 2), "×", size(ws.m_arr, 3), ")")

Base.summary(ws::MergeWorkspace{FT}) where FT =
    string("MergeWorkspace{", FT, "}(", size(ws.m_merged, 1), "×", size(ws.m_merged, 2), "×", size(ws.m_merged, 3), ")")

Base.summary(ws::WindowStorage{FT}) where FT =
    string("WindowStorage{", FT, "}(", length(ws.all_m), " windows)")

"""
    allocate_transform_workspace(grid, T, Nz_native) -> SpectralTransformWorkspace

Allocate the scratch state needed to synthesize one day's hourly spectral input
onto the native target grid.
"""
function allocate_transform_workspace(grid::LatLonTargetGeometry, T::Int, Nz_native::Int)
    Nx = nlon(grid)
    Ny = nlat(grid)

    u_cc = Array{Float64}(undef, Nx, Ny, Nz_native)
    v_cc = Array{Float64}(undef, Nx, Ny, Nz_native)
    u_stag = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    v_stag = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    dp = Array{Float64}(undef, Nx, Ny, Nz_native)
    m_arr = Array{Float64}(undef, Nx, Ny, Nz_native)
    am_arr = Array{Float64}(undef, Nx + 1, Ny, Nz_native)
    bm_arr = Array{Float64}(undef, Nx, Ny + 1, Nz_native)
    cm_arr = Array{Float64}(undef, Nx, Ny, Nz_native + 1)
    sp = Array{Float64}(undef, Nx, Ny)

    nt = Threads.nthreads()
    nt_max = nt + 2  # small margin over nthreads() for thread-pool headroom
    P_buf = zeros(Float64, T + 1, T + 1)
    fft_buf = zeros(ComplexF64, Nx)
    field_2d = Array{Float64}(undef, Nx, Ny)
    P_buf_t = [zeros(Float64, T + 1, T + 1) for _ in 1:nt_max]
    fft_buf_t = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    fft_out_t = [zeros(ComplexF64, Nx) for _ in 1:nt_max]
    u_spec_t = [zeros(ComplexF64, T + 1, T + 1) for _ in 1:nt_max]
    v_spec_t = [zeros(ComplexF64, T + 1, T + 1) for _ in 1:nt_max]
    field_2d_t = [Array{Float64}(undef, Nx, Ny) for _ in 1:nt_max]
    bfft_plans = [plan_bfft(fft_buf_t[i]) for i in 1:nt_max]

    return SpectralTransformWorkspace(u_cc, v_cc, u_stag, v_stag, dp,
                                      m_arr, am_arr, bm_arr, cm_arr, sp,
                                      P_buf, fft_buf, field_2d,
                                      P_buf_t, fft_buf_t, fft_out_t,
                                      u_spec_t, v_spec_t, field_2d_t,
                                      bfft_plans)
end

"""
    allocate_merge_workspace(grid, Nz_native, Nz, FT) -> MergeWorkspace{FT}

Allocate merged-level work arrays and delta buffers for one day.
"""
function allocate_merge_workspace(grid::LatLonTargetGeometry, Nz_native::Int, Nz::Int, ::Type{FT}) where FT
    Nx = nlon(grid)
    Ny = nlat(grid)

    return MergeWorkspace{FT}(Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz),
                              Array{FT}(undef, Nx, Ny, Nz + 1),
                              Array{FT}(undef, Nx, Ny, Nz_native),
                              Array{FT}(undef, Nx + 1, Ny, Nz_native),
                              Array{FT}(undef, Nx, Ny + 1, Nz_native),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz),
                              Array{FT}(undef, Nx, Ny, Nz + 1),
                              Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx, Ny, Nz),
                              Array{FT}(undef, Nx + 1, Ny, Nz),
                              Array{FT}(undef, Nx, Ny + 1, Nz))
end

"""
    allocate_window_storage(Nt, FT; include_qv=false, tm5_convection=false) -> WindowStorage{FT}

Allocate per-window storage for the daily output payloads. When
`tm5_convection=true`, also allocate the four TM5 section vectors.
"""
function allocate_window_storage(Nt::Int, ::Type{FT};
                                 include_qv::Bool=false,
                                 tm5_convection::Bool=false) where FT
    _qv_vec() = include_qv ? Vector{Array{FT, 3}}(undef, Nt) : Vector{Array{FT, 3}}()
    _tm5_vec() = tm5_convection ? Vector{Array{FT, 3}}(undef, Nt) : Vector{Array{FT, 3}}()
    return WindowStorage{FT}(Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 3}}(undef, Nt),
                             Vector{Array{FT, 2}}(undef, Nt),
                             _qv_vec(), _qv_vec(),
                             _tm5_vec(), _tm5_vec(), _tm5_vec(), _tm5_vec())
end

"""
    allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)

Allocate the humidity workspace for the current day. Returns `NoQVWorkspace`
when humidity is not required.
"""
function allocate_qv_workspace(grid::LatLonTargetGeometry,
                               settings,
                               date::Date,
                               Nz_native::Int,
                               Nz::Int,
                               ::Type{FT}) where FT
    setup = resolve_qv_requirements(date, settings)
    Nx = nlon(grid)
    Ny = nlat(grid)

    setup.needs_qv || return NoQVWorkspace{FT}()

    # Detect thermo/target grid mismatch
    Nx_thermo, Ny_thermo = _detect_thermo_grid_size(setup.thermo_path)
    needs_interp = (Nx_thermo != Nx || Ny_thermo != Ny)
    qv_thermo = needs_interp ? Array{Float64}(undef, Nx_thermo, Ny_thermo, Nz_native) :
                               Array{Float64}(undef, 0, 0, 0)
    if needs_interp
        @info @sprintf("  QV interpolation: thermo %d×%d → target %d×%d",
                       Nx_thermo, Ny_thermo, Nx, Ny)
    end

    return NativeQVWorkspace{FT}(setup.thermo_path,
                                 Array{Float64}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz_native),
                                 Array{FT}(undef, Nx, Ny, Nz),
                                 settings.include_qv,
                                 qv_thermo,
                                 needs_interp)
end

"""
    _detect_thermo_grid_size(thermo_path) -> (Nx, Ny)

Read the lon/lat dimensions of the thermo NetCDF file.
"""
function _detect_thermo_grid_size(thermo_path::String)
    NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            return (size(q_var, 1), size(q_var, 2))
        else
            return (size(q_var, 4), size(q_var, 3))
        end
    end
end

read_window_qv!(::NoQVWorkspace, args...) = nothing

"""
    read_window_qv!(qv, win_idx, Nx, Ny, Nz_native)

Load the humidity field associated with one analysis window into the native
humidity workspace. When the thermo grid differs from the target grid,
bilinear-interpolates QV to the target resolution.
"""
function read_window_qv!(qv::NativeQVWorkspace,
                         win_idx::Int,
                         Nx::Int,
                         Ny::Int,
                         Nz_native::Int)
    if qv.needs_interp
        Nx_t = size(qv.qv_thermo, 1)
        Ny_t = size(qv.qv_thermo, 2)
        qv.qv_thermo .= read_qv_from_thermo(qv.thermo_path, win_idx, Nx_t, Ny_t, Nz_native; FT=Float64)
        _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, Nx, Ny, Nx_t, Ny_t, Nz_native)
    else
        qv.qv_native .= read_qv_from_thermo(qv.thermo_path, win_idx, Nx, Ny, Nz_native; FT=Float64)
    end
    return nothing
end

"""
    _interpolate_ll_qv!(dst, src, Nx_dst, Ny_dst, Nx_src, Ny_src, Nz)

Bilinear-interpolate QV from a regular LL source grid to a regular LL target grid.
Both grids are assumed to span [-180,180) × [-90,90] with uniform spacing.
"""
function _interpolate_ll_qv!(dst::Array{Float64, 3}, src::Array{Float64, 3},
                              Nx_dst::Int, Ny_dst::Int,
                              Nx_src::Int, Ny_src::Int,
                              Nz::Int)
    Δlon_src = 360.0 / Nx_src
    Δlat_src = 180.0 / (Ny_src - 1)
    Δlon_dst = 360.0 / Nx_dst
    Δlat_dst = 180.0 / (Ny_dst - 1)

    @inbounds for k in 1:Nz, j in 1:Ny_dst, i in 1:Nx_dst
        # Target cell center in degrees
        lon = -180.0 + (i - 0.5) * Δlon_dst
        lat = -90.0 + (j - 1) * Δlat_dst

        # Source fractional indices
        if_val = (lon + 180.0) / Δlon_src + 0.5
        jf_val = (lat + 90.0) / Δlat_src + 1.0

        i0 = clamp(floor(Int, if_val), 1, Nx_src)
        j0 = clamp(floor(Int, jf_val), 1, Ny_src - 1)
        i1 = i0 == Nx_src ? 1 : i0 + 1  # periodic longitude
        j1 = min(j0 + 1, Ny_src)

        wi = if_val - floor(if_val)
        wj = jf_val - j0

        dst[i, j, k] = (1 - wi) * (1 - wj) * src[i0, j0, k] +
                        wi       * (1 - wj) * src[i1, j0, k] +
                        (1 - wi) * wj       * src[i0, j1, k] +
                        wi       * wj       * src[i1, j1, k]
    end
    return nothing
end

read_next_day_qv!(::NoQVWorkspace, args...) = nothing

"""
    read_next_day_qv!(qv, date, settings, Nx, Ny, Nz_native)

Load next day's hour-0 humidity so the final window can form forward deltas
with consistent dry-basis metadata.
"""
function read_next_day_qv!(qv::NativeQVWorkspace,
                           date::Date,
                           settings,
                           Nx::Int,
                           Ny::Int,
                           Nz_native::Int)
    next_thermo_path = joinpath(settings.thermo_dir,
                                "era5_thermo_ml_$(Dates.format(date + Day(1), "yyyymmdd")).nc")
    isfile(next_thermo_path) || error("Thermo file not found: $next_thermo_path")
    if qv.needs_interp
        Nx_t = size(qv.qv_thermo, 1)
        Ny_t = size(qv.qv_thermo, 2)
        qv.qv_thermo .= read_qv_from_thermo(next_thermo_path, 1, Nx_t, Ny_t, Nz_native; FT=Float64)
        _interpolate_ll_qv!(qv.qv_native, qv.qv_thermo, Nx, Ny, Nx_t, Ny_t, Nz_native)
    else
        qv.qv_native .= read_qv_from_thermo(next_thermo_path, 1, Nx, Ny, Nz_native; FT=Float64)
    end
    return nothing
end

apply_dry_basis_if_needed!(::Val{:moist}, native, qv) = nothing

function apply_dry_basis_if_needed!(::Val{:dry},
                                    native::SpectralTransformWorkspace,
                                    qv::NativeQVWorkspace)
    apply_dry_basis_native!(native.m_arr, native.am_arr, native.bm_arr, qv.qv_native)
    return nothing
end

function apply_dry_basis_if_needed!(::Val{:dry},
                                    native::SpectralTransformWorkspace,
                                    ::NoQVWorkspace)
    error("Dry-mass preprocessing requires a humidity workspace")
end

"""
    apply_dry_basis_if_needed!(basis, native, qv)

Conditionally convert the native hourly mass and horizontal flux fields from
moist to dry basis.
"""
apply_dry_basis_if_needed!(basis::Symbol, native, qv) =
    apply_dry_basis_if_needed!(Val(basis), native, qv)

merge_qv_output!(::NoQVWorkspace, merged, vertical, settings) = nothing

"""
    merge_qv_output!(qv, merged, vertical, settings)

Merge native humidity to the output vertical grid when `qv` is requested in the
binary payload.
"""
function merge_qv_output!(qv::NativeQVWorkspace{FT},
                          merged::MergeWorkspace{FT},
                          vertical,
                          settings) where FT
    settings.include_qv || return nothing
    @. qv.qv_native_ft = FT(qv.qv_native)
    merge_qv!(qv.qv_merged, qv.qv_native_ft, merged.m_native, vertical.merge_map)
    return nothing
end

store_qv_output!(storage::WindowStorage, ::NoQVWorkspace, win_idx::Int) = nothing

"""
    store_qv_output!(storage, qv, win_idx)

Persist the merged `qv` field for one window into the daily storage bundle.
"""
function store_qv_output!(storage::WindowStorage,
                          qv::NativeQVWorkspace,
                          win_idx::Int)
    isempty(storage.all_qv_start) && return nothing
    storage.all_qv_start[win_idx] = copy(qv.qv_merged)
    return nothing
end

"""
    apply_structured_pole_constraints!(am, bm)

Enforce the structured lat-lon polar boundary conditions after vertical merging.
"""
function apply_structured_pole_constraints!(am::AbstractArray{FT, 3},
                                            bm::AbstractArray{FT, 3}) where FT
    Ny = size(am, 2)
    @views am[:, 1, :] .= zero(FT)
    @views am[:, Ny, :] .= zero(FT)
    @views bm[:, 1, :] .= zero(FT)
    @views bm[:, Ny + 1, :] .= zero(FT)
    return nothing
end

"""
    merge_native_window!(merged, native, qv, vertical, settings)

Collapse native hourly fields to the merged transport levels, recompute merged
`cm`, and optionally produce merged `qv`.
"""
function merge_native_window!(merged::MergeWorkspace{FT},
                              native::SpectralTransformWorkspace,
                              qv::AbstractQVWorkspace{FT},
                              vertical,
                              settings) where FT
    @. merged.m_native = FT(native.m_arr)
    @. merged.am_native = FT(native.am_arr)
    @. merged.bm_native = FT(native.bm_arr)

    merge_cell_field!(merged.m_merged, merged.m_native, vertical.merge_map)
    merge_cell_field!(merged.am_merged, merged.am_native, vertical.merge_map)
    merge_cell_field!(merged.bm_merged, merged.bm_native, vertical.merge_map)

    apply_structured_pole_constraints!(merged.am_merged, merged.bm_merged)

    recompute_cm_from_divergence!(merged.cm_merged, merged.am_merged, merged.bm_merged, merged.m_merged;
                                  B_ifc=vertical.merged_vc.B)
    @views merged.cm_merged[:, :, 1] .= zero(FT)
    @views merged.cm_merged[:, :, size(merged.cm_merged, 3)] .= zero(FT)

    merge_qv_output!(qv, merged, vertical, settings)

    return nothing
end

"""
    store_window_fields!(storage, merged, ps, qv, win_idx)

Copy one processed window into the daily storage bundle.
"""
function store_window_fields!(storage::WindowStorage{FT},
                              merged::MergeWorkspace{FT},
                              ps::AbstractMatrix,
                              qv::AbstractQVWorkspace{FT},
                              win_idx::Int) where FT
    storage.all_m[win_idx] = copy(merged.m_merged)
    storage.all_am[win_idx] = copy(merged.am_merged)
    storage.all_bm[win_idx] = copy(merged.bm_merged)
    storage.all_cm[win_idx] = copy(merged.cm_merged)
    storage.all_ps[win_idx] = FT.(ps)
    store_qv_output!(storage, qv, win_idx)
    return nothing
end

"""
    log_mass_fix_configuration(settings)

Emit a short description of the dry-surface-pressure mass-fix mode selected for
the current run.
"""
function log_mass_fix_configuration(settings)
    if settings.mass_fix_enable
        if settings.include_qv || settings.mass_basis == :dry
            @info @sprintf("  Mass fix ENABLED: pin ⟨ps_dry⟩→%.2f Pa using hourly native qv",
                           settings.target_ps_dry_pa)
        else
            target_total = settings.target_ps_dry_pa / (1.0 - settings.qv_global_climatology)
            @info @sprintf("  Mass fix ENABLED: pin ⟨ps_dry⟩→%.2f Pa (⟨ps_total⟩→%.2f Pa, qv_global=%.5f)",
                           settings.target_ps_dry_pa, target_total, settings.qv_global_climatology)
        end
    else
        @info "  Mass fix DISABLED"
    end
    return nothing
end

"""
    should_log_window(win_idx, Nt) -> Bool

Return `true` for the subset of windows that should emit per-window timing
messages.
"""
@inline should_log_window(win_idx::Int, Nt::Int) =
    win_idx <= 3 || win_idx == Nt || win_idx % 8 == 0

apply_mass_fix_if_needed!(::NoQVWorkspace, args...) = nothing

"""
    apply_mass_fix_if_needed!(qv_workspace, transform, grid, vertical, settings, ps_offsets, slot)

Apply the configured global dry-mass fix to the current hourly surface-pressure
field and recompute native mass/flux fields afterward.

When humidity is available, the target dry surface pressure is matched using the
actual hourly native `q` field. Otherwise the fallback global `qv` climatology
is used.
"""
function apply_mass_fix_if_needed!(qv::NoQVWorkspace,
                                   transform::SpectralTransformWorkspace,
                                   grid::LatLonTargetGeometry,
                                   vertical,
                                   settings,
                                   ps_offsets::Vector{Float64},
                                   ps_offset_slot::Int)
    settings.mass_fix_enable || return nothing
    ps_offset = pin_global_mean_ps!(transform.sp, grid.area;
                                    target_ps_dry_pa=settings.target_ps_dry_pa,
                                    qv_global=settings.qv_global_climatology)
    recompute_native_mass_fields!(transform, vertical.ab, grid, settings.half_dt)
    ps_offsets[ps_offset_slot] = ps_offset
    return nothing
end

function apply_mass_fix_if_needed!(qv::NativeQVWorkspace,
                                   transform::SpectralTransformWorkspace,
                                   grid::LatLonTargetGeometry,
                                   vertical,
                                   settings,
                                   ps_offsets::Vector{Float64},
                                   ps_offset_slot::Int)
    settings.mass_fix_enable || return nothing
    ps_offset = pin_global_mean_ps_using_qv!(transform.sp, grid.area,
                                             vertical.ab.dA, vertical.ab.dB, qv.qv_native;
                                             target_ps_dry_pa=settings.target_ps_dry_pa)
    recompute_native_mass_fields!(transform, vertical.ab, grid, settings.half_dt)
    ps_offsets[ps_offset_slot] = ps_offset
    return nothing
end

"""
    process_window!(win_idx, hour, spec, grid, vertical, settings, transform, merged, qv, storage, ps_offsets)

Run the full hourly processing chain for one analysis window:
spectral synthesis, humidity load, optional dry-mass fix, optional dry-basis
conversion, vertical merging, and storage.
"""
function process_window!(win_idx::Int,
                         hour::Int,
                         spec,
                         grid::LatLonTargetGeometry,
                         vertical,
                         settings,
                         transform::SpectralTransformWorkspace,
                         merged::MergeWorkspace{FT},
                         qv::AbstractQVWorkspace{FT},
                         storage::WindowStorage{FT},
                         ps_offsets::Vector{Float64};
                         physics_reader = nothing,
                         tm5_ws = nothing,
                         tm5_stats = nothing) where FT
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    t0 = time()

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
        spec.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_window_qv!(qv, win_idx, Nx, Ny, vertical.Nz_native)
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, win_idx)
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    merge_native_window!(merged, transform, qv, vertical, settings)
    store_window_fields!(storage, merged, transform.sp, qv, win_idx)

    # Plan 24 Commit 4: TM5 convection step — runs only when the
    # caller wired up a physics BIN reader and TM5 workspace.  Uses
    # `transform.sp` for surface pressure (Commit-2 BIN omits PS).
    # `win_idx` (1..Nt) is the correct physics-BIN hour index;
    # ERA5 spectral `hour` is 0-indexed so cannot be used directly.
    if settings.tm5_convection_enable && physics_reader !== nothing
        _store_window_tm5_fields!(storage, win_idx,
                                   physics_reader, tm5_ws,
                                   transform.sp, vertical, tm5_stats, FT)
    end

    elapsed = round(time() - t0, digits=2)
    should_log_window(win_idx, length(storage.all_m)) &&
        @info(@sprintf("    Window %d/%d (hour %02d): %.2fs  ps_offset=%+.3f Pa",
                       win_idx, length(storage.all_m), hour, elapsed, ps_offsets[win_idx]))

    return nothing
end

# Plan 24 Commit 4 helper — per-window TM5 compute + store.
# `ps_target` is `transform.sp` at the preprocessor's target grid
# (== ERA5 native 720×361 for the Commit-4-supported shape).
# `win_idx` is the 1-indexed physics-BIN hour slot (matches the
# 24 hourly slices in the Commit-2 daily BIN).
function _store_window_tm5_fields!(storage::WindowStorage{FT},
                                    win_idx::Int,
                                    physics_reader,
                                    tm5_ws::TM5PreprocessingWorkspace{FT},
                                    ps_target::AbstractMatrix,
                                    vertical,
                                    tm5_stats,
                                    ::Type{FT}) where FT
    compute_tm5_merged_hour_on_source!(
        tm5_ws, physics_reader, win_idx, ps_target,
        vertical.ab.a_ifc, vertical.ab.b_ifc,
        vertical.Nz_native, vertical.merge_map;
        stats = tm5_stats)

    Nx_t, Ny_t, Nz = size(tm5_ws.entu_merged_src)
    storage.all_entu[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_detu[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_entd[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    storage.all_detd[win_idx] = Array{FT, 3}(undef, Nx_t, Ny_t, Nz)
    tm5_copy_or_regrid_ll!(storage.all_entu[win_idx], tm5_ws.entu_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_detu[win_idx], tm5_ws.detu_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_entd[win_idx], tm5_ws.entd_merged_src, tm5_ws)
    tm5_copy_or_regrid_ll!(storage.all_detd[win_idx], tm5_ws.detd_merged_src, tm5_ws)
    return nothing
end

"""
    next_day_merged_fields(next_day_hour0, ...)

Process the next day's hour-0 spectral and humidity fields so the current day's
final window can form forward deltas and carry a consistent mass-fix offset.

Returns `nothing` early when `next_day_hour0 === nothing` (no next-day data
was loaded, e.g. at the end of the processed range).
"""
function next_day_merged_fields(next_day_hour0,
                                date::Date,
                                grid::LatLonTargetGeometry,
                                vertical,
                                settings,
                                transform::SpectralTransformWorkspace,
                                merged::MergeWorkspace{FT},
                                qv::AbstractQVWorkspace{FT},
                                ps_offsets::Vector{Float64}) where FT
    next_day_hour0 === nothing && return nothing
    Nx = size(transform.sp, 1)
    Ny = size(transform.sp, 2)
    @info "  Computing next day hour 0 for last-window delta..."

    spectral_to_native_fields!(
        transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
        transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
        next_day_hour0.lnsp, next_day_hour0.vo, next_day_hour0.d,
        next_day_hour0.T, vertical.level_range, vertical.ab, grid, settings.half_dt,
        transform.P_buf, transform.fft_buf, transform.field_2d,
        transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
        transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
        transform.bfft_plans)

    read_next_day_qv!(qv, date, settings, Nx, Ny, vertical.Nz_native)
    apply_mass_fix_if_needed!(qv, transform, grid, vertical, settings, ps_offsets, length(ps_offsets))
    apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
    merge_native_window!(merged, transform, qv, vertical, settings)

    return (m=copy(merged.m_merged),
            am=copy(merged.am_merged),
            bm=copy(merged.bm_merged),
            cm=copy(merged.cm_merged),
            qv=settings.include_qv ? copy(qv.qv_merged) : nothing)
end

"""
    fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)

Fill the cell-mass target used by the Poisson horizontal-flux balance step.

The stored `am/bm/cm` fields are half-sweep transport amounts. A full Strang
substep applies the horizontal fluxes twice, so the Poisson target must be the
forward window mass difference divided by `2 * steps_per_window`.
"""
function fill_window_mass_tendency!(dm_dt_buf::Array{FT, 3},
                                    storage::WindowStorage{FT},
                                    last_hour_next,
                                    win_idx::Int,
                                    steps_per_window::Int) where FT
    Nt = length(storage.all_m)
    scale = poisson_balance_target_scale(steps_per_window, FT)

    if win_idx < Nt
        dm_dt_buf .= (storage.all_m[win_idx + 1] .- storage.all_m[win_idx]) .* scale
    elseif last_hour_next !== nothing
        dm_dt_buf .= (last_hour_next.m .- storage.all_m[win_idx]) .* scale
    else
        fill!(dm_dt_buf, zero(FT))
    end

    return nothing
end

"""
    verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, ::Type{FT})

Plan 39 Commit E — write-time replay gate for structured LL storage.
Iterates every window k and asserts

    m[k] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[k+1]    (k < Nt)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ last_hour_next.m   (k == Nt, if available)
    m[Nt] − 2·steps·(∇·am + ∇·bm + ∂_k cm) ≈ m[Nt]              (otherwise zero-tendency fallback)

to within a Poisson-balance tolerance floor derived from `FT` (roughly
`1e-10` for `Float64`, `1e-4` for `Float32`). Errors loudly with a per-window
diagnostic if the contract is violated — this was the gate that would have
caught the dry-basis Δb×pit closure bug before it reached the runtime.

Bypass with env var `ATMOSTR_NO_WRITE_REPLAY_CHECK=1` for diagnostic runs.
"""
function verify_storage_continuity_ll!(storage::WindowStorage{FT},
                                        last_hour_next,
                                        steps_per_window::Int,
                                        ::Type{FT}) where FT
    if get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") == "1"
        @info "  Write-time replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
        return nothing
    end
    Nt = length(storage.all_m)
    Nt == 0 && return nothing
    tol_rel = replay_tolerance(FT)
    div_scratch = Array{Float64}(undef, size(storage.all_m[1]))
    layout = structured_replay_layout()
    run_replay_gate(Nt; tol_rel=tol_rel,
                    summary_label="  Write-time replay gate",
                    failure_prefix="Write-time replay gate") do k
        m_next = if k < Nt
            storage.all_m[k + 1]
        elseif last_hour_next !== nothing
            last_hour_next.m
        else
            storage.all_m[k]
        end
        verify_window_continuity(layout, div_scratch,
                                 storage.all_m[k],
                                 storage.all_cm[k],
                                 m_next,
                                 steps_per_window,
                                 storage.all_am[k],
                                 storage.all_bm[k])
    end
    return nothing
end

"""
    apply_poisson_balance!(storage, last_hour_next, steps_per_window)

Apply the TM5-style Poisson horizontal-flux correction to every stored window
so horizontal convergence matches the per-half-sweep mass target implied by the
forward window endpoints.
"""
function apply_poisson_balance!(storage::WindowStorage{FT},
                                last_hour_next,
                                steps_per_window::Int) where FT
    Nx, Ny, Nz = size(storage.all_m[1])
    dm_dt_buf = Array{FT}(undef, Nx, Ny, Nz)
    div_scratch = Array{Float64}(undef, Nx, Ny, Nz)
    poisson_ws = LLPoissonWorkspace(Nx, Ny)
    replay_layout = structured_replay_layout()

    @info "  Applying Poisson mass flux balance..."
    for win_idx in eachindex(storage.all_m)
        fill_window_mass_tendency!(dm_dt_buf, storage, last_hour_next, win_idx, steps_per_window)
        balance_mass_fluxes!(storage.all_am[win_idx], storage.all_bm[win_idx], dm_dt_buf, poisson_ws)
        @views storage.all_bm[win_idx][:, 1, :] .= zero(FT)
        @views storage.all_bm[win_idx][:, Ny + 1, :] .= zero(FT)
        # Plan 39 dry-basis fix (2026-04-22): use explicit-dm closure, not
        # the hybrid Δb×pit one. The Δb×pit closure assumes
        # dm[k] = dB[k] × Σ_k dm[k], which holds under moist hybrid coords
        # but is violated by ~27% under dry basis because qv[k] varies with
        # level. That mismatch caused the 0.75% day-boundary air_mass jump
        # observed on F64 probe; see plan39_reconnect.md memory entry.
        recompute_cm_from_dm_target!(replay_layout, div_scratch,
                                     storage.all_cm[win_idx], storage.all_m[win_idx], dm_dt_buf,
                                     storage.all_am[win_idx], storage.all_bm[win_idx])
        @views storage.all_cm[win_idx][:, :, 1] .= zero(FT)
        @views storage.all_cm[win_idx][:, :, Nz + 1] .= zero(FT)
    end

    # Plan 39 Commit E: write-time replay gate. Under the `:window_constant`
    # contract, starting from `storage.all_m[k]` and integrating the stored
    # fluxes (am, bm, cm) over one window via palindrome continuity must
    # reproduce `storage.all_m[k+1]` (or `last_hour_next.m` for k=Nt) to
    # within the Poisson-balance tolerance floor. Fails loudly if the fix
    # regresses or a new preprocessor path breaks the contract.
    verify_storage_continuity_ll!(storage, last_hour_next, steps_per_window, FT)
    @info "  Poisson balance complete for $(length(storage.all_m)) windows"

    return nothing
end

"""
    compute_window_deltas!(merged, storage, win_idx, last_hour_next)

Form the forward-in-time `dam`, `dbm`, `dcm`, and `dm` payloads for one window.
"""
function compute_window_deltas!(merged::MergeWorkspace{FT},
                                storage::WindowStorage{FT},
                                win_idx::Int,
                                last_hour_next) where FT
    Nt = length(storage.all_m)

    if win_idx < Nt
        merged.dam_merged .= storage.all_am[win_idx + 1] .- storage.all_am[win_idx]
        merged.dbm_merged .= storage.all_bm[win_idx + 1] .- storage.all_bm[win_idx]
        merged.dcm_merged .= storage.all_cm[win_idx + 1] .- storage.all_cm[win_idx]
        merged.dm_merged  .= storage.all_m[win_idx + 1]  .- storage.all_m[win_idx]
    elseif last_hour_next !== nothing
        merged.dam_merged .= last_hour_next.am .- storage.all_am[win_idx]
        merged.dbm_merged .= last_hour_next.bm .- storage.all_bm[win_idx]
        merged.dcm_merged .= last_hour_next.cm .- storage.all_cm[win_idx]
        merged.dm_merged  .= last_hour_next.m  .- storage.all_m[win_idx]
    else
        fill!(merged.dam_merged, zero(FT))
        fill!(merged.dbm_merged, zero(FT))
        fill!(merged.dcm_merged, zero(FT))
        fill!(merged.dm_merged, zero(FT))
    end

    return nothing
end

function fill_qv_endpoints!(storage::WindowStorage{FT}, last_hour_next) where FT
    isempty(storage.all_qv_start) && return nothing
    Nt = length(storage.all_qv_start)

    for win_idx in 1:Nt-1
        storage.all_qv_end[win_idx] = copy(storage.all_qv_start[win_idx + 1])
    end

    if last_hour_next !== nothing && hasproperty(last_hour_next, :qv) && last_hour_next.qv !== nothing
        storage.all_qv_end[Nt] = copy(last_hour_next.qv)
    else
        storage.all_qv_end[Nt] = copy(storage.all_qv_start[Nt])
    end

    return nothing
end

"""
    write_window!(io, win_idx, storage, settings, merged, last_hour_next) -> Int64

Write one window's payload blocks to the output stream in v4 on-disk order.
"""
function write_window!(io::IO,
                       win_idx::Int,
                       storage::WindowStorage{FT},
                       settings,
                       merged::MergeWorkspace{FT},
                       last_hour_next) where FT
    bytes_written = Int64(0)
    bytes_written += write_array!(io, storage.all_m[win_idx])
    bytes_written += write_array!(io, storage.all_am[win_idx])
    bytes_written += write_array!(io, storage.all_bm[win_idx])
    bytes_written += write_array!(io, storage.all_cm[win_idx])
    bytes_written += write_array!(io, storage.all_ps[win_idx])
    if settings.include_qv
        bytes_written += write_array!(io, storage.all_qv_start[win_idx])
        bytes_written += write_array!(io, storage.all_qv_end[win_idx])
    end

    compute_window_deltas!(merged, storage, win_idx, last_hour_next)
    bytes_written += write_array!(io, merged.dam_merged)
    bytes_written += write_array!(io, merged.dbm_merged)
    bytes_written += write_array!(io, merged.dcm_merged)
    bytes_written += write_array!(io, merged.dm_merged)

    # Plan 24 Commit 4: TM5 convection sections (order must match
    # _transport_push_optional_sections! in TransportBinary.jl:557-578).
    if settings.tm5_convection_enable
        bytes_written += write_array!(io, storage.all_entu[win_idx])
        bytes_written += write_array!(io, storage.all_detu[win_idx])
        bytes_written += write_array!(io, storage.all_entd[win_idx])
        bytes_written += write_array!(io, storage.all_detd[win_idx])
    end

    return bytes_written
end

"""
    write_day_binary!(bin_path, header_json, storage, settings, merged, last_hour_next)

Write the padded header and all window payloads for one daily binary file.
Returns the total number of bytes written.
"""
function write_day_binary!(bin_path::String,
                           header_json,
                           storage::WindowStorage{FT},
                           settings,
                           merged::MergeWorkspace{FT},
                           last_hour_next) where FT
    @info "  Writing binary..."
    bytes_written = Int64(0)

    open(bin_path, "w") do io
        hdr_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(hdr_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, hdr_buf)
        bytes_written += HEADER_SIZE

        for win_idx in eachindex(storage.all_m)
            bytes_written += write_window!(io, win_idx, storage, settings, merged, last_hour_next)
        end

        flush(io)
    end

    return bytes_written
end

@inline function fill_cs_window_mass_tendency!(dm_panels::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                               m_next::NTuple{NP, <:AbstractArray{FT, 3}},
                                               steps_per_window::Int) where {FT, NP}
    inv_two_steps = one(FT) / FT(2 * steps_per_window)
    for p in 1:NP
        @inbounds for idx in eachindex(dm_panels[p])
            dm_panels[p][idx] = (m_next[p][idx] - m_cur[p][idx]) * inv_two_steps
        end
    end
    return nothing
end

@inline function convert_cs_mass_target_to_delta!(m_target::NTuple{NP, <:AbstractArray{FT, 3}},
                                                  m_cur::NTuple{NP, <:AbstractArray{FT, 3}}) where {FT, NP}
    for p in 1:NP
        @inbounds for idx in eachindex(m_target[p])
            m_target[p][idx] -= m_cur[p][idx]
        end
    end
    return nothing
end

function verify_write_replay_cs!(m_cur::NTuple{NP, <:AbstractArray{FT, 3}},
                                 am::NTuple{NP, <:AbstractArray},
                                 bm::NTuple{NP, <:AbstractArray},
                                 cm::NTuple{NP, <:AbstractArray},
                                 m_next::NTuple{NP, <:AbstractArray},
                                 steps_per_window::Int,
                                 tol_rel::Real,
                                 win_idx::Int) where {FT, NP}
    diag = verify_window_continuity_cs(m_cur, am, bm, cm, m_next, steps_per_window)
    diag.max_rel_err <= tol_rel ||
        error("Write-time replay gate FAILED for CS window $(win_idx): " *
              "rel=$(diag.max_rel_err) > tol=$(tol_rel) at cell $(diag.worst_idx) " *
              "(abs=$(diag.max_abs_err) kg). Stored CS fluxes do not integrate to " *
              "the target mass endpoint under palindrome continuity.")
    return diag
end

"""
    process_day(date, grid, settings, vertical; next_day_hour0=nothing)

Fallback target-grid method that rejects unsupported geometries after the
configuration layer has already parsed them.
"""
function process_day(date::Date,
                     grid::AbstractTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    ensure_supported_target(grid)
    return nothing
end

"""
    process_day(date, grid::CubedSphereTargetGeometry, settings, vertical; ...)

Spectral→CS transport binary: spectral synthesis to an internal LL staging grid,
conservative regridding to CS panels, global Poisson balance, cm diagnosis,
and streaming binary write. No on-disk LL intermediate.
"""
function process_day(date::Date,
                     grid::CubedSphereTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    FT = settings.output_float_type
    Nc = grid.Nc
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target=settings.T_target)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)
    Nt = spec.n_times

    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    # --- Build the internal LL staging grid ---
    staging_grid = build_target_geometry(Val(:latlon),
        Dict{String,Any}("type" => "latlon",
                          "nlon" => grid.staging_nlon,
                          "nlat" => grid.staging_nlat), FT)
    Nx_stg = nlon(staging_grid)
    Ny_stg = nlat(staging_grid)
    @info @sprintf("  Staging grid: %d×%d LL → C%d CS (%d panels)",
                   Nx_stg, Ny_stg, Nc, CS_PANEL_COUNT)

    # --- Build conservative regridder (LL→CS, cached) ---
    t_reg = time()
    regridder = build_regridder(staging_grid.mesh, grid.mesh;
                                normalize=false,
                                cache_dir=grid.cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    @info @sprintf("  Regridder: %d×%d  nnz=%d (%.1fs)",
                   n_src, n_dst, length(regridder.intersections.nzval), time() - t_reg)

    # --- Allocate workspaces ---
    # LL staging workspaces (reuse existing LL infrastructure)
    transform = allocate_transform_workspace(staging_grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(staging_grid, Nz_native, Nz, FT)
    qv = allocate_qv_workspace(staging_grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)

    # CS workspaces
    cs_ws = allocate_cs_preprocess_workspace(Nc, Nx_stg, Ny_stg, Nz, n_src, n_dst, FT)

    # Vertical coordinate and CS geometry
    vc_merged = vertical.merged_vc
    A_ifc = Float64.(vc_merged.A)
    B_ifc = Float64.(vc_merged.B)
    gravity = FT(GRAV)
    dt_factor = FT(settings.met_interval / (2 * steps_per_met))
    Δx = grid.mesh.Δx  # (Nc, Nc) matrix
    Δy = grid.mesh.Δy  # (Nc, Nc) matrix

    # --- Open streaming CS binary writer ---
    writer = open_streaming_cs_transport_binary(
        bin_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=settings.met_interval,
        half_dt_seconds=settings.half_dt,
        steps_per_window=steps_per_met,
        include_flux_delta=true,
        mass_basis=Symbol(settings.mass_basis),
        extra_header=Dict{String, Any}(
            "preprocessor"     => "preprocess_transport_binary.jl",
            "source_type"      => "era5_spectral",
            "target_type"      => "cubed_sphere",
            "staging_nlon"     => Nx_stg,
            "staging_nlat"     => Ny_stg,
            "regrid_method"    => "conservative",
            "poisson_balanced" => true,
            "mass_fix_enabled" => settings.mass_fix_enable,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(FT)
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path),
                   expected_total / 1e9, Nt)

    log_mass_fix_configuration(settings)
    @info "  Streaming: spectral → LL staging → CS regrid → balance → write..."
    write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
    write_replay_on || @info "  Write-time CS replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    replay_tol = replay_tolerance(FT)

    # --- Helper: synthesize one window to staging LL, merge, then regrid to CS ---
    function _synth_and_regrid_to_cs!(win_idx, hour, m_out, ps_out, am_out, bm_out)
        # Spectral → staging LL (native levels)
        spectral_to_native_fields!(
            transform.m_arr, transform.am_arr, transform.bm_arr, transform.cm_arr, transform.sp,
            transform.u_cc, transform.v_cc, transform.u_stag, transform.v_stag, transform.dp,
            spec.lnsp_all[hour], spec.vo_by_hour[hour], spec.d_by_hour[hour],
            spec.T, vertical.level_range, vertical.ab, staging_grid, settings.half_dt,
            transform.P_buf, transform.fft_buf, transform.field_2d,
            transform.P_buf_t, transform.fft_buf_t, transform.fft_out_t,
            transform.u_spec_t, transform.v_spec_t, transform.field_2d_t,
            transform.bfft_plans)

        # Mass fix + merge vertical levels
        read_window_qv!(qv, win_idx, Nx_stg, Ny_stg, Nz_native)
        apply_mass_fix_if_needed!(qv, transform, staging_grid, vertical, settings, ps_offsets, win_idx)
        apply_dry_basis_if_needed!(settings.mass_basis, transform, qv)
        merge_native_window!(merged, transform, qv, vertical, settings)

        # Conservative regrid scalars: m, ps → CS panels
        regrid_3d_to_cs_panels!(m_out, regridder, merged.m_merged, cs_ws, Nc)
        regrid_2d_to_cs_panels!(ps_out, regridder, transform.sp, cs_ws, Nc)

        # Per-level mean correction on regridded m.
        # On a closed sphere, Σ div_h = 0 (topological). For the Poisson
        # system to be fully solvable, we need Σ(m_next - m_cur) = 0 at each
        # level. Conservative regridding preserves global mass but shifts the
        # per-level distribution by O(10⁻⁶) relative. We absorb this by
        # adjusting regridded m so each level's global sum matches the staging
        # LL grid's sum. The correction is applied HERE (not in a hidden
        # projection) so the stored m and the balance target are consistent.
        _enforce_perlevel_mass_consistency!(m_out, merged.m_merged, Nc, Nz)

        # Recover LL cell-center winds from merged fluxes
        stg_lats = staging_grid.lats
        Δy_ll = FT(staging_grid.mesh.radius * deg2rad(staging_grid.mesh.Δφ))
        Δlon_ll = FT(deg2rad(staging_grid.mesh.Δλ))

        recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
            merged.am_merged, merged.bm_merged, transform.sp,
            A_ifc, B_ifc, stg_lats, Δy_ll, Δlon_ll,
            FT(staging_grid.mesh.radius), gravity, dt_factor)

        # Regrid geographic winds to CS panels and rotate to panel-local.
        # ConservativeRegridding.regrid! already divides by dst_areas internally,
        # so the regridded winds are correctly area-averaged (intensive quantity).
        # Then rotate east/north → panel-local (x, y) using the gnomonic Jacobian.
        regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, regridder, cs_ws.v_cc, cs_ws, Nc)
        rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      Nc, Nz)

        # Reconstruct CS face fluxes from regridded winds
        reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                               cs_ws.dp_panels, ps_out,
                               A_ifc, B_ifc, Δx, Δy, gravity, dt_factor, Nc, Nz)
    end

    # --- Pre-allocate sliding buffer (no per-window allocation) ---
    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)

    @inline function _copy_panels!(dst, src)
        for p in 1:CS_PANEL_COUNT
            copyto!(dst[p], src[p])
        end
    end

    worst_pre = 0.0
    worst_post = 0.0
    worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)

    # --- Process first window ---
    t0 = time()
    _synth_and_regrid_to_cs!(1, spec.hours[1],
        cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
    @info @sprintf("    Window  1/%d (hour %02d): synth+regrid %.2fs  offset=%+.3f Pa",
                   Nt, spec.hours[1], time() - t0, ps_offsets[1])

    _copy_panels!(cur_m,  cs_ws.m_panels)
    _copy_panels!(cur_ps, cs_ws.ps_panels)
    _copy_panels!(cur_am, cs_ws.am_panels)
    _copy_panels!(cur_bm, cs_ws.bm_panels)

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t0 = time()
        _synth_and_regrid_to_cs!(win, spec.hours[win],
            cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
        t_synth = time() - t0

        # Copy m_next for balance (unmodified — the Poisson CG internally
        # projects the RHS to mean-zero, handling the topological constraint
        # Σ div_h = 0 on a closed sphere without modifying the stored m)
        _copy_panels!(cs_ws.m_next_panels, cs_ws.m_panels)

        # Balance the PREVIOUS window using (m_cur, m_next)
        t_bal = time()
        bal_diag = balance_cs_global_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            grid.face_table, grid.cell_degree, steps_per_met,
            grid.poisson_scratch; tol=1e-14, max_iter=20000)
        t_bal = time() - t_bal

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        # Sync ALL boundary mirrors (including de-duplicated faces) so that
        # per-panel flux divergence and the advection kernel telescope correctly.
        sync_all_cs_boundary_mirrors!(cur_am, cur_bm, grid.mesh.connectivity, Nc, Nz)

        # Diagnose cm from balanced am/bm and raw mass tendency.
        fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
        for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
        diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
        if write_replay_on
            diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                                  cs_ws.m_next_panels,
                                                  steps_per_met, replay_tol, win - 1)
            if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
                worst_replay_rel = diag_replay.max_rel_err
                worst_replay_abs = diag_replay.max_abs_err
                worst_replay_win = win - 1
                worst_replay_idx = diag_replay.worst_idx
            end
        end
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        # Write balanced previous window
        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

        should_log_window(win - 1, Nt) &&
            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | synth %2d (%.2fs)",
                           win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_synth)

        # Swap: copy just-synthesized panels into cur (no allocation)
        _copy_panels!(cur_m,  cs_ws.m_panels)
        _copy_panels!(cur_ps, cs_ws.ps_panels)
        _copy_panels!(cur_am, cs_ws.am_panels)
        _copy_panels!(cur_bm, cs_ws.bm_panels)
    end

    # --- Balance & write LAST window (next-day closure when available) ---
    last_hour_next = next_day_merged_fields(next_day_hour0, date, staging_grid, vertical,
                                            settings, transform, merged, qv, ps_offsets)
    if last_hour_next !== nothing
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, regridder, last_hour_next.m, cs_ws, Nc)
        _enforce_perlevel_mass_consistency!(cs_ws.m_next_panels, last_hour_next.m, Nc, Nz)
    else
        _copy_panels!(cs_ws.m_next_panels, cur_m)
    end
    t_bal = time()
    bal_diag = balance_cs_global_mass_fluxes!(
        cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
        grid.face_table, grid.cell_degree, steps_per_met,
        grid.poisson_scratch; tol=1e-14, max_iter=5000)
    t_bal = time() - t_bal

    worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
    worst_post = max(worst_post, bal_diag.max_post_residual)
    worst_iter = max(worst_iter, bal_diag.max_cg_iter)

    # Sync ALL boundary mirrors (including de-duplicated faces) — same as main loop.
    sync_all_cs_boundary_mirrors!(cur_am, cur_bm, grid.mesh.connectivity, Nc, Nz)

    fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
    for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
    diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
    if write_replay_on
        diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                              cs_ws.m_next_panels,
                                              steps_per_met, replay_tol, Nt)
        if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
            worst_replay_rel = diag_replay.max_rel_err
            worst_replay_abs = diag_replay.max_abs_err
            worst_replay_win = Nt
            worst_replay_idx = diag_replay.worst_idx
        end
    end
    convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

    window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                 dm=cs_ws.m_next_panels)
    write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre=%.2e post=%.2e iter=%d",
                   Nt, Nt, t_bal, bal_diag.max_pre_residual,
                   bal_diag.max_post_residual, bal_diag.max_cg_iter)

    # --- Finalize ---
    close_streaming_transport_binary!(writer)

    if settings.mass_fix_enable
        ps_offsets_day = @view ps_offsets[1:Nt]
        @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                       minimum(ps_offsets_day), maximum(ps_offsets_day), sum(ps_offsets_day) / Nt)
    end

    @info @sprintf("  Poisson balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                   worst_pre, worst_post, worst_iter)
    if write_replay_on
        replay_msg = worst_replay_win > 0 ?
            @sprintf("max rel=%.3e abs=%.3e kg win=%d cell=%s",
                     worst_replay_rel, worst_replay_abs, worst_replay_win, worst_replay_idx) :
            "no windows checked"
        @info "  Write-time replay gate: $replay_msg"
    end

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path),
                   actual / 1e9, time() - t_day)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return bin_path
end

"""
    regrid_ll_binary_to_cs(ll_binary_path, cs_grid, out_path; FT=Float64, mass_basis=nothing)

Regrid an existing LL transport binary to a cubed-sphere binary.

Reads each window from the LL binary, recovers cell-center winds from am/bm,
conservatively regrids m/ps/winds to CS panels, rotates winds to panel-local
coordinates, reconstructs CS face fluxes, applies global Poisson balance,
diagnoses cm from continuity, and stream-writes the CS binary.

This reuses the entire CS regrid/balance/write infrastructure from the
spectral→CS path — the only difference is the data source (binary reader
instead of spectral synthesis).

Timestep metadata (`dt_met_seconds`, `steps_per_window`) is read directly
from the source header. The output CS binary carries the same values, so
the per-substep flux semantics survive the regrid without rescaling.

## Keyword arguments
- `FT::Type = Float64` — on-disk float type for the output CS binary.
- `mass_basis::Union{Nothing, Symbol} = nothing` — output mass-basis label.
  `nothing` (default) = match the source. Setting this to a value that
  differs from the source's `mass_basis` currently errors: actual
  dry↔moist conversion requires loading the source's `qv` and applying
  `apply_dry_basis_native!`, which this function does not do. Invariant
  14 mandates `:dry` end-to-end; use a dry source.
- `allow_terminal_zero_tendency::Bool = false` — diagnostic-only escape hatch
  for legacy LL sources that do not carry `dm`. Production-safe regrids should
  leave this at `false` so the final CS window is closed against an explicit
  endpoint target instead of an inferred zero-tendency fallback.
"""
function regrid_ll_binary_to_cs(ll_binary_path::String,
                                cs_grid::CubedSphereTargetGeometry,
                                out_path::String;
                                FT::Type{<:AbstractFloat} = Float64,
                                mass_basis::Union{Nothing, Symbol} = nothing,
                                allow_terminal_zero_tendency::Bool = false)
    t_start = time()
    Nc = cs_grid.Nc

    # --- Open LL binary reader ---
    reader = TransportBinaryReader(ll_binary_path; FT=FT)
    h = reader.header
    Nx_ll = h.Nx
    Ny_ll = h.Ny
    Nz = h.nlevel
    Nt = h.nwindow
    A_ifc = Float64.(h.A_ifc)
    B_ifc = Float64.(h.B_ifc)

    # Refuse silent basis relabeling. The function reads raw `m/am/bm/ps`
    # from the source and never touches `qv`, so mismatched basis produces
    # a mislabeled binary (invariant 14 violation). Matching or unset is OK.
    source_basis = Symbol(h.mass_basis)
    output_basis = mass_basis === nothing ? source_basis : Symbol(mass_basis)
    output_basis === source_basis || throw(ArgumentError(
        "regrid_ll_binary_to_cs: requested mass_basis=$(output_basis) " *
        "differs from source header mass_basis=$(source_basis). This " *
        "function does not perform dry↔moist conversion (it would need " *
        "to load `qv` from the source and apply `apply_dry_basis_native!` " *
        "to m/am/bm). Regenerate the source on the desired basis, or omit " *
        "the `mass_basis` kwarg to match the source."))
    source_has_dm = :dm in h.payload_sections
    source_has_dm || allow_terminal_zero_tendency || throw(ArgumentError(
        "regrid_ll_binary_to_cs requires source `dm` payloads to close the final " *
        "CS window safely. Source $(basename(ll_binary_path)) lacks `dm`; " *
        "regenerate the LL binary with flux deltas or pass " *
        "`allow_terminal_zero_tendency=true` for diagnostic-only regrids."
    ))
    source_has_dm && delta_semantics(reader) === :forward_window_endpoint_difference ||
        !source_has_dm || throw(ArgumentError(
            "regrid_ll_binary_to_cs requires source delta_semantics = " *
            ":forward_window_endpoint_difference when `dm` is present, got " *
            "$(delta_semantics(reader))."
        ))

    # Timestep metadata comes from the source header. The stored `am/bm`
    # are per-substep mass (flux_kind = :substep_mass_amount) with the
    # source's own `steps_per_window`, and the CS writer reuses the same
    # substep count — so the per-substep semantics match end-to-end.
    met_interval = Float64(h.dt_met_seconds)
    steps_per_met = Int(h.steps_per_window)
    dt_factor = FT(met_interval / (2 * steps_per_met))
    gravity = FT(GRAV)

    @info @sprintf("  LL source: %s (%d×%d×%d, %d windows)",
                   basename(ll_binary_path), Nx_ll, Ny_ll, Nz, Nt)
    @info @sprintf("  CS target: C%d (%d panels, %d levels)", Nc, CS_PANEL_COUNT, Nz)

    # --- Build LL source mesh for regridder ---
    # Reconstruct the LL mesh from the binary header metadata
    ll_mesh = LatLonMesh(; FT=FT,
                          size=(Nx_ll, Ny_ll),
                          longitude=(-180, 180),
                          latitude=(-90, 90),
                          radius=FT(R_EARTH))
    ll_lats = FT.(ll_mesh.φᶜ)
    Δy_ll = FT(ll_mesh.radius * deg2rad(ll_mesh.Δφ))
    Δlon_ll = FT(deg2rad(ll_mesh.Δλ))

    # --- Build conservative regridder (LL → CS) ---
    t_reg = time()
    regridder = build_regridder(ll_mesh, cs_grid.mesh;
                                normalize=false,
                                cache_dir=cs_grid.cache_dir)
    n_src = length(regridder.src_areas)
    n_dst = length(regridder.dst_areas)
    @info @sprintf("  Regridder: %d→%d  nnz=%d (%.1fs)",
                   n_src, n_dst, length(regridder.intersections.nzval), time() - t_reg)

    # --- Allocate workspaces ---
    cs_ws = allocate_cs_preprocess_workspace(Nc, Nx_ll, Ny_ll, Nz, n_src, n_dst, FT)
    Δx = cs_grid.mesh.Δx
    Δy = cs_grid.mesh.Δy

    # Pre-allocate LL read buffers
    m_ll  = Array{FT}(undef, Nx_ll, Ny_ll, Nz)
    am_ll = Array{FT}(undef, Nx_ll + 1, Ny_ll, Nz)
    bm_ll = Array{FT}(undef, Nx_ll, Ny_ll + 1, Nz)
    cm_ll = Array{FT}(undef, Nx_ll, Ny_ll, Nz + 1)
    ps_ll = Array{FT}(undef, Nx_ll, Ny_ll)
    dm_ll = source_has_dm ? Array{FT}(undef, Nx_ll, Ny_ll, Nz) : nothing

    # --- Build vertical coordinate from binary header ---
    vc_merged = HybridSigmaPressure(A_ifc, B_ifc)

    # --- Open streaming CS binary writer ---
    mkpath(dirname(out_path))
    writer = open_streaming_cs_transport_binary(
        out_path, Nc, CS_PANEL_COUNT, Nz, Nt, vc_merged;
        FT=FT,
        dt_met_seconds=met_interval,
        half_dt_seconds=met_interval / 2,
        steps_per_window=steps_per_met,
        include_flux_delta=true,
        mass_basis=output_basis,
        extra_header=Dict{String, Any}(
            "preprocessor"      => "regrid_ll_binary_to_cs",
            "source_type"       => "ll_transport_binary",
            "source_path"       => ll_binary_path,
            "target_type"       => "cubed_sphere",
            "regrid_method"     => "conservative",
            "poisson_balanced"  => true,
        ))

    bytes_per_window = writer.elems_per_window * sizeof(FT)
    expected_total = writer.header_bytes + Nt * bytes_per_window
    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(out_path),
                   expected_total / 1e9, Nt)
    @info "  Streaming: LL binary → CS regrid → balance → write..."
    write_replay_on = get(ENV, "ATMOSTR_NO_WRITE_REPLAY_CHECK", "0") != "1"
    write_replay_on || @info "  Write-time CS replay gate SKIPPED (ATMOSTR_NO_WRITE_REPLAY_CHECK=1)"
    replay_tol = replay_tolerance(FT)

    # --- Helper: read one LL window and regrid to CS ---
    function _read_and_regrid_to_cs!(win_idx, m_out, ps_out, am_out, bm_out)
        load_window!(reader, win_idx; m=m_ll, ps=ps_ll, am=am_ll, bm=bm_ll, cm=cm_ll)

        # Conservative regrid scalars: m, ps → CS panels
        regrid_3d_to_cs_panels!(m_out, regridder, m_ll, cs_ws, Nc)
        regrid_2d_to_cs_panels!(ps_out, regridder, ps_ll, cs_ws, Nc)
        _enforce_perlevel_mass_consistency!(m_out, m_ll, Nc, Nz)

        # Recover LL cell-center winds from binary's am/bm
        recover_ll_cell_center_winds!(cs_ws.u_cc, cs_ws.v_cc,
            am_ll, bm_ll, ps_ll,
            A_ifc, B_ifc, ll_lats, Δy_ll, Δlon_ll,
            FT(ll_mesh.radius), gravity, dt_factor)

        # Regrid geographic winds to CS + rotate to panel-local
        regrid_3d_to_cs_panels!(cs_ws.u_cs_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        regrid_3d_to_cs_panels!(cs_ws.v_cs_panels, regridder, cs_ws.v_cc, cs_ws, Nc)
        rotate_winds_to_panel_local!(cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                                      Nc, Nz)

        # Reconstruct CS face fluxes
        reconstruct_cs_fluxes!(am_out, bm_out, cs_ws.u_cs_panels, cs_ws.v_cs_panels,
                               cs_ws.dp_panels, ps_out,
                               A_ifc, B_ifc, Δx, Δy, gravity, dt_factor, Nc, Nz)
    end

    # --- Pre-allocate sliding buffer ---
    cur_m  = ntuple(_ -> zeros(FT, Nc, Nc, Nz), CS_PANEL_COUNT)
    cur_ps = ntuple(_ -> zeros(FT, Nc, Nc), CS_PANEL_COUNT)
    cur_am = ntuple(_ -> zeros(FT, Nc + 1, Nc, Nz), CS_PANEL_COUNT)
    cur_bm = ntuple(_ -> zeros(FT, Nc, Nc + 1, Nz), CS_PANEL_COUNT)
    cur_cm = ntuple(_ -> zeros(FT, Nc, Nc, Nz + 1), CS_PANEL_COUNT)

    @inline function _copy_panels_regrid!(dst, src)
        for p in 1:CS_PANEL_COUNT; copyto!(dst[p], src[p]); end
    end

    worst_pre = 0.0; worst_post = 0.0; worst_iter = 0
    worst_replay_rel = 0.0
    worst_replay_abs = 0.0
    worst_replay_win = 0
    worst_replay_idx = (0, 0, 0, 0)

    # --- Process first window ---
    t0 = time()
    _read_and_regrid_to_cs!(1, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
    @info @sprintf("    Window  1/%d: read+regrid %.2fs", Nt, time() - t0)

    _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
    _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
    _copy_panels_regrid!(cur_am, cs_ws.am_panels)
    _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)

    # --- Sliding-window loop: windows 2..Nt ---
    for win in 2:Nt
        t0 = time()
        _read_and_regrid_to_cs!(win, cs_ws.m_panels, cs_ws.ps_panels, cs_ws.am_panels, cs_ws.bm_panels)
        t_read = time() - t0

        _copy_panels_regrid!(cs_ws.m_next_panels, cs_ws.m_panels)

        t_bal = time()
        bal_diag = balance_cs_global_mass_fluxes!(
            cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
            cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
            cs_grid.poisson_scratch; tol=1e-14, max_iter=20000)
        t_bal = time() - t_bal

        worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
        worst_post = max(worst_post, bal_diag.max_post_residual)
        worst_iter = max(worst_iter, bal_diag.max_cg_iter)

        sync_all_cs_boundary_mirrors!(cur_am, cur_bm, cs_grid.mesh.connectivity, Nc, Nz)

        fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
        for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
        diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
        if write_replay_on
            diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                                  cs_ws.m_next_panels,
                                                  steps_per_met, replay_tol, win - 1)
            if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
                worst_replay_rel = diag_replay.max_rel_err
                worst_replay_abs = diag_replay.max_abs_err
                worst_replay_win = win - 1
                worst_replay_idx = diag_replay.worst_idx
            end
        end
        convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

        window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                     dm=cs_ws.m_next_panels)
        write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

        should_log_window(win - 1, Nt) &&
            @info @sprintf("    Window %2d/%d: wrote (bal %.2fs pre=%.2e post=%.2e iter=%d) | read %2d (%.2fs)",
                           win - 1, Nt, t_bal, bal_diag.max_pre_residual,
                           bal_diag.max_post_residual, bal_diag.max_cg_iter, win, t_read)

        _copy_panels_regrid!(cur_m,  cs_ws.m_panels)
        _copy_panels_regrid!(cur_ps, cs_ws.ps_panels)
        _copy_panels_regrid!(cur_am, cs_ws.am_panels)
        _copy_panels_regrid!(cur_bm, cs_ws.bm_panels)
    end

    # --- Balance & write LAST window ---
    if source_has_dm
        deltas = load_flux_delta_window!(reader, Nt; dm=dm_ll)
        (deltas !== nothing && haskey(deltas, :dm)) || throw(ArgumentError(
            "regrid_ll_binary_to_cs: source header for $(basename(ll_binary_path)) " *
            "declares `dm`, but window $(Nt) could not be loaded."
        ))
        @. cs_ws.u_cc = m_ll + dm_ll
        regrid_3d_to_cs_panels!(cs_ws.m_next_panels, regridder, cs_ws.u_cc, cs_ws, Nc)
        _enforce_perlevel_mass_consistency!(cs_ws.m_next_panels, cs_ws.u_cc, Nc, Nz)
    else
        @warn "regrid_ll_binary_to_cs: using zero-tendency fallback for the final CS window because source `dm` is unavailable."
        _copy_panels_regrid!(cs_ws.m_next_panels, cur_m)
    end
    t_bal = time()
    bal_diag = balance_cs_global_mass_fluxes!(
        cur_am, cur_bm, cur_m, cs_ws.m_next_panels,
        cs_grid.face_table, cs_grid.cell_degree, steps_per_met,
        cs_grid.poisson_scratch; tol=1e-14, max_iter=5000)
    t_bal = time() - t_bal

    worst_pre  = max(worst_pre,  bal_diag.max_pre_residual)
    worst_post = max(worst_post, bal_diag.max_post_residual)
    worst_iter = max(worst_iter, bal_diag.max_cg_iter)

    sync_all_cs_boundary_mirrors!(cur_am, cur_bm, cs_grid.mesh.connectivity, Nc, Nz)

    fill_cs_window_mass_tendency!(cs_ws.dm_panels, cur_m, cs_ws.m_next_panels, steps_per_met)
    for p in 1:CS_PANEL_COUNT; fill!(cur_cm[p], zero(FT)); end
    diagnose_cs_cm!(cur_cm, cur_am, cur_bm, cs_ws.dm_panels, cur_m, Nc, Nz)
    if write_replay_on
        diag_replay = verify_write_replay_cs!(cur_m, cur_am, cur_bm, cur_cm,
                                              cs_ws.m_next_panels,
                                              steps_per_met, replay_tol, Nt)
        if worst_replay_win == 0 || diag_replay.max_rel_err > worst_replay_rel
            worst_replay_rel = diag_replay.max_rel_err
            worst_replay_abs = diag_replay.max_abs_err
            worst_replay_win = Nt
            worst_replay_idx = diag_replay.worst_idx
        end
    end
    convert_cs_mass_target_to_delta!(cs_ws.m_next_panels, cur_m)

    window_nt = (m=cur_m, am=cur_am, bm=cur_bm, cm=cur_cm, ps=cur_ps,
                 dm=cs_ws.m_next_panels)
    write_streaming_cs_window!(writer, window_nt, Nc, CS_PANEL_COUNT)

    @info @sprintf("    Window %2d/%d (last): bal %.2fs  pre=%.2e post=%.2e iter=%d",
                   Nt, Nt, t_bal, bal_diag.max_pre_residual,
                   bal_diag.max_post_residual, bal_diag.max_cg_iter)

    close_streaming_transport_binary!(writer)
    close(reader)

    @info @sprintf("  Poisson balance summary: pre=%.3e  post=%.3e  max_iter=%d",
                   worst_pre, worst_post, worst_iter)
    if write_replay_on
        replay_msg = worst_replay_win > 0 ?
            @sprintf("max rel=%.3e abs=%.3e kg win=%d cell=%s",
                     worst_replay_rel, worst_replay_abs, worst_replay_win, worst_replay_idx) :
            "no windows checked"
        @info "  Write-time replay gate: $replay_msg"
    end

    actual = filesize(out_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(out_path),
                   actual / 1e9, time() - t_start)
    actual == expected_total ||
        @warn @sprintf("File size mismatch: expected %d bytes, got %d", expected_total, actual)

    return out_path
end

"""
    process_day(date, grid::LatLonTargetGeometry, settings, vertical; next_day_hour0=nothing)

Run the full one-day preprocessing workflow for the structured lat-lon target:
read spectral input, process all windows, apply Poisson balance, and write the
final binary.
"""
function process_day(date::Date,
                     grid::LatLonTargetGeometry,
                     settings,
                     vertical;
                     next_day_hour0=nothing)
    FT = settings.output_float_type
    Nz_native = vertical.Nz_native
    Nz = vertical.Nz
    Nx = nlon(grid)
    Ny = nlat(grid)
    steps_per_met = exact_steps_per_window(settings.met_interval, settings.dt)
    date_str = Dates.format(date, "yyyymmdd")

    vo_d_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_vo_d.gb")
    lnsp_path = joinpath(settings.spectral_dir, "era5_spectral_$(date_str)_lnsp.gb")

    if !isfile(vo_d_path) || !isfile(lnsp_path)
        @warn "Missing GRIB files for $date_str, skipping"
        return nothing
    end

    t_day = time()
    @info "  Reading spectral data for $date_str..."
    spec = read_day_spectral_streaming(vo_d_path, lnsp_path; T_target=settings.T_target)
    @info @sprintf("  Spectral data read: T=%d, %d hours (%.1fs)",
                   spec.T, spec.n_times, time() - t_day)

    Nt = spec.n_times
    counts = window_element_counts(grid, Nz;
                                    include_qv=settings.include_qv,
                                    tm5_convection=settings.tm5_convection_enable)
    byte_sizes = window_byte_sizes(counts, FT, Nt)
    counts = merge(counts, (bytes_per_window = byte_sizes.bytes_per_window,))

    mkpath(settings.out_dir)
    bin_path = output_binary_path(date, settings.out_dir, settings.min_dp, FT)

    if existing_complete_output(bin_path, byte_sizes.total_bytes)
        @info "  SKIP (exists, correct size): $(basename(bin_path))"
        return bin_path
    end

    @info @sprintf("  Output: %s (%.2f GB, %d windows)", basename(bin_path), byte_sizes.total_bytes / 1e9, Nt)

    provenance = script_provenance()
    sizes = (Nx = Nx, Ny = Ny, Nz = Nz, Nz_native = Nz_native, Nt = Nt,
             steps_per_met = steps_per_met)
    header = build_v4_header(date, grid, vertical, settings, FT, counts, sizes, provenance)
    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header JSON too large: $(length(header_json)) >= $(HEADER_SIZE)")

    transform = allocate_transform_workspace(grid, spec.T, Nz_native)
    merged = allocate_merge_workspace(grid, Nz_native, Nz, FT)
    storage = allocate_window_storage(Nt, FT;
                                       include_qv=settings.include_qv,
                                       tm5_convection=settings.tm5_convection_enable)
    qv = allocate_qv_workspace(grid, settings, date, Nz_native, Nz, FT)
    ps_offsets = zeros(Float64, Nt + 1)

    # Plan 24 Commit 4: TM5 convection setup (LL target == ERA5 native
    # 720×361 only — see NOTES.md for the scope narrowing).  When
    # enabled, open the day's physics BIN, shape-check against the
    # target, and allocate the per-day workspace + cleanup stats.
    physics_reader = nothing
    tm5_ws         = nothing
    tm5_stats      = nothing
    if settings.tm5_convection_enable
        physics_reader = open_era5_physics_binary(settings.tm5_physics_bin_dir, date)
        Nlon_src = physics_reader.header.Nlon
        Nlat_src = physics_reader.header.Nlat
        (Nlon_src == Nx && Nlat_src == Ny) || error(
            "Plan 24 Commit 4 requires LL target == physics BIN shape. " *
            "BIN is ($Nlon_src, $Nlat_src), target is ($Nx, $Ny). " *
            "Either (a) use a 720×361 LL target config, or (b) wait for " *
            "Commit 4b/4c (regrid + PS sourcing for coarser / non-LL targets).")
        tm5_ws    = allocate_tm5_workspace(Nlon_src, Nlat_src, Nz_native, Nz, FT)
        tm5_stats = TM5CleanupStats()
    end

    log_mass_fix_configuration(settings)
    @info "  Computing spectral -> gridpoint -> merged for $Nt windows..."

    try
        for (win_idx, hour) in enumerate(spec.hours)
            process_window!(win_idx, hour, spec, grid, vertical, settings,
                            transform, merged, qv, storage, ps_offsets;
                            physics_reader = physics_reader,
                            tm5_ws         = tm5_ws,
                            tm5_stats      = tm5_stats)
        end

        if settings.mass_fix_enable
            @info @sprintf("  Mass-fix offsets (Pa) min/max/mean: %+.3f / %+.3f / %+.3f",
                           minimum(ps_offsets[1:Nt]),
                           maximum(ps_offsets[1:Nt]),
                           sum(ps_offsets[1:Nt]) / Nt)
        end

        tm5_stats === nothing || log_tm5_cleanup_stats(tm5_stats, date_str)

        last_hour_next = next_day_merged_fields(next_day_hour0, date, grid, vertical,
                                                settings, transform, merged, qv, ps_offsets)

        apply_poisson_balance!(storage, last_hour_next, sizes.steps_per_met)
        fill_qv_endpoints!(storage, last_hour_next)

        header["ps_offsets_pa_per_window"] = ps_offsets[1:Nt]
        header["ps_offsets_next_day_hour0_pa"] = ps_offsets[Nt + 1]
        header_json = JSON3.write(header)
        length(header_json) < HEADER_SIZE ||
            error("Header JSON too large after offsets update: $(length(header_json)) >= $(HEADER_SIZE)")

        write_day_binary!(bin_path, header_json, storage, settings, merged, last_hour_next)
    finally
        physics_reader === nothing || close_era5_physics_binary(physics_reader)
    end

    actual = filesize(bin_path)
    @info @sprintf("  Done: %s (%.2f GB, %.1fs)", basename(bin_path), actual / 1e9, time() - t_day)
    actual == byte_sizes.total_bytes ||
        error(@sprintf("SIZE MISMATCH: expected %d bytes, got %d", byte_sizes.total_bytes, actual))

    last_merged = (m = storage.all_m[Nt], am = storage.all_am[Nt], bm = storage.all_bm[Nt])
    return bin_path, last_merged
end

# =========================================================================
# TOML-driven entry point — called by the CLI script
# =========================================================================

"""
    process_day(cfg::Dict; day_override=nothing)

Top-level TOML-driven entry point for the unified preprocessor CLI.
Parses the config, constructs target geometry and vertical setup,
then dispatches to the grid-specific `process_day(date, grid, ...)` method.

Called by `scripts/preprocessing/preprocess_transport_binary.jl`.
"""
function process_day(cfg::Dict{String, Any};
                     day_override::Union{String, Nothing}=nothing)
    grid = build_target_geometry(cfg["grid"], Float64)
    settings = resolve_runtime_settings(cfg)
    settings = merge(settings, (T_target = target_spectral_truncation(grid),))
    vertical = build_vertical_setup(settings.coeff_path, settings.level_range, settings.min_dp, cfg["grid"])

    log_preprocessor_configuration(settings, grid, vertical)
    ensure_supported_target(grid)

    # Date selection
    dates = if day_override !== nothing
        [Date(day_override)]
    else
        day_filter = parse_day_filter(day_override === nothing ? String[] : ["--day", day_override])
        select_processing_dates(available_spectral_dates(settings.spectral_dir), day_filter)
    end

    @info @sprintf("Processing %d days: %s to %s", length(dates), first(dates), last(dates))
    t_total = time()

    for (idx, date) in enumerate(dates)
        @info @sprintf("[%d/%d] %s", idx, length(dates), date)
        next_day_h0 = next_day_hour0(date, dates, settings.spectral_dir, settings.T_target)
        next_day_h0 !== nothing && @info("  Next day hour 0 available for last-window delta")
        process_day(date, grid, settings, vertical; next_day_hour0=next_day_h0)
    end

    elapsed = time() - t_total
    @info @sprintf("All done! %d days in %.1fs (%.1fs/day)", length(dates), elapsed, elapsed / length(dates))
end
