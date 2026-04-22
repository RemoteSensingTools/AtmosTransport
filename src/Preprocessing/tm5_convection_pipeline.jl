# ---------------------------------------------------------------------------
# Plan 24 Commit 4 — TM5 convection pipeline wiring.
#
# Bridges Commits 1–3 (column math, physics BIN reader, grid-level loop
# + native→merged remap) to the `process_day` preprocessor.  Shared
# across LL / RG / CS targets.
#
# Per-hour flow on the ERA5-native horizontal grid (Nlon_src, Nlat_src,
# typically 720×361 matching the physics BIN):
#
#   native-L137 UDMF/DDMF/UDRF/DDRF + T + Q + PS (from physics BIN)
#       ↓  tm5_native_fields_for_hour!  (Commit 3)
#   native-L137 entu/detu/entd/detd (kg/m²/s)
#       ↓  merge_tm5_field_3d!  (Commit 3)
#   merged-Nz entu/detu/entd/detd on the source (ERA5) grid
#       ↓  ConservativeRegridding (per-topology) — NOT bilinear, see
#          feedback_conservative_regrid_for_mass_fluxes.md
#   merged-Nz entu/detu/entd/detd on the target grid (LL / RG / CS)
#
# The regrid step is topology-specific and happens inline in each
# process_day variant; the PIPELINE helper here only produces the
# source-grid merged fields.
# ---------------------------------------------------------------------------

"""
    TM5PreprocessingWorkspace{FT, R}

Per-day scratch for the TM5 convection preprocessor.  Holds the
per-column vectors reused across all columns of an hour, plus the
4×(Nlon_src, Nlat_src, Nz_native) native-vertical buffers and the
4×(Nlon_src, Nlat_src, Nz) merged-vertical buffers on the ERA5
source grid.  `regridder` is the conservative regridder from the
ERA5 LL source mesh to the target mesh, or `nothing` when source
and target are the same LL grid (identity fast-path).
"""
struct TM5PreprocessingWorkspace{FT, R}
    col_scratch     :: NTuple{11, Vector{FT}}
    entu_native     :: Array{FT, 3}
    detu_native     :: Array{FT, 3}
    entd_native     :: Array{FT, 3}
    detd_native     :: Array{FT, 3}
    entu_merged_src :: Array{FT, 3}
    detu_merged_src :: Array{FT, 3}
    entd_merged_src :: Array{FT, 3}
    detd_merged_src :: Array{FT, 3}
    regridder       :: R
end

"""
    allocate_tm5_workspace(Nlon_src, Nlat_src, Nz_native, Nz, FT;
                           regridder=nothing) -> TM5PreprocessingWorkspace

Allocate the TM5 preprocessing workspace.  Pass `regridder=nothing`
for identity (source and target grids match).  Otherwise pass a
`ConservativeRegridding.Regridder` built via
`build_regridder(source_mesh, target_mesh)` from
`src/Regridding/`.
"""
function allocate_tm5_workspace(Nlon_src::Integer, Nlat_src::Integer,
                                Nz_native::Integer, Nz::Integer,
                                ::Type{FT};
                                regridder = nothing) where {FT <: AbstractFloat}
    col_scratch = (
        Vector{FT}(undef, Nz_native + 1),    # udmf_col
        Vector{FT}(undef, Nz_native + 1),    # ddmf_col
        Vector{FT}(undef, Nz_native),        # udrf_col
        Vector{FT}(undef, Nz_native),        # ddrf_col
        Vector{FT}(undef, Nz_native),        # t_col
        Vector{FT}(undef, Nz_native),        # q_col
        Vector{FT}(undef, Nz_native),        # dz_col
        Vector{FT}(undef, Nz_native),        # entu_col
        Vector{FT}(undef, Nz_native),        # detu_col
        Vector{FT}(undef, Nz_native),        # entd_col
        Vector{FT}(undef, Nz_native),        # detd_col
    )
    return TM5PreprocessingWorkspace{FT, typeof(regridder)}(
        col_scratch,
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz_native),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        Array{FT, 3}(undef, Nlon_src, Nlat_src, Nz),
        regridder,
    )
end

"""
    compute_tm5_merged_hour_on_source!(ws, reader, hour, ps_hour, ak_full, bk_full,
                                         Nz_native, merge_map; stats)

Phase 1 of the per-hour TM5 step.  Reads native-L137 ERA5 physics
fields for `hour` from the mmap-backed `reader`, runs the TM5 math
column-by-column into `ws.*_native`, then collapses to merged Nz
into `ws.*_merged_src`.  All work is on the ERA5-native horizontal
grid (720×361 for standard physics BINs).

Commit 4 narrowed scope: source and target grids MUST match
(Nx, Ny) == (720, 361).  PS comes from the caller (typically the
preprocessor's `transform.sp` after spectral synthesis) because
the Commit-2 physics BIN does not carry PS.  See
`docs/plans/24_TM5_PREPROCESSOR/NOTES.md` §"Deviations from plan
doc §4.4" for why RG/CS are follow-on commits.

`stats::Union{Nothing, NamedTuple}` is the TM5CleanupStats bundle;
when non-nothing, counters accumulate across all columns of all
hours processed.

Shape guards:

- `reader` fields shape `(Nlon_src, Nlat_src, Nz_native, Nt)` —
  must match `ws.entu_native`'s leading dims.
- `ps_hour` shape `(Nlon_src, Nlat_src)`.
- `merge_map` length `Nz_native`.
"""
function compute_tm5_merged_hour_on_source!(
        ws::TM5PreprocessingWorkspace{FT},
        reader,
        hour::Integer,
        ps_hour::AbstractArray{<:Real, 2},
        ak_full::AbstractVector,
        bk_full::AbstractVector,
        Nz_native::Integer,
        merge_map::AbstractVector{<:Integer};
        stats = nothing) where {FT <: AbstractFloat}

    udmf_4d = get_era5_physics_field(reader, :udmf)
    ddmf_4d = get_era5_physics_field(reader, :ddmf)
    udrf_4d = get_era5_physics_field(reader, :udrf_rate)
    ddrf_4d = get_era5_physics_field(reader, :ddrf_rate)
    t_4d    = get_era5_physics_field(reader, :t)
    q_4d    = get_era5_physics_field(reader, :q)

    # Slice to 3D per-hour view (zero-copy — Julia reshape-view).
    udmf_h = @view udmf_4d[:, :, :, hour]
    ddmf_h = @view ddmf_4d[:, :, :, hour]
    udrf_h = @view udrf_4d[:, :, :, hour]
    ddrf_h = @view ddrf_4d[:, :, :, hour]
    t_h    = @view t_4d[:, :, :, hour]
    q_h    = @view q_4d[:, :, :, hour]

    size(udmf_h, 1) == size(ws.entu_native, 1) || error(
        "Physics BIN Nlon=$(size(udmf_h,1)) ≠ workspace Nlon=$(size(ws.entu_native,1))")
    size(udmf_h, 2) == size(ws.entu_native, 2) || error(
        "Physics BIN Nlat=$(size(udmf_h,2)) ≠ workspace Nlat=$(size(ws.entu_native,2))")
    size(ps_hour)   == (size(udmf_h, 1), size(udmf_h, 2)) || error(
        "ps_hour shape $(size(ps_hour)) ≠ expected $(size(udmf_h,1)), $(size(udmf_h,2)))")

    FT_reader = eltype(udmf_h)
    FT_reader === FT || error(
        "TM5 workspace FT=$FT does not match physics BIN eltype=$(FT_reader); " *
        "match FT to Float32 (current production) or add conversion at call site")

    # The Commit-3 grid helper expects FT-typed ps_hour; accept any
    # real and convert here so callers can pass Float64 transform.sp.
    ps_h_ft = eltype(ps_hour) === FT ? ps_hour : FT.(ps_hour)

    tm5_native_fields_for_hour!(
        ws.entu_native, ws.detu_native, ws.entd_native, ws.detd_native,
        udmf_h, ddmf_h, udrf_h, ddrf_h, t_h, q_h, ps_h_ft,
        ak_full, bk_full, Nz_native;
        stats = stats, scratch = ws.col_scratch,
    )

    merge_tm5_field_3d!(ws.entu_merged_src, ws.entu_native, merge_map)
    merge_tm5_field_3d!(ws.detu_merged_src, ws.detu_native, merge_map)
    merge_tm5_field_3d!(ws.entd_merged_src, ws.entd_native, merge_map)
    merge_tm5_field_3d!(ws.detd_merged_src, ws.detd_native, merge_map)
    return nothing
end

"""
    log_tm5_cleanup_stats(stats, date_str)

Pretty-print the per-day TM5 cleanup counters produced by
`TM5CleanupStats()`.  Zero-valued counters are omitted to keep the
output compact; a fully-clean day yields a single line.
"""
function log_tm5_cleanup_stats(stats, date_str::AbstractString)
    n_cols = stats.columns_processed[]
    n_cols == 0 && return nothing

    nonzero = NamedTuple{
        (:no_updraft, :no_downdraft,
         :levels_udmf_clipped, :levels_ddmf_clipped,
         :levels_udrf_clipped, :levels_ddrf_clipped,
         :levels_entu_neg, :levels_detu_neg,
         :levels_entd_neg, :levels_detd_neg)
    }((
        stats.no_updraft[], stats.no_downdraft[],
        stats.levels_udmf_clipped[], stats.levels_ddmf_clipped[],
        stats.levels_udrf_clipped[], stats.levels_ddrf_clipped[],
        stats.levels_entu_neg[], stats.levels_detu_neg[],
        stats.levels_entd_neg[], stats.levels_detd_neg[],
    ))

    any_nonzero = any(v -> v > 0, values(nonzero))
    if !any_nonzero
        @info "  TM5 convection $date_str: $n_cols columns, all clean"
        return nothing
    end

    summary_parts = String["$n_cols cols"]
    for (k, v) in pairs(nonzero)
        v > 0 && push!(summary_parts, "$k=$v")
    end
    @info "  TM5 convection $date_str: $(join(summary_parts, ", "))"
    return nothing
end

"""
    tm5_copy_or_regrid_ll!(dst_3d, ws_field, ws)

LL phase-2 helper: copy (identity, when shapes match) or regrid
(conservative) a single source-grid merged TM5 field into the
per-window target array `dst_3d` of shape `(Nx, Ny, Nz)`.
"""
function tm5_copy_or_regrid_ll!(dst_3d::AbstractArray{FT, 3},
                                 ws_field::AbstractArray{FT, 3},
                                 ws::TM5PreprocessingWorkspace{FT}) where FT
    if ws.regridder === nothing
        size(dst_3d) == size(ws_field) ||
            error("TM5 LL identity path requires matching shapes, got " *
                  "dst=$(size(dst_3d)) vs src=$(size(ws_field))")
        copyto!(dst_3d, ws_field)
    else
        apply_regridder!(dst_3d, ws.regridder, ws_field)
    end
    return dst_3d
end
