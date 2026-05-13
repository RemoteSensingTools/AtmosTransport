#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Plan 26 P0.D3 — departures IO round-trip + schema-validation tests.
#
# Coverage:
#   * `CSDepartureRecord` keyword constructor: dtype coercion, finite
#     checks on simulated/departure/normalized/value/sigma, range
#     checks on lat / panel / step / i / j.
#   * `build_departure_set` alignment + finite gatekeeper. Length
#     mismatches and non-finite simulated values are rejected.
#   * `write_departures` / `read_departures` round-trip through a v1
#     NetCDF file. Every field is bit-exact at its declared dtype
#     precision.
#   * On-disk layout matches `schemas/cs_departures_v1.toml`: dim
#     names, variable dtypes, root attributes.
#   * Row order + observation `id`s round-trip in the original order
#     even when the input is constructed in non-monotonic id order
#     (the user-specified D3 hard requirement).
#   * Reader rejects schema violations: wrong-version, missing
#     schema attr, missing required attr (each of the seven required
#     root attrs), wrong departure-sign-convention, missing required
#     variable, wrong `date_component` dim.
#   * Optional `run_id` / `iteration` round-trip when given, absent
#     when not.
#   * Sign convention: `departure == simulated_value - observed_value`
#     and `normalized_departure == departure / value_sigma` are
#     computed correctly inside `build_departure_set`.
# ---------------------------------------------------------------------------

using Test
using Dates: DateTime
using NCDatasets: NCDataset, defDim, defVar

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
const AT = AtmosTransport

const FT_TEST = Float64
const NC_TEST = 6
const DT_TEST = 3600.0
const T0_TEST_STR = "2024-06-15 00:00:00"
const T0_TEST = DateTime(2024, 6, 15, 0, 0, 0)

_mesh() = AT.CubedSphereMesh(; Nc = NC_TEST, FT = FT_TEST)

# Build an aligned (CSObservationSet, observations, simulated) triple
# from a list of (id, panel, i, j, observed, sigma, hour_offset)
# targets. The records and the bound observations are aligned
# row-for-row because no filtering happens between them.
function _aligned_inputs(targets; tracer = "CO2",
                          instrument = "TCCON",
                          sim_offset = 0.5)
    mesh = _mesh()
    records = AT.CSObservationRecord[]
    sims = Float64[]
    for t in targets
        lons, lats = AT.panel_cell_center_lonlat(mesh, t.panel)
        lat_, lon_ = lats[t.i, t.j], lons[t.i, t.j]
        # date = T0 + hour_offset hours.
        h = Int(t.hour_offset)
        push!(records, AT.CSObservationRecord(
            id = t.id,
            date_components = (Int16(2024), Int16(6), Int16(15),
                                Int16(h),    Int16(0), Int16(0)),
            lat = Float32(lat_), lon = Float32(lon_), alt = 0.0f0,
            value = Float64(t.observed), value_sigma = Float64(t.sigma),
            instrument_type = instrument, tracer = tracer))
        push!(sims, Float64(t.observed) + sim_offset)
    end
    set = AT.CSObservationSet(records; time_origin = T0_TEST_STR)
    obs = AT.bind_to_mesh(set, mesh, T0_TEST, DT_TEST; nsteps = 24)
    return (set = set, observations = obs, simulated = sims,
            mesh = mesh, sim_offset = sim_offset)
end

# ---------------------------------------------------------------------------
# CSDepartureRecord constructor
# ---------------------------------------------------------------------------

@testset "CSDepartureRecord keyword constructor" begin
    r = AT.CSDepartureRecord(
        id = 7, tracer = "CO2", instrument_type = "TCCON",
        date_components = (2024, 6, 15, 12, 0, 0),
        lat = 34.0, lon = -118.0, alt = 100.0,
        step = 5, panel = 3, i = 4, j = 6,
        observed_value = 420.0, simulated_value = 420.5,
        departure = 0.5, value_sigma = 1.0,
        normalized_departure = 0.5)
    @test r.id === Int64(7)
    @test r.tracer == "CO2"
    @test r.instrument_type == "TCCON"
    @test r.date_components === (Int16(2024), Int16(6), Int16(15),
                                  Int16(12),   Int16(0), Int16(0))
    @test r.lat === Float32(34.0)
    @test r.lon === Float32(-118.0)
    @test r.alt === Float32(100.0)
    @test r.step === Int64(5)
    @test r.panel === Int8(3)
    @test r.i === Int32(4)
    @test r.j === Int32(6)
    @test r.observed_value === 420.0
    @test r.simulated_value === 420.5
    @test r.departure === 0.5
    @test r.value_sigma === 1.0
    @test r.normalized_departure === 0.5

    _bad(; kwargs...) = AT.CSDepartureRecord(;
        id = 1, tracer = "T", instrument_type = "I",
        date_components = (2024, 1, 1, 0, 0, 0),
        lat = 0.0, lon = 0.0, alt = 0.0,
        step = 1, panel = 1, i = 1, j = 1,
        observed_value = 1.0, simulated_value = 1.0,
        departure = 0.0, value_sigma = 1.0,
        normalized_departure = 0.0, kwargs...)
    @test_throws ArgumentError _bad(simulated_value = NaN)
    @test_throws ArgumentError _bad(simulated_value = Inf)
    @test_throws ArgumentError _bad(departure = NaN)
    @test_throws ArgumentError _bad(normalized_departure = Inf)
    @test_throws ArgumentError _bad(observed_value = NaN)
    @test_throws ArgumentError _bad(value_sigma = -1.0)
    @test_throws ArgumentError _bad(value_sigma = Inf)
    @test_throws ArgumentError _bad(lat = NaN)
    @test_throws ArgumentError _bad(lat = 91.0)
    @test_throws ArgumentError _bad(lon = Inf)
    @test_throws ArgumentError _bad(step = 0)
    @test_throws ArgumentError _bad(panel = 0)
    @test_throws ArgumentError _bad(panel = 7)
    @test_throws ArgumentError _bad(i = 0)
    @test_throws ArgumentError _bad(j = -1)
end

# ---------------------------------------------------------------------------
# Arithmetic invariant — pinned by the v1 sign convention. A record
# that contradicts `departure = simulated - observed` or
# `normalized_departure = departure / sigma` must be rejected at
# construction, which also covers `read_departures` (the reader
# constructs one record per row).
# ---------------------------------------------------------------------------

@testset "CSDepartureRecord — sign-convention arithmetic invariant" begin
    _ctor(; departure, normalized_departure,
            observed_value = 10.0, simulated_value = 11.0,
            value_sigma = 2.0) = AT.CSDepartureRecord(
        id = 1, tracer = "T", instrument_type = "I",
        date_components = (2024, 1, 1, 0, 0, 0),
        lat = 0.0, lon = 0.0, alt = 0.0,
        step = 1, panel = 1, i = 1, j = 1,
        observed_value = observed_value, simulated_value = simulated_value,
        departure = departure, value_sigma = value_sigma,
        normalized_departure = normalized_departure)

    # Reference values: sim - obs = 1.0, normalized = 0.5.
    @test _ctor(departure = 1.0, normalized_departure = 0.5) isa
          AT.CSDepartureRecord

    # Sign flip on `departure` — the exact case flagged by the review.
    @test_throws ArgumentError _ctor(departure = -1.0,
                                      normalized_departure = -0.5)

    # `normalized_departure` inconsistent with `departure / sigma`.
    @test_throws ArgumentError _ctor(departure = 1.0,
                                      normalized_departure = 0.3)

    # Magnitude off by a factor of 2.
    @test_throws ArgumentError _ctor(departure = 2.0,
                                      normalized_departure = 1.0)

    # Within rounding tolerance — must still accept.
    @test _ctor(departure = 1.0 + 1e-13,
                normalized_departure = 0.5 + 5e-14) isa AT.CSDepartureRecord
end

# ---------------------------------------------------------------------------
# build_departure_set: alignment + finite gatekeeper + sign convention
# ---------------------------------------------------------------------------

@testset "build_departure_set — alignment + finite gates" begin
    inputs = _aligned_inputs([
        (id = 1, panel = 1, i = 2, j = 3, observed = 410.0, sigma = 0.5,
         hour_offset = 0),
        (id = 2, panel = 4, i = 5, j = 1, observed = 411.0, sigma = 0.7,
         hour_offset = 2),
        (id = 3, panel = 6, i = 1, j = 6, observed = 415.0, sigma = 1.0,
         hour_offset = 5),
    ])
    # Sanity: bind_to_mesh preserved all three rows.
    @test length(inputs.observations) == 3

    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24)
    @test dep isa AT.CSDepartureSet
    @test length(dep) == 3
    @test dep.mesh_Nc == NC_TEST
    @test dep.mesh_panel_convention == "GnomonicPanelConvention"
    @test dep.mesh_cs_definition_tag == "equiangular_gnomonic"
    @test dep.t_start == T0_TEST_STR
    @test dep.dt_seconds == DT_TEST
    @test dep.nsteps == 24
    @test dep.run_id === nothing
    @test dep.iteration === nothing

    # Sign convention: simulated - observed; normalized = departure / sigma.
    for (k, r) in enumerate(dep.records)
        @test r.observed_value == inputs.set.records[k].value
        @test r.simulated_value == inputs.simulated[k]
        @test r.departure ≈ inputs.sim_offset atol = 1e-12
        @test r.normalized_departure ≈ inputs.sim_offset /
                                       inputs.set.records[k].value_sigma atol = 1e-12
    end

    # Length-mismatch rejections.
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, inputs.observations[1:2],
        inputs.simulated, inputs.mesh,
        T0_TEST_STR, DT_TEST, 24)
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, inputs.observations,
        inputs.simulated[1:2], inputs.mesh,
        T0_TEST_STR, DT_TEST, 24)

    # Non-finite simulated rejection.
    bad_sim = copy(inputs.simulated)
    bad_sim[2] = NaN
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, inputs.observations, bad_sim, inputs.mesh,
        T0_TEST_STR, DT_TEST, 24)
    bad_sim[2] = Inf
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, inputs.observations, bad_sim, inputs.mesh,
        T0_TEST_STR, DT_TEST, 24)

    # nsteps must be positive.
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, inputs.observations, inputs.simulated, inputs.mesh,
        T0_TEST_STR, DT_TEST, 0)
end

# ---------------------------------------------------------------------------
# Alignment provenance — observations must come from the same records,
# in the same order. Reversing the observations vector produces row 1
# with id/value from `set.records[1]` but step/panel/cell from a
# different observation. `build_departure_set` must catch this.
# ---------------------------------------------------------------------------

@testset "build_departure_set — alignment provenance check" begin
    # Targets crafted so each record has a DIFFERENT observed value
    # and sigma — otherwise reversing would coincidentally satisfy
    # `obs.value == rec.value`.
    inputs = _aligned_inputs([
        (id = 1, panel = 1, i = 2, j = 3, observed = 410.0, sigma = 0.5,
         hour_offset = 0),
        (id = 2, panel = 4, i = 5, j = 1, observed = 411.0, sigma = 0.7,
         hour_offset = 1),
        (id = 3, panel = 6, i = 1, j = 6, observed = 412.0, sigma = 1.0,
         hour_offset = 2),
    ])

    # Reversed observations vector — values no longer match rows.
    reversed_obs = reverse(inputs.observations)
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, reversed_obs, inputs.simulated,
        inputs.mesh, T0_TEST_STR, DT_TEST, 24)

    # Reversed simulated stays aligned to records (sim/observed not
    # constrained to match record's value pre-departure), but the
    # observations-vs-records check still triggers when observations
    # are scrambled.
    scrambled_obs = inputs.observations[[2, 1, 3]]
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, scrambled_obs, inputs.simulated,
        inputs.mesh, T0_TEST_STR, DT_TEST, 24)

    # An obs hand-built with a deliberately wrong value (right step,
    # wrong value) is also caught — provenance is about the value /
    # sigma fields, not the index.
    wrong_value_obs = copy(inputs.observations)
    wrong_value_obs[2] = AT.CSObservation(
        inputs.observations[2].step,
        inputs.observations[2].objective,
        999.0,                                 # value differs from record
        Float64(inputs.set.records[2].value_sigma))
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, wrong_value_obs, inputs.simulated,
        inputs.mesh, T0_TEST_STR, DT_TEST, 24)

    # Wrong sigma (record sigma = 0.7, obs sigma = 99) — also caught.
    wrong_sigma_obs = copy(inputs.observations)
    wrong_sigma_obs[2] = AT.CSObservation(
        inputs.observations[2].step,
        inputs.observations[2].objective,
        Float64(inputs.set.records[2].value),
        99.0)
    @test_throws ArgumentError AT.build_departure_set(
        inputs.set, wrong_sigma_obs, inputs.simulated,
        inputs.mesh, T0_TEST_STR, DT_TEST, 24)
end

# ---------------------------------------------------------------------------
# Per-record bounds against run metadata. The CSDepartureSet ctor
# rejects records whose step > nsteps or i/j > mesh_Nc, which also
# covers `read_departures` (the reader builds a CSDepartureSet from
# the parsed records).
# ---------------------------------------------------------------------------

@testset "CSDepartureSet — per-record bounds vs run metadata" begin
    _dep_rec(; step = 1, i = 1, j = 1, panel = 1) =
        AT.CSDepartureRecord(
            id = 1, tracer = "T", instrument_type = "I",
            date_components = (2024, 1, 1, 0, 0, 0),
            lat = 0.0, lon = 0.0, alt = 0.0,
            step = step, panel = panel, i = i, j = j,
            observed_value = 10.0, simulated_value = 11.0,
            departure = 1.0, value_sigma = 2.0,
            normalized_departure = 0.5)

    function _set(rec; mesh_Nc = NC_TEST, nsteps = 24)
        return AT.CSDepartureSet([rec];
            mesh_Nc = mesh_Nc,
            mesh_panel_convention = "GnomonicPanelConvention",
            mesh_cs_definition_tag = "equiangular_gnomonic",
            t_start = T0_TEST_STR,
            dt_seconds = DT_TEST,
            nsteps = nsteps)
    end

    # In bounds — accepted.
    @test _set(_dep_rec()) isa AT.CSDepartureSet
    @test _set(_dep_rec(step = 24, i = NC_TEST, j = NC_TEST)) isa
          AT.CSDepartureSet

    # step > nsteps -> reject.
    @test_throws ArgumentError _set(_dep_rec(step = 999))
    @test_throws ArgumentError _set(_dep_rec(step = 25); nsteps = 24)

    # i / j out of mesh.Nc bounds -> reject.
    @test_throws ArgumentError _set(_dep_rec(i = 999))
    @test_throws ArgumentError _set(_dep_rec(j = NC_TEST + 1))

    # Negative / zero step is already caught by the CSDepartureRecord
    # ctor — sanity-check it's still caught (defense in depth).
    @test_throws ArgumentError _dep_rec(step = 0)
end

# ---------------------------------------------------------------------------
# Round-trip: bit-exact across every field
# ---------------------------------------------------------------------------

@testset "CSDepartureSet bit-exact NetCDF round-trip" begin
    inputs = _aligned_inputs([
        (id = 10, panel = 1, i = 2, j = 3, observed = 410.1, sigma = 0.5,
         hour_offset = 0),
        (id = 20, panel = 3, i = 4, j = 5, observed = 411.2, sigma = 0.7,
         hour_offset = 1),
        (id = 30, panel = 6, i = 1, j = 6, observed = 415.0, sigma = 1.0,
         hour_offset = 4),
    ])
    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24;
                                  run_id = "run-A", iteration = 3)

    mktempdir() do dir
        path = joinpath(dir, "dep_v1.nc")
        AT.write_departures(path, dep)
        @test isfile(path)

        back = AT.read_departures(path)
        @test back isa AT.CSDepartureSet
        @test length(back) == length(dep)
        @test back.mesh_Nc == dep.mesh_Nc
        @test back.mesh_panel_convention == dep.mesh_panel_convention
        @test back.mesh_cs_definition_tag == dep.mesh_cs_definition_tag
        @test back.t_start == dep.t_start
        @test back.dt_seconds == dep.dt_seconds
        @test back.nsteps == dep.nsteps
        @test back.run_id == "run-A"
        @test back.iteration == 3

        for (orig, got) in zip(dep, back)
            @test orig.id == got.id
            @test orig.tracer == got.tracer
            @test orig.instrument_type == got.instrument_type
            @test orig.date_components == got.date_components
            @test orig.lat === got.lat
            @test orig.lon === got.lon
            @test orig.alt === got.alt
            @test orig.step == got.step
            @test orig.panel == got.panel
            @test orig.i == got.i
            @test orig.j == got.j
            @test orig.observed_value === got.observed_value
            @test orig.simulated_value === got.simulated_value
            @test orig.departure === got.departure
            @test orig.value_sigma === got.value_sigma
            @test orig.normalized_departure === got.normalized_departure
        end
    end
end

# ---------------------------------------------------------------------------
# Row order + id preservation (user-specified D3 hard requirement)
# ---------------------------------------------------------------------------

@testset "write_departures / read_departures preserves row order + ids" begin
    # Non-monotonic id order: 99, 1, 42, 7 — the file must read back
    # in this exact order.
    inputs = _aligned_inputs([
        (id = 99, panel = 1, i = 1, j = 1, observed = 410.0, sigma = 0.5,
         hour_offset = 0),
        (id =  1, panel = 3, i = 2, j = 4, observed = 411.0, sigma = 0.6,
         hour_offset = 1),
        (id = 42, panel = 5, i = 3, j = 2, observed = 412.0, sigma = 0.7,
         hour_offset = 2),
        (id =  7, panel = 6, i = 6, j = 6, observed = 413.0, sigma = 0.8,
         hour_offset = 3),
    ])
    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24)

    mktempdir() do dir
        path = joinpath(dir, "ordered.nc")
        AT.write_departures(path, dep)
        back = AT.read_departures(path)
        @test [r.id for r in back] == [99, 1, 42, 7]
    end
end

# ---------------------------------------------------------------------------
# Optional run_id / iteration
# ---------------------------------------------------------------------------

@testset "write_departures omits optional attrs when not provided" begin
    inputs = _aligned_inputs([
        (id = 1, panel = 1, i = 1, j = 1, observed = 410.0, sigma = 0.5,
         hour_offset = 0)])
    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24)  # no run_id / iteration

    mktempdir() do dir
        path = joinpath(dir, "no_optional.nc")
        AT.write_departures(path, dep)
        NCDataset(path, "r") do ds
            @test !haskey(ds.attrib, "run_id")
            @test !haskey(ds.attrib, "iteration")
        end
        back = AT.read_departures(path)
        @test back.run_id === nothing
        @test back.iteration === nothing
    end
end

# ---------------------------------------------------------------------------
# On-disk layout matches schema
# ---------------------------------------------------------------------------

@testset "v1 NetCDF on-disk layout matches schema" begin
    inputs = _aligned_inputs([
        (id = 1, panel = 1, i = 2, j = 3, observed = 410.1, sigma = 0.5,
         hour_offset = 0),
        (id = 2, panel = 3, i = 4, j = 5, observed = 411.2, sigma = 0.7,
         hour_offset = 1),
    ])
    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24;
                                  run_id = "run-B", iteration = 7)

    mktempdir() do dir
        path = joinpath(dir, "layout.nc")
        AT.write_departures(path, dep)
        NCDataset(path, "r") do ds
            @test ds.attrib["cs_departures_schema"] == "v1"
            @test ds.attrib["mesh_Nc"] == NC_TEST
            @test ds.attrib["mesh_panel_convention"] == "GnomonicPanelConvention"
            @test ds.attrib["mesh_cs_definition_tag"] == "equiangular_gnomonic"
            @test ds.attrib["t_start"] == T0_TEST_STR
            @test ds.attrib["dt_seconds"] == DT_TEST
            @test ds.attrib["nsteps"] == 24
            @test ds.attrib["departure_sign_convention"] == "simulated_minus_observed"
            @test ds.attrib["run_id"] == "run-B"
            @test ds.attrib["iteration"] == 7

            @test ds.dim["obs"] == length(dep)
            @test ds.dim["date_component"] == 6

            @test eltype(ds["id"][:])                  === Int64
            @test eltype(ds["date_components"][:, :])  === Int16
            @test eltype(ds["lat"][:])                 === Float32
            @test eltype(ds["lon"][:])                 === Float32
            @test eltype(ds["alt"][:])                 === Float32
            @test eltype(ds["step"][:])                === Int64
            @test eltype(ds["panel"][:])               === Int8
            @test eltype(ds["i"][:])                   === Int32
            @test eltype(ds["j"][:])                   === Int32
            @test eltype(ds["observed_value"][:])      === Float64
            @test eltype(ds["simulated_value"][:])     === Float64
            @test eltype(ds["departure"][:])           === Float64
            @test eltype(ds["value_sigma"][:])         === Float64
            @test eltype(ds["normalized_departure"][:]) === Float64
            @test ds["tracer"][1] isa AbstractString
            @test ds["instrument_type"][1] isa AbstractString

            @test size(ds["date_components"][:, :]) == (6, length(dep))
        end
    end
end

# ---------------------------------------------------------------------------
# Reader rejects schema violations
# ---------------------------------------------------------------------------

@testset "read_departures rejects schema violations" begin
    inputs = _aligned_inputs([
        (id = 1, panel = 1, i = 2, j = 3, observed = 410.1, sigma = 0.5,
         hour_offset = 0),
        (id = 2, panel = 3, i = 4, j = 5, observed = 411.2, sigma = 0.7,
         hour_offset = 1),
    ])
    dep = AT.build_departure_set(inputs.set, inputs.observations,
                                  inputs.simulated, inputs.mesh,
                                  T0_TEST_STR, DT_TEST, 24)

    mktempdir() do dir
        good = joinpath(dir, "good.nc")
        AT.write_departures(good, dep)
        @test AT.read_departures(good) isa AT.CSDepartureSet

        # Wrong schema version.
        bad_version = joinpath(dir, "bad_version.nc")
        cp(good, bad_version)
        NCDataset(bad_version, "a") do ds
            ds.attrib["cs_departures_schema"] = "v2"
        end
        @test_throws ArgumentError AT.read_departures(bad_version)

        # Missing schema attribute.
        no_attr = joinpath(dir, "no_attr.nc")
        cp(good, no_attr)
        NCDataset(no_attr, "a") do ds
            delete!(ds.attrib, "cs_departures_schema")
        end
        @test_throws ArgumentError AT.read_departures(no_attr)

        # Wrong sign convention.
        bad_sign = joinpath(dir, "bad_sign.nc")
        cp(good, bad_sign)
        NCDataset(bad_sign, "a") do ds
            ds.attrib["departure_sign_convention"] = "observed_minus_simulated"
        end
        @test_throws ArgumentError AT.read_departures(bad_sign)

        # Each required attribute missing -> reject.
        required = ("mesh_Nc", "mesh_panel_convention", "mesh_cs_definition_tag",
                    "t_start", "dt_seconds", "nsteps", "departure_sign_convention")
        for attr in required
            tmp = joinpath(dir, "no_" * attr * ".nc")
            cp(good, tmp)
            NCDataset(tmp, "a") do ds
                delete!(ds.attrib, attr)
            end
            @test_throws ArgumentError AT.read_departures(tmp)
        end

        # Non-existent file.
        @test_throws ArgumentError AT.read_departures(joinpath(dir, "missing.nc"))

        # Each required variable missing -> reject. We can't really
        # delete a variable from a NetCDF in place via NCDatasets, so
        # build each scenario by writing a fresh file that omits the
        # target variable. The simplest path: write the good file,
        # copy the byte stream, then prove that the reader requires
        # the variable by rewriting the file with that variable
        # absent. We do this by hand-constructing minimal NetCDFs.
        function _write_missing_var(path, omitted_var)
            isfile(path) && rm(path)
            NCDataset(path, "c") do ds
                defDim(ds, "obs", 1)
                defDim(ds, "date_component", 6)
                ds.attrib["cs_departures_schema"] = "v1"
                ds.attrib["mesh_Nc"] = Int64(NC_TEST)
                ds.attrib["mesh_panel_convention"] = "GnomonicPanelConvention"
                ds.attrib["mesh_cs_definition_tag"] = "equiangular_gnomonic"
                ds.attrib["t_start"] = T0_TEST_STR
                ds.attrib["dt_seconds"] = DT_TEST
                ds.attrib["nsteps"] = Int64(24)
                ds.attrib["departure_sign_convention"] = "simulated_minus_observed"
                spec = (
                    ("id", Int64, ("obs",), [Int64(1)]),
                    ("tracer", String, ("obs",), ["CO2"]),
                    ("instrument_type", String, ("obs",), ["TCCON"]),
                    ("date_components", Int16, ("date_component", "obs"),
                     reshape(Int16[2024, 1, 1, 0, 0, 0], 6, 1)),
                    ("lat", Float32, ("obs",), Float32[0.0]),
                    ("lon", Float32, ("obs",), Float32[0.0]),
                    ("alt", Float32, ("obs",), Float32[0.0]),
                    ("step", Int64, ("obs",), [Int64(1)]),
                    ("panel", Int8, ("obs",), Int8[1]),
                    ("i", Int32, ("obs",), Int32[1]),
                    ("j", Int32, ("obs",), Int32[1]),
                    ("observed_value", Float64, ("obs",), [10.0]),
                    ("simulated_value", Float64, ("obs",), [11.0]),
                    ("departure", Float64, ("obs",), [1.0]),
                    ("value_sigma", Float64, ("obs",), [2.0]),
                    ("normalized_departure", Float64, ("obs",), [0.5]),
                )
                for (name, T, dims, data) in spec
                    name == omitted_var && continue
                    v = defVar(ds, name, T, dims)
                    v[:] = data
                end
            end
            return path
        end

        required_vars = ("id", "tracer", "instrument_type", "date_components",
                         "lat", "lon", "alt",
                         "step", "panel", "i", "j",
                         "observed_value", "simulated_value", "departure",
                         "value_sigma", "normalized_departure")
        for v in required_vars
            tmp = joinpath(dir, "no_var_" * v * ".nc")
            _write_missing_var(tmp, v)
            @test_throws ArgumentError AT.read_departures(tmp)
        end

        # Wrong date_component dim length -> reject.
        wrong_dc = joinpath(dir, "wrong_dc.nc")
        NCDataset(wrong_dc, "c") do ds
            defDim(ds, "obs", 1)
            defDim(ds, "date_component", 5)  # wrong; must be 6
            ds.attrib["cs_departures_schema"] = "v1"
            ds.attrib["mesh_Nc"] = Int64(NC_TEST)
            ds.attrib["mesh_panel_convention"] = "GnomonicPanelConvention"
            ds.attrib["mesh_cs_definition_tag"] = "equiangular_gnomonic"
            ds.attrib["t_start"] = T0_TEST_STR
            ds.attrib["dt_seconds"] = DT_TEST
            ds.attrib["nsteps"] = Int64(24)
            ds.attrib["departure_sign_convention"] = "simulated_minus_observed"
            spec = (
                ("id", Int64, ("obs",), [Int64(1)]),
                ("tracer", String, ("obs",), ["CO2"]),
                ("instrument_type", String, ("obs",), ["TCCON"]),
                ("date_components", Int16, ("date_component", "obs"),
                 reshape(Int16[2024, 1, 1, 0, 0], 5, 1)),
                ("lat", Float32, ("obs",), Float32[0.0]),
                ("lon", Float32, ("obs",), Float32[0.0]),
                ("alt", Float32, ("obs",), Float32[0.0]),
                ("step", Int64, ("obs",), [Int64(1)]),
                ("panel", Int8, ("obs",), Int8[1]),
                ("i", Int32, ("obs",), Int32[1]),
                ("j", Int32, ("obs",), Int32[1]),
                ("observed_value", Float64, ("obs",), [10.0]),
                ("simulated_value", Float64, ("obs",), [11.0]),
                ("departure", Float64, ("obs",), [1.0]),
                ("value_sigma", Float64, ("obs",), [2.0]),
                ("normalized_departure", Float64, ("obs",), [0.5]),
            )
            for (name, T, dims, data) in spec
                v = defVar(ds, name, T, dims)
                v[:] = data
            end
        end
        @test_throws ArgumentError AT.read_departures(wrong_dc)
    end
end
