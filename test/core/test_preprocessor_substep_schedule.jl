using Test
using AtmosTransport.Preprocessing:
    SubstepSchedulePolicy,
    initial_substeps,
    next_substeps,
    required_substeps,
    rescale_substep_amounts!
using AtmosTransport.MetDrivers:
    set_transport_header_steps_per_window_schedule!,
    set_transport_header_split_substep_recommendations!,
    validate_transport_contract!

@testset "shared preprocessor substep schedule policy" begin
    policy = SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.95,
        min_steps_per_window = 1,
        max_steps_per_window = 64)

    @test initial_substeps(policy, 4) == 4
    @test required_substeps(policy, 8, 1.233) == 11
    @test next_substeps(policy, 8, 0.25) == 8
    @test next_substeps(policy, 8, Inf) == 64

    fixed = SubstepSchedulePolicy(
        adaptive_substeps = false,
        substep_cfl_target = 0.95,
        min_steps_per_window = 1,
        max_steps_per_window = 64)
    @test next_substeps(fixed, 8, 10.0) == 8

    a = fill(2.0, 2, 2)
    b = fill(4.0, 2, 2)
    rescale_substep_amounts!((a, b), 4, 10)
    @test all(==(0.8), a)
    @test all(==(1.6), b)

    @test_throws ArgumentError SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.0)
    @test_throws ArgumentError SubstepSchedulePolicy(
        adaptive_substeps = true,
        substep_cfl_target = 0.95,
        min_steps_per_window = 8,
        max_steps_per_window = 4)
end

@testset "transport header schedule patch keeps Poisson contract synchronized" begin
    header = Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 3,
        "source_flux_sampling" => "window_start_endpoint",
        "air_mass_sampling" => "window_start_endpoint",
        "flux_sampling" => "window_constant",
        "flux_kind" => "substep_mass_amount",
        "delta_semantics" => "forward_window_endpoint_difference",
        "humidity_sampling" => "none",
        "poisson_balance_target_scale" => 1 / 8,
        "poisson_balance_target_semantics" =>
            "forward_window_mass_difference / (2 * steps_per_window)",
        "nwindow" => 3,
        "steps_per_window" => 4,
        "steps_per_window_by_window" => [4, 4, 4],
        "time_step_schedule" => "constant",
        "poisson_balance_target_scale_by_window" => [1 / 8, 1 / 8, 1 / 8],
    )

    set_transport_header_steps_per_window_schedule!(header, [4, 7, 5])
    @test header["steps_per_window"] == 7
    @test header["steps_per_window_by_window"] == [4, 7, 5]
    @test header["time_step_schedule"] == "per_window"
    @test header["poisson_balance_target_semantics"] ==
          "forward_window_mass_difference / (2 * steps_per_window_by_window[win])"
    @test header["poisson_balance_target_scale_by_window"] ==
          [1 / 8, 1 / 14, 1 / 10]
    @test header["poisson_balance_target_scale"] == 1 / 14
end

@testset "split-substep recommendations (BACKLOG 11b)" begin
    header = Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 3,
        "source_flux_sampling" => "window_start_endpoint",
        "air_mass_sampling" => "window_start_endpoint",
        "flux_sampling" => "window_constant",
        "flux_kind" => "full_window_mass_amount",
        "delta_semantics" => "forward_window_endpoint_difference",
        "humidity_sampling" => "none",
        "poisson_balance_target_scale" => 1.0,
        "poisson_balance_target_semantics" => "forward_window_mass_difference",
        "nwindow" => 3,
        "steps_per_window" => 4,
        "steps_per_window_by_window" => [4, 4, 4],
        "time_step_schedule" => "constant",
        "poisson_balance_target_scale_by_window" => [1.0, 1.0, 1.0],
    )
    set_transport_header_steps_per_window_schedule!(header, [4, 8, 6])

    # must be called after the schedule
    bare = Dict{String, Any}("magic" => "MFLX")
    @test_throws ArgumentError set_transport_header_split_substep_recommendations!(
        bare, [1, 1, 1], [1, 1, 1])

    # length mismatch
    @test_throws ArgumentError set_transport_header_split_substep_recommendations!(
        header, [2, 3], [4, 8, 6])
    # zero entries
    @test_throws ArgumentError set_transport_header_split_substep_recommendations!(
        header, [0, 3, 2], [4, 8, 6])
    # a direction may not exceed the combined schedule (it is dominated:
    # outgoing_xy + outgoing_z >= each part)
    @test_throws ArgumentError set_transport_header_split_substep_recommendations!(
        header, [2, 9, 2], [4, 8, 6])

    set_transport_header_split_substep_recommendations!(header, [2, 3, 2], [4, 8, 6])
    @test header["recommended_substeps_xy_by_window"] == [2, 3, 2]
    @test header["recommended_substeps_z_by_window"] == [4, 8, 6]

    # the contract validator rejects a lone array and bad lengths
    lone = copy(header)
    delete!(lone, "recommended_substeps_z_by_window")
    @test_throws ArgumentError validate_transport_contract!(lone)
    badlen = copy(header)
    badlen["recommended_substeps_xy_by_window"] = [2, 3]
    @test_throws ArgumentError validate_transport_contract!(badlen)
end
