source = read("test/runtests.jl", String)
include_string(Main, first(split(source, "selected = _selected_tiers(ARGS)")), abspath("test/runtests.jl"))
for f in _tier_files(:core)
    basename(f) >= "test_readme_current.jl" || continue
    @info "Running $f"
    run_test_file_isolated(f)
end
run_test_file_isolated("regridding/runtests.jl")
@info "Resumed core and regridding suite complete."
