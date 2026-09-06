using TOML, Statistics, Printf

# Recompute the table from archived samples; sample 0 is compilation/warmup.
const here = @__DIR__
rows = Dict[]
for nt in (6, 32), tile in ("256", "32", "32x2")
    samples = [TOML.parsefile(joinpath(here, "samples", tile,
                                     "collab_result_$(nt)_$(sample).toml"))
               for sample in 1:2]
    row = Dict("tracers" => nt, "tile" => tile,
               "median_seconds" => median(r["wall_seconds"] for r in samples),
               "median_host_bytes" => median(r["host_allocated_bytes"] for r in samples),
               "samples_seconds" => [r["wall_seconds"] for r in samples])
    push!(rows, row)
    @printf("Nt=%2d tile=%4s: %.6f s, %.6f GB cumulative host allocation\n",
            nt, tile, row["median_seconds"], row["median_host_bytes"] / 1e9)
end
open(joinpath(here, "summary.toml"), "w") do io
    TOML.print(io, Dict("measurements" => rows))
end
