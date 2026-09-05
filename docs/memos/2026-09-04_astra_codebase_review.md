# Astra codebase review — 2026-09-04

Recovered from the local Astra session started at 16:22 on September 4, 2026.
Session ID: `01a06ebb-0bc5-7b20-880b-a72d23c23062`.

This preserves the original review and handoff, rather than a fresh validation.
Line references describe the working tree at review time and may have shifted.
Reported measurements below are from that session; they have not been rerun
in the recovery session. The user subsequently requested “fix these now.”
GPU testing for continued work must use the A100, per the current instruction.

## Recovered review

There are worthwhile improvements in all three areas. The typed grid/state/operator design is a strong foundation; I would prioritize configuration reliability, clearer ownership of setup code, and reduced memory traffic.

I reviewed the current working tree, including uncommitted work, and ran targeted CPU probes. I made no repository changes.

For scientists, the first priority is making the documentation and configuration trustworthy:

- The [README](/home/cfranken/code/gitHub/AtmosTransportModel/README.md:50) advertises slopes and PPM on reduced Gaussian grids, but the [runtime rejects both](/home/cfranken/code/gitHub/AtmosTransportModel/src/Operators/Advection/StrangSplitting.jl:1483). I reproduced both errors.
- [Configuration validation](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/DrivenRunner.jl:341) reports success for invalid physics names when input paths exist. A misspelled `scheem = "ppm"` also silently selects default upwind. Rejecting unknown keys and validating every physics specification before allocation would prevent misleading runs.
- The [contributor guide](/home/cfranken/code/gitHub/AtmosTransportModel/CONTRIBUTING.md:49) describes obsolete types, directories, and interfaces. Replace its operator example with a small, executable example using the current API.

I would also give each operator a short scientific description covering its equation, units, array dimensions, vertical ordering, and execution cadence. In particular, distinguish stored tracer quantity—dry VMR multiplied by dry-air mass—from physical species mass. These details should be discoverable without following historical development notes.

The structural changes with the clearest benefit are:

| Current pressure point | Suggested organization |
|---|---|
| [DrivenRunner.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/DrivenRunner.jl:857), 1,457 lines with separate structured/CS execution loops | Separate configuration, model construction, output handling, and execution. Share the file/window lifecycle through topology-specific setup methods. |
| [InitialConditionIO.jl](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/InitialConditionIO.jl:1), 2,049 lines including emissions | Separate initial conditions, surface-flux loading, unit conversion, and regridding adapters. |
| Runtime imports preprocessing for panel-unpacking helpers | Move those small shared helpers into geometry/regridding, removing the dependency shown [here](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/InitialConditionIO.jl:52). |
| Diffusion scratch belongs to the advection workspace | Give diffusion its own workspace. Currently, a [diffusion-only CS model allocates an advection workspace](/home/cfranken/code/gitHub/AtmosTransportModel/src/Models/TransportModel.jl:282). |

These changes would make ownership clearer while retaining the existing dispatch-based numerical implementation.

For speed, I found both measurable improvements and candidates worth profiling:

| Opportunity | Evidence and proposed change |
|---|---|
| **Improve CPU column diagnostics** | [The current loop](/home/cfranken/code/gitHub/AtmosTransportModel/src/Output/diagnostics.jl:18) traverses the strided vertical dimension innermost. A temporary experiment using contiguous traversal was **1.7–3.3× faster**, with identical results on the sampled inputs. At `180×180×72`, Float32 took **2.94 → 1.14 ms**. |
| **Capture only requested output** | [Snapshot capture](/home/cfranken/code/gitHub/AtmosTransportModel/src/Output/snapshots.jl:101) copies full air-mass and every tracer field to the CPU, even for column-only output. Compute requested reductions on the backend and transfer compact results. C180/L72 with four Float32 tracers currently needs approximately **267 MiB per snapshot**, before diagnostic temporaries. |
| **Reuse multi-tracer air-mass backups** | [Lin–Rood allocates a fresh backup each step](/home/cfranken/code/gitHub/AtmosTransportModel/src/Operators/Advection/StrangSplitting.jl:1189); RG has the same pattern. A warmed four-tracer C24/L32 CPU step allocated **1.19 MB**, including a roughly **0.69 MB** backup. Store it in the workspace. |
| **Cache preprocessing scratch** | [Poisson balancing](/home/cfranken/code/gitHub/AtmosTransportModel/src/Preprocessing/cs_poisson_balance.jl:812) recreates per-thread scratch on every call. With 16 thread slots at C180, the additional scratch arrays alone total about **156 MiB per call**. Allocate them once per preprocessing workspace. |
| **Reduce Lin–Rood launch/synchronization overhead** | [Its horizontal update](/home/cfranken/code/gitHub/AtmosTransportModel/src/Operators/Advection/LinRood.jl:743) contains repeated synchronizations and shared panel scratch; tracers run individually. Panel batching and tracer fusion are promising, but need GPU profiling and conservation/adjoint checks. |

There is also a startup and development-speed opportunity: the [canonical runner](/home/cfranken/code/gitHub/AtmosTransportModel/scripts/run_transport.jl:91) and 91 core test files directly include the package source. Using `using AtmosTransport`, while retaining isolated test globals, would enable package caching and reduce repeated module construction.

Before larger performance changes, improve the benchmark coverage. Its [I/O benchmark](/home/cfranken/code/gitHub/AtmosTransportModel/benchmarking/src/AtmosTransportBenchmarks.jl:358) measures Julia serialization, and its payload retains only the first tracer. Add a representative binary-reader → `DrivenSimulation` → snapshot-writer benchmark, covering all three topologies, tracer counts, allocations, and selected-output modes.

My recommended implementation order is configuration/docs, diagnostic traversal and scratch reuse, output capture, then runtime restructuring and GPU optimization. The measured speedups above are single-thread CPU microbenchmarks on an AMD EPYC; production and GPU speedups remain unmeasured.

## Handoff when work paused

Now is a good time—I’ll pause here.

The runtime file reorganization is saved on disk but not yet tested. Configuration, documentation, and performance fixes are still pending. No long-running commands from this work are active.

After reconnecting in tmux, resume this conversation and say “continue the AtmosTransport fixes.”
