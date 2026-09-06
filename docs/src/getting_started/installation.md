# [Installation](@id Installation)

This guide installs AtmosTransport from source and verifies that Julia can load
it. The first simulation on the next page uses the CPU and synthetic data, so
you do not need a GPU, meteorological credentials, Python, or NetCDF command-line
tools.

## 1. Install Julia and Git

AtmosTransport requires Julia 1.10 or later. Install Julia with
[juliaup](https://github.com/JuliaLang/juliaup), the installer recommended by
the Julia project, then open a new terminal and check the version:

```bash
julia --version
```

Any result beginning with `julia version 1.10` or newer is suitable. You also
need [Git](https://git-scm.com/downloads):

```bash
git --version
```

!!! tip "New to terminals or Julia?"
    Read [Julia orientation](@ref Julia-orientation) for the difference between
    terminal commands, the `julia>` prompt, and Julia's `pkg>` prompt. You can
    return here after the two-minute overview.

## 2. Clone the repository

AtmosTransport is not installed from Julia's General registry yet. Clone the
source and enter the repository directory:

```bash
git clone https://github.com/RemoteSensingTools/AtmosTransport.jl.git
cd AtmosTransport.jl
```

All commands in this documentation assume that your terminal remains in this
directory. A quick check is:

```bash
test -f Project.toml && echo "repository root found"
```

On Windows PowerShell, use `Test-Path Project.toml` instead.

## 3. Install the Julia environment

Instantiate the packages declared by `Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The first run downloads packages and precompiles them. It can take several
minutes and may appear quiet for short periods. Later starts reuse the compiled
cache. The transport runner, transport preprocessor, LL-to-CS binary regridder,
CS inversion driver, and binary-to-NetCDF converter import the installed project
package and reuse that cache. Launch these scripts with `--project=.` from this
checkout so Julia selects the matching source and dependencies.

## 4. Verify the package loads

```bash
julia --project=. -e 'using AtmosTransport; println("AtmosTransport is ready")'
```

Expected final line:

```text
AtmosTransport is ready
```

That is enough to continue. The [Quickstart](@ref Quickstart) performs a more
useful end-to-end check by creating a current transport binary, running the
model, and writing NetCDF output.

## Optional: run the regression suite

The complete synthetic CPU suite is intended for contributors and takes much
longer than the load check:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

It requires no private meteorological data. Real-data and large diagnostic
campaigns are separate opt-in test tiers.

## Optional: GPU support

Start on the CPU. Once the quickstart works, a run can request an available
backend through TOML:

```toml
[architecture]
use_gpu = true
backend = "auto"    # or "cuda" / "metal"

[numerics]
float_type = "Float32"
```

- NVIDIA runs use CUDA. `Float32` is the practical production default on most
  GPUs; `Float64` is supported where hardware performance permits.
- Apple Silicon runs use Metal and require `Float32`.
- The runtime fails clearly if a requested GPU backend is unavailable; it does
  not silently change the execution path to CPU.

Backend packages are optional weak dependencies, so a fresh CPU installation
does not fetch them. Add only the backend you need to your Julia user
environment:

```bash
julia -e 'using Pkg; Pkg.add("CUDA")'  # NVIDIA
julia -e 'using Pkg; Pkg.add("Metal")' # Apple Silicon
```

Then diagnose it after the base installation succeeds:

```bash
julia --project=. -e 'using CUDA; CUDA.versioninfo()'
julia --project=. -e 'using Metal; Metal.versioninfo()'
```

Run only the command appropriate for your hardware.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `julia: command not found` | Open a new terminal after installing Julia, or finish the juliaup PATH setup. |
| `Package AtmosTransport not found` | Run from the cloned repository and include `--project=.`. |
| A dependency “does not seem to be installed” | Re-run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. |
| Julia loads the wrong checkout | Run `Base.active_project()` in the REPL; it should end in this repository's `Project.toml`. |
| GPU package fails to load | Keep `[architecture].use_gpu = false` and finish the CPU quickstart before debugging drivers. |

## Next step

Run the [Quickstart](@ref Quickstart). If the syntax is unfamiliar, keep the
[Julia orientation](@ref Julia-orientation) page open alongside it.
