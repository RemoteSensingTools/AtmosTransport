# [Julia orientation](@id Julia-orientation)

You can run AtmosTransport without already knowing Julia. This page explains
the few pieces of Julia syntax and tooling used throughout the documentation.
For a broader introduction, see the official
[Julia getting-started guide](https://docs.julialang.org/en/v1/manual/getting-started/).

## Terminal, Julia prompt, and package prompt

The documentation uses commands from three different prompts:

| Prompt | What it is | Example in these docs |
|---|---|---|
| `$` or a `bash` block | Your operating-system terminal | `julia --project=. examples/generate_synthetic_quickstart.jl` |
| `julia>` | Julia's interactive REPL | `using AtmosTransport` |
| `pkg>` | Julia's package manager, opened by typing `]` in the REPL | `instantiate` |

Do not type the displayed prompt itself. For example, at `julia>` enter only
`using AtmosTransport`.

Start the Julia REPL from a terminal:

```bash
julia --project=.
```

Exit with `Ctrl-D` or `exit()`. Type `?` to enter help mode and `]` to enter
package mode; press Backspace to return to the Julia prompt.

## What `--project=.` means

Julia packages live in **environments**. The repository's `Project.toml`
declares the exact packages AtmosTransport needs. In

```bash
julia --project=. scripts/run_transport.jl my_run.toml
```

`--project=.` says “use the project in the current directory.” Run commands
from the repository root—the directory containing `Project.toml`, `src/`, and
`scripts/`. This keeps AtmosTransport dependencies separate from packages in
your global Julia environment.

The first installation command,

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

asks Julia's built-in package manager to install that environment. The `-e`
flag evaluates the quoted Julia expression and exits. You normally run it once
after cloning and again only when dependencies change.

## Reading common Julia syntax

You will encounter these patterns:

```julia
using AtmosTransport                 # load exported user-facing names

mesh = LatLonMesh(Nx=36, Ny=18)      # construct an object with keyword arguments
scheme = SlopesScheme()               # capitalized names usually construct types
caps = inspect_binary("forcing.bin") # ordinary function call

mass_basis = :dry                     # a Symbol: a lightweight named value
step!(simulation)                     # ! means the function mutates an argument
```

A semicolon separates positional arguments from keyword arguments:

```julia
driver = TransportBinaryDriver(path; FT=Float32, arch=CPU())
```

Julia arrays use 1-based indices. Unicode characters such as `Δt` may appear
in scientific code, but you do not need a special keyboard: type `\Delta` and
press Tab in the Julia REPL or a Julia-aware editor.

## Multiple dispatch in one minute

AtmosTransport uses multiple dispatch instead of classes with methods attached
to objects. Several methods can share a function name, and Julia selects the
method from the argument types:

```julia
apply!(state, forcing, grid, diffusion, Δt)
apply!(state, forcing, grid, convection, Δt)
```

The same idea selects lat-lon, reduced-Gaussian, or cubed-sphere kernels and
CPU or GPU storage. A run TOML is parsed once into concrete Julia types; the
inner loop then dispatches on those types rather than repeatedly interpreting
strings. You do not need to understand the type parameters printed by Julia to
run the model. The [Architecture tour](@ref Architecture-tour) shows where this
matters in the package.

## Scripts, TOML, and library calls

There are two normal ways to use the repository:

1. **Command line (recommended first).** Edit a TOML file and pass it to
   `scripts/run_transport.jl`. TOML is configuration data, not Julia code.
2. **Julia library.** Start with `using AtmosTransport`, construct objects, and
   call `run_driven_simulation` or lower-level functions directly.

The command-line runner uses the same library API—it parses the TOML and calls
`run_driven_simulation`. Start with the CLI, then move to direct Julia calls
when you need a custom experiment.

## Getting help

Inside the Julia REPL:

```julia
?inspect_binary
?TransportModel
```

Useful diagnostics:

```julia
versioninfo()       # Julia, operating system, CPU, and threading information
using Pkg
Pkg.status()        # packages in the active environment
```

If Julia cannot find `AtmosTransport`, first confirm that your terminal is in
the repository root and that Julia was started with `--project=.`.

## Next step

Continue to the [Quickstart](@ref Quickstart). It uses terminal commands and a
TOML file, so no additional Julia syntax is required.
