# TM5 Advection: Tracer Mass vs Mixing Ratio and Multi-Tracer Handling

Date: 2026-03-14

## Question

Does TM5 advection use tracer mass or mixing ratio, and how are multiple tracers handled?

## Short answer

TM5 advection is mass-based. It advects and updates:

- air mass `m`
- tracer mass `rm`
- tracer slope moments (`rxm`, `rym`, `rzm`)

Mixing ratio is treated as a derived quantity (`rm/m`), not the transported prognostic variable in advection.

## Evidence in code

### 1. X-direction advection (`dynamu`)

Pointers are set to air mass and tracer mass fields:

- `m => m_dat(region)%data`
- `rm => mass_dat(region)%rm_t`

File: `deps/tm5/base/src/advectx.F90:523-527`

Air mass update is done explicitly:

- `mnew = m + am(in) - am(out)`

File: `deps/tm5/base/src/advectx.F90:629-631`

Tracer flux uses mass-based Courant scaling and tracer mass:

- `alpha = am/m`
- `f = alpha*(rm + ...*rxm)`

File: `deps/tm5/base/src/advectx.F90:663-665`

Tracer mass is updated from flux divergence:

- `rm = rm + (f_left - f_right)`

File: `deps/tm5/base/src/advectx.F90:706`

Then new air mass is stored:

- `m = mnew`

File: `deps/tm5/base/src/advectx.F90:733-734`

### 2. Y-direction advection

Same pattern: tracer fluxes and tendencies are computed on `rm` with `beta = bm/m`, then `rm` is updated by flux divergence.

File: `deps/tm5/base/src/advecty.F90:503-506, 619`

### 3. Z-direction advection

In the 1D column kernel:

- `mnew(l) = m(l) + cm(l-1) - cm(l)`
- `f(l,n)` computed from `rm` and `rzm`
- `rm(l,n) = rm(l,n) + f(l-1,n) - f(l,n)`
- `m = mnew`

File: `deps/tm5/base/src/advectz.F90:441, 463-464, 480, 496`

Mixing ratio appears explicitly as a diagnostic print:

- `maxval(rm/m)`

File: `deps/tm5/base/src/advectz.F90:386`

## How TM5 handles multiple tracers

### Local data layout

Tracer mass arrays are 4D with tracer as the last dimension and local tracer count `nt`:

- `rm_t(..., nt)`, `rxm_t(..., nt)`, `rym_t(..., nt)`, `rzm_t(..., nt)`

File: `deps/tm5/base/src/global_data.F90:320-323`

Under MPI, `nt = ntracetloc`:

File: `deps/tm5/base/src/global_data.F90:316`

### Advection loops over local tracers

Each direction loops over local tracer index:

- `do n=1,ntracetloc`

Files:

- `deps/tm5/base/src/advectx.F90:638`
- `deps/tm5/base/src/advecty.F90:494`
- `deps/tm5/base/src/advectz.F90:451`

Ranks with no local tracers return early:

- `if ( ntracetloc == 0 ) return`

Files:

- `deps/tm5/base/src/advectx.F90:150`
- `deps/tm5/base/src/advecty.F90:148`
- `deps/tm5/base/src/advectz.F90:116`

### MPI tracer decomposition

Tracer counts are partitioned across ranks:

- `determine_lmar(ntracet_ar, npes, ntracet)`
- `ntracetloc = ntracet_ar(myid)`

File: `deps/tm5/base/src/mpi_comm.F90:571, 578`

Global-to-local mapping and rank ownership:

- `tracer_loc(n)` gives local index for active global tracer `n`
- `proc_tracer(...)` gives rank that owns each tracer

File: `deps/tm5/base/src/mpi_comm.F90:663-683`

Tracer communicator is created only for ranks with `ntracetloc /= 0`:

File: `deps/tm5/base/src/mpi_comm.F90:716-729`

## Practical implication

For consistency work against other models or meteo products, TM5 advection state should be interpreted as tracer mass transport coupled to air-mass transport, with mixing ratio reconstructed diagnostically when needed.
