# TM5 local setup for validation

Steps to obtain, build, and run TM5 locally so we can compare outputs with our reference run (Option A). All menial steps are scripted where possible.

## Prerequisites

- **Mercurial (hg)** for cloning (or download a zip snapshot)
- **Intel oneAPI** (`ifx` Fortran compiler, MKL). Activate with `source /opt/intel/oneapi/setvars.sh --force`
- **OpenMPI** (`module load mpi/openmpi-x86_64`)
- **NetCDF-Fortran** compiled with `ifx` (system packages use gfortran `.mod` files which are incompatible). See "NetCDF-Fortran with ifx" below.
- **Python 3.5+** for the Pycasso build system
- **makedepf90** (Fortran dependency scanner). Build from source: `git clone https://github.com/outpaddling/makedepf90.git && cd makedepf90 && make && cp makedepf90 ~/.local/bin/`
- **ecCodes** with Fortran bindings (only needed for meteo preprocessing; see [METEO_PREPROCESSING.md](METEO_PREPROCESSING.md)). Installed with ifx at `~/.local/eccodes/` (version 2.34.0)

**Obtaining TM5 source from SourceForge:**

- **Create a free SourceForge account** (no payment): https://sourceforge.net/user/registration/
  Then run `./scripts/setup_tm5_local.sh`; when `hg clone` asks for credentials, use your account.
- **When hg asks for a username:** Try leaving it blank and pressing Enter (anonymous read).
- **Files page:** The [TM5 Files](https://sourceforge.net/projects/tm5/files/) page has user guide and data tarballs; the **source code** is only in the Mercurial repo.

## 1. Obtain TM5 source

TM5 should end up at `deps/tm5` (or set `TM5_ROOT`).

**Option A – Clone (requires Mercurial):**
```bash
# From project root
./scripts/setup_tm5_local.sh
# Or: hg clone https://hg.code.sf.net/p/tm5/code deps/tm5
```

**Option B – Zip (e.g. SourceForge snapshot):**
```bash
cd deps && unzip -q /path/to/tm5-*.zip
mv tm5-cy3_4dvar-* tm5   # or whatever the single top-level folder is named
# TM5 source is now at deps/tm5
```

Cycle 3 4DVAR (adjoint) repo:
```bash
hg clone https://sourceforge.net/p/tm5/cy3_4dvar/ci/default/tree/ "$TM5_ROOT/cy3_4dvar"
```

## 2. Build NetCDF-Fortran with ifx

System NetCDF-Fortran packages (compiled with gfortran) produce `.mod` files
that are incompatible with Intel `ifx`. Build from source:

```bash
source /opt/intel/oneapi/setvars.sh --force
cd /tmp
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.5.2.tar.gz
tar xzf v4.5.2.tar.gz && cd netcdf-fortran-4.5.2
mkdir build && cd build
FC=ifx CC=icx cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/netcdf-fortran-ifx \
  -DNETCDF_C_LIBRARY=/usr/lib64/libnetcdf.so \
  -DNETCDF_C_INCLUDE_DIR=/usr/include
make -j4 && make install
```

Result: `$HOME/.local/netcdf-fortran-ifx/include/netcdf.mod` and
`$HOME/.local/netcdf-fortran-ifx/lib64/libnetcdff.so`.

## 3. Configure and build TM5

**Machine configuration** is in `deps/tm5/rc/include/machine.linux-ifx.rc`:
- Points NetCDF4 flags at the ifx-compiled install above
- ecCodes include/lib paths for `~/.local/eccodes/`
- Uses `-qmkl=parallel` for MKL (LAPACK95 for convection solver)
- Explicit `-lmkl_lapack95_lp64` in MKL libs (needed for `dgetrf_f95`/`dgetrs_f95`)
- MPI handled by `mpif90` wrapper (OpenMPI)

**Run setup** from project root:

```bash
./scripts/run_tm5_setup.sh [rcfile] [-n]
```

- Default rcfile: `rc/nam1x1-dummy_tr.rc` (dummy tracer, 1x1 deg global)
- `-n` forces a clean rebuild
- The script sources Intel oneAPI and loads OpenMPI automatically
- Executable lands at `/tmp/tm5_cfranken/var4d/dummy_tr/nam1x1/ml137/tropo25a/tm5-var4d.x`

**Known source fixes applied to the cy3_4dvar snapshot:**

| File | Fix | Reason |
|------|-----|--------|
| `base/src/tmm.F90` | Wrapped `TMM_MF_TM5_HDF_Init/Done` calls in `#ifdef with_hdf4` | Calls were inside `#ifdef with_tmm_tm5` but not guarded by `with_hdf4`; caused linker errors without HDF4 |
| `base/src/tmm.F90` | Fixed `IntLat`/`IntLon` calls | Wrapped raw spectral arrays into TshGrid objects to match the generic interface |
| `base/src/advect_tools.F90` | Commented out `use io_hdf` | HDF4 module not available; not actually used in this routine |
| `base/src/file_grib.F90` | Migrated from GRIBEX to ecCodes API | `pbOpen`->`codes_open_file`, `pbGrib`+`GribEx`->`codes_grib_new_from_file`+`codes_get`, `pbWrite`+`GribEx(encode)`->`codes_grib_new_from_samples`+`codes_set`+`codes_write`, `pbClose`->`codes_close_file`; used `use eccodes, only: ...` to avoid clash with TM5's `grib_Check` |
| `base/src/tmm_mf_ecmwf_tmpp.F90`, `tmm_mf_ecmwf_tm5.F90` | Fixed `GetPid` calls | Was called as subroutine but is a function |
| `rc/include/folders.rc` | Added `proj/openmp_speedups` to `my.projects.basic` | Provides `mass_to_pressure` subroutine needed by `modelIntegration.F90` |
| `rc/include/folders.rc` | Added `proj/era5_met` to projects | ERA5 meteo preprocessing support |
| `rc/include/folders.rc` | Changed `my.home` to `${PWD}` | Resolved absolute path issues in the cy3_4dvar snapshot |
| `rc/include/folders.rc` | Simplified `my.projects.output` | cy3_4dvar only has `proj/output/src/`, not `flask/satellite/etc.` |
| `base/py/helper/rc.py` | Allow duplicate keys with same value (warning instead of error) | Widespread duplicate key issue in cy3_4dvar RC files |
| `tm5-expert.rc` | Added `with_grib` library mapping | Links ecCodes for GRIB I/O |

**RC config changes for ERA5+GRIB:**

| File | Change |
|------|--------|
| `machine.linux-ifx.rc` | ecCodes include/lib paths |
| `meteo-ea-nc.rc` | Added `with_tmm_ecmwf with_grib` macros, uncommented diffusion.dir |
| `nam1x1-dummy_tr.rc` | Switched from meteo-ei.rc to meteo-ea-nc.rc, LEVS changed to tropo25a |

## 4. Meteorological data

**TM5 does NOT read standard ERA5 NetCDF files.** It requires preprocessed
meteo with mass fluxes computed from spectral ECMWF data. See
[METEO_PREPROCESSING.md](METEO_PREPROCESSING.md) for:

- Why mass fluxes must be computed in spectral space (Bregman et al. 2003)
- How to run TM5 in preprocessing mode (`tmm.output : T`)
- What ERA5 GRIB fields to download (spectral VO, D, LNSP + gridpoint convective/surface)
- Prerequisites: ecCodes library with Fortran bindings

TM5 compiles and links successfully with full ERA5+GRIB support. Without preprocessed meteo, it cannot run a forward simulation.

**ERA5 GRIB download:** Use `scripts/download_era5_grib_tm5.py` to fetch the required spectral and gridpoint fields.

## 5. Run minimal forward case

Once meteo is preprocessed (step 4):

```bash
cd /tmp/tm5_cfranken/var4d/dummy_tr/nam1x1/ml137/tropo25a
./submit_tm5 ./tm5-var4d.x tm5-var4d.rc
```

- Single tracer, 1-2 days, 1x1 deg global
- Use same time window as our Julia reference run (see REFERENCE_RUN.md)
- TM5 writes output NetCDF to the configured output directory

## 6. Compare with our model

After running both models on the same ERA5 period:

```bash
julia --project=. scripts/compare_tm5_output.jl \
  data/era5/output/reference_era5_output.nc \
  /path/to/tm5_output.nc
```

See `scripts/compare_tm5_output.jl` for required variable names and optional regridding.
