# ERA5 Spectral Humidity/Pressure Consistency Notes

## Scope

This note documents how ERA5 provides the relevant fields for transport, and how
our current AtmosTransport ERA5 spectral pipeline uses them.

Focus:

- spectral mass-flux inputs (`VO`, `D`, `LNSP`)
- specific humidity (`Q`)
- surface pressure (`SP` vs `exp(LNSP)`)
- timing consistency across advection windows and substeps

## Main Findings

1. ERA5 does not provide all needed fields in spectral form.
   - Spectral: `VO` (138), `D` (155), `LNSP` (152)
   - Gridpoint: `Q` (133), `T` (130), `SP` (134)
2. Current spectral preprocessor uses only `VO/D/LNSP` to generate `m, am, bm, cm, ps`.
   - `ps` is computed as `exp(lnsp)` on the target grid.
   - No `q` is written by default in this preprocessed spectral file.
3. In the current `preprocessed_latlon` runtime path, humidity is optional.
   - `qv` is used only if the file contains one of: `qv`, `QV`, `q`, `specific_humidity`.
4. Timing in the spectral pipeline is analysis-snapshot based.
   - 6-hourly met windows (`met_interval = 21600 s`), with multiple advection substeps per window.
   - The same window fields are reused through all substeps in that window.

## ERA5 Path Goal

**Project goal:** make the ERA5 spectral path hourly and fully validity-time consistent.

Target state:

1. Use hourly ERA5 validity times for the core transport fields (`VO`, `D`, `LNSP`) instead of the current `00/06/12/18` subset.
2. Align humidity (`Q`) and pressure (`PS`/`LNSP`) to the same hourly validity times used for advection.
3. Run with hourly met windows (`met_interval = 3600 s`) while keeping advection substeps independent (`dt` remains configurable).
4. Keep forecast-step products (e.g., convective fluxes) on an explicit mapping/interpolation rule to the hourly analysis timeline.

## Where This Is Defined Locally

### Spectral download (VO/D/LNSP only)

- [scripts/downloads/download_era5_spectral.py](../../scripts/downloads/download_era5_spectral.py)
  - fields and params: lines `9-13`
  - request: `levtype=ml`, `type=an`, `stream=oper`, `param=138/155` and `152`: lines `80-91`, `102-113`, `133-143`, `155-165`

### TM5-style helper download showing spectral vs gridpoint split

- [scripts/downloads/download_era5_grib_tm5.py](../../scripts/downloads/download_era5_grib_tm5.py)
  - spectral VO/D/LNSP requests: lines `64-97`
  - gridpoint `T/Q` (`130/133`) on `N320`: lines `99-119`
  - gridpoint surface pressure `SP` (`134`) on `N320`: lines `121-139`
  - explicit grid naming `T0639` vs `N0320`: lines `14`, `61-62`

### Spectral preprocessing use of fields

- [scripts/preprocessing/preprocess_spectral_massflux.jl](../../scripts/preprocessing/preprocess_spectral_massflux.jl)
  - reads spectral GRIB (`138/155/152`): lines `17-18`
  - maps per-time fields by `dataTime` (00/06/12/18): lines `257-259`
  - computes `sp = exp(lnsp)`: lines `667-670`
  - computes and writes `m, am, bm, cm, ps` only: lines `546-563`, `700-707`

### Runtime use of humidity in preprocessed lat-lon driver

- [src/IO/preprocessed_latlon_driver.jl](../../src/IO/preprocessed_latlon_driver.jl)
  - load `qv` if available; otherwise return false: lines `384-404`

### Spectral CATRINE config timing

- [config/preprocessing/catrine_spectral.toml](../../config/preprocessing/catrine_spectral.toml)
  - `dt = 900 s`, `met_interval = 21600 s`: lines `25-28`

## Timing Semantics

### Analysis fields

For the fields above (`type=an`):

- values are instantaneous analysis snapshots at validity times
- no forecast `step` accumulation semantics are involved

In this pipeline, each 6-hourly snapshot is a met window, and advection substeps
within that window use the same loaded `m/am/bm/cm/ps`.

### Forecast-only fields (if used)

Some fields (for example convective mass flux in current helper scripts) use
`type=fc` and explicit `step` values:

- [scripts/downloads/download_era5_mars_cmfmc.py](../../scripts/downloads/download_era5_mars_cmfmc.py)
  - `type=fc`, `step=3/6/9/12`, `time=06/18`: lines `92-101`

Those should be treated separately from analysis-time fields when building a
fully consistent forcing timeline.

## Recommended Consistency Recipe

1. Keep advection flux construction on one basis:
   - spectral `VO/D/LNSP` at `00/06/12/18` (`type=an`, `levtype=ml`)
2. Keep pressure basis consistent:
   - use `ps = exp(lnsp)` from the same spectral analysis stream
   - do not mix a different-time `SP` source for advection pressure
3. If humidity is needed for dry/moist conversions:
   - retrieve `Q` (`param=133`) at the same validity times (`00/06/12/18`, `type=an`)
   - append/store `q` in preprocessed files so runtime uses matching-time humidity
4. Keep forecast-step fields (e.g., convective fluxes) on an explicit, documented
   interpolation/sampling rule to the same met-window times.

## External ERA5 References

- ECMWF ERA5 data documentation:
  - <https://confluence.ecmwf.int/display/CKB/ERA5%3A%2Bdata%2Bdocumentation>
- ECMWF parameter database (param IDs and names):
  - <https://codes.ecmwf.int/grib/param-db/>
