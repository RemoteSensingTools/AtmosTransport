# TM5 4D-Var Dry VMR Audit

## Conclusion

TM5 4D-Var does contain dry-air handling, but not as a single consistent model-output conversion used for all observation comparisons.

- `live behavior`: the generic Fortran sampling/output paths still write modeled mixing ratios as `rm / m`, where `m` is total model air mass, not dry air mass.
- `live behavior`: real dry-air logic exists in parts of the satellite/XCO2 observation stack, especially tracer-specific Python code and some f2py averaging-kernel operators.
- `stale/dead config`: the advertised `output.point.dryair.mixratio` and `output.satellite.dryair.mixratio` controls are not wired into a universal live dry-air output path.
- `likely mismatch`: some satellite operators appear to combine wet model profiles from the track file with dry priors or dry retrieval quantities.

## Generic Wet `rm / m` Paths

These are the main live model-sampling paths in `deps/tm5-cy3-4dvar`, and they all sample modeled mole fractions as `rm / m`:

- Satellite sampling:
  - [`user_output_satellite.F90:833`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L833)
  - [`user_output_satellite.F90:857`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L857)
  - Column mean `sum_rm / sum_m` at [`user_output_satellite.F90:877`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L877)
- Flask / aircraft / point-style sampling:
  - [`user_output_flask.F90:1318`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_flask.F90#L1318)
  - Linear interpolation still uses `rm(...) / m(...)` at [`user_output_flask.F90:1325`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_flask.F90#L1325)
- Station time series:
  - [`user_output_station.F90:730`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_station.F90#L730)
  - Linear interpolation still uses `rm(...) / m(...)` at [`user_output_station.F90:738`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_station.F90#L738)
- TCCON profile output:
  - [`user_output_tccon.F90:559`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_tccon.F90#L559)
  - Linear interpolation still uses `rm(...) / m(...)` at [`user_output_tccon.F90:566`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_tccon.F90#L566)

For in-situ/point mismatch generation, the Python point stack then directly compares observed `mixing_ratio` with the modeled `mixing_ratio` from the track file:

- Observations read from input file at [`PointObs_base.py:126`](../deps/tm5-cy3-4dvar/base/py/main/PointObs_base.py#L126)
- Modeled values read from track file at [`PointObs_base.py:174`](../deps/tm5-cy3-4dvar/base/py/main/PointObs_base.py#L174)
- Mismatch formed directly at [`PointObs_base.py:236`](../deps/tm5-cy3-4dvar/base/py/main/PointObs_base.py#L236)

There is no live humidity-based dry-air correction in that generic point path.

## Where Dry-Air Logic Exists

Dry-air concepts do exist in the 4D-Var code, but mainly in the satellite observation operator stack rather than in the generic model-output samplers.

### Satellite track humidity ingestion

- The satellite base class reads `specific_humidity` from the track-file `meteo` group and converts it to a water mole fraction proxy at [`Satellite_base.py:446`](../deps/tm5-cy3-4dvar/base/py/main/Satellite_base.py#L446).

### TCCON CO2 operator

- The CO2 TCCON operator explicitly states that the prior is given in wet-air mixing ratios and converts it to dry-air mixing ratios at:
  - [`Satellite.py:587`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Satellite.py#L587)
  - [`Satellite.py:598`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Satellite.py#L598)
  - [`Satellite.py:600`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Satellite.py#L600)

### CO2 satellite observation builders

- The CO2 observation builders label many retrieved quantities as dry-air mole fraction:
  - [`Observations.py:2520`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Observations.py#L2520)
  - [`Observations.py:2532`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Observations.py#L2532)
  - [`Observations.py:6857`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Observations.py#L6857)

### f2py averaging-kernel operators

- The TROPOMI/CH4 averaging-kernel interface explicitly accepts dry-air subcolumns:
  - [`averaging_kernels.pyf:86`](../deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.pyf#L86)
  - [`averaging_kernels.F90:571`](../deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90#L571)
- Inside the operator, retrieval pressure thickness is reconstructed from dry-air subcolumns at:
  - [`averaging_kernels.F90:635`](../deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90#L635)
  - [`averaging_kernels.F90:637`](../deps/tm5-cy3-4dvar/base/f2py/src/averaging_kernels.F90#L637)

## Dead Or Stale `dryair` Flags

Some rc/config knobs advertise dry-air output support, but the live code does not provide a universal implementation.

- Point rc flag exists in config:
  - [`point_parameters.rc:8`](../deps/tm5-cy3-4dvar/rc/include/point_parameters.rc#L8)
- Satellite rc flag exists in config:
  - [`satellite_parameters_meteo.rc:7`](../deps/tm5-cy3-4dvar/rc/include/satellite_parameters_meteo.rc#L7)
- Satellite code mentions a dry-air sampling flag in comments, but the variable is commented out:
  - [`user_output_satellite.F90:53`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L53)
  - [`user_output_satellite.F90:88`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L88)
- The adjoint path also has the `ReadRc` call commented out:
  - [`adj_user_output_satellite.F90:62`](../deps/tm5-cy3-4dvar/proj/output/src/adj_user_output_satellite.F90#L62)

I did not find active point/station/flask code that consumes `output.point.dryair.mixratio`, and I did not find active satellite Fortran code that turns `output.satellite.dryair.mixratio` into a general dry-air track-file output mode.

## Humidity Timing

When humidity is written to observation track/output files, it is time-weighted over the sampling window, not endpoint-interpolated.

- Satellite path:
  - Accumulation with `weight` at [`user_output_satellite.F90:962`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L962)
  - Averaging by total weight at [`user_output_satellite.F90:1022`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L1022)
  - Output written at [`user_output_satellite.F90:1153`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L1153)
- Flask path:
  - Accumulation at [`user_output_flask.F90:1410`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_flask.F90#L1410)
  - Averaging at [`user_output_flask.F90:864`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_flask.F90#L864)
- Station path:
  - Accumulation at [`user_output_station.F90:805`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_station.F90#L805)
  - Averaging at [`user_output_station.F90:468`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_station.F90#L468)
- TCCON path:
  - Accumulation at [`user_output_tccon.F90:617`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_tccon.F90#L617)
  - Averaging through the per-step weighted accumulators before write-out, matching the same pattern as the satellite path

## Likely Basis Mismatches

### Wet model profiles versus dry priors / dry observations

- `live behavior`: the Fortran satellite sampler writes model profiles as wet `rm / m` in [`user_output_satellite.F90:1126`](../deps/tm5-cy3-4dvar/proj/output/src/user_output_satellite.F90#L1126).
- `live behavior`: the CO2/TCCON Python operator dry-corrects the prior using water information at [`Satellite.py:600`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Satellite.py#L600).
- `likely mismatch`: the same operator uses `squeezed_model_profile` directly in the modeled column calculation at [`Satellite.py:601`](../deps/tm5-cy3-4dvar/proj/tracer/CO2/py/Satellite.py#L601), so the model profile basis may remain wet while the prior/measurement basis is dry.

### Approximate water-mole-fraction conversion

- `likely mismatch`: [`Satellite_base.py:447`](../deps/tm5-cy3-4dvar/base/py/main/Satellite_base.py#L447) converts specific humidity to `mole_frac_water` using `(28.94/18.0152) * q`.
- If `q` is true specific humidity (`kg_H2O / kg_moist_air`), the exact dry-air conversion would involve `q / (1 - q)`. The current expression is therefore an approximation, not an exact dry-air conversion.

## Implications For AtmosTransport

The current Julia codebase already has explicit dry-air conversion machinery:

- LL dry-mass setup in [`src/Models/run_loop.jl:148`](../src/Models/run_loop.jl#L148)
- LL output conversion to dry VMR in [`src/Models/physics_phases.jl:1736`](../src/Models/physics_phases.jl#L1736)
- LL dry-air correction kernels in [`src/Advection/latlon_dry_air.jl:24`](../src/Advection/latlon_dry_air.jl#L24)

That means the local Julia repo is already capable of deliberate dry-basis behavior in a way that the generic TM5 4D-Var model-output comparison stack is not.

The clean comparison to carry forward is:

- TM5 forward transport and generic sampling are primarily moist/total-air-mass based.
- TM5 4D-Var dry-air handling is path-specific and concentrated in satellite/XCO2 operator code.
- If AtmosTransport wants TM5 parity, the default reference is moist transport with `rm / m`.
- If AtmosTransport wants deliberate dry transport or dry output everywhere, that is a separate design choice and should not be described as default TM5 behavior.

