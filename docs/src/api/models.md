# Models API


The `Models` module contains the runtime stepper —
`TransportModel`, `DrivenSimulation`, the IC pipeline, and the
recipe builders that turn a TOML config into a fully-wired runtime
object.

```@autodocs
Modules = [
    AtmosTransport.Models,
    AtmosTransport.Models.InitialConditionIO,
    AtmosTransport.Models.BinaryPathExpander,
    AtmosTransport.Models.InputStaging,
    AtmosTransport.Models.DrivenRunner,
]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
