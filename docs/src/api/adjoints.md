# Adjoints and checkpointing API

The adjoint surface covers cubed-sphere objectives, observations, footprints,
preconditioning, and 4D-Var solvers. `Tape` owns checkpoint schedules and
device, pinned-host, and memory-mapped tape storage.

## Adjoints

```@autodocs
Modules = [AtmosTransport.Adjoints]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Tape and checkpointing

```@autodocs
Modules = [AtmosTransport.Tape]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
