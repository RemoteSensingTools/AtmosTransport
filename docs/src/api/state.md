# State API


For narrative coverage including the dry-basis contract and the
tracer accessor surface, see [State & basis](@ref).

```@autodocs
Modules = [AtmosTransport.State]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```

## Fields

Time-varying and panel-wise field types used by `State` (Kz caches,
profile fields, PBL parameters, …).

```@autodocs
Modules = [AtmosTransport.State.Fields]
Order   = [:module, :constant, :type, :function, :macro]
Private = false
```
