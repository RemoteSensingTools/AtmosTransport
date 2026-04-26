# Operators API


For narrative coverage of the operator hierarchy, the `apply!`
contract, and the Strang palindrome, see [Concepts: Operators](@ref).
For per-scheme advection properties, [Theory: Advection schemes](@ref).

## `Operators` (top-level)

```@autodocs
Modules = [AtmosTransport.Operators]
Order   = [:type, :function]
Private = false
```

## Advection

```@autodocs
Modules = [AtmosTransport.Operators.Advection]
Order   = [:type, :function]
Private = false
```

## Convection

```@autodocs
Modules = [AtmosTransport.Operators.Convection]
Order   = [:type, :function]
Private = false
```

## Diffusion

```@autodocs
Modules = [AtmosTransport.Operators.Diffusion]
Order   = [:type, :function]
Private = false
```

## SurfaceFlux

```@autodocs
Modules = [AtmosTransport.Operators.SurfaceFlux]
Order   = [:type, :function]
Private = false
```
