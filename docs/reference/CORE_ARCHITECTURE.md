# Core architecture contract

AtmosTransport follows a composition-first model similar to Oceananigans:
the grid owns geometry and architecture; the model owns prognostic fields and
operators; operators own their workspaces; drivers expose typed forcing; and
preprocessing produces the exact runtime quantities.

## Stable boundaries

1. `AtmosGrid` combines a horizontal mesh, vertical coordinate, and one
   execution architecture.
2. `CellState` owns air mass and packed tracer mass. Physical basis is encoded
   in types and cannot be inferred from array names.
3. Face-state types encode placement: structured directional faces,
   face-indexed meshes, or cubed-sphere panels.
4. Operators dispatch on grid/state/forcing types. A physics equation must not
   contain backend selection or file-format parsing.
5. Each operator owns scratch required by its algorithm. Scratch is not
   borrowed from an unrelated operator.
6. A meteorological driver yields a typed transport window and declares its
   capabilities. It does not repair obsolete binary semantics.
7. Preprocessing owns unit conversion, basis conversion, balance, and exact
   sub-grid-physics conversion. Runtime consumes those declared quantities.

## Public API rule

Public names describe physical roles, placements, or algorithms. String
configuration is converted to concrete types at the boundary. Compatibility
aliases, ignored keywords, and nominal interfaces that throw for required
operations are not retained while the package has no external users.

Every exported physical routine documents quantity, units, placement, array
shape, vertical orientation, sign, mutation/allocation behavior, conservation
invariant, and its stable upstream reference.

## Dependency direction

```text
Architectures  Grids  Parameters
       \        |       /
              State
                |
          Operators  MetDrivers
                \      /
                Models
                  |
         Preprocessing / Output
```

Preprocessing may use model-independent grid and physics utilities, but the
runtime core never depends on preprocessing. Topology-specific code implements
small dispatched interfaces instead of adding topology branches throughout
physics kernels.

This structure is informed by Oceananigans' explicit field locations,
composed model components, and small model interface:

- <https://clima.github.io/OceananigansDocumentation/stable/fields>
- <https://clima.github.io/OceananigansDocumentation/stable/models/models_overview>
- <https://clima.github.io/OceananigansDocumentation/stable/developer_docs/model_interface>
