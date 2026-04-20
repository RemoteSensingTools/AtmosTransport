# ---------------------------------------------------------------------------
# ConvectionForcing — per-window convective mass-flux container.
#
# Plan 18 runtime data flow (see plan 18 v5.1 §2.4 Decision 22 and
# §2.17 Decision 23): convection forcing lives at TWO runtime slots:
#
#   1. On the transport window: `window.convection::Union{Nothing,
#      ConvectionForcing}`, populated by `load_transport_window!`
#      once per met window.
#   2. On the model: `model.convection_forcing::ConvectionForcing`,
#      refreshed every substep by `DrivenSimulation._refresh_forcing!`
#      via `copy_convection_forcing!`.
#
# Both slots carry the same struct type. Capability (which of
# `cmfmc`, `dtrain`, `tm5_fields` are non-nothing) is invariant for
# the lifetime of a run (Decision 27).
#
# Commit 1 ships the minimal struct: two-argument default constructor
# (all-nothing placeholder) and a three-argument positional
# constructor. Commit 2 extends with: invariant-enforcing validating
# constructors, `copy_convection_forcing!`, `allocate_convection_forcing_like`,
# `Adapt.adapt_structure` for GPU transfer, `has_convection_forcing`,
# and the window-struct extensions.
# ---------------------------------------------------------------------------

"""
    ConvectionForcing{CM, DT, TM}

Container for one window (or one substep) of convective mass-flux
forcing. Three optional payload slots:

- `cmfmc :: Union{Nothing, AbstractArray{FT, 3}}` — cloud updraft mass
  flux at level interfaces, shape `(Nx, Ny, Nz+1)`. GCHP / CMFMCConvection
  consumer.
- `dtrain :: Union{Nothing, AbstractArray{FT, 3}}` — detraining mass
  flux at layer centers, shape `(Nx, Ny, Nz)`. When `nothing`,
  `CMFMCConvection` falls through to Tiedtke-style single-flux
  transport.
- `tm5_fields :: Union{Nothing, NamedTuple{(:entu, :detu, :entd, :detd)}}` —
  four-field entrainment/detrainment arrays at layer centers
  `(Nx, Ny, Nz)`. TM5 / `TM5Convection` consumer.

Invariants (Commit 2 will add validating constructors):

- `dtrain !== nothing` requires `cmfmc !== nothing`.
- `tm5_fields !== nothing` and `cmfmc !== nothing` may coexist for
  dual-capability binaries (plan 18 v5.1 Decision 28 — the sim
  selects which capability to consume via the installed operator).
- Capability is INVARIANT for the lifetime of a `DrivenSimulation`
  (plan 18 v5.1 Decision 27). `copy_convection_forcing!` enforces
  strict match.

Default construction: `ConvectionForcing()` produces an all-nothing
placeholder. This is the initial value of `TransportModel.convection_forcing`;
`DrivenSimulation` allocates real buffers at construction (plan 18
v5.1 Decision 26) via `allocate_convection_forcing_like` (shipped
Commit 2).
"""
struct ConvectionForcing{CM, DT, TM}
    cmfmc      :: CM
    dtrain     :: DT
    tm5_fields :: TM
end

# Julia auto-generates the 3-argument outer constructor
# `ConvectionForcing(cmfmc, dtrain, tm5_fields)` from the struct
# definition — re-defining it here would be a method overwrite.
# Commit 2 will add validating inner constructors (plan 18 v5.1
# §2.4 invariants) and `Adapt.adapt_structure`.

# Zero-argument default: all-nothing placeholder used as the initial
# model-side slot before sim construction allocates real buffers
# (plan 18 v5.1 §2.20 Decision 26).
ConvectionForcing() = ConvectionForcing(nothing, nothing, nothing)

"""
    has_convection_forcing(forcing::ConvectionForcing) -> Bool

Whether `forcing` carries any non-nothing payload. Returns `false`
for the all-nothing placeholder. Used by `_refresh_forcing!` as a
gate so the copy path is skipped for models without an active
convection operator.
"""
has_convection_forcing(forcing::ConvectionForcing) =
    forcing.cmfmc !== nothing ||
    forcing.dtrain !== nothing ||
    forcing.tm5_fields !== nothing

export ConvectionForcing, has_convection_forcing
