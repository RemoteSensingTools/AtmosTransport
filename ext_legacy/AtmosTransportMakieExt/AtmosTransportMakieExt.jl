"""
CairoMakie + GeoMakie extension for AtmosTransport visualization.

Provides implementations for `plot_field`, `plot_output`, and `animate_output`.
Loaded automatically when both CairoMakie and GeoMakie are imported.
"""
module AtmosTransportMakieExt

import AtmosTransport
using AtmosTransport.Visualization: PREDEFINED_DOMAINS

using CairoMakie
using GeoMakie
using NCDatasets
using Dates
using Printf

include("utils.jl")
include("plot_field.jl")
include("plot_output.jl")
include("animate.jl")

end # module AtmosTransportMakieExt
