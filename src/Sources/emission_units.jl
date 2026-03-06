# =====================================================================
# Emission unit conventions and conversion constants
#
# Standard emission unit: kg(species) / m² / s
#
# All emission sources must convert to this unit before storage (binary)
# or application (runtime). The emission kernel computes:
#
#   rm += flux[kg/m²/s] × dt[s] × area[m²] × (M_AIR / M_species)
#
# where rm = mixing_ratio × air_mass (prognostic variable).
#
# Conversion constants are verified at module load time using Unitful.jl.
# =====================================================================

using Unitful: @u_str, ustrip

# --- Time constants ---

"""Seconds per Julian year (365.25 days), Unitful-verified."""
const SECONDS_PER_YEAR  = Float64(ustrip(u"s", 1u"yr"))           # 31557600.0

"""Seconds per average month (1/12 Julian year), Unitful-verified."""
const SECONDS_PER_MONTH = Float64(ustrip(u"s", 1u"yr" / 12))     # 2629800.0

# --- Mass constants ---

"""Kilograms per metric tonne (1 Mg = 1000 kg), Unitful-verified."""
const KG_PER_TONNE      = Float64(ustrip(u"kg", 1u"Mg"))          # 1000.0

# --- Molar mass ratios ---

"""kgC → kgCO2 conversion factor (M_CO2 / M_C)."""
const KGC_TO_KGCO2      = 44.01 / 12.011                          # 3.6641

# --- Conversion functions ---

"""
    tonnes_per_year_to_kgm2s(value, area_m2) → Float64

Convert emission rate from Tonnes/year per grid cell to kg/m²/s.

Used by: EDGAR (SF6, CO2, CH4) — raw data in Tonnes/cell/year.
"""
function tonnes_per_year_to_kgm2s(value, area_m2)
    return value * KG_PER_TONNE / (SECONDS_PER_YEAR * area_m2)
end

"""
    kgC_per_m2s_to_kgCO2_per_m2s(value) → same type

Convert carbon mass flux to CO2 mass flux.

Used by: LMDZ posterior fluxes (kgC/m²/s → kgCO2/m²/s).
"""
kgC_per_m2s_to_kgCO2_per_m2s(value) = value * KGC_TO_KGCO2

"""
    kgCO2_per_month_m2_to_kgm2s(value) → same type

Convert monthly CO2 flux density to per-second.

Used by: GridFED (kgCO2/month/m² → kgCO2/m²/s).
"""
kgCO2_per_month_m2_to_kgm2s(value) = value / SECONDS_PER_MONTH

# --- Compile-time verification ---

# These assertions run at module load time and catch any constant errors.
@assert SECONDS_PER_YEAR ≈ 365.25 * 86400
@assert SECONDS_PER_MONTH ≈ 365.25 * 86400 / 12
@assert KG_PER_TONNE == 1000.0
@assert 3.66 < KGC_TO_KGCO2 < 3.67  # sanity check
