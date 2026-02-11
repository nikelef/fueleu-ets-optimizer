from __future__ import annotations

from dataclasses import dataclass

# Regulatory constants and defaults (2026+ focus)

# ETS scope factors (share of emissions in scope)
ETS_SCOPE_INTRA_EU = 1.0     # 100% intra-EU
ETS_SCOPE_BERTH = 1.0        # 100% at berth in EU/EEA
ETS_SCOPE_EXTRA_EU = 0.5     # 50% EU/EEA <-> non-EU/EEA

# ETS surrender phase-in factor by reporting year (emissions year)
def ets_surrender_factor(reporting_year: int) -> float:
    # For completeness; user requested start from 2026 onwards.
    if reporting_year <= 2023:
        return 0.0
    if reporting_year == 2024:
        return 0.40
    if reporting_year == 2025:
        return 0.70
    return 1.00  # 2026+

# GWP100 per Commission Delegated Regulation (EU) 2020/1044
GWP100_CH4 = 28.0
GWP100_N2O = 265.0

# FuelEU reference value (2020 fleet average) used in many summaries: 91.16 gCO2e/MJ
# (The regulation uses a reference value; many official/industry docs cite 91.16.)
FUELEU_REFERENCE_GHG_INTENSITY = 91.16  # gCO2eq/MJ

def fueleu_reduction_percent(reporting_year: int) -> float:
    """
    Stepwise limits:
    - 2% from 1 Jan 2025
    - 6% from 1 Jan 2030
    - 14.5% from 1 Jan 2035
    - 31% from 1 Jan 2040
    - 62% from 1 Jan 2045
    - 80% from 1 Jan 2050
    """
    if reporting_year < 2025:
        return 0.0
    if 2025 <= reporting_year <= 2029:
        return 0.02
    if 2030 <= reporting_year <= 2034:
        return 0.06
    if 2035 <= reporting_year <= 2039:
        return 0.145
    if 2040 <= reporting_year <= 2044:
        return 0.31
    if 2045 <= reporting_year <= 2049:
        return 0.62
    return 0.80  # 2050+

def fueleu_target_intensity(reporting_year: int) -> float:
    red = fueleu_reduction_percent(reporting_year)
    return FUELEU_REFERENCE_GHG_INTENSITY * (1.0 - red)

# FuelEU penalty constants (Annex IV Part B):
# 41,000 MJ per tonne VLSFO equivalent, 2,400 EUR per tonne VLSFO equivalent
FUELEU_PENALTY_MJ_PER_TONNE_VLSFO_EQ = 41_000.0
FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQ = 2_400.0

@dataclass(frozen=True)
class DefaultPrices:
    eua_price_eur_per_tco2e: float = 85.0
    pool_price_eur_per_tco2e: float = 150.0  # internal transfer price for FuelEU pooling (editable)
