from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from .config import (
    ETS_SCOPE_INTRA_EU,
    ETS_SCOPE_EXTRA_EU,
    ETS_SCOPE_BERTH,
    GWP100_CH4,
    GWP100_N2O,
    ets_surrender_factor,
)

@dataclass
class EtsInputs:
    reporting_year: int
    eua_price_eur_per_tco2e: float

    # Fuel consumption in tonnes by voyage bucket (annual totals)
    # buckets: "intra_eu", "extra_eu", "berth"
    consumption_tonnes: Dict[str, Dict[str, float]]  # bucket -> fuel_key -> tonnes

    # Optional: LNG methane fraction for slip conversion (default 1.0 means slip is CH4)
    lng_slip_ch4_mass_fraction: float = 1.0

def _bucket_scope_factor(bucket: str) -> float:
    if bucket == "intra_eu":
        return ETS_SCOPE_INTRA_EU
    if bucket == "extra_eu":
        return ETS_SCOPE_EXTRA_EU
    if bucket == "berth":
        return ETS_SCOPE_BERTH
    raise ValueError(f"Unknown bucket: {bucket}")

def compute_ets(fuels: Dict[str, Dict[str, Any]], inp: EtsInputs) -> Dict[str, Any]:
    """
    EU ETS cost for maritime (2026+):
    - CO2, CH4, N2O in scope
    - Convert CH4, N2O to CO2e using GWP100 (28 and 265)
    - Apply geographic scope factors (1.0 / 0.5 / 1.0)
    - Apply surrender phase-in factor (2026+: 1.0)
    """
    surrender = ets_surrender_factor(inp.reporting_year)

    totals = {
        "co2_t": 0.0,
        "ch4_t": 0.0,
        "n2o_t": 0.0,
        "co2e_t": 0.0,
        "co2e_t_scoped": 0.0,
        "co2e_t_surrender": 0.0,
        "ets_cost_eur": 0.0,
    }

    breakdown = []

    for bucket, by_fuel in inp.consumption_tonnes.items():
        scope = _bucket_scope_factor(bucket)
        for fuel_key, tonnes in by_fuel.items():
            if tonnes <= 0:
                continue
            if fuel_key not in fuels:
                raise KeyError(f"Fuel not found: {fuel_key}")
            f = fuels[fuel_key]

            # Convert tonnes fuel -> grams fuel
            g_fuel = tonnes * 1_000_000.0

            # Combustion / TTW factors (g gas per g fuel)
            co2_g = g_fuel * float(f["cf_co2_g_per_gfuel"])
            ch4_g = g_fuel * float(f.get("cf_ch4_g_per_gfuel", 0.0))
            n2o_g = g_fuel * float(f.get("cf_n2o_g_per_gfuel", 0.0))

            # Slip (as % of mass of fuel used) -> treated as CH4 mass by default
            slip_pct = float(f.get("slip_pct", 0.0))
            if slip_pct > 0:
                slipped_g = g_fuel * (slip_pct / 100.0)
                ch4_g += slipped_g * float(inp.lng_slip_ch4_mass_fraction)

            co2_t = co2_g / 1e6
            ch4_t = ch4_g / 1e6
            n2o_t = n2o_g / 1e6

            co2e_t = co2_t + (ch4_t * GWP100_CH4) + (n2o_t * GWP100_N2O)
            co2e_scoped = co2e_t * scope
            co2e_surrender = co2e_scoped * surrender

            totals["co2_t"] += co2_t
            totals["ch4_t"] += ch4_t
            totals["n2o_t"] += n2o_t
            totals["co2e_t"] += co2e_t
            totals["co2e_t_scoped"] += co2e_scoped
            totals["co2e_t_surrender"] += co2e_surrender

            breakdown.append({
                "bucket": bucket,
                "fuel_key": fuel_key,
                "fuel_name": f["fuel_name"],
                "tonnes_fuel": tonnes,
                "co2_t": co2_t,
                "ch4_t": ch4_t,
                "n2o_t": n2o_t,
                "co2e_t": co2e_t,
                "scope_factor": scope,
                "co2e_t_scoped": co2e_scoped,
                "surrender_factor": surrender,
                "co2e_t_surrender": co2e_surrender,
            })

    totals["ets_cost_eur"] = totals["co2e_t_surrender"] * float(inp.eua_price_eur_per_tco2e)

    return {
        "surrender_factor": surrender,
        "totals": totals,
        "breakdown": breakdown,
    }
