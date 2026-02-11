from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

from .config import (
    ETS_SCOPE_INTRA_EU,
    ETS_SCOPE_EXTRA_EU,
    ETS_SCOPE_BERTH,
    fueleu_target_intensity,
    FUELEU_PENALTY_MJ_PER_TONNE_VLSFO_EQ,
    FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQ,
)

@dataclass
class FuelEUInputs:
    reporting_year: int

    # Fuel consumption in tonnes by voyage bucket (annual totals)
    # FuelEU uses same 100%/50% energy scoping concept as ETS in many implementations
    consumption_tonnes: Dict[str, Dict[str, float]]  # bucket -> fuel_key -> tonnes

    # Wind-assist parameters (optional “surprise” policy)
    wind_pwind_over_pprop: float = 0.0  # ratio
    apply_wind_reward: bool = False

def _bucket_scope_factor(bucket: str) -> float:
    if bucket == "intra_eu":
        return ETS_SCOPE_INTRA_EU
    if bucket == "extra_eu":
        return ETS_SCOPE_EXTRA_EU
    if bucket == "berth":
        return ETS_SCOPE_BERTH
    raise ValueError(f"Unknown bucket: {bucket}")

def wind_reward_factor(pwind_over_pprop: float) -> float:
    """
    FuelEU Annex I: reward factor depends on PWind/PProp bands:
    - 0.05 -> 0.99
    - 0.10 -> 0.97
    - >=0.15 -> 0.95
    Otherwise 1.00
    """
    r = max(0.0, pwind_over_pprop)
    if r >= 0.15:
        return 0.95
    if r >= 0.10:
        return 0.97
    if r >= 0.05:
        return 0.99
    return 1.00

def compute_fueleu(fuels: Dict[str, Dict[str, Any]], inp: FuelEUInputs) -> Dict[str, Any]:
    """
    FuelEU:
    - Compute scoped energy (MJ)
    - Compute WtW emissions: WtT (gCO2eq/MJ * MJ) + TtW CO2eq from combustion (derived from Cf_* and slip)
      Here we approximate TtW CO2eq by converting CO2, CH4, N2O to CO2eq using the same GWP100 constants used in ETS,
      but you can replace with certified CO2eq,TtW,j values if you maintain them.
    - Actual GHG intensity = total_gco2eq / total_MJ
    - Target intensity = reference * (1 - reduction%)
    - Compliance balance per Annex IV Part A:
      CB[gCO2eq] = (GHGIE_target - GHGIE_actual) * total_energy_MJ
      (positive => surplus, negative => deficit)
    - Penalty per Annex IV Part B:
      FuelEU Penalty = |CB| / (GHGIE_actual * 41,000) * 2,400
    """
    # Local GWP constants (same as ETS)
    from .config import GWP100_CH4, GWP100_N2O

    target = fueleu_target_intensity(inp.reporting_year)

    total_mj = 0.0
    total_gco2eq = 0.0
    rows = []

    for bucket, by_fuel in inp.consumption_tonnes.items():
        scope = _bucket_scope_factor(bucket)
        for fuel_key, tonnes in by_fuel.items():
            if tonnes <= 0:
                continue
            f = fuels[fuel_key]

            # tonnes -> grams -> MJ
            g_fuel = tonnes * 1_000_000.0
            mj = g_fuel * float(f["lcv_mj_per_g"])
            mj_scoped = mj * scope

            # WtT emissions (gCO2eq/MJ * MJ)
            wtt = float(f["wtt_gco2eq_per_mj"]) * mj_scoped

            # TtW emissions from Cf factors + slip
            co2_g = g_fuel * float(f["cf_co2_g_per_gfuel"])
            ch4_g = g_fuel * float(f.get("cf_ch4_g_per_gfuel", 0.0))
            n2o_g = g_fuel * float(f.get("cf_n2o_g_per_gfuel", 0.0))

            slip_pct = float(f.get("slip_pct", 0.0))
            if slip_pct > 0:
                slipped_g = g_fuel * (slip_pct / 100.0)
                ch4_g += slipped_g  # treat as CH4 by default

            # Convert to CO2eq grams
            ttw_gco2eq = co2_g + (ch4_g * GWP100_CH4) + (n2o_g * GWP100_N2O)
            # Scope the TtW by the same bucket factor
            ttw_gco2eq_scoped = ttw_gco2eq * scope

            gco2eq = wtt + ttw_gco2eq_scoped

            total_mj += mj_scoped
            total_gco2eq += gco2eq

            rows.append({
                "bucket": bucket,
                "fuel_key": fuel_key,
                "fuel_name": f["fuel_name"],
                "tonnes_fuel": tonnes,
                "scope_factor": scope,
                "energy_mj_scoped": mj_scoped,
                "wtt_gco2eq": wtt,
                "ttw_gco2eq": ttw_gco2eq_scoped,
                "wtw_gco2eq": gco2eq,
            })

    if total_mj <= 0:
        raise ValueError("Total scoped energy is zero. Please enter fuel consumption.")

    ghgie_actual = total_gco2eq / total_mj  # gCO2eq/MJ

    # Apply wind reward (reduces the intensity index by multiplying)
    rw = wind_reward_factor(inp.wind_pwind_over_pprop) if inp.apply_wind_reward else 1.0
    ghgie_actual_rewarded = ghgie_actual * rw

    compliance_balance_g = (target - ghgie_actual_rewarded) * total_mj  # gCO2eq (Annex IV Part A)

    penalty_eur = 0.0
    if compliance_balance_g < 0:
        # Annex IV Part B (GHG intensity penalty)
        penalty_eur = (abs(compliance_balance_g) / (ghgie_actual_rewarded * FUELEU_PENALTY_MJ_PER_TONNE_VLSFO_EQ)) \
                      * FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQ

    return {
        "target_gco2eq_per_mj": target,
        "wind_reward_factor": rw,
        "actual_gco2eq_per_mj": ghgie_actual,
        "actual_rewarded_gco2eq_per_mj": ghgie_actual_rewarded,
        "total_energy_mj": total_mj,
        "total_wtw_gco2eq": total_gco2eq,
        "compliance_balance_gco2eq": compliance_balance_g,
        "penalty_eur": penalty_eur,
        "rows": rows,
    }
