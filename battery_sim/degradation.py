import math
from typing import Sequence

import rainflow

DAYS_PER_YEAR = 365.25

def validate_degradation_spec(spec: dict) -> None:
    for key in (
        "cycle_fade_per_efc_at_100dod",
        "dod_exponent",
        "cycle_reference_temp_degC",  
        "cycle_activation_energy_over_R_K",  
        "calendar_fade_at_1yr",
        "calendar_reference_temp_degC",
        "calendar_activation_energy_over_R_K",
        "calendar_low_soc_reference",
        "calendar_low_soc_factor",
        "calendar_high_soc_reference",
        "calendar_high_soc_factor",
        "c_rate_exponent",
        "c_rate_reference",  
    ):
        if key not in spec:
            raise KeyError(f"Missing degradation spec key: {key}")
        if not math.isfinite(spec[key]):
            raise ValueError(f"{key} must be finite.")

    if not (0 <= spec["cycle_fade_per_efc_at_100dod"] < 1):
        raise ValueError("cycle_fade_per_efc_at_100dod must be in [0, 1).")

    if spec["dod_exponent"] <= 0:
        raise ValueError("dod_exponent must be positive.")

    if spec["cycle_reference_temp_degC"] <= -273.15:
        raise ValueError("cycle_reference_temp_degC must be above absolute zero.")
    if spec["cycle_activation_energy_over_R_K"] < 0:
        raise ValueError("cycle_activation_energy_over_R_K must be non-negative.")

    if not (0 <= spec["calendar_fade_at_1yr"] < 1):
        raise ValueError("calendar_fade_at_1yr must be in [0, 1).")

    if spec["calendar_reference_temp_degC"] <= -273.15:
        raise ValueError("calendar_reference_temp_degC must be above absolute zero.")

    if spec["calendar_activation_energy_over_R_K"] < 0:
        raise ValueError("calendar_activation_energy_over_R_K must be non-negative.")
    if not (0 <= spec["calendar_high_soc_reference"] <= 1):
        raise ValueError("calendar_high_soc_reference must be in [0, 1].")

    if spec["calendar_high_soc_factor"] < 1:
        raise ValueError("calendar_high_soc_factor must be >= 1.")
    if not (0 <= spec["calendar_low_soc_reference"] <= 1):
        raise ValueError("calendar_low_soc_reference must be in [0, 1].")
    if spec["calendar_low_soc_factor"] < 1:
        raise ValueError("calendar_low_soc_factor must be >= 1.")
    
    if spec["calendar_low_soc_reference"] >= spec["calendar_high_soc_reference"]:
        raise ValueError(
            "calendar_low_soc_reference must be < calendar_high_soc_reference"
        )
    if spec["c_rate_exponent"] < 0:
        raise ValueError("c_rate_exponent must be non-negative.")
    if spec["c_rate_reference"] <= 0:
        raise ValueError("c_rate_reference must be positive.")
    if not (0.0 < spec["calendar_low_soc_reference"] < spec["calendar_high_soc_reference"] < 1.0):
        raise ValueError("Require 0 < calendar_low_soc_reference < calendar_high_soc_reference < 1.")

def _soc_calendar_factor(
    soc: float,
    low_soc_reference: float,
    high_soc_reference: float,
    low_soc_factor: float,
    high_soc_factor: float,
) -> float:
    if not math.isfinite(soc) or not (0.0 <= soc <= 1.0):
        raise ValueError("soc must be finite and in [0, 1].")
    
    if soc < low_soc_reference:
        return low_soc_factor - (low_soc_factor - 1.0) * (soc / low_soc_reference)
    
    if soc > high_soc_reference:
        return 1.0 + (high_soc_factor - 1.0) * (soc - high_soc_reference) / (1.0 - high_soc_reference)
    
    return 1.0   

def _arrhenius_factor(
    temp_degC: float,
    reference_temp_degC: float,
    activation_energy_over_R_K: float,
) -> float:
    if not math.isfinite(temp_degC):
        raise ValueError("battery temperature must be finite.")
    if temp_degC <= -273.15:
        raise ValueError("battery temperature must be above absolute zero.")
    
    temp_K = temp_degC + 273.15
    ref_K = reference_temp_degC + 273.15

    return math.exp(activation_energy_over_R_K * (1.0 / ref_K - 1.0 / temp_K))


def initial_degradation_state() -> dict:
    return {
        "capacity_factor": 1.0,
        "cumulative_efc": 0.0,
        "cycle_fade": 0.0,
        "calendar_fade": 0.0,
        "calendar_days_elapsed": 0.0,
    }

def _c_rate_factor(
    mean_power_kW: float,
    nominal_capacity_kWh: float,
    c_rate_reference: float,
    c_rate_exponent: float,
) -> float:
    if not math.isfinite(nominal_capacity_kWh) or nominal_capacity_kWh <= 0:
        raise ValueError("nominal_capacity_kWh must be positive and finite.")
    
    c_rate = abs(mean_power_kW) / nominal_capacity_kWh
    
    if c_rate <= 0:
        return 1.0
    
    return max(1.0, (c_rate / c_rate_reference) ** c_rate_exponent)


def update_degradation_for_period(
    state: dict,
    spec: dict,
    soc_fraction_series: Sequence[float],
    battery_temp_degC_series: Sequence[float],
    power_kW_series: Sequence[float], 
    nominal_capacity_kWh: float,  
    period_days: float,
) -> tuple[dict, dict]:
    if not math.isfinite(period_days) or period_days < 0:
        raise ValueError("period_days must be finite and non-negative.")
    for soc in soc_fraction_series:
        if not math.isfinite(soc) or not (0.0 <= soc <= 1.0):
            raise ValueError("soc_fraction_series values must be finite and in [0, 1].")
    if len(battery_temp_degC_series) == 0:
        raise ValueError("battery_temp_degC_series must not be empty.")
    for temp in battery_temp_degC_series:
        if not math.isfinite(temp) or temp <= -273.15:
            raise ValueError("battery_temp_degC_series values must be finite and above absolute zero.")
    if len(soc_fraction_series) == 0:
        raise ValueError("soc_fraction_series must not be empty.")
    if len(soc_fraction_series) != len(battery_temp_degC_series):
        raise ValueError("soc_fraction_series and battery_temp_degC_series must have same length.")
    if len(power_kW_series) != len(soc_fraction_series):
        raise ValueError("power_kW_series and soc_fraction_series must have same length.")
    for p in power_kW_series:
        if not math.isfinite(p):
            raise ValueError("power_kW_series values must be finite.")
    if not math.isfinite(nominal_capacity_kWh) or nominal_capacity_kWh <= 0:
        raise ValueError("nominal_capacity_kWh must be positive and finite.")
    

    validate_degradation_spec(spec)
    new_state = state.copy()
    capacity_factor_before = new_state["capacity_factor"]

    old_years = state["calendar_days_elapsed"] / DAYS_PER_YEAR
    new_years = (state["calendar_days_elapsed"] + period_days) / DAYS_PER_YEAR
    base_calendar_fade = (
        spec["calendar_fade_at_1yr"] * math.sqrt(new_years)
        - spec["calendar_fade_at_1yr"] * math.sqrt(old_years)
    )

    mean_calendar_stress_factor = sum(
        _arrhenius_factor(
            temp,
            spec["calendar_reference_temp_degC"],
            spec["calendar_activation_energy_over_R_K"],
        )
        * _soc_calendar_factor(
            soc,
            spec["calendar_low_soc_reference"],  
            spec["calendar_high_soc_reference"],      
            spec["calendar_low_soc_factor"],      
            spec["calendar_high_soc_factor"],
        )
        for soc, temp in zip(soc_fraction_series, battery_temp_degC_series)
    ) / len(soc_fraction_series)

    period_calendar_fade = base_calendar_fade * mean_calendar_stress_factor

    period_efc = 0.0
    period_cycle_fade = 0.0

    if len(soc_fraction_series) >= 2:
        for rng, _mean, count, i_start, i_end in rainflow.extract_cycles(soc_fraction_series):
            dod = rng
            if dod <= 0:
                continue

            cycle_temps = battery_temp_degC_series[i_start:i_end+1]
            mean_cycle_temp_degC = sum(cycle_temps) / len(cycle_temps)
            cycle_powers = power_kW_series[i_start:i_end+1]
            mean_cycle_power_kW = sum(abs(p) for p in cycle_powers) / len(cycle_powers)

            cycle_temp_factor = _arrhenius_factor(
                mean_cycle_temp_degC,
                spec["cycle_reference_temp_degC"],
                spec["cycle_activation_energy_over_R_K"],
            )
            cycle_c_rate_factor = _c_rate_factor(
                mean_cycle_power_kW,
                nominal_capacity_kWh,
                spec["c_rate_reference"],
                spec["c_rate_exponent"],
            )
            period_efc += dod * count
            period_cycle_fade += (
                spec["cycle_fade_per_efc_at_100dod"] 
                * count 
                * dod ** spec["dod_exponent"]
                * cycle_temp_factor
                * cycle_c_rate_factor
            )

    new_state["cumulative_efc"] += period_efc
    new_state["cycle_fade"] += period_cycle_fade
    new_state["calendar_fade"] += period_calendar_fade
    new_state["calendar_days_elapsed"] += period_days
    new_state["capacity_factor"] = max(
        0.0,
        (1.0 - new_state["cycle_fade"]) * (1.0 - new_state["calendar_fade"])
    )

    info = {
        "efc": period_efc,
        "cycle_fade": period_cycle_fade,
        "calendar_fade": period_calendar_fade,
        "mean_calendar_stress_factor": mean_calendar_stress_factor,
        "capacity_factor_before": capacity_factor_before,
        "capacity_factor_after": new_state["capacity_factor"],
    }

    return new_state, info