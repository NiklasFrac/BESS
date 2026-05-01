import math
from typing import Sequence

import rainflow

DAYS_PER_YEAR = 365.25


def validate_degradation_spec(spec: dict) -> None:
    if spec["cycle_fade_per_efc_at_100dod"] < 0:
        raise ValueError("cycle_fade_per_efc_at_100dod must be non-negative.")
    if spec["dod_exponent"] < 0:
        raise ValueError("dod_exponent must be non-negative.")
    if spec["calendar_fade_per_year"] < 0:
        raise ValueError("calendar_fade_per_year must be non-negative.")


def initial_degradation_state() -> dict:
    return {
        "capacity_factor": 1.0,
        "cumulative_efc": 0.0,
        "cycle_fade": 0.0,
        "calendar_fade": 0.0,
        "calendar_days_elapsed": 0.0,
    }


def update_degradation_for_period(
    state: dict,
    spec: dict,
    soc_fraction_series: Sequence[float],
    period_days: float,
) -> dict:
    if not math.isfinite(period_days) or period_days < 0:
        raise ValueError("period_days must be finite and non-negative.")
    for soc in soc_fraction_series:
        if not math.isfinite(soc) or not (0.0 <= soc <= 1.0):
            raise ValueError("soc_fraction_series values must be finite and in [0, 1].")

    capacity_factor_before = state["capacity_factor"]

    old_years = state["calendar_days_elapsed"] / DAYS_PER_YEAR
    new_years = (state["calendar_days_elapsed"] + period_days) / DAYS_PER_YEAR
    period_calendar_fade = (
        spec["calendar_fade_per_year"] * math.sqrt(new_years)
        - spec["calendar_fade_per_year"] * math.sqrt(old_years)
    )

    period_efc = 0.0
    period_cycle_fade = 0.0

    if len(soc_fraction_series) >= 2:
        for dod, count in rainflow.count_cycles(soc_fraction_series):
            if dod <= 0:
                continue
            period_efc += dod * count
            period_cycle_fade += spec["cycle_fade_per_efc_at_100dod"] * count * dod ** spec["dod_exponent"]

    state["cumulative_efc"] += period_efc
    state["cycle_fade"] += period_cycle_fade
    state["calendar_fade"] += period_calendar_fade
    state["calendar_days_elapsed"] += period_days
    state["capacity_factor"] = (1.0 - state["cycle_fade"]) * (1.0 - state["calendar_fade"])

    return {
        "efc": period_efc,
        "cycle_fade": period_cycle_fade,
        "calendar_fade": period_calendar_fade,
        "capacity_factor_before": capacity_factor_before,
        "capacity_factor_after": state["capacity_factor"],
    }