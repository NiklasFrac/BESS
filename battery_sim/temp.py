import math


def validate_thermal_spec(spec: dict) -> None:
    if not math.isfinite(spec["initial_temp_degC"]):
        raise ValueError("initial_temp_degC must be finite.")
    if not math.isfinite(spec["thermal_time_constant_h"]) or spec["thermal_time_constant_h"] <= 0:
        raise ValueError("thermal_time_constant_h must be positive and finite.")
    if not math.isfinite(spec["heat_capacity_kwh_per_degC"]) or spec["heat_capacity_kwh_per_degC"] <= 0:
        raise ValueError("heat_capacity_kwh_per_degC must be positive and finite.")
    if not math.isfinite(spec["heat_to_battery_fraction"]) or not (0.0 <= spec["heat_to_battery_fraction"] <= 1.0):
        raise ValueError("heat_to_battery_fraction must be in [0, 1].")

def step_temperature(
    state: dict,
    spec: dict,
    ambient_temp_degC: float,
    heat_loss_kwh: float,
    dt_h: float,
) -> dict:
    if not math.isfinite(ambient_temp_degC):
        raise ValueError("ambient_temp_degC must be finite.")
    if not math.isfinite(heat_loss_kwh) or heat_loss_kwh < 0:
        raise ValueError("heat_loss_kwh must be non-negative and finite.")
    if not math.isfinite(dt_h) or dt_h <= 0:
        raise ValueError("dt_h must be positive and finite.")

    temp_before = state["battery_temp_degC"]
    if not math.isfinite(temp_before):
        raise ValueError("battery_temp_degC must be finite.")

    heat_to_battery_kwh = heat_loss_kwh * spec["heat_to_battery_fraction"]
    heat_to_battery_kw = heat_to_battery_kwh / dt_h

    thermal_resistance = spec["thermal_time_constant_h"] / spec["heat_capacity_kwh_per_degC"]
    equilibrium_temp = ambient_temp_degC + thermal_resistance * heat_to_battery_kw
    decay = math.exp(-dt_h / spec["thermal_time_constant_h"])
    temp_after = equilibrium_temp + (temp_before - equilibrium_temp) * decay



    return {
        "battery_temp_degC": temp_after,
        "battery_temp_before_degC": temp_before,
        "ambient_temp_degC": ambient_temp_degC,
        "heat_loss_kwh": heat_loss_kwh,
        "heat_to_battery_kwh": heat_to_battery_kwh,
    }
