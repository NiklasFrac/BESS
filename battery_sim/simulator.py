
import numpy as np
import pandas as pd

from .battery_core import validate_spec, step as step_battery
from .temp import validate_thermal_spec, step_temperature
from .degradation import (
        validate_degradation_spec,
        initial_degradation_state,
        update_degradation_for_period
        )


def simulate(
    action_df: pd.DataFrame,
    battery_spec: dict,
    thermal_spec: dict,
    degradation_spec: dict,
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    cols = ["timestamp_utc", "action_kw", "ambient_temp_degC"]
    missing = set(cols) - set(action_df.columns)
    if missing:
        raise ValueError(f"action_df missing columns: {sorted(missing)}")

    dt_h = float(dt_h)
    if not np.isfinite(dt_h) or dt_h <= 0:
        raise ValueError("dt_h must be positive and finite.")

    battery_spec = battery_spec.copy()
    thermal_spec = thermal_spec.copy()
    degradation_spec = degradation_spec.copy()

    validate_spec(battery_spec)
    validate_thermal_spec(thermal_spec)
    validate_degradation_spec(degradation_spec)

    df = action_df[cols].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["action_kw"] = pd.to_numeric(df["action_kw"], errors="raise")
    df["ambient_temp_degC"] = pd.to_numeric(df["ambient_temp_degC"], errors="raise")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df.empty:
        raise ValueError("action_df must not be empty.")
    if df["timestamp_utc"].isna().any():
        raise ValueError("timestamp_utc contains invalid timestamps.")
    if not np.isfinite(df["action_kw"]).all():
        raise ValueError("action_kw contains NaN or infinite values.")
    if not np.isfinite(df["ambient_temp_degC"]).all():
        raise ValueError("ambient_temp_degC contains NaN or infinite values.")

    nominal_capacity_kwh = battery_spec["capacity_kwh"]
    degradation_state = initial_degradation_state()
    thermal_state = {"battery_temp_degC": thermal_spec["initial_temp_degC"]}
    state = {"soc_kwh": battery_spec["capacity_kwh"] * battery_spec["soc_min"]}

    battery_rows, temperature_rows, degradation_rows = [], [], []
    month_soc, month_temp, month_power = [], [], []
    current_month = None

    def close_period(ts):
        nonlocal degradation_state, month_soc, month_temp, month_power

        if not month_soc:
            return

        degradation_state, info = update_degradation_for_period(
            state=degradation_state,
            spec=degradation_spec,
            soc_fraction_series=month_soc,
            battery_temp_degC_series=month_temp,
            power_kW_series=month_power,
            nominal_capacity_kWh=nominal_capacity_kwh,
            period_days=len(month_soc) * dt_h / 24.0,
        )

        battery_spec["capacity_kwh"] = nominal_capacity_kwh * degradation_state["capacity_factor"]
        soc_min = battery_spec["capacity_kwh"] * battery_spec["soc_min"]
        soc_max = battery_spec["capacity_kwh"] * battery_spec["soc_max"]
        state["soc_kwh"] = min(max(state["soc_kwh"], soc_min), soc_max)

        degradation_rows.append({
            "timestamp_utc": ts,
            "period_year": current_month[0],
            "period_month": current_month[1],
            **info,
            **degradation_state,
        })

        month_soc, month_temp, month_power = [], [], []

    for row in df.itertuples(index=False):
        ts = row.timestamp_utc
        row_month = (ts.year, ts.month)

        if current_month is None:
            current_month = row_month
        elif row_month != current_month:
            close_period(ts)
            current_month = row_month

        battery_temp = thermal_state["battery_temp_degC"]

        result = step_battery(
            state=state,
            spec=battery_spec,
            action_kw=float(row.action_kw),
            dt_h=dt_h,
            battery_temp_degC=battery_temp,
        )

        actual_kw = (result["charge_ac_kwh"] - result["discharge_ac_kwh"]) / dt_h

        thermal_state = step_temperature(
            state=thermal_state,
            spec=thermal_spec,
            ambient_temp_degC=float(row.ambient_temp_degC),
            heat_loss_kwh=result["loss_kwh"],
            dt_h=dt_h,
        )

        state["soc_kwh"] = result["soc_after_kwh"]
        soc_fraction = result["soc_after_kwh"] / battery_spec["capacity_kwh"]

        month_soc.append(soc_fraction)
        month_temp.append(battery_temp)
        month_power.append(actual_kw)

        battery_rows.append({
            "timestamp_utc": ts,
            "action_kw": float(row.action_kw),
            "actual_kw": actual_kw,
            "charge_ac_kwh": result["charge_ac_kwh"],
            "discharge_ac_kwh": result["discharge_ac_kwh"],
            "loss_kwh": result["loss_kwh"],
            "soc_kwh": result["soc_after_kwh"],
            "soc_fraction": soc_fraction,
            "capacity_kwh": battery_spec["capacity_kwh"],
        })

        temperature_rows.append({
            "timestamp_utc": ts,
            "battery_temp_degC": battery_temp,
        })
    close_period(df["timestamp_utc"].iloc[-1] + pd.to_timedelta(dt_h, unit="h"))

    return (
        pd.DataFrame(battery_rows),
        pd.DataFrame(temperature_rows),
        pd.DataFrame(degradation_rows),
    )