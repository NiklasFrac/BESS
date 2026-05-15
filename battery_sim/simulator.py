from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .battery_core import validate_spec, step as step_battery
from .degradation import (
    initial_degradation_state,
    update_degradation_for_period,
    validate_degradation_spec,
)
from .temp import step_temperature, validate_thermal_spec


@dataclass
class BatterySimulationState:
    soc_kwh: float
    battery_temp_degC: float
    nominal_capacity_kwh: float
    capacity_kwh: float
    degradation_state: dict = field(default_factory=initial_degradation_state)
    current_month: tuple[int, int] | None = None
    month_soc: list[float] = field(default_factory=list)
    month_temp: list[float] = field(default_factory=list)
    month_power: list[float] = field(default_factory=list)


def _prepare_action_df(action_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp_utc", "action_kw", "ambient_temp_degC"]
    missing = set(cols) - set(action_df.columns)
    if missing:
        raise ValueError(f"action_df missing columns: {sorted(missing)}")

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

    return df


def initial_simulation_state(
    battery_spec: dict,
    thermal_spec: dict,
    *,
    start_soc_kwh: float | None = None,
) -> BatterySimulationState:
    validate_spec(battery_spec)
    validate_thermal_spec(thermal_spec)

    nominal_capacity_kwh = float(battery_spec["capacity_kwh"])
    soc_kwh = (
        nominal_capacity_kwh * battery_spec["soc_min"]
        if start_soc_kwh is None
        else float(start_soc_kwh)
    )
    if not np.isfinite(soc_kwh) or not (
        nominal_capacity_kwh * battery_spec["soc_min"]
        <= soc_kwh
        <= nominal_capacity_kwh * battery_spec["soc_max"]
    ):
        raise ValueError("start_soc_kwh must satisfy battery SoC limits.")

    return BatterySimulationState(
        soc_kwh=soc_kwh,
        battery_temp_degC=float(thermal_spec["initial_temp_degC"]),
        nominal_capacity_kwh=nominal_capacity_kwh,
        capacity_kwh=nominal_capacity_kwh,
    )


def _close_degradation_period(
    state: BatterySimulationState,
    battery_spec: dict,
    degradation_spec: dict,
    dt_h: float,
    timestamp_utc,
) -> dict | None:
    if not state.month_soc:
        return None
    current_month = state.current_month
    if current_month is None:
        raise RuntimeError("Cannot close degradation period without current_month.")
    period_year, period_month = current_month

    degradation_state, info = update_degradation_for_period(
        state=state.degradation_state,
        spec=degradation_spec,
        soc_fraction_series=state.month_soc,
        battery_temp_degC_series=state.month_temp,
        power_kW_series=state.month_power,
        nominal_capacity_kWh=state.nominal_capacity_kwh,
        period_days=len(state.month_soc) * dt_h / 24.0,
    )

    state.degradation_state = degradation_state
    state.capacity_kwh = (
        state.nominal_capacity_kwh * state.degradation_state["capacity_factor"]
    )

    soc_min = state.capacity_kwh * battery_spec["soc_min"]
    soc_max = state.capacity_kwh * battery_spec["soc_max"]
    state.soc_kwh = min(max(state.soc_kwh, soc_min), soc_max)

    row = {
        "timestamp_utc": pd.to_datetime(timestamp_utc, utc=True),
        "period_year": period_year,
        "period_month": period_month,
        **info,
        **state.degradation_state,
    }

    state.month_soc = []
    state.month_temp = []
    state.month_power = []

    return row


def simulate_period(
    action_df: pd.DataFrame,
    battery_spec: dict,
    thermal_spec: dict,
    degradation_spec: dict,
    dt_h: float,
    state: BatterySimulationState,
    *,
    finalize_period: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BatterySimulationState]:
    dt_h = float(dt_h)
    if not np.isfinite(dt_h) or dt_h <= 0:
        raise ValueError("dt_h must be positive and finite.")

    battery_spec = battery_spec.copy()
    battery_spec["capacity_kwh"] = state.capacity_kwh

    validate_spec(battery_spec)
    validate_thermal_spec(thermal_spec)
    validate_degradation_spec(degradation_spec)

    df = _prepare_action_df(action_df)

    battery_rows, temperature_rows, degradation_rows = [], [], []

    def close_period(timestamp_utc) -> None:
        degradation_row = _close_degradation_period(
            state,
            battery_spec,
            degradation_spec,
            dt_h,
            timestamp_utc,
        )
        if degradation_row is not None:
            degradation_rows.append(degradation_row)

    for row in df.itertuples(index=False):
        ts = row.timestamp_utc
        row_month = (ts.year, ts.month)

        if state.current_month is None:
            state.current_month = row_month
        elif row_month != state.current_month:
            close_period(ts)
            state.current_month = row_month

        battery_temp = state.battery_temp_degC
        battery_spec["capacity_kwh"] = state.capacity_kwh
        core_state = {"soc_kwh": state.soc_kwh}

        result = step_battery(
            state=core_state,
            spec=battery_spec,
            action_kw=float(row.action_kw),
            dt_h=dt_h,
            battery_temp_degC=battery_temp,
        )

        actual_kw = (result["charge_ac_kwh"] - result["discharge_ac_kwh"]) / dt_h

        thermal_result = step_temperature(
            state={"battery_temp_degC": state.battery_temp_degC},
            spec=thermal_spec,
            ambient_temp_degC=float(row.ambient_temp_degC),
            heat_loss_kwh=result["loss_kwh"],
            dt_h=dt_h,
        )

        state.soc_kwh = result["soc_after_kwh"]
        state.battery_temp_degC = thermal_result["battery_temp_degC"]
        soc_fraction = result["soc_after_kwh"] / state.capacity_kwh

        state.month_soc.append(soc_fraction)
        state.month_temp.append(battery_temp)
        state.month_power.append(actual_kw)

        battery_rows.append(
            {
                "timestamp_utc": ts,
                "action_kw": float(row.action_kw),
                "actual_kw": actual_kw,
                "charge_ac_kwh": result["charge_ac_kwh"],
                "discharge_ac_kwh": result["discharge_ac_kwh"],
                "loss_kwh": result["loss_kwh"],
                "soc_kwh": result["soc_after_kwh"],
                "soc_fraction": soc_fraction,
                "capacity_kwh": state.capacity_kwh,
            }
        )

        temperature_rows.append(
            {
                "timestamp_utc": ts,
                "battery_temp_degC": battery_temp,
            }
        )

    if finalize_period:
        close_period(df["timestamp_utc"].iloc[-1] + pd.to_timedelta(dt_h, unit="h"))

    return (
        pd.DataFrame(battery_rows),
        pd.DataFrame(temperature_rows),
        pd.DataFrame(degradation_rows),
        state,
    )


def simulate(
    action_df: pd.DataFrame,
    battery_spec: dict,
    thermal_spec: dict,
    degradation_spec: dict,
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = initial_simulation_state(battery_spec, thermal_spec)
    battery_df, temperature_df, degradation_df, _state = simulate_period(
        action_df=action_df,
        battery_spec=battery_spec,
        thermal_spec=thermal_spec,
        degradation_spec=degradation_spec,
        dt_h=dt_h,
        state=state,
        finalize_period=True,
    )

    return (
        battery_df,
        temperature_df,
        degradation_df,
    )
