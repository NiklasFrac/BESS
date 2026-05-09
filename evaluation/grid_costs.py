from pathlib import Path
import math

import pandas as pd


def _load_with_utc(
    load_path: Path,
    timestamp_col: str,
    load_col: str,
    timezone: str,
) -> pd.DataFrame:
    df = pd.read_csv(load_path)
    if "timestamp_utc" in df:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    else:
        raw_ts = df[timestamp_col].astype(str)
        ts = pd.to_datetime(
            raw_ts.str.replace(" b", "", regex=False),
            format="%d.%m.%Y %H:%M:%S",
        ).dt.tz_localize(
            timezone,
            ambiguous=(~raw_ts.str.endswith(" b")).to_numpy(),
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    return pd.DataFrame({"timestamp_utc": ts, "load_kw": pd.to_numeric(df[load_col])})


def _add_grid_cost_columns(
    df: pd.DataFrame,
    energy_price_eur_per_kwh: float,
    demand_charge_eur_per_kw_year: float,
    dt_h: float,
) -> pd.DataFrame:
    df["grid_import_kwh"] = df["grid_import_kw"] * dt_h
    df["energy_cost_eur"] = df["grid_import_kwh"] * energy_price_eur_per_kwh
    df["grid_peak_so_far_kw"] = df["grid_import_kw"].cummax()
    df["demand_increment_kw"] = df["grid_peak_so_far_kw"].diff().fillna(df["grid_peak_so_far_kw"]).clip(lower=0)
    df["demand_increment_cost_eur"] = df["demand_increment_kw"] * demand_charge_eur_per_kw_year
    return df


def write_load_grid_costs(
    load_path: Path,
    out_path: Path,
    *,
    energy_price_eur_per_kwh: float,
    demand_charge_eur_per_kw_year: float,
    dt_h: float,
    timestamp_col: str = "timestamp",
    load_col: str = "load_kw",
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    out = _load_with_utc(load_path, timestamp_col, load_col, timezone)
    out["grid_import_kw"] = out["load_kw"].clip(lower=0)
    out = _add_grid_cost_columns(out, energy_price_eur_per_kwh, demand_charge_eur_per_kw_year, dt_h)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.3f")
    return out


def write_system_grid_costs(
    load_path: Path,
    pv_path: Path,
    battery_path: Path,
    out_path: Path,
    *,
    energy_price_eur_per_kwh: float,
    demand_charge_eur_per_kw_year: float,
    dt_h: float,
    timestamp_col: str = "timestamp",
    load_col: str = "load_kw",
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    load = _load_with_utc(load_path, timestamp_col, load_col, timezone)
    pv = pd.read_csv(pv_path, usecols=["timestamp_utc", "pv_kw"])
    pv["timestamp_utc"] = pd.to_datetime(pv["timestamp_utc"], utc=True)
    pv_min = int(pv["timestamp_utc"].sort_values().diff().dt.total_seconds().dropna().mode().iloc[0] / 60)
    target_min = int(round(dt_h * 60))
    base_min = math.gcd(pv_min, target_min)
    pv = pd.concat(
        pv.assign(timestamp_utc=pv["timestamp_utc"] - pd.Timedelta(minutes=pv_min) + pd.Timedelta(minutes=i * base_min))
        for i in range(pv_min // base_min)
    ).set_index("timestamp_utc").resample(f"{target_min}min").mean().dropna().reset_index()
    pv["timestamp_utc"] += pd.Timedelta(minutes=target_min)
    battery = pd.read_csv(battery_path, usecols=["timestamp_utc", "actual_kw"])
    battery["timestamp_utc"] = pd.to_datetime(battery["timestamp_utc"], utc=True)
    out = battery.merge(pv, on="timestamp_utc").rename(columns={"actual_kw": "actual_battery_action_kw"})
    for df in (load, out):
        df["merge_key"] = df["timestamp_utc"].dt.strftime("%m-%d %H:%M")
        df["_n"] = df.groupby("merge_key").cumcount()
    out = out.merge(load[["merge_key", "_n", "load_kw"]], on=["merge_key", "_n"]).drop(columns=["merge_key", "_n"])
    out["grid_import_kw"] = (out["load_kw"] - out["pv_kw"] + out["actual_battery_action_kw"]).clip(lower=0)
    out = _add_grid_cost_columns(out, energy_price_eur_per_kwh, demand_charge_eur_per_kw_year, dt_h)
    out = out[
        [
            "timestamp_utc",
            "load_kw",
            "pv_kw",
            "actual_battery_action_kw",
            "grid_import_kw",
            "grid_import_kwh",
            "energy_cost_eur",
            "grid_peak_so_far_kw",
            "demand_increment_kw",
            "demand_increment_cost_eur",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.3f")
    return out
