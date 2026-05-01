from pathlib import Path
import logging

import numpy as np
import pandas as pd
import yaml

try:
    from .battery_core import validate_spec, step as step_battery
    from .temp import validate_thermal_spec, step_temperature
except ImportError:
    from battery_core import validate_spec, step as step_battery
    from temp import validate_thermal_spec, step_temperature


LOGGER = logging.getLogger(__name__)
ENERGY_COLUMNS = ["timestamp_utc", "e_net_ac_kwh", "TT_10"]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with configs/config.yaml and data folder.")


def simulate(repo_root: Path, config: dict) -> tuple[list[dict], list[dict]]:
    dt_h = pd.to_timedelta(config["time"]["freq"]).total_seconds() / 3600

    battery_spec = config["batterie"]
    thermal_spec = config["thermal"]
    validate_spec(battery_spec)

    energy_path = repo_root / config["paths"]["energy"]
    LOGGER.info("Input: %s", energy_path)

    energy_curve = pd.read_csv(energy_path, usecols=ENERGY_COLUMNS)
    energy_curve["e_net_ac_kwh"] = (
        pd.to_numeric(energy_curve["e_net_ac_kwh"], errors="coerce").fillna(0.0)
    )
    energy_curve["ambient_temp_degC"] = pd.to_numeric(
        energy_curve["TT_10"],
        errors="coerce"
    )

    #ÜBERGANGSLÖSUNG!
    energy_curve["ambient_temp_degC"] = energy_curve["ambient_temp_degC"].mask(
        ~np.isfinite(energy_curve["ambient_temp_degC"])
    )
    energy_curve["ambient_temp_degC"] = energy_curve["ambient_temp_degC"].ffill().bfill()
    if not energy_curve.empty and energy_curve["ambient_temp_degC"].isna().any():
        raise ValueError("No finite ambient temperature values available in TT_10.")
    
    state = {"soc_kwh": battery_spec["capacity_kwh"] * battery_spec["soc_min"]}
    battery_rows = []
    temperature_rows = []

    validate_thermal_spec(thermal_spec)
    thermal_state = {"battery_temp_degC": thermal_spec["initial_temp_degC"]}

    for energy_row in energy_curve.itertuples(index=False):
        pv_energy_kwh = energy_row.e_net_ac_kwh
        action_kw = pv_energy_kwh / dt_h
        battery_temp_degC = thermal_state["battery_temp_degC"]

        result = step_battery(
            state=state,
            spec=battery_spec, 
            action_kw=action_kw, 
            dt_h=dt_h,
            battery_temp_degC=battery_temp_degC)

        thermal_state = step_temperature(
            state=thermal_state,
            spec=thermal_spec,
            ambient_temp_degC=energy_row.ambient_temp_degC,        
            heat_loss_kwh=result["loss_kwh"],
            dt_h=dt_h,
        )
        
        state = {"soc_kwh": result["soc_after_kwh"]}

        battery_rows.append({
            "timestamp_utc": energy_row.timestamp_utc,
            "pv_energy_kwh": pv_energy_kwh,
            "action_kw": action_kw,
            "charge_ac_kwh": result["charge_ac_kwh"],
            "discharge_ac_kwh": result["discharge_ac_kwh"],
            "loss_kwh": result["loss_kwh"],
            "soc_kwh": result["soc_after_kwh"],
            "soc_fraction": result["soc_after_kwh"] / battery_spec["capacity_kwh"],
            "charge_allowed": result["charge_allowed"],
            "discharge_allowed": result["discharge_allowed"],
            "charge_temp_limited_ac_kwh": result["charge_temp_limited_ac_kwh"],
            "discharge_temp_limited_ac_kwh": result["discharge_temp_limited_ac_kwh"],
        })

        temperature_rows.append({
            "timestamp_utc": energy_row.timestamp_utc,
            "ambient_temp_degC": energy_row.ambient_temp_degC,
            "battery_temp_before_degC": battery_temp_degC,
            "battery_temp_after_degC": thermal_state["battery_temp_degC"],
            "heat_loss_kwh": result["loss_kwh"],
        })

    
    return battery_rows, temperature_rows


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)

    config_path = repo_root / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
        datefmt=config["logging"]["datefmt"],
    )

    battery_rows, temperature_rows = simulate(repo_root, config)

    battery_results = pd.DataFrame(battery_rows)
    temperature_results = pd.DataFrame(temperature_rows)

    LOGGER.info("Battery rows: %d", len(battery_results))
    LOGGER.info("Temperature rows: %d", len(temperature_results))
    LOGGER.info("Temperature dtypes:\n%s", temperature_results.dtypes)

    if battery_results.empty or temperature_results.empty:
        LOGGER.warning("Simulation fertig, aber Ergebnis ist leer.")
        return

    battery_output_path = repo_root / config["paths"]["bat_sim"]
    temp_output_path = repo_root / config["paths"]["bat_temp"]

    battery_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Schreibe Battery output: %s", battery_output_path)
    battery_results.to_parquet(battery_output_path, index=False)
    LOGGER.info("Battery output fertig: %s", battery_output_path)

    LOGGER.info("Schreibe Temperature output: %s", temp_output_path)
    temperature_results.to_parquet(temp_output_path, index=False)
    LOGGER.info("Temperature output fertig: %s", temp_output_path)

if __name__ == "__main__":
    main()
