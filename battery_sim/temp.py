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




    forecast_df = resample_power_temp(
        pd.read_csv(paths.energy),
        "timestamp_utc",
        "p_ac_w",
        "TT_10",
        source_min,
        target_min,
    )
    forecast_df["pv_kw"] = forecast_df["p_ac_w"] / 1000.0
    forecast_df = forecast_df.drop(columns=["p_ac_w"])

    load_df = pd.read_csv(repo_root / cfg["paths"]["single"])
    is_b = load_df["timestamp"].str.endswith(" b")
    load_df["timestamp_utc"] = pd.to_datetime(load_df["timestamp"]
                            .str.replace(" b", "", regex=False),
                            format="%d.%m.%Y %H:%M:%S").dt.tz_localize("Europe/Berlin",
                            ambiguous=(~is_b).to_numpy(),
                            nonexistent="shift_forward").dt.tz_convert("UTC")

    forecast_df["merge_key"] = forecast_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")
    load_df["merge_key"] = load_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")

    forecast_df = forecast_df.merge(
        load_df[["merge_key", "load_kw"]],
        on="merge_key",
        how="inner",
    ).drop(columns=["merge_key"])

    optimizer_results = []

    for _, day_df in forecast_df.groupby(forecast_df["timestamp_utc"].dt.date):

        result = optimize_energy_system(
            system_params=system_params,
            economic_params=economic_params,
            initial_states=initial_states,
            forecast_df=day_df,
        )

        optimizer_results.append(result)

        initial_states = OptimizerInitialStates(
            e_start_kwh=result["e_end_kwh"],
            p_peak_year_before_kw=result["p_peak_new_kw"],
        )
