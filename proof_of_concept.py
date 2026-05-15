import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from battery_sim.simulator import initial_simulation_state, simulate_period
from pv_sim.runner import PvSimParams, PvSimPaths, run_pv_sim
from download.run_downloads import main as run_downloads
from evaluation.grid_costs import write_load_grid_costs, write_system_grid_costs
from optimizer.forecast_df import resample_power_temp
from optimizer.optimizer import (
    OptimizerEconomicParams,
    OptimizerInitialStates,
    OptimizerSystemParams,
    optimize_energy_system,
)
from evaluation.result_plots import make_eval_plots


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())
    logging.basicConfig(**cfg["logging"], force=True)
    log = logging.getLogger(__name__)

    path_cfg = cfg["paths"]
    paths = PvSimPaths(
        metadata=repo_root / path_cfg["metadata"],
        meteo=repo_root / path_cfg["meteo"],
        solar=repo_root / path_cfg["solar"],
        horizon=repo_root / path_cfg["pvgis"],
        true_sun_position=repo_root / path_cfg["true_sun_position"],
        apparent=repo_root / path_cfg["apparent"],
        dni=repo_root / path_cfg["dni"],
        poa=repo_root / path_cfg["poa"],
        effective_irradiance=repo_root / path_cfg["effective_irradiance"],
        energy=repo_root / path_cfg["energy"],
        pv_output=repo_root / path_cfg["pv_output"],
        energy_plot=repo_root / path_cfg["energy_plot"],
        horizon_plot=repo_root / path_cfg["horizon_plot"],
    )

    params = PvSimParams(
        station_name=cfg["station"]["name"],
        start_utc=cfg["time"]["start_utc"],
        end_utc=cfg["time"]["end_utc"],
        freq=cfg["time"]["freq"],
        timestamp_col=cfg["dni"]["timestamp_col"],
        missing_value=cfg["dni"]["dwd_missing_value"],
        solar_unit=cfg["dni"]["solar_unit"],
        surface_tilt=cfg["pv"]["surface_tilt"],
        surface_azimuth=cfg["pv"]["surface_azimuth"],
        albedo=cfg["pv"]["albedo"],
        module_pdc0=cfg["pv"]["module_pdc0"],
        module_count=cfg["pv"]["module_count"],
        gamma_pdc=cfg["pv"]["gamma_pdc"],
        annual_age_loss_pct=cfg["losses"]["annual_age_loss_pct"],
        pac0_each=cfg["inverter"]["pac0_each"],
        inverter_count=cfg["inverter"]["inverter_count"],
        eta_inv_nom=cfg["inverter"]["eta_inv_nom"],
    )

    system_params = OptimizerSystemParams(
        dt_h=cfg["time"]["interval_minutes"] / 60,
        e_nom_kwh=cfg["batterie"]["capacity_kwh"],
        soc_min=cfg["batterie"]["soc_min"],
        soc_max=cfg["batterie"]["soc_max"],
        p_grid_max_kw=cfg["grid"]["p_grid_max_kw"],
        p_charge_max_kw=cfg["batterie"]["charge"]["max_kw"],
        p_discharge_max_kw=cfg["batterie"]["discharge"]["max_kw"],
        eta_charge=cfg["batterie"]["charge"]["eta_nominal"],
        eta_discharge=cfg["batterie"]["discharge"]["eta_nominal"],
    )
    economic_params = OptimizerEconomicParams(**cfg["tariff"], **cfg["economics"])

    initial_states = OptimizerInitialStates(
        e_start_kwh=cfg["batterie"]["capacity_kwh"] * cfg["optimizer"]["start_soc"],
        p_peak_year_before_kw=cfg["optimizer"]["p_peak_year_before_kw"],
    )

    log.info("Starte Downloads")
    run_downloads()
    log.info("Starte PV-Simulation")
    run_pv_sim(paths, params, logging_config=cfg["logging"])
    source_min = int(pd.Timedelta(cfg["time"]["freq"]).total_seconds() / 60)
    target_min = cfg["time"]["interval_minutes"]

    forecast_df = resample_power_temp(
        pd.read_csv(paths.pv_output),
        "timestamp_utc",
        "pv_kw",
        "ambient_temp_degC",
        source_min,
        target_min,
    )

    load_df = pd.read_csv(repo_root / cfg["paths"]["single"])
    is_b = load_df["timestamp"].str.endswith(" b")
    load_df["timestamp_utc"] = (
        pd.to_datetime(
            load_df["timestamp"].str.replace(" b", "", regex=False),
            format="%d.%m.%Y %H:%M:%S",
        )
        .dt.tz_localize(
            "Europe/Berlin", ambiguous=(~is_b).to_numpy(), nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )

    forecast_df["merge_key"] = forecast_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")
    load_df["merge_key"] = load_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")

    forecast_df = forecast_df.merge(
        load_df[["merge_key", "load_kw"]],
        on="merge_key",
        how="inner",
    ).drop(columns=["merge_key"])

    battery_state = initial_simulation_state(
        cfg["batterie"],
        cfg["thermal"],
        start_soc_kwh=initial_states.e_start_kwh,
    )
    p_peak_actual_kw = initial_states.p_peak_year_before_kw
    battery_results, temperature_results, degradation_results = [], [], []
    optimizer_dispatch_results = []
    day_groups = list(forecast_df.groupby(forecast_df["timestamp_utc"].dt.date))
    log.info("Starte Optimizer/Batterie-Simulation: %d Tage", len(day_groups))

    for i, (day, day_df) in enumerate(day_groups):
        if i % 30 == 0 or i == len(day_groups) - 1:
            log.info(
                "Fortschritt Optimizer/Batterie: %d/%d Tage", i + 1, len(day_groups)
            )

        result = optimize_energy_system(
            system_params=replace(system_params, e_nom_kwh=battery_state.capacity_kwh),
            economic_params=economic_params,
            initial_states=OptimizerInitialStates(
                e_start_kwh=battery_state.soc_kwh,
                p_peak_year_before_kw=p_peak_actual_kw,
            ),
            forecast_df=day_df,
        )
        optimizer_dispatch_results.append(result["dispatch"])

        action_df = result["action"].copy()
        action_df["ambient_temp_degC"] = day_df["ambient_temp_degC"].to_numpy()

        battery_df, temp_df, degradation_df, battery_state = simulate_period(
            action_df,
            cfg["batterie"],
            cfg["thermal"],
            cfg["degradation"],
            system_params.dt_h,
            battery_state,
            finalize_period=i == len(day_groups) - 1
            or day_groups[i + 1][0].month != day.month,
        )

        realized_df = day_df[["timestamp_utc", "pv_kw", "load_kw"]].merge(
            battery_df[["timestamp_utc", "actual_kw"]],
            on="timestamp_utc",
        )
        p_peak_actual_kw = max(
            p_peak_actual_kw,
            (realized_df["load_kw"] - realized_df["pv_kw"] + realized_df["actual_kw"])
            .clip(lower=0)
            .max(),
        )
        battery_results.append(battery_df)
        temperature_results.append(temp_df)
        degradation_results.append(degradation_df)

    battery_path = repo_root / cfg["paths"]["bat_sim"]
    battery_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_path = repo_root / cfg["paths"]["optimizer_dispatch"]
    optimizer_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Schreibe Ergebnisse")

    pd.concat(optimizer_dispatch_results).to_csv(
        optimizer_path, index=False, float_format="%.3f"
    )
    pd.concat(battery_results).to_csv(battery_path, index=False, float_format="%.3f")
    pd.concat(temperature_results).to_csv(
        repo_root / cfg["paths"]["bat_temp"], index=False, float_format="%.3f"
    )
    pd.concat(degradation_results).to_csv(
        repo_root / cfg["paths"]["bat_degradation"], index=False, float_format="%.3f"
    )

    cost_args = {**cfg["tariff"], "dt_h": system_params.dt_h}
    write_load_grid_costs(
        repo_root / path_cfg["single"],
        repo_root / path_cfg["ems_baseline_dispatch"],
        **cost_args,
    )
    write_system_grid_costs(
        repo_root / path_cfg["single"],
        paths.pv_output,
        battery_path,
        repo_root / path_cfg["ems_system_dispatch"],
        **cost_args,
    )

    log.info("Erzeuge Auswertungen")
    make_eval_plots(
        repo_root / path_cfg["ems_baseline_dispatch"],
        repo_root / path_cfg["ems_system_dispatch"],
        repo_root / path_cfg["costs_plot"],
        repo_root / path_cfg["duration_plot"],
        repo_root / path_cfg["kpi_table_plot"],
    )
    log.info("Fertig")


if __name__ == "__main__":
    main()
