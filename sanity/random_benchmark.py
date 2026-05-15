import argparse
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from battery_sim.simulator import initial_simulation_state, simulate_period
from battery_sim.simulator import BatterySimulationState
from optimizer.forecast_df import resample_power_temp
from optimizer.optimizer import (
    OptimizerEconomicParams,
    OptimizerInitialStates,
    OptimizerSystemParams,
    optimize_energy_system,
)


def _load_cfg(repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())


def _build_system_params(cfg: dict) -> OptimizerSystemParams:
    return OptimizerSystemParams(
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


def _load_forecast_df(repo_root: Path, cfg: dict) -> pd.DataFrame:
    path_cfg = cfg["paths"]
    source_min = int(pd.Timedelta(cfg["time"]["freq"]).total_seconds() / 60)
    target_min = cfg["time"]["interval_minutes"]

    forecast_df = resample_power_temp(
        pd.read_csv(repo_root / path_cfg["pv_output"]),
        "timestamp_utc",
        "pv_kw",
        "ambient_temp_degC",
        source_min,
        target_min,
    )

    load_df = pd.read_csv(repo_root / path_cfg["single"])
    raw_ts = load_df["timestamp"].astype(str)
    is_b = raw_ts.str.endswith(" b")
    load_df["timestamp_utc"] = (
        pd.to_datetime(
            raw_ts.str.replace(" b", "", regex=False),
            format="%d.%m.%Y %H:%M:%S",
        )
        .dt.tz_localize(
            "Europe/Berlin",
            ambiguous=(~is_b).to_numpy(),
            nonexistent="shift_forward",
        )
        .dt.tz_convert("UTC")
    )

    forecast_df["merge_key"] = forecast_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")
    load_df["merge_key"] = load_df["timestamp_utc"].dt.strftime("%m-%d %H:%M")

    return (
        forecast_df.merge(
            load_df[["merge_key", "load_kw"]],
            on="merge_key",
            how="inner",
        )
        .drop(columns=["merge_key"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def _day_groups(df: pd.DataFrame) -> list[tuple[date, pd.DataFrame]]:
    return list(df.groupby(df["timestamp_utc"].dt.date))


def _finalize_month(
    i: int, day: date, day_groups: list[tuple[date, pd.DataFrame]]
) -> bool:
    return i == len(day_groups) - 1 or day_groups[i + 1][0].month != day.month


def _evaluate_run(
    *,
    policy: str,
    seed: int | None,
    forecast_df: pd.DataFrame,
    battery_df: pd.DataFrame,
    cfg: dict,
    system_params: OptimizerSystemParams,
    final_state: BatterySimulationState | None = None,
) -> dict:
    realized = forecast_df[["timestamp_utc", "pv_kw", "load_kw"]].merge(
        battery_df[
            [
                "timestamp_utc",
                "action_kw",
                "actual_kw",
                "charge_ac_kwh",
                "discharge_ac_kwh",
                "loss_kwh",
                "soc_kwh",
            ]
        ],
        on="timestamp_utc",
        how="inner",
    )
    realized["grid_import_kw"] = (
        realized["load_kw"] - realized["pv_kw"] + realized["actual_kw"]
    ).clip(lower=0)
    realized["grid_import_kwh"] = realized["grid_import_kw"] * system_params.dt_h
    realized["energy_cost_eur"] = (
        realized["grid_import_kwh"] * cfg["tariff"]["energy_price_eur_per_kwh"]
    )
    realized["grid_peak_so_far_kw"] = realized["grid_import_kw"].cummax()
    realized["demand_increment_kw"] = (
        realized["grid_peak_so_far_kw"]
        .diff()
        .fillna(realized["grid_peak_so_far_kw"])
        .clip(lower=0)
    )
    realized["demand_increment_cost_eur"] = (
        realized["demand_increment_kw"] * cfg["tariff"]["demand_charge_eur_per_kw_year"]
    )

    action_error_kwh = (
        (realized["action_kw"] - realized["actual_kw"]).abs() * system_params.dt_h
    ).sum()
    ac_throughput_kwh = (
        realized["charge_ac_kwh"].sum() + realized["discharge_ac_kwh"].sum()
    )
    degradation_throughput_kwh = (
        system_params.eta_charge * realized["charge_ac_kwh"].sum()
        + realized["discharge_ac_kwh"].sum() / system_params.eta_discharge
    )
    usable_capacity_kwh = cfg["batterie"]["capacity_kwh"] * (
        cfg["batterie"]["soc_max"] - cfg["batterie"]["soc_min"]
    )
    wear_cost_eur_per_kwh = cfg["economics"]["battery_replacement_cost_eur"] / (
        2.0 * usable_capacity_kwh * cfg["economics"]["expected_efc"]
    )
    battery_wear_proxy_eur = degradation_throughput_kwh * wear_cost_eur_per_kwh
    if final_state is None:
        final_capacity_kwh = np.nan
        final_capacity_loss_kwh = np.nan
        final_capacity_factor = np.nan
        cumulative_efc = np.nan
        cycle_fade = np.nan
        calendar_fade = np.nan
        actual_degradation_cost_eur = np.nan
    else:
        final_capacity_kwh = final_state.capacity_kwh
        final_capacity_loss_kwh = (
            final_state.nominal_capacity_kwh - final_state.capacity_kwh
        )
        final_capacity_factor = final_state.degradation_state["capacity_factor"]
        cumulative_efc = final_state.degradation_state["cumulative_efc"]
        cycle_fade = final_state.degradation_state["cycle_fade"]
        calendar_fade = final_state.degradation_state["calendar_fade"]
        actual_degradation_cost_eur = (1.0 - final_capacity_factor) * cfg["economics"][
            "battery_replacement_cost_eur"
        ]
    grid_cost_eur = (
        realized["energy_cost_eur"].sum() + realized["demand_increment_cost_eur"].sum()
    )

    return {
        "policy": policy,
        "seed": seed,
        "grid_cost_eur": grid_cost_eur,
        "battery_wear_proxy_eur": battery_wear_proxy_eur,
        "total_cost_with_wear_eur": grid_cost_eur + battery_wear_proxy_eur,
        "actual_degradation_cost_eur": actual_degradation_cost_eur,
        "total_cost_with_actual_degradation_eur": grid_cost_eur
        + actual_degradation_cost_eur,
        "energy_cost_eur": realized["energy_cost_eur"].sum(),
        "demand_cost_eur": realized["demand_increment_cost_eur"].sum(),
        "grid_import_kwh": realized["grid_import_kwh"].sum(),
        "grid_peak_kw": realized["grid_import_kw"].max(),
        "battery_ac_throughput_kwh": ac_throughput_kwh,
        "battery_degradation_throughput_kwh": degradation_throughput_kwh,
        "battery_loss_kwh": realized["loss_kwh"].sum(),
        "action_clipped_kwh": action_error_kwh,
        "final_soc_kwh": realized["soc_kwh"].iloc[-1],
        "final_capacity_kwh": final_capacity_kwh,
        "final_capacity_loss_kwh": final_capacity_loss_kwh,
        "final_capacity_factor": final_capacity_factor,
        "cumulative_efc": cumulative_efc,
        "cycle_fade": cycle_fade,
        "calendar_fade": calendar_fade,
    }


def _evaluate_pv_only(
    forecast_df: pd.DataFrame,
    cfg: dict,
    system_params: OptimizerSystemParams,
) -> dict:
    zero_battery = pd.DataFrame(
        {
            "timestamp_utc": forecast_df["timestamp_utc"],
            "action_kw": 0.0,
            "actual_kw": 0.0,
            "charge_ac_kwh": 0.0,
            "discharge_ac_kwh": 0.0,
            "loss_kwh": 0.0,
            "soc_kwh": np.nan,
        }
    )
    return _evaluate_run(
        policy="pv_only_no_battery",
        seed=None,
        forecast_df=forecast_df,
        battery_df=zero_battery,
        cfg=cfg,
        system_params=system_params,
    )


def _run_optimizer_policy(
    forecast_df: pd.DataFrame,
    cfg: dict,
    system_params: OptimizerSystemParams,
    log: logging.Logger,
) -> tuple[dict, pd.DataFrame]:
    economic_params = OptimizerEconomicParams(**cfg["tariff"], **cfg["economics"])
    initial_states = OptimizerInitialStates(
        e_start_kwh=cfg["batterie"]["capacity_kwh"] * cfg["optimizer"]["start_soc"],
        p_peak_year_before_kw=cfg["optimizer"]["p_peak_year_before_kw"],
    )

    battery_state = initial_simulation_state(
        cfg["batterie"],
        cfg["thermal"],
        start_soc_kwh=initial_states.e_start_kwh,
    )
    p_peak_actual_kw = initial_states.p_peak_year_before_kw
    battery_results = []
    dispatch_results = []
    groups = _day_groups(forecast_df)

    for i, (day, day_df) in enumerate(groups):
        if i % 30 == 0 or i == len(groups) - 1:
            log.info("Optimizer-Replay: %d/%d Tage", i + 1, len(groups))

        result = optimize_energy_system(
            system_params=replace(system_params, e_nom_kwh=battery_state.capacity_kwh),
            economic_params=economic_params,
            initial_states=OptimizerInitialStates(
                e_start_kwh=battery_state.soc_kwh,
                p_peak_year_before_kw=p_peak_actual_kw,
            ),
            forecast_df=day_df,
        )
        dispatch_results.append(result["dispatch"])

        action_df = result["action"].copy()
        action_df["ambient_temp_degC"] = day_df["ambient_temp_degC"].to_numpy()
        battery_df, _temp_df, _degradation_df, battery_state = simulate_period(
            action_df,
            cfg["batterie"],
            cfg["thermal"],
            cfg["degradation"],
            system_params.dt_h,
            battery_state,
            finalize_period=_finalize_month(i, day, groups),
        )

        realized = day_df[["timestamp_utc", "pv_kw", "load_kw"]].merge(
            battery_df[["timestamp_utc", "actual_kw"]],
            on="timestamp_utc",
        )
        p_peak_actual_kw = max(
            p_peak_actual_kw,
            (realized["load_kw"] - realized["pv_kw"] + realized["actual_kw"])
            .clip(lower=0)
            .max(),
        )
        battery_results.append(battery_df)

    battery_df = pd.concat(battery_results, ignore_index=True)
    dispatch_df = pd.concat(dispatch_results, ignore_index=True)
    return (
        _evaluate_run(
            policy="optimizer",
            seed=None,
            forecast_df=forecast_df,
            battery_df=battery_df,
            cfg=cfg,
            system_params=system_params,
            final_state=battery_state,
        ),
        dispatch_df,
    )


def _random_day_actions(
    day_df: pd.DataFrame,
    *,
    start_soc_kwh: float,
    capacity_kwh: float,
    battery_spec: dict,
    dt_h: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    soc_kwh = start_soc_kwh
    e_min = capacity_kwh * battery_spec["soc_min"]
    e_max = capacity_kwh * battery_spec["soc_max"]
    eta_charge = battery_spec["charge"]["eta_nominal"]
    eta_discharge = battery_spec["discharge"]["eta_nominal"]

    rows = []
    for row in day_df.itertuples(index=False):
        surplus_kw = max(float(row.pv_kw) - float(row.load_kw), 0.0)
        deficit_kw = max(float(row.load_kw) - float(row.pv_kw), 0.0)

        max_charge_kw = min(
            float(battery_spec["charge"]["max_kw"]),
            surplus_kw,
            max((e_max - soc_kwh) / (eta_charge * dt_h), 0.0),
        )
        max_discharge_kw = min(
            float(battery_spec["discharge"]["max_kw"]),
            deficit_kw,
            max((soc_kwh - e_min) * eta_discharge / dt_h, 0.0),
        )

        if max_charge_kw > 0.0:
            action_kw = float(rng.uniform(0.0, max_charge_kw))
            soc_kwh += action_kw * dt_h * eta_charge
        elif max_discharge_kw > 0.0:
            action_kw = -float(rng.uniform(0.0, max_discharge_kw))
            soc_kwh += action_kw * dt_h / eta_discharge
        else:
            action_kw = 0.0

        soc_kwh = min(max(soc_kwh, e_min), e_max)
        rows.append(
            {
                "timestamp_utc": row.timestamp_utc,
                "action_kw": action_kw,
                "ambient_temp_degC": row.ambient_temp_degC,
            }
        )

    return pd.DataFrame(rows)


def _run_random_policy(
    forecast_df: pd.DataFrame,
    cfg: dict,
    system_params: OptimizerSystemParams,
    *,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    start_soc_kwh = cfg["batterie"]["capacity_kwh"] * cfg["optimizer"]["start_soc"]
    battery_state = initial_simulation_state(
        cfg["batterie"],
        cfg["thermal"],
        start_soc_kwh=start_soc_kwh,
    )
    battery_results = []
    groups = _day_groups(forecast_df)

    for i, (day, day_df) in enumerate(groups):
        action_df = _random_day_actions(
            day_df,
            start_soc_kwh=battery_state.soc_kwh,
            capacity_kwh=battery_state.capacity_kwh,
            battery_spec=cfg["batterie"],
            dt_h=system_params.dt_h,
            rng=rng,
        )
        battery_df, _temp_df, _degradation_df, battery_state = simulate_period(
            action_df,
            cfg["batterie"],
            cfg["thermal"],
            cfg["degradation"],
            system_params.dt_h,
            battery_state,
            finalize_period=_finalize_month(i, day, groups),
        )
        battery_results.append(battery_df)

    return _evaluate_run(
        policy="random_feasible",
        seed=seed,
        forecast_df=forecast_df,
        battery_df=pd.concat(battery_results, ignore_index=True),
        cfg=cfg,
        system_params=system_params,
        final_state=battery_state,
    )


def _summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    optimizer_grid_cost = float(
        metrics_df.loc[metrics_df["policy"] == "optimizer", "grid_cost_eur"].iloc[0]
    )
    optimizer_wear_cost = float(
        metrics_df.loc[
            metrics_df["policy"] == "optimizer", "total_cost_with_wear_eur"
        ].iloc[0]
    )
    optimizer_actual_degradation_cost = float(
        metrics_df.loc[
            metrics_df["policy"] == "optimizer",
            "total_cost_with_actual_degradation_eur",
        ].iloc[0]
    )
    optimizer_final_capacity = float(
        metrics_df.loc[metrics_df["policy"] == "optimizer", "final_capacity_kwh"].iloc[
            0
        ]
    )
    optimizer_capacity_loss = float(
        metrics_df.loc[
            metrics_df["policy"] == "optimizer", "final_capacity_loss_kwh"
        ].iloc[0]
    )
    optimizer_cumulative_efc = float(
        metrics_df.loc[metrics_df["policy"] == "optimizer", "cumulative_efc"].iloc[0]
    )
    random_grid_costs = metrics_df.loc[
        metrics_df["policy"] == "random_feasible", "grid_cost_eur"
    ]
    random_wear_costs = metrics_df.loc[
        metrics_df["policy"] == "random_feasible", "total_cost_with_wear_eur"
    ]
    random_actual_degradation_costs = metrics_df.loc[
        metrics_df["policy"] == "random_feasible",
        "total_cost_with_actual_degradation_eur",
    ]
    random_final_capacities = metrics_df.loc[
        metrics_df["policy"] == "random_feasible", "final_capacity_kwh"
    ]
    random_capacity_losses = metrics_df.loc[
        metrics_df["policy"] == "random_feasible", "final_capacity_loss_kwh"
    ]
    random_cumulative_efcs = metrics_df.loc[
        metrics_df["policy"] == "random_feasible", "cumulative_efc"
    ]
    pv_only_cost = float(
        metrics_df.loc[
            metrics_df["policy"] == "pv_only_no_battery", "grid_cost_eur"
        ].iloc[0]
    )

    return pd.DataFrame(
        [
            {
                "optimizer_grid_cost_eur": optimizer_grid_cost,
                "optimizer_total_cost_with_wear_eur": optimizer_wear_cost,
                "optimizer_total_cost_with_actual_degradation_eur": optimizer_actual_degradation_cost,
                "optimizer_final_capacity_kwh": optimizer_final_capacity,
                "optimizer_capacity_loss_kwh": optimizer_capacity_loss,
                "optimizer_cumulative_efc": optimizer_cumulative_efc,
                "pv_only_cost_eur": pv_only_cost,
                "optimizer_grid_saving_vs_pv_only_eur": pv_only_cost
                - optimizer_grid_cost,
                "random_runs": len(random_grid_costs),
                "random_best_grid_cost_eur": random_grid_costs.min(),
                "random_p05_grid_cost_eur": random_grid_costs.quantile(0.05),
                "random_median_grid_cost_eur": random_grid_costs.median(),
                "random_mean_grid_cost_eur": random_grid_costs.mean(),
                "random_p95_grid_cost_eur": random_grid_costs.quantile(0.95),
                "random_worst_grid_cost_eur": random_grid_costs.max(),
                "optimizer_grid_saving_vs_random_mean_eur": random_grid_costs.mean()
                - optimizer_grid_cost,
                "random_runs_better_than_optimizer": int(
                    (random_grid_costs < optimizer_grid_cost).sum()
                ),
                "optimizer_grid_cost_percentile_vs_random": 100.0
                * (random_grid_costs < optimizer_grid_cost).mean(),
                "random_best_total_cost_with_wear_eur": random_wear_costs.min(),
                "random_median_total_cost_with_wear_eur": random_wear_costs.median(),
                "random_mean_total_cost_with_wear_eur": random_wear_costs.mean(),
                "random_runs_better_than_optimizer_with_wear": int(
                    (random_wear_costs < optimizer_wear_cost).sum()
                ),
                "optimizer_wear_cost_percentile_vs_random": 100.0
                * (random_wear_costs < optimizer_wear_cost).mean(),
                "random_best_total_cost_with_actual_degradation_eur": random_actual_degradation_costs.min(),
                "random_median_total_cost_with_actual_degradation_eur": random_actual_degradation_costs.median(),
                "random_mean_total_cost_with_actual_degradation_eur": random_actual_degradation_costs.mean(),
                "random_runs_better_than_optimizer_with_actual_degradation": int(
                    (
                        random_actual_degradation_costs
                        < optimizer_actual_degradation_cost
                    ).sum()
                ),
                "optimizer_actual_degradation_cost_percentile_vs_random": 100.0
                * (
                    random_actual_degradation_costs < optimizer_actual_degradation_cost
                ).mean(),
                "random_median_final_capacity_kwh": random_final_capacities.median(),
                "random_median_capacity_loss_kwh": random_capacity_losses.median(),
                "random_median_cumulative_efc": random_cumulative_efcs.median(),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = _load_cfg(repo_root)
    logging.basicConfig(**cfg["logging"], force=True)
    log = logging.getLogger(__name__)

    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    system_params = _build_system_params(cfg)
    forecast_df = _load_forecast_df(repo_root, cfg)
    log.info("Benchmark-Zeitreihe: %d Intervalle", len(forecast_df))

    metrics = [_evaluate_pv_only(forecast_df, cfg, system_params)]

    optimizer_metrics, optimizer_dispatch = _run_optimizer_policy(
        forecast_df, cfg, system_params, log
    )
    metrics.append(optimizer_metrics)

    for i in range(args.runs):
        seed = args.seed + i
        if i % 10 == 0 or i == args.runs - 1:
            log.info("Random-Replay: %d/%d Runs", i + 1, args.runs)
        metrics.append(_run_random_policy(forecast_df, cfg, system_params, seed=seed))

    results_dir = repo_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    summary_df = _summary(metrics_df)

    metrics_path = results_dir / "random_benchmark_runs.csv"
    summary_path = results_dir / "random_benchmark_summary.csv"
    optimizer_dispatch_path = results_dir / "random_benchmark_optimizer_dispatch.csv"

    metrics_df.to_csv(metrics_path, index=False, float_format="%.3f")
    summary_df.to_csv(summary_path, index=False, float_format="%.3f")
    optimizer_dispatch.to_csv(optimizer_dispatch_path, index=False, float_format="%.3f")

    log.info("Runs: %s", metrics_path)
    log.info("Summary: %s", summary_path)
    log.info("Optimizer dispatch: %s", optimizer_dispatch_path)
    log.info(
        "Optimizer %.2f EUR Grid, Random-Median %.2f EUR Grid, Random besser: %d/%d",
        summary_df["optimizer_grid_cost_eur"].iloc[0],
        summary_df["random_median_grid_cost_eur"].iloc[0],
        summary_df["random_runs_better_than_optimizer"].iloc[0],
        args.runs,
    )


if __name__ == "__main__":
    main()
