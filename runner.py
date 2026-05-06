from pathlib import Path
import logging
import pandas as pd
import yaml

from battery_sim.simulator import simulate as simulate_battery
from download.run_downloads import main as run_downloads
from pv_sim.run_pv import PVRunConfig, PVRunPaths, run_pv
from optimizer.optimizer_core import optimize_energy_system

def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Repo-Root nicht gefunden.")

def build_forecast_df(pv_df: pd.DataFrame, load_df: pd.DataFrame, load_col: str, dt_h: float) -> pd.DataFrame:
    pv = pv_df[["timestamp_utc", "e_net_ac_kwh"]].copy()
    pv["timestamp_utc"] = pd.to_datetime(pv["timestamp_utc"], utc=True)
    pv["pv_kw"] = (pd.to_numeric(pv["e_net_ac_kwh"], errors="coerce").fillna(0.0) / dt_h).clip(lower=0)

    load = load_df[["timestamp_utc", load_col]].copy()
    load["timestamp_utc"] = pd.to_datetime(load["timestamp_utc"], utc=True)
    load = load.rename(columns={load_col: "load_kw"})

    out = pv[["timestamp_utc", "pv_kw"]].merge(
        load[["timestamp_utc", "load_kw"]],
        on="timestamp_utc",
        how="inner",
    )

    out["forecast_source"] = "mock_actual_load"
    return out

def forecast_pv_freq(config: dict) -> str:
    return {
        "hourly": "1h",
        "minutely_15": "15min",
    }.get(config["time"]["freq_fc"], config["time"]["freq_fc"])


def build_pv_config(config: dict, *, freq: str, solar_unit: str) -> PVRunConfig:
    pv = config["pv"]
    inverter = config["inverter"]

    return PVRunConfig(
        station_name=str(config["station"]["name"]),
        start_utc=config["time"]["start_utc"],
        end_utc=config["time"]["end_utc"],
        freq=freq,
        timestamp_col=config["dni"]["timestamp_col"],
        missing_value=float(config["dni"]["dwd_missing_value"]),
        solar_unit=solar_unit,
        surface_tilt=float(pv["surface_tilt"]),
        surface_azimuth=float(pv["surface_azimuth"]),
        albedo=float(pv["albedo"]),
        module_pdc0=float(pv["module_pdc0"]),
        module_count=int(pv["module_count"]),
        gamma_pdc=float(pv["gamma_pdc"]),
        annual_age_loss_pct=float(config["losses"]["annual_age_loss_pct"]),
        pac0_each=float(inverter["pac0_each"]),
        inverter_count=int(inverter["inverter_count"]),
        eta_inv_nom=float(inverter["eta_inv_nom"]),
    )


def build_pv_paths(
    repo_root: Path,
    config: dict,
    *,
    suffix: str | None = None,
    include_plots: bool = False,
) -> PVRunPaths:
    paths = config["paths"]

    def p(name: str) -> Path:
        return repo_root / paths[f"{name}_{suffix}" if suffix else name]

    return PVRunPaths(
        metadata=repo_root / paths["metadata"],
        horizon=repo_root / paths["pvgis"],
        meteo=p("meteo"),
        solar=p("solar"),
        true_sun_position=p("true_sun_position"),
        apparent=p("apparent"),
        dni=p("dni"),
        poa=p("poa"),
        effective_irradiance=p("effective_irradiance"),
        energy=p("energy"),
        horizon_plot=repo_root / paths["horizon_plot"] if include_plots and suffix is None else None,
        energy_plot=p("energy_plot") if include_plots else None,
    )


def build_system_params(config: dict) -> dict[str, float]:
    b = config["batterie"]
    return {
        "dt_h": config["time"]["interval_minutes"] / 60.0,
        "e_nom_kwh": b["capacity_kwh"],
        "soc_min": b["soc_min"],
        "soc_max": b["soc_max"],
        "p_grid_max_kw": config["grid"]["p_grid_max_kw"],
        "p_charge_max_kw": b["charge"]["max_kw"],
        "p_discharge_max_kw": b["discharge"]["max_kw"],
        "eta_charge": b["charge"]["eta_nominal"],
        "eta_discharge": b["discharge"]["eta_nominal"],
    }


def build_economic_params(config: dict) -> dict[str, float]:
    return {
        "energy_price_eur_per_kwh": config["tariff"]["energy_price_eur_per_kwh"],
        "demand_charge_eur_per_kw_year": config["tariff"]["demand_charge_eur_per_kw_year"],
        "battery_replacement_cost_eur": config["economics"]["battery_replacement_cost_eur"],
        "expected_efc": config["economics"]["expected_efc"],
    }


def build_initial_states(config: dict) -> dict[str, float]:
    b = config["batterie"]
    return {
        "e_start_kwh": b["capacity_kwh"] * config["optimizer"]["start_soc"],
        "p_peak_year_before_kw": config["optimizer"]["p_peak_year_before_kw"],
    }

def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    config_path = repo_root / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    log = logging.getLogger(__name__)
    log_cfg = config["logging"]
    logging.basicConfig(
        level=log_cfg["level"],
        format=log_cfg["format"],
        datefmt=log_cfg["datefmt"],
    )

    log.info("=== Full Pipeline Start ===")

    run_downloads()

    log.info("=== PV Actual ===")
    run_pv(
        build_pv_paths(repo_root, config, include_plots=True),
        build_pv_config(
            config,
            freq=config["time"]["freq"],
            solar_unit=config["dni"]["solar_unit"],
        ),
    )

    log.info("=== PV Forecast ===")
    run_pv(
        build_pv_paths(repo_root, config, suffix="fc", include_plots=True),
        build_pv_config(
            config,
            freq=forecast_pv_freq(config),
            solar_unit=config["dni"].get("solar_unit_fc", "wm2"),
        ),
    )
    log.info("=== Optimizer ===")
    pv_fc = pd.read_csv(repo_root / config["paths"]["energy_fc"])
    load = pd.read_csv(repo_root / config["paths"]["raw"])
    print(load.columns.tolist())
    print(load.head())
    
    dt_h = config["time"]["interval_minutes"] / 60.0

    forecast_df = build_forecast_df(
        pv_df=pv_fc,
        load_df=load,
        load_col="LG 1",
        dt_h=dt_h,
    )
    result = optimize_energy_system(
        system_params=build_system_params(config),
        economic_params=build_economic_params(config),
        initial_states=build_initial_states(config),
        forecast_df=forecast_df,
    )

    action_df = result["action"]


    log.info("=== Full Pipeline Done ===")


if __name__ == "__main__":
    main()
