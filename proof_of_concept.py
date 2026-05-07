from pathlib import Path

import yaml

from pv_sim.runner import PvSimParams, PvSimPaths, run_pv_sim
from download.run_downloads import main as run_downloads

def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

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

    run_downloads()
    run_pv_sim(paths, params)


if __name__ == "__main__":
    main()
