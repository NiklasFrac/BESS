from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pv_sim.compute_dni import compute_dni
from pv_sim.compute_effective_irradiance import compute_effective_irradiance
from pv_sim.compute_poa import compute_poa
from pv_sim.modul_sim import compute_energy
from pv_sim.seen_pos import compute_apparent_sun_position
from pv_sim.true_pos import compute_true_sun_position
from pv_sim.visualization.energy_prod_visual import plot_energy_overview
from pv_sim.visualization.horizon_visual import plot_horizon_profile

PathLike = str | Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PVRunPaths:
    metadata: PathLike
    meteo: PathLike
    solar: PathLike
    horizon: PathLike
    true_sun_position: PathLike
    apparent: PathLike
    dni: PathLike
    poa: PathLike
    effective_irradiance: PathLike
    energy: PathLike
    horizon_plot: PathLike | None = None
    energy_plot: PathLike | None = None

    def __post_init__(self) -> None:
        for name in (
            "metadata",
            "meteo",
            "solar",
            "horizon",
            "true_sun_position",
            "apparent",
            "dni",
            "poa",
            "effective_irradiance",
            "energy",
            "horizon_plot",
            "energy_plot",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True)
class PVRunConfig:
    station_name: str
    start_utc: str
    end_utc: str
    freq: str
    timestamp_col: str
    missing_value: float
    solar_unit: str
    surface_tilt: float
    surface_azimuth: float
    albedo: float
    module_pdc0: float
    module_count: int
    gamma_pdc: float
    annual_age_loss_pct: float
    pac0_each: float
    inverter_count: int
    eta_inv_nom: float


def _validate_run_config(config: PVRunConfig) -> None:
    if config.solar_unit not in {"jcm2", "wm2"}:
        raise ValueError("solar_unit must be 'jcm2' or 'wm2'.")
    if config.module_count <= 0:
        raise ValueError("module_count must be greater than 0.")
    if config.inverter_count <= 0:
        raise ValueError("inverter_count must be greater than 0.")
    if config.freq.strip() == "":
        raise ValueError("freq must not be empty.")


def run_pv(
    paths: PVRunPaths,
    config: PVRunConfig,
    *,
    show_plots: bool = False,
) -> None:
    _validate_run_config(config)

    log.info("=== True Solar Position ===")
    compute_true_sun_position(
        metadata_path=paths.metadata,
        out_path=paths.true_sun_position,
        station_name=config.station_name,
        start_utc=config.start_utc,
        end_utc=config.end_utc,
        freq=config.freq,
    )

    log.info("=== Apparent Solar Position ===")
    compute_apparent_sun_position(
        meteo_path=paths.meteo,
        true_solar_path=paths.true_sun_position,
        out_path=paths.apparent,
        freq=config.freq,
        missing_value=float(config.missing_value),
    )

    log.info("=== DNI ===")
    compute_dni(
        solar_path=paths.solar,
        sun_position_path=paths.true_sun_position,
        out_path=paths.dni,
        ts_col=config.timestamp_col,
        missing=float(config.missing_value),
        solar_unit=config.solar_unit,
    )

    log.info("=== POA ===")
    compute_poa(
        dni_path=paths.dni,
        apparent_path=paths.apparent,
        horizon_path=paths.horizon,
        out_path=paths.poa,
        surface_tilt=float(config.surface_tilt),
        surface_azimuth=float(config.surface_azimuth),
        albedo=float(config.albedo),
    )

    log.info("=== Effective Irradiance ===")
    compute_effective_irradiance(
        true_sun_path=paths.true_sun_position,
        apparent_path=paths.apparent,
        meteo_path=paths.meteo,
        poa_path=paths.poa,
        out_path=paths.effective_irradiance,
        surface_tilt=float(config.surface_tilt),
        surface_azimuth=float(config.surface_azimuth),
    )

    log.info("=== PV Energy ===")
    compute_energy(
        meteo_path=paths.meteo,
        poa_path=paths.poa,
        effective_irradiance_path=paths.effective_irradiance,
        out_path=paths.energy,
        module_pdc0=float(config.module_pdc0),
        module_count=int(config.module_count),
        gamma_pdc=float(config.gamma_pdc),
        annual_age_loss_pct=float(config.annual_age_loss_pct),
        pac0_each=float(config.pac0_each),
        inverter_count=int(config.inverter_count),
        eta_inv_nom=float(config.eta_inv_nom),
        freq=config.freq,
    )

    if paths.horizon_plot is not None:
        log.info("=== Horizon Visualisation ===")
        plot_horizon_profile(
            horizon_path=paths.horizon,
            plot_path=paths.horizon_plot,
            station_name=config.station_name,
            show=show_plots,
        )

    if paths.energy_plot is not None:
        log.info("=== Energy Visualisation ===")
        plot_energy_overview(
            energy_path=paths.energy,
            plot_path=paths.energy_plot,
            show=show_plots,
        )


__all__ = ["PVRunConfig", "PVRunPaths", "run_pv"]
