from dataclasses import dataclass
from pathlib import Path

from .compute_dni import compute_dni
from .compute_effective_irradiance import compute_effective_irradiance
from .compute_poa import compute_poa
from .modul_sim import compute_energy
from .seen_pos import compute_apparent_sun_position
from .true_pos import compute_true_sun_position
from .visualization.energy_prod_visual import plot_energy_overview
from .visualization.horizon_visual import plot_horizon_profile


@dataclass(frozen=True)
class PvSimPaths:
    metadata: Path
    meteo: Path
    solar: Path
    horizon: Path
    true_sun_position: Path
    apparent: Path
    dni: Path
    poa: Path
    effective_irradiance: Path
    energy: Path
    energy_plot: Path
    horizon_plot: Path


@dataclass(frozen=True)
class PvSimParams:
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


def run_pv_sim(paths: PvSimPaths, params: PvSimParams) -> None:
    compute_true_sun_position(
        metadata_path=paths.metadata,
        out_path=paths.true_sun_position,
        station_name=params.station_name,
        start_utc=params.start_utc,
        end_utc=params.end_utc,
        freq=params.freq,
    )
    compute_apparent_sun_position(
        meteo_path=paths.meteo,
        true_solar_path=paths.true_sun_position,
        out_path=paths.apparent,
        freq=params.freq,
        missing_value=params.missing_value,
    )
    compute_dni(
        solar_path=paths.solar,
        sun_position_path=paths.true_sun_position,
        out_path=paths.dni,
        ts_col=params.timestamp_col,
        missing=params.missing_value,
        solar_unit=params.solar_unit,
    )
    compute_poa(
        dni_path=paths.dni,
        apparent_path=paths.apparent,
        horizon_path=paths.horizon,
        out_path=paths.poa,
        surface_tilt=params.surface_tilt,
        surface_azimuth=params.surface_azimuth,
        albedo=params.albedo,
    )
    compute_effective_irradiance(
        true_sun_path=paths.true_sun_position,
        apparent_path=paths.apparent,
        meteo_path=paths.meteo,
        poa_path=paths.poa,
        out_path=paths.effective_irradiance,
        surface_tilt=params.surface_tilt,
        surface_azimuth=params.surface_azimuth,
    )
    compute_energy(
        meteo_path=paths.meteo,
        poa_path=paths.poa,
        effective_irradiance_path=paths.effective_irradiance,
        out_path=paths.energy,
        module_pdc0=params.module_pdc0,
        module_count=params.module_count,
        gamma_pdc=params.gamma_pdc,
        annual_age_loss_pct=params.annual_age_loss_pct,
        pac0_each=params.pac0_each,
        inverter_count=params.inverter_count,
        eta_inv_nom=params.eta_inv_nom,
        freq=params.freq,
    )
    plot_energy_overview(paths.energy, paths.energy_plot)
    plot_horizon_profile(paths.horizon, paths.horizon_plot, params.station_name)
