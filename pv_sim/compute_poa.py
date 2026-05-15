import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

log = logging.getLogger(__name__)


def compute_poa(
    dni_path: Path,
    apparent_path: Path,
    horizon_path: Path,
    out_path: Path,
    surface_tilt: float,
    surface_azimuth: float,
    albedo: float,
) -> None:
    dni = pd.read_csv(dni_path, parse_dates=["timestamp_utc"])
    apparent = pd.read_csv(apparent_path, parse_dates=["timestamp_utc"])
    horizon = pd.read_csv(horizon_path)

    dni = dni[["timestamp_utc", "ghi_wm2", "dhi_wm2", "dni_wm2"]].copy()
    apparent = apparent[
        [
            "timestamp_utc",
            "apparent_zenith_deg",
            "apparent_elevation_deg",
            "apparent_azimuth_deg",
        ]
    ].copy()
    horizon = horizon[["azimuth_deg", "horizon_height_deg"]].copy()

    dni = dni.sort_values("timestamp_utc").reset_index(drop=True)
    apparent = apparent.sort_values("timestamp_utc").reset_index(drop=True)
    horizon = horizon.sort_values("azimuth_deg").reset_index(drop=True)

    df = dni.merge(apparent, on="timestamp_utc", how="inner", validate="one_to_one")

    az = horizon["azimuth_deg"].to_numpy(dtype=float)
    hh = horizon["horizon_height_deg"].to_numpy(dtype=float)

    az_ext = np.r_[az, az[0] + 360.0]
    hh_ext = np.r_[hh, hh[0]]

    sun_az = np.mod(df["apparent_azimuth_deg"].to_numpy(dtype=float), 360.0)
    horizon_interp = np.interp(sun_az, az_ext, hh_ext)

    df["horizon_height_deg_interp"] = horizon_interp

    shaded = df["apparent_elevation_deg"] <= df["horizon_height_deg_interp"]

    df["dni_shaded"] = df["dni_wm2"]
    df.loc[shaded, "dni_shaded"] = 0.0
    df.loc[df["dni_wm2"].isna(), "dni_shaded"] = np.nan

    df["ghi_wm2"] = df["ghi_wm2"].clip(lower=0)
    df["dhi_wm2"] = df["dhi_wm2"].clip(lower=0)
    df["dni_shaded"] = df["dni_shaded"].clip(lower=0)

    df["relative_airmass"] = pvlib.atmosphere.get_relative_airmass(
        df["apparent_zenith_deg"]
    ).to_numpy()

    df["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        pd.DatetimeIndex(df["timestamp_utc"])
    ).to_numpy()

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=df["apparent_zenith_deg"],
        solar_azimuth=df["apparent_azimuth_deg"],
        dni=df["dni_shaded"],
        ghi=df["ghi_wm2"],
        dhi=df["dhi_wm2"],
        dni_extra=df["dni_extra"],
        airmass=df["relative_airmass"],
        albedo=albedo,
        model="perez-driesse",
    )

    poa.insert(0, "timestamp_utc", df["timestamp_utc"].to_numpy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    poa.to_csv(out_path, index=False)
    log.info(
        "Gespeichert: %s | Zeilen: %d | shaded: %d", out_path, len(poa), shaded.sum()
    )
