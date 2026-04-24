from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pvlib

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with 'data' folder.")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    dni_path = repo_root / cfg["paths"]["dni"]
    apparent_path = repo_root / cfg["paths"]["apparent"]
    horizon_path = repo_root / cfg["paths"]["pvgis"]
    poa_path = repo_root / cfg["paths"]["poa"]
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

    # zyklische Interpolation des Horizonts über Azimut
    az = horizon["azimuth_deg"].to_numpy(dtype=float)
    hh = horizon["horizon_height_deg"].to_numpy(dtype=float)

    # für 360°-Wraparound erweitern
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

    #RELATIVE AIRMASS für Perez
    df["relative_airmass"] = pvlib.atmosphere.get_relative_airmass(
        df["apparent_zenith_deg"]
    ).to_numpy()
    #DNI EXTRA für Perez
    df["dni_extra"] = pvlib.irradiance.get_extra_radiation(
        pd.DatetimeIndex(df["timestamp_utc"])
    ).to_numpy()

    #POA berechnung
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=cfg["pv"]["surface_tilt"],
        surface_azimuth=cfg["pv"]["surface_azimuth"],
        solar_zenith=df["apparent_zenith_deg"],
        solar_azimuth=df["apparent_azimuth_deg"],
        dni=df["dni_shaded"],
        ghi=df["ghi_wm2"],
        dhi=df["dhi_wm2"],
        dni_extra=df["dni_extra"],
        airmass=df["relative_airmass"],
        albedo=cfg["pv"]["albedo"],
        model="perez-driesse",
    )

    poa.insert(0, "timestamp_utc", df["timestamp_utc"].to_numpy())
    poa.to_csv(poa_path, index=False)

if __name__ == "__main__":
    main()