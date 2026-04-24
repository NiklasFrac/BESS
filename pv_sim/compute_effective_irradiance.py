from pathlib import Path

import yaml
import pvlib
import pandas as pd

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with 'data' folder.")

def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    surface_tilt = cfg["pv"]["surface_tilt"]
    surface_azimuth = cfg["pv"]["surface_azimuth"]
    true_sun_path = repo_root / cfg["paths"]["true_sun_position"]
    poa_path = repo_root / cfg["paths"]["poa"]

    df = pd.read_csv(
        true_sun_path,
        usecols=["timestamp_utc", "solar_zenith_deg", "solar_azimuth_deg"],
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc").reset_index(drop=True)

    df["aoi_deg"] = pvlib.irradiance.aoi(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=df["solar_zenith_deg"],
        solar_azimuth=df["solar_azimuth_deg"],
    )

    apparent_path = repo_root / cfg["paths"]["apparent"]
    meteo_path = repo_root / cfg["paths"]["meteo"]

    apparent_df = pd.read_csv(
        apparent_path,
        usecols=["timestamp_utc", "apparent_zenith_deg"],
        parse_dates=["timestamp_utc"],
    )

    meteo_df = pd.read_csv(
        meteo_path,
        usecols=["timestamp_utc", "PP_10"],
        parse_dates=["timestamp_utc"],
    )

    df = df.merge(apparent_df, on="timestamp_utc", how="left")
    df = df.merge(meteo_df, on="timestamp_utc", how="left")

    df["pressure_pa"] = df["PP_10"] * 100.0

    df["airmass_relative"] = pvlib.atmosphere.get_relative_airmass(
        df["apparent_zenith_deg"],
        model="kastenyoung1989",
    )

    df["airmass_absolute"] = pvlib.atmosphere.get_absolute_airmass(
        df["airmass_relative"],
        pressure=df["pressure_pa"],
    )

    poa_df = pd.read_csv(
        poa_path,
        usecols=["timestamp_utc", "poa_direct", "poa_diffuse"],
        parse_dates=["timestamp_utc"],
    )

    df = df.merge(poa_df, on="timestamp_utc", how="left")

    #Approx
    df["iam"] = pvlib.iam.physical(df["aoi_deg"])
    df["effective_irradiance"] = df["poa_direct"] * df["iam"] + df["poa_diffuse"]

    out_path = repo_root / cfg["paths"]["effective_irradiance"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
if __name__ == "__main__":
    main()