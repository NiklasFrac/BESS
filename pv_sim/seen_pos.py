import logging
from pathlib import Path

import pandas as pd
import pvlib
import yaml


def _check_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required - set(df.columns)):
        raise ValueError(f"{label}: fehlende Spalten: {missing}")

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find <repo-root>. Expected a parent directory containing a 'data' folder."
    )


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())


    logging.basicConfig(
        level=cfg["logging"]["level"],
        format=cfg["logging"]["format"],
        datefmt=cfg["logging"]["datefmt"],
    )
    log = logging.getLogger(__name__)

    meteo_path = repo_root / cfg["paths"]["meteo"]
    true_solar_path = repo_root / cfg["paths"]["true_sun_position"]
    output_path = repo_root / cfg["paths"]["apparent"]

    meteo      = pd.read_csv(meteo_path, parse_dates=["timestamp_utc"])
    true_solar = pd.read_csv(
        true_solar_path,
        parse_dates=["solar_position_reference_utc", "timestamp_utc"],
    )

    _check_columns(meteo,      {"timestamp_utc", "TT_10", "PP_10"}, "Meteo")
    _check_columns(
        true_solar,
        {
            "solar_position_reference_utc",
            "timestamp_utc",
            "latitude",
            "longitude",
            "height_m_amsl",
            "solar_azimuth_deg",
            "solar_zenith_deg",
            "solar_elevation_deg",
        },
        "Solar",
    )

    row           = true_solar.iloc[0]
    latitude      = float(row["latitude"])
    longitude     = float(row["longitude"])
    height_m_amsl = float(row["height_m_amsl"])

    log.info("Lat/Lon/Höhe: %.6f, %.6f, %.1f m", latitude, longitude, height_m_amsl)


    for col in ["TT_10", "PP_10"]:
        meteo[col] = pd.to_numeric(meteo[col], errors="coerce")
        meteo[col] = meteo[col].replace(-999, pd.NA)

    meteo = meteo.sort_values("timestamp_utc").reset_index(drop=True)

    full_index = pd.date_range(
        start=meteo["timestamp_utc"].min(),
        end=meteo["timestamp_utc"].max(),
        freq="10min",
    )
    meteo = (
        meteo
        .set_index("timestamp_utc")
        .reindex(full_index)
        .rename_axis("timestamp_utc")
        .reset_index()
    )

    meteo["TT_10_mid"] = (meteo["TT_10"] + meteo["TT_10"].shift(1)) / 2
    meteo["PP_10_mid"] = (meteo["PP_10"] + meteo["PP_10"].shift(1)) / 2

    df = pd.merge(
        true_solar[
            [
                "solar_position_reference_utc",
                "timestamp_utc",
                "solar_zenith_deg",
                "solar_elevation_deg",
                "solar_azimuth_deg",
            ]
        ],
        meteo[["timestamp_utc", "TT_10_mid", "PP_10_mid"]],
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
    ).sort_values("timestamp_utc").reset_index(drop=True)

    df["pressure_Pa"] = df["PP_10_mid"] * 100.0
    df["met_data_complete"] = df["TT_10_mid"].notna() & df["PP_10_mid"].notna() & (df["pressure_Pa"] > 0)

    mask = df["met_data_complete"]

    df["apparent_zenith_deg"]       = df["solar_zenith_deg"]
    df["apparent_elevation_deg"]    = df["solar_elevation_deg"]
    df["apparent_azimuth_deg"]      = df["solar_azimuth_deg"]
    df["refraction_correction_deg"] = 0.0

    n_fallback = int((~mask).sum())
    if n_fallback:
        log.warning("Kein Meteo für %d Zeilen — true position als Fallback", n_fallback)
        
    if mask.any():
        spa = pvlib.solarposition.spa_python(
            time=pd.DatetimeIndex(df.loc[mask, "solar_position_reference_utc"]),
            latitude=latitude,
            longitude=longitude,
            altitude=height_m_amsl,
            pressure=df.loc[mask, "pressure_Pa"].to_numpy(),
            temperature=df.loc[mask, "TT_10_mid"].to_numpy(),
            delta_t=None,
            how="numpy",
        )
        df.loc[mask, "apparent_zenith_deg"]    = spa["apparent_zenith"].to_numpy()
        df.loc[mask, "apparent_elevation_deg"] = spa["apparent_elevation"].to_numpy()
        df.loc[mask, "apparent_azimuth_deg"]   = spa["azimuth"].to_numpy()
        df.loc[mask, "refraction_correction_deg"] = (
            spa["apparent_elevation"].to_numpy() - spa["elevation"].to_numpy()
        )

    final_columns = [
        "solar_position_reference_utc",
        "timestamp_utc",
        "solar_zenith_deg", "solar_elevation_deg",
        "apparent_zenith_deg", "apparent_elevation_deg", "apparent_azimuth_deg",
        "refraction_correction_deg",
        "met_data_complete",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[final_columns].to_csv(output_path, index=False)

    log.info("Gespeichert: %s | Zeilen: %d | mit Meteo: %d | ohne: %d",
             output_path, len(df), int(df["met_data_complete"].sum()), int((~df["met_data_complete"]).sum()))


if __name__ == "__main__":
    main()