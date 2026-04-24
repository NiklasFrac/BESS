import logging
from pathlib import Path


import pandas as pd
import pvlib
import yaml


REQUIRED_COLUMNS = {
    "station_id",
    "latitude",
    "longitude",
    "height_m_amsl",
}


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find <repo-root>. Expected a parent directory containing a 'data' folder."
    )



def _load_station_row(metadata_path: Path, station_id: str) -> pd.Series:
    df = pd.read_csv(metadata_path, dtype={"station_id": str})

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"metadata_stations.csv is missing required columns: {missing_str}"
        )

    matched = df.loc[df["station_id"] == station_id].copy()

    if matched.empty:
        raise ValueError(
            f"Station ID '{station_id}' not found in {metadata_path}."
        )

    return matched.iloc[0]



def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(script_path.parent)

    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    logging.basicConfig(
        level=cfg["logging"]["level"],
        format=cfg["logging"]["format"],
        datefmt=cfg["logging"]["datefmt"],
    )
    logger = logging.getLogger(__name__)

    metadata_path = repo_root / cfg["paths"]["metadata"]

    station = _load_station_row(metadata_path, str(cfg["station"]["id"]))

    latitude  = float(station["latitude"])
    longitude = float(station["longitude"])
    altitude  = float(station["height_m_amsl"])
    start_utc = pd.to_datetime(cfg["time"]["start_utc"], utc=True)
    end_utc = pd.to_datetime(cfg["time"]["end_utc"], utc=True)

    freq = pd.to_timedelta(cfg["time"]["freq"])
    if freq <= pd.Timedelta(0):
        raise ValueError(f"Invalid freq: {cfg['time']['freq']}")

    times_start = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq=freq,
        inclusive="left",
    )

    times_ref = times_start + freq / 2
    times_end = times_start + freq

    solar_position = pvlib.solarposition.get_solarposition(
        time=times_ref,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    result = pd.DataFrame(
        {
            "solar_position_reference_utc": times_ref,
            "timestamp_utc":      times_end,
            "station_id":         str(station["station_id"]),
            "latitude":           latitude,
            "longitude":          longitude,
            "height_m_amsl":      altitude,
            "solar_zenith_deg":   solar_position["zenith"].to_numpy(),
            "solar_elevation_deg": solar_position["elevation"].to_numpy(),
            "solar_azimuth_deg":  solar_position["azimuth"].to_numpy(),
        }
    )


    output_path = repo_root / cfg["paths"]["true_sun_position"]

    if output_path.exists():
        logger.warning("Output already exists, overwriting: %s", output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    logger.info(f"Wrote {len(result):,} rows to {output_path}")

if __name__ == "__main__":
    main()
