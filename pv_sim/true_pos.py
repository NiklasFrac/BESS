import logging
from pathlib import Path

import pandas as pd
import pvlib


REQUIRED_COLUMNS = {
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "height_m_amsl",
}

log = logging.getLogger(__name__)


def _load_station_row(metadata_path: Path, station_name: str) -> pd.Series:
    df = pd.read_csv(metadata_path, dtype={"station_id": str})

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"metadata_stations.csv is missing required columns: {missing_str}"
        )

    station_query = station_name.strip()
    normalized_query = station_query.casefold()

    station_names = df["station_name"].astype(str).str.strip().str.casefold()
    matched = df.loc[station_names == normalized_query].copy()

    if matched.empty and station_query:
        matched = df.loc[
            df["station_id"].astype(str).str.strip() == station_query
        ].copy()

    if matched.empty:
        raise ValueError(f"Station '{station_name}' not found in {metadata_path}.")

    if len(matched) > 1:
        raise ValueError(f"Station '{station_name}' is not unique in {metadata_path}.")

    return matched.iloc[0]


def compute_true_sun_position(
    metadata_path: Path,
    out_path: Path,
    station_name: str,
    start_utc: str,
    end_utc: str,
    freq: str,
) -> None:
    station = _load_station_row(metadata_path, station_name)

    latitude = float(station["latitude"])
    longitude = float(station["longitude"])
    altitude = float(station["height_m_amsl"])

    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc = pd.to_datetime(end_utc, utc=True)

    freq_td = pd.to_timedelta(freq)
    if freq_td <= pd.Timedelta(0):
        raise ValueError(f"Invalid freq: {freq}")

    times_start = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq=freq_td,
        inclusive="left",
    )

    times_ref = times_start + freq_td / 2
    times_end = times_start + freq_td

    solar_position = pvlib.solarposition.get_solarposition(
        time=times_ref,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    result = pd.DataFrame(
        {
            "solar_position_reference_utc": times_ref,
            "timestamp_utc": times_end,
            "station_id": str(station["station_id"]),
            "latitude": latitude,
            "longitude": longitude,
            "height_m_amsl": altitude,
            "solar_zenith_deg": solar_position["zenith"].to_numpy(),
            "solar_elevation_deg": solar_position["elevation"].to_numpy(),
            "solar_azimuth_deg": solar_position["azimuth"].to_numpy(),
        }
    )

    if out_path.exists():
        log.warning("Output already exists, overwriting: %s", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    log.info("Wrote %d rows to %s", len(result), out_path)