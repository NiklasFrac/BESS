from pathlib import Path

import pandas as pd
import requests
import yaml


FORECAST_DAY_SUFFIX = "previous_day1"
DEFAULT_FORECAST_FREQ = "hourly"
OPEN_METEO_VARIABLES = [
    "temperature_2m",
    "surface_pressure",
    "wind_speed_10m",
    "shortwave_radiation",
    "diffuse_radiation",
    "cloud_cover",
]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with configs/config.yaml and data folder.")


def _load_station(cfg: dict, repo_root: Path) -> pd.Series:
    meta_path = repo_root / cfg["paths"]["metadata"]
    station_id = str(cfg["station"]["id"])
    meta = pd.read_csv(meta_path, dtype={"station_id": str})
    station = meta.loc[meta["station_id"] == station_id]
    if station.empty:
        raise ValueError(f"Station {station_id} nicht in {meta_path} gefunden.")
    return station.iloc[0]


def _forecast_column(variable: str) -> str:
    return f"{variable}_{FORECAST_DAY_SUFFIX}"



def _forecast_time_params(cfg: dict, freq: str) -> dict[str, str]:
    start_utc = pd.to_datetime(cfg["time"]["start_utc"], utc=True)
    end_utc = pd.to_datetime(cfg["time"]["end_utc"], utc=True)

    if freq == "minutely_15":
        if (end_utc - pd.Timedelta(minutes=15)) < start_utc:
            raise ValueError("Forecast end_utc must be at least 15 minutes after start_utc.")
        return {
            "start_minutely_15": start_utc.strftime("%Y-%m-%dT%H:%M"),
            "end_minutely_15": (end_utc - pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M"),
        }

    return {
        "start_date": start_utc.date().isoformat(),
        "end_date": (end_utc - pd.Timedelta(days=1)).date().isoformat(),
    }


def _normalise_forecast(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)

    meteo = pd.DataFrame(
        {
            "timestamp_utc": raw["timestamp_utc"],
            "TT_10": pd.to_numeric(raw[_forecast_column("temperature_2m")], errors="coerce"),
            "PP_10": pd.to_numeric(raw[_forecast_column("surface_pressure")], errors="coerce"),
            "FF_10": pd.to_numeric(raw[_forecast_column("wind_speed_10m")], errors="coerce"),
        }
    )

    solar = pd.DataFrame(
        {
            "timestamp_utc": raw["timestamp_utc"],
            "GS_10": raw[_forecast_column("shortwave_radiation")],
            "DS_10": raw[_forecast_column("diffuse_radiation")],
        }
    )

    return (
        meteo.sort_values("timestamp_utc").reset_index(drop=True),
        solar.sort_values("timestamp_utc").reset_index(drop=True),
    )


def download_open_meteo_forecast(cfg: dict, repo_root: Path) -> None:
    station = _load_station(cfg, repo_root)
    payload_key = cfg["time"]["freq_fc"]
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "timezone": cfg["time"]["timezone"],
        "wind_speed_unit": "ms",
        payload_key: ",".join(_forecast_column(v) for v in OPEN_METEO_VARIABLES),
        **_forecast_time_params(cfg, payload_key),
    }

    r = requests.get(cfg["url"]["weather_forecast"], params=params, timeout=(10, 300))
    r.raise_for_status()
    raw = pd.DataFrame(r.json()[payload_key]).rename(columns={"time": "timestamp_utc"})

    meteo, solar = _normalise_forecast(raw)

    meteo_path = repo_root / cfg["paths"]["meteo_fc"]
    solar_path = repo_root / cfg["paths"]["solar_fc"]

    for path, df in [(meteo_path, meteo), (solar_path, solar)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text(encoding="utf-8"))
    download_open_meteo_forecast(cfg, repo_root)


if __name__ == "__main__":
    main()
