import requests
import pandas as pd
from pathlib import Path
import yaml

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "config.yaml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with configs/config.yaml and data folder.")

def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)

    config_path = repo_root / "configs" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    meta_path = repo_root / cfg["paths"]["metadata"]
    output_path = repo_root / cfg["paths"]["weather_fc"]
    url = cfg["url"]["weather_forecast"]

    start_date = pd.to_datetime(cfg["time"]["start_utc"]).date().isoformat()
    end_date = (pd.to_datetime(cfg["time"]["end_utc"]) - pd.Timedelta(days=1)).date().isoformat()
    
    station_id = str(cfg["station"]["id"])
    meta = pd.read_csv(meta_path, dtype={"station_id": str})
    station = meta.loc[meta["station_id"] == station_id].iloc[0]

    latitude = station["latitude"]
    longitude = station["longitude"]
    vars_ = [
        "temperature_2m",
        "wind_speed_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": cfg["time"]["timezone"],
        "hourly": ",".join(f"{v}_previous_day1" for v in vars_),
    }

    r = requests.get(url, params=params, timeout=(10, 300))
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "timestamp_utc"})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()