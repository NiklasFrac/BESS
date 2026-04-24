import logging
from pathlib import Path

import pandas as pd
import pvlib
import yaml


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Repo-Root nicht gefunden. Erwartet ein Verzeichnis mit 'data'-Ordner."
    )

def _load_station_coords(metadata_path: Path, station_id: str) -> tuple[float, float]:
    df = pd.read_csv(metadata_path, dtype={"station_id": str})
    row = df[df["station_id"].str.strip() == station_id.strip()]
    if row.empty:
        raise ValueError(f"Station ID {station_id!r} not found in {metadata_path}")
    lat = float(row["latitude"].iloc[0])
    lon = float(row["longitude"].iloc[0])
    return lat, lon


def download_pvgis_horizon(cfg: dict, repo_root: Path) -> pd.DataFrame:

    log = logging.getLogger(__name__)

    station_id = cfg["station"]["id"]
    station_name = cfg["station"]["name"]
    metadata_path = repo_root / cfg["paths"]["metadata"]
    output_path =  repo_root / cfg["paths"]["pvgis"]

    log.info("Loading coordinates for station %s (%s)", station_id, station_name)
    lat, lon = _load_station_coords(metadata_path, station_id)
    log.info("Coordinates: lat=%.4f, lon=%.4f", lat, lon)

    log.info("Fetching PVGIS horizon data...")
    horizon, _ = pvlib.iotools.get_pvgis_horizon(latitude=lat, longitude=lon)

    if not isinstance(horizon, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(horizon)!r}")

    df = (
        horizon.rename("horizon_height_deg")
        .rename_axis("azimuth_deg")
        .reset_index()
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d horizon points to %s", len(df), output_path)

    return df


if __name__ == "__main__":
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())
    log_cfg = cfg["logging"]
    logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"], datefmt=log_cfg["datefmt"])
    df = download_pvgis_horizon(cfg, repo_root)
