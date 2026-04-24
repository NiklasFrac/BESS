import logging
from pathlib import Path

import yaml

from download.horizon import download_pvgis_horizon
from download.meta_data import download_station_metadata
from download.solar import download_dwd_10min_solar
from download.smard import download_smard_day_ahead_prices
from download.weather import download_dwd_temp_pressure_wind

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Repo-Root nicht gefunden. Erwartet ein Verzeichnis mit 'data'-Ordner."
    )

def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    log_cfg = cfg["logging"]
    logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"], datefmt=log_cfg["datefmt"])
    log = logging.getLogger(__name__)

    steps = [
        ("DWD Metadaten Download", lambda: download_station_metadata(cfg, repo_root)),
        ("DWD Wetter Download", lambda: download_dwd_temp_pressure_wind(cfg, repo_root)),
        ("DWD Solar Download", lambda: download_dwd_10min_solar(cfg, repo_root)),
        ("PVGIS Horizont Download", lambda: download_pvgis_horizon(cfg, repo_root)),
        ("SMARD Download", lambda: download_smard_day_ahead_prices(cfg, repo_root)),
    ]

    for name, fn in steps:
        log.info("=== %s ===", name)
        fn()

if __name__ == "__main__":
    main()
