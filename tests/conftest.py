from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")


@dataclass
class PvTestRepo:
    root: Path
    config: dict[str, Any]

    def path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def write_config(self) -> None:
        config_path = self.root / "configs" / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")

    def write_csv(self, relative_path: str, data: pd.DataFrame | list[dict[str, Any]]) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        df.to_csv(path, index=False)
        return path

    def set_path(self, key: str, relative_path: str) -> None:
        self.config["paths"][key] = relative_path
        self.write_config()


@pytest.fixture()
def pv_test_repo(tmp_path: Path) -> PvTestRepo:
    for dirname in ("configs", "data", "results"):
        (tmp_path / dirname).mkdir()

    config: dict[str, Any] = {
        "logging": {
            "level": "INFO",
            "format": "%(levelname)s:%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "paths": {
            "metadata": "data/metadata_stations.csv",
            "meteo": "data/meteo.csv",
            "true_sun_position": "data/true_sun.csv",
            "apparent": "data/apparent.csv",
            "solar": "data/solar.csv",
            "dni": "data/dni.csv",
            "pvgis": "data/horizon.csv",
            "poa": "data/poa.csv",
            "effective_irradiance": "data/effective_irradiance.csv",
            "energy": "results/energy_curve.csv",
        },
        "time": {
            "start_utc": "2020-01-01 00:00:00",
            "end_utc": "2020-01-01 00:30:00",
            "freq": "10min",
        },
        "station": {"id": "00232", "name": "Augsburg"},
        "dni": {"timestamp_col": "timestamp_utc", "dwd_missing_value": -999.0},
        "pv": {
            "surface_tilt": 20,
            "surface_azimuth": 180,
            "albedo": 0.20,
            "module_pdc0": 500.0,
            "gamma_pdc": -0.003,
            "module_count": 2,
        },
        "inverter": {
            "pac0_each": 800.0,
            "inverter_count": 1,
            "eta_inv_nom": 0.96,
        },
        "losses": {"annual_age_loss_pct": 0.5},
    }
    repo = PvTestRepo(root=tmp_path, config=config)
    repo.write_config()
    return repo


@pytest.fixture()
def patch_repo_root(monkeypatch: pytest.MonkeyPatch, pv_test_repo: PvTestRepo):
    def patch(module: Any) -> None:
        monkeypatch.setattr(module, "_find_repo_root", lambda _start: pv_test_repo.root)

    return patch
