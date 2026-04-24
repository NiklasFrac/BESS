"""
tests/test_true_sun_position.py
--------------------------------
Testsuite für pipeline/true_sun_position.py

Getestet:
  - _find_repo_root
  - _load_station_row

Nicht getestet (bewusst):
  - pvlib.solarposition  → fremde Library
  - Parquet-Schreiben    → Standardbibliothek
  - main()               → Integrationstest, läuft gegen echte Daten
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pv_sim.true_pos import _find_repo_root, _load_station_row


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_METADATA = pd.DataFrame(
    {
        "station_id":   ["00232", "00233"],
        "station_name": ["Augsburg", "München"],
        "latitude":     [48.4253, 48.1351],
        "longitude":    [10.9417, 11.5820],
        "height_m_amsl":[462.0,   520.0  ],
    }
)


@pytest.fixture()
def metadata_file(tmp_path: Path) -> Path:
    """Schreibt eine valide metadata_stations.csv in ein tmp-Verzeichnis."""
    path = tmp_path / "metadata_stations.csv"
    VALID_METADATA.to_csv(path, index=False)
    return path


@pytest.fixture()
def repo_root_with_data(tmp_path: Path) -> Path:
    """Verzeichnisstruktur mit data/-Unterordner."""
    (tmp_path / "data").mkdir()
    return tmp_path


# ── _find_repo_root ───────────────────────────────────────────────────────────

class TestFindRepoRoot:
    def test_finds_root_from_direct_child(self, repo_root_with_data: Path) -> None:
        subdir = repo_root_with_data / "pipeline"
        subdir.mkdir()
        assert _find_repo_root(subdir) == repo_root_with_data

    def test_finds_root_from_nested_child(self, repo_root_with_data: Path) -> None:
        nested = repo_root_with_data / "pipeline" / "submodule"
        nested.mkdir(parents=True)
        assert _find_repo_root(nested) == repo_root_with_data

    def test_finds_root_when_start_is_root(self, repo_root_with_data: Path) -> None:
        assert _find_repo_root(repo_root_with_data) == repo_root_with_data

    def test_raises_when_no_data_dir(self, tmp_path: Path) -> None:
        # tmp_path hat kein data/ → soll FileNotFoundError werfen
        with pytest.raises(FileNotFoundError, match="repo-root"):
            _find_repo_root(tmp_path)


# ── _load_station_row ─────────────────────────────────────────────────────────

class TestLoadStationRow:

    # --- Happy path ---

    def test_finds_existing_station(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "Augsburg")
        assert row["station_id"] == "00232"
        assert float(row["latitude"]) == pytest.approx(48.4253)

    def test_returns_correct_values(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "München")
        assert float(row["longitude"]) == pytest.approx(11.5820)
        assert float(row["height_m_amsl"]) == pytest.approx(520.0)

    # --- Case insensitivity ---

    def test_case_insensitive_lower(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "augsburg")
        assert row["station_id"] == "00232"

    def test_case_insensitive_upper(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "AUGSBURG")
        assert row["station_id"] == "00232"

    def test_case_insensitive_mixed(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "aUgSbUrG")
        assert row["station_id"] == "00232"

    # --- Whitespace ---

    def test_strips_leading_trailing_whitespace(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "  Augsburg  ")
        assert row["station_id"] == "00232"

    # --- Fehlerfall: Station nicht gefunden ---

    def test_raises_on_unknown_station(self, metadata_file: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _load_station_row(metadata_file, "Berlin")

    def test_raises_on_empty_string(self, metadata_file: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _load_station_row(metadata_file, "")

    # --- Fehlerfall: Duplikate ---

    def test_raises_on_duplicate_station(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "station_id":    ["00232", "00999"],
                "station_name":  ["Augsburg", "Augsburg"],
                "latitude":      [48.4253, 48.4300],
                "longitude":     [10.9417, 10.9500],
                "height_m_amsl": [462.0,   462.0  ],
            }
        )
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="not unique"):
            _load_station_row(path, "Augsburg")

    # --- Fehlerfall: fehlende Spalten ---

    def test_raises_on_missing_required_column(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "station_id":   ["00232"],
                "station_name": ["Augsburg"],
                # latitude, longitude, height_m_amsl fehlen
            }
        )
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            _load_station_row(path, "Augsburg")

    def test_error_message_lists_missing_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"station_id": ["00232"], "station_name": ["Augsburg"]})
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError) as exc_info:
            _load_station_row(path, "Augsburg")

        msg = str(exc_info.value)
        assert "latitude" in msg
        assert "longitude" in msg
        assert "height_m_amsl" in msg

    # --- Edge case: leere CSV ---

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            columns=["station_id", "station_name", "latitude", "longitude", "height_m_amsl"]
        )
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="not found"):
            _load_station_row(path, "Augsburg")