from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pv_sim.true_pos import _load_station_row


VALID_METADATA = pd.DataFrame(
    {
        "station_id": ["00232", "00233"],
        "station_name": ["Augsburg", "Muenchen"],
        "latitude": [48.4253, 48.1351],
        "longitude": [10.9417, 11.5820],
        "height_m_amsl": [462.0, 520.0],
    }
)


@pytest.fixture()
def metadata_file(tmp_path: Path) -> Path:
    path = tmp_path / "metadata_stations.csv"
    VALID_METADATA.to_csv(path, index=False)
    return path


class TestLoadStationRow:
    def test_finds_existing_station(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "Augsburg")
        assert row["station_id"] == "00232"
        assert float(row["latitude"]) == pytest.approx(48.4253)

    def test_returns_correct_values(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "Muenchen")
        assert float(row["longitude"]) == pytest.approx(11.5820)
        assert float(row["height_m_amsl"]) == pytest.approx(520.0)

    def test_case_insensitive_lower(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "augsburg")
        assert row["station_id"] == "00232"

    def test_case_insensitive_upper(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "AUGSBURG")
        assert row["station_id"] == "00232"

    def test_case_insensitive_mixed(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "aUgSbUrG")
        assert row["station_id"] == "00232"

    def test_strips_leading_trailing_whitespace(self, metadata_file: Path) -> None:
        row = _load_station_row(metadata_file, "  Augsburg  ")
        assert row["station_id"] == "00232"

    def test_raises_on_unknown_station(self, metadata_file: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _load_station_row(metadata_file, "Berlin")

    def test_raises_on_empty_string(self, metadata_file: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _load_station_row(metadata_file, "")

    def test_raises_on_duplicate_station(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "station_id": ["00232", "00999"],
                "station_name": ["Augsburg", "Augsburg"],
                "latitude": [48.4253, 48.4300],
                "longitude": [10.9417, 10.9500],
                "height_m_amsl": [462.0, 462.0],
            }
        )
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="not unique"):
            _load_station_row(path, "Augsburg")

    def test_raises_on_missing_required_column(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "station_id": ["00232"],
                "station_name": ["Augsburg"],
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

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "latitude",
                "longitude",
                "height_m_amsl",
            ]
        )
        path = tmp_path / "metadata_stations.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="not found"):
            _load_station_row(path, "Augsburg")
