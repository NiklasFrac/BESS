from pathlib import Path

import pandas as pd
import pytest

import validation


def write_meteo(path: Path, times: pd.DatetimeIndex, tt: list, pp: list, ff: list):
    pd.DataFrame(
        {
            "timestamp_utc": times,
            "TT_10": tt,
            "PP_10": pp,
            "FF_10": ff,
        }
    ).to_csv(path, index=False)


def write_metadata(path: Path):
    pd.DataFrame(
        [
            {
                "station_id": "00001",
                "station_name": "Target",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
            },
            {
                "station_id": "00002",
                "station_name": "Near",
                "latitude": 48.01,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
            },
            {
                "station_id": "00003",
                "station_name": "Far",
                "latitude": 48.02,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
            },
        ]
    ).to_csv(path, index=False)


def cfg(tmp_path: Path, end_utc: str) -> dict:
    return {
        "station": {"id": "00001"},
        "time": {
            "start_utc": "2024-01-01 00:00:00",
            "end_utc": end_utc,
            "freq": "10min",
        },
        "validation": {"max_gap_length": "60min", "max_distance": 10.0},
        "paths": {
            "meteo_raw": "raw.csv",
            "meteo": "meteo.csv",
            "metadata": "metadata.csv",
        },
        "url": {
            "air_temp_url": "https://example.test/tu/file.zip",
            "wind_url": "https://example.test/wind/file.zip",
        },
    }


def test_validate_meteo_interpolates_gap_up_to_max_gap(tmp_path):
    times = pd.date_range("2024-01-01 00:00:00+00:00", periods=9, freq="10min")
    write_metadata(tmp_path / "metadata.csv")
    write_meteo(
        tmp_path / "raw.csv",
        times,
        [0.0, None, None, None, None, None, None, 7.0, 8.0],
        [1000.0] * 9,
        [2.0] * 9,
    )

    report, errors = validation._validate_meteo(
        cfg(tmp_path, "2024-01-01 01:30:00"),
        tmp_path,
    )

    out = pd.read_csv(tmp_path / "meteo.csv")
    assert errors == []
    assert out["TT_10"].isna().sum() == 0
    assert out["TT_10"].tolist() == pytest.approx([0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert report["columns"]["TT_10"]["interpolated"][0]["count"] == 6
    assert report["columns"]["TT_10"]["filled_from_station"] == []


def test_validate_meteo_fills_large_gap_from_first_complete_station(
    tmp_path,
    monkeypatch,
):
    times = pd.date_range("2024-01-01 00:00:00+00:00", periods=10, freq="10min")
    write_metadata(tmp_path / "metadata.csv")
    write_meteo(
        tmp_path / "raw.csv",
        times,
        [0.0, None, None, None, None, None, None, None, 8.0, 9.0],
        [1000.0] * 10,
        [2.0] * 10,
    )
    calls = []

    def fake_url(base, prefix, station_id, start, end):
        calls.append(station_id)
        return f"https://example.test/{station_id}.zip"

    def fake_read(url, station_id, col, start, end):
        gap = times[1:8]
        values = [None, 21, 22, 23, 24, 25, 26] if station_id == "00002" else range(30, 37)
        return pd.DataFrame({col: values}, index=gap)

    monkeypatch.setattr(validation, "_dwd_url", fake_url)
    monkeypatch.setattr(validation, "_read_dwd_col", fake_read)

    report, errors = validation._validate_meteo(
        cfg(tmp_path, "2024-01-01 01:40:00"),
        tmp_path,
    )

    out = pd.read_csv(tmp_path / "meteo.csv")
    fill = report["columns"]["TT_10"]["filled_from_station"][0]
    assert errors == []
    assert calls[:2] == ["00002", "00003"]
    assert fill["station_id"] == "00003"
    assert fill["count"] == 7
    assert out.loc[1:7, "TT_10"].tolist() == pytest.approx(list(range(30, 37)))
