from __future__ import annotations

import pandas as pd
import pytest

from pv_sim import seen_pos


def test_check_columns_reports_missing_columns():
    df = pd.DataFrame({"timestamp_utc": ["2020-01-01 00:00:00+00:00"]})

    with pytest.raises(ValueError, match="fehlende Spalten"):
        seen_pos._check_columns(df, {"timestamp_utc", "TT_10", "PP_10"}, "Meteo")


def test_main_uses_midpoint_meteo_and_falls_back_when_incomplete(
    pv_test_repo, patch_repo_root, monkeypatch
):
    patch_repo_root(seen_pos)
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [
            {"timestamp_utc": "2020-01-01 00:00:00+00:00", "TT_10": 10.0, "PP_10": 1000.0},
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "TT_10": 12.0, "PP_10": 1002.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "TT_10": -999.0, "PP_10": 1004.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "TT_10": 14.0, "PP_10": 1006.0},
            {"timestamp_utc": "2020-01-01 00:40:00+00:00", "TT_10": 16.0, "PP_10": -1008.0},
            {"timestamp_utc": "2020-01-01 00:50:00+00:00", "TT_10": 18.0, "PP_10": -1000.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/true_sun.csv",
        [
            {
                "solar_position_reference_utc": "2020-01-01 00:05:00+00:00",
                "timestamp_utc": "2020-01-01 00:00:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 100.0,
                "solar_zenith_deg": 90.0,
                "solar_elevation_deg": 0.0,
            },
            {
                "solar_position_reference_utc": "2020-01-01 00:15:00+00:00",
                "timestamp_utc": "2020-01-01 00:10:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 110.0,
                "solar_zenith_deg": 80.0,
                "solar_elevation_deg": 10.0,
            },
            {
                "solar_position_reference_utc": "2020-01-01 00:25:00+00:00",
                "timestamp_utc": "2020-01-01 00:20:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 120.0,
                "solar_zenith_deg": 70.0,
                "solar_elevation_deg": 20.0,
            },
            {
                "solar_position_reference_utc": "2020-01-01 00:35:00+00:00",
                "timestamp_utc": "2020-01-01 00:30:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 130.0,
                "solar_zenith_deg": 60.0,
                "solar_elevation_deg": 30.0,
            },
            {
                "solar_position_reference_utc": "2020-01-01 00:45:00+00:00",
                "timestamp_utc": "2020-01-01 00:40:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 140.0,
                "solar_zenith_deg": 50.0,
                "solar_elevation_deg": 40.0,
            },
            {
                "solar_position_reference_utc": "2020-01-01 00:55:00+00:00",
                "timestamp_utc": "2020-01-01 00:50:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 150.0,
                "solar_zenith_deg": 40.0,
                "solar_elevation_deg": 50.0,
            },
        ],
    )

    captured = {}

    def fake_spa_python(time, latitude, longitude, altitude, pressure, temperature, delta_t, how):
        captured["time"] = pd.DatetimeIndex(time)
        captured["pressure"] = list(pressure)
        captured["temperature"] = list(temperature)
        assert latitude == pytest.approx(48.0)
        assert longitude == pytest.approx(11.0)
        assert altitude == pytest.approx(500.0)
        assert how == "numpy"
        return pd.DataFrame(
            {
                "apparent_zenith": [79.5],
                "apparent_elevation": [10.5],
                "azimuth": [111.0],
                "elevation": [10.1],
            },
            index=time,
        )

    monkeypatch.setattr(seen_pos.pvlib.solarposition, "spa_python", fake_spa_python)

    seen_pos.main()

    output = pd.read_csv(pv_test_repo.path("data/apparent.csv"), parse_dates=["timestamp_utc"])
    assert captured["time"].tolist() == [
        pd.Timestamp("2020-01-01 00:15:00+00:00"),
    ]
    assert captured["temperature"] == [pytest.approx(11.0)]
    assert captured["pressure"] == [pytest.approx(100100.0)]

    complete = output.loc[output["timestamp_utc"] == pd.Timestamp("2020-01-01 00:10:00+00:00")].iloc[0]
    assert bool(complete["met_data_complete"]) is True
    assert complete["apparent_zenith_deg"] == pytest.approx(79.5)
    assert complete["apparent_elevation_deg"] == pytest.approx(10.5)
    assert complete["apparent_azimuth_deg"] == pytest.approx(111.0)
    assert complete["refraction_correction_deg"] == pytest.approx(0.4)

    fallback = output.loc[output["timestamp_utc"] == pd.Timestamp("2020-01-01 00:00:00+00:00")].iloc[0]
    assert bool(fallback["met_data_complete"]) is False
    assert fallback["apparent_zenith_deg"] == pytest.approx(fallback["solar_zenith_deg"])
    assert fallback["apparent_elevation_deg"] == pytest.approx(fallback["solar_elevation_deg"])
    assert fallback["refraction_correction_deg"] == pytest.approx(0.0)

    invalid_pressure = output.loc[
        output["timestamp_utc"] == pd.Timestamp("2020-01-01 00:50:00+00:00")
    ].iloc[0]
    assert bool(invalid_pressure["met_data_complete"]) is False


def test_main_uses_true_position_when_no_meteo_row_is_complete(
    pv_test_repo, patch_repo_root, monkeypatch
):
    patch_repo_root(seen_pos)
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [{"timestamp_utc": "2020-01-01 00:10:00+00:00", "TT_10": -999.0, "PP_10": 1000.0}],
    )
    pv_test_repo.write_csv(
        "data/true_sun.csv",
        [
            {
                "solar_position_reference_utc": "2020-01-01 00:15:00+00:00",
                "timestamp_utc": "2020-01-01 00:10:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 110.0,
                "solar_zenith_deg": 80.0,
                "solar_elevation_deg": 10.0,
            }
        ],
    )

    def fail_spa_python(*_args, **_kwargs):
        raise AssertionError("spa_python must not run without complete meteo data")

    monkeypatch.setattr(seen_pos.pvlib.solarposition, "spa_python", fail_spa_python)

    seen_pos.main()

    output = pd.read_csv(pv_test_repo.path("data/apparent.csv"))
    row = output.iloc[0]
    assert bool(row["met_data_complete"]) is False
    assert row["apparent_zenith_deg"] == pytest.approx(80.0)
    assert row["apparent_elevation_deg"] == pytest.approx(10.0)
    assert row["apparent_azimuth_deg"] == pytest.approx(110.0)
    assert row["refraction_correction_deg"] == pytest.approx(0.0)
