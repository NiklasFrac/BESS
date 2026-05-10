import pandas as pd
import pytest

import pv_sim.seen_pos as seen_pos_module
from pv_sim.seen_pos import _check_columns, compute_apparent_sun_position


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_check_columns_reports_missing_columns():
    with pytest.raises(ValueError, match="fehlende Spalten"):
        _check_columns(pd.DataFrame({"timestamp_utc": []}), {"timestamp_utc", "TT_10"}, "Meteo")


def test_compute_apparent_sun_position_uses_midpoint_meteo_and_true_fallback(
    tmp_path,
    monkeypatch,
):
    meteo_path = tmp_path / "meteo.csv"
    true_path = tmp_path / "true.csv"
    out_path = tmp_path / "out" / "apparent.csv"
    write_csv(
        meteo_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "TT_10": 10.0, "PP_10": 1000.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "TT_10": 20.0, "PP_10": 1002.0},
            {"timestamp_utc": "2024-01-01 02:00:00+00:00", "TT_10": -999.0, "PP_10": 1004.0},
        ],
    )
    write_csv(
        true_path,
        [
            {
                "solar_position_reference_utc": "2024-01-01 00:30:00+00:00",
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 170.0,
                "solar_zenith_deg": 50.0,
                "solar_elevation_deg": 40.0,
            },
            {
                "solar_position_reference_utc": "2024-01-01 01:30:00+00:00",
                "timestamp_utc": "2024-01-01 02:00:00+00:00",
                "latitude": 48.0,
                "longitude": 11.0,
                "height_m_amsl": 500.0,
                "solar_azimuth_deg": 180.0,
                "solar_zenith_deg": 60.0,
                "solar_elevation_deg": 30.0,
            },
        ],
    )

    captured = {}

    def fake_spa_python(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "apparent_zenith": [49.5],
                "apparent_elevation": [40.5],
                "azimuth": [171.0],
                "elevation": [40.0],
            }
        )

    monkeypatch.setattr(seen_pos_module.pvlib.solarposition, "spa_python", fake_spa_python)

    compute_apparent_sun_position(
        meteo_path=meteo_path,
        true_solar_path=true_path,
        out_path=out_path,
        freq="1h",
        missing_value=-999.0,
    )

    out = pd.read_csv(out_path)
    assert out["met_data_complete"].tolist() == [True, False]
    assert out.loc[0, "apparent_zenith_deg"] == pytest.approx(49.5)
    assert out.loc[0, "apparent_elevation_deg"] == pytest.approx(40.5)
    assert out.loc[0, "apparent_azimuth_deg"] == pytest.approx(171.0)
    assert out.loc[0, "refraction_correction_deg"] == pytest.approx(0.5)
    assert out.loc[1, "apparent_zenith_deg"] == pytest.approx(60.0)
    assert out.loc[1, "refraction_correction_deg"] == pytest.approx(0.0)
    assert captured["latitude"] == pytest.approx(48.0)
    assert captured["pressure"].tolist() == pytest.approx([100100.0])
    assert captured["temperature"].tolist() == pytest.approx([15.0])
