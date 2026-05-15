import numpy as np
import pandas as pd
import pytest

import pv_sim.compute_dni as compute_dni_module
from pv_sim.compute_dni import compute_dni


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_dni_converts_jcm2_inputs_and_preserves_missing_values(
    tmp_path,
    monkeypatch,
):
    solar_path = tmp_path / "solar.csv"
    sun_path = tmp_path / "true_sun.csv"
    out_path = tmp_path / "out" / "dni.csv"
    write_csv(
        solar_path,
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "GS_10": 36.0,
                "DS_10": 18.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "GS_10": -999.0,
                "DS_10": 9.0,
            },
        ],
    )
    write_csv(
        sun_path,
        [
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "solar_zenith_deg": 70.0},
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "solar_zenith_deg": 60.0},
        ],
    )

    def fake_dni(*, ghi, dhi, zenith):
        return ghi - dhi + zenith

    monkeypatch.setattr(compute_dni_module.pvlib.irradiance, "dni", fake_dni)

    compute_dni(
        solar_path=solar_path,
        sun_position_path=sun_path,
        out_path=out_path,
        ts_col="timestamp_utc",
        missing=-999.0,
        solar_unit="jcm2",
    )

    out = pd.read_csv(out_path)
    assert out.columns.tolist() == ["timestamp_utc", "ghi_wm2", "dhi_wm2", "dni_wm2"]
    assert out["ghi_wm2"].tolist()[0] == pytest.approx(100.0)
    assert out["dhi_wm2"].tolist()[0] == pytest.approx(50.0)
    assert out["dni_wm2"].tolist()[0] == pytest.approx(110.0)
    assert np.isnan(out["ghi_wm2"].tolist()[1])
    assert np.isnan(out["dni_wm2"].tolist()[1])


def test_compute_dni_rejects_unknown_solar_unit(tmp_path):
    solar_path = tmp_path / "solar.csv"
    sun_path = tmp_path / "true_sun.csv"
    write_csv(
        solar_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "GS_10": 1.0, "DS_10": 0.5},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "GS_10": 1.0, "DS_10": 0.5},
        ],
    )
    write_csv(
        sun_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "solar_zenith_deg": 60.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "solar_zenith_deg": 60.0},
        ],
    )

    with pytest.raises(ValueError, match="solar_unit"):
        compute_dni(
            solar_path, sun_path, tmp_path / "dni.csv", "timestamp_utc", -999.0, "bad"
        )
