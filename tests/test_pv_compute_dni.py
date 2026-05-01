from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_sim import compute_dni


def test_read_utc_strips_columns_and_parses_timezone(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(" timestamp_utc , value \n2020-01-01 00:00:00,1\n", encoding="utf-8")

    df = compute_dni._read_utc(path, "timestamp_utc")

    assert list(df.columns) == ["timestamp_utc", "value"]
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00+00:00")


def test_main_converts_dwd_energy_to_irradiance_and_dni(pv_test_repo, patch_repo_root):
    patch_repo_root(compute_dni)
    pv_test_repo.write_csv(
        "data/solar.csv",
        [
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "GS_10": -999.0, "DS_10": 18.0},
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "GS_10": 60.0, "DS_10": 18.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "GS_10": "bad", "DS_10": 5.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/true_sun.csv",
        [
            {"timestamp_utc": "2020-01-01 00:40:00+00:00", "solar_zenith_deg": 60.0},
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "solar_zenith_deg": 60.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "solar_zenith_deg": 60.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "solar_zenith_deg": 60.0},
        ],
    )

    compute_dni.main()

    output = pd.read_csv(pv_test_repo.path("data/dni.csv"), parse_dates=["timestamp_utc"])
    assert output["timestamp_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2020-01-01 00:10:00+00:00",
                "2020-01-01 00:20:00+00:00",
                "2020-01-01 00:30:00+00:00",
                "2020-01-01 00:40:00+00:00",
            ],
            utc=True,
        )
    )

    first = output.iloc[0]
    assert first["ghi_wm2"] == pytest.approx(1000.0)
    assert first["dhi_wm2"] == pytest.approx(300.0)
    assert first["dni_wm2"] == pytest.approx((1000.0 - 300.0) / np.cos(np.deg2rad(60.0)))

    assert np.isnan(output.iloc[1]["ghi_wm2"])
    assert output.iloc[1]["dhi_wm2"] == pytest.approx(300.0)
    assert np.isnan(output.iloc[2]["ghi_wm2"])
    assert np.isnan(output.iloc[3]["ghi_wm2"])
    assert np.isnan(output.iloc[3]["dhi_wm2"])


def test_main_rejects_duplicate_timestamps_in_merge(pv_test_repo, patch_repo_root):
    patch_repo_root(compute_dni)
    pv_test_repo.write_csv(
        "data/solar.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "GS_10": 60.0, "DS_10": 18.0},
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "GS_10": 61.0, "DS_10": 18.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/true_sun.csv",
        [{"timestamp_utc": "2020-01-01 00:10:00+00:00", "solar_zenith_deg": 60.0}],
    )

    with pytest.raises(pd.errors.MergeError):
        compute_dni.main()
