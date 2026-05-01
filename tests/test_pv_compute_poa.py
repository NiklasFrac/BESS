from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_sim import compute_poa


def test_main_applies_horizon_shading_clipping_and_pvlib_inputs(
    pv_test_repo, patch_repo_root, monkeypatch
):
    patch_repo_root(compute_poa)
    pv_test_repo.write_csv(
        "data/dni.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "ghi_wm2": 100.0, "dhi_wm2": 50.0, "dni_wm2": 800.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "ghi_wm2": 100.0, "dhi_wm2": 50.0, "dni_wm2": 700.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "ghi_wm2": 100.0, "dhi_wm2": 50.0, "dni_wm2": 600.0},
            {"timestamp_utc": "2020-01-01 00:40:00+00:00", "ghi_wm2": 100.0, "dhi_wm2": 50.0, "dni_wm2": np.nan},
            {"timestamp_utc": "2020-01-01 00:50:00+00:00", "ghi_wm2": -10.0, "dhi_wm2": -2.0, "dni_wm2": -5.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/apparent.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "apparent_zenith_deg": 84.0, "apparent_elevation_deg": 6.0, "apparent_azimuth_deg": 90.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "apparent_zenith_deg": 85.0, "apparent_elevation_deg": 5.0, "apparent_azimuth_deg": 90.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "apparent_zenith_deg": 85.0, "apparent_elevation_deg": 5.0, "apparent_azimuth_deg": 359.0},
            {"timestamp_utc": "2020-01-01 00:40:00+00:00", "apparent_zenith_deg": 80.0, "apparent_elevation_deg": 10.0, "apparent_azimuth_deg": 180.0},
            {"timestamp_utc": "2020-01-01 00:50:00+00:00", "apparent_zenith_deg": 79.0, "apparent_elevation_deg": 11.0, "apparent_azimuth_deg": 0.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/horizon.csv",
        [
            {"azimuth_deg": 0.0, "horizon_height_deg": 10.0},
            {"azimuth_deg": 90.0, "horizon_height_deg": 5.0},
            {"azimuth_deg": 180.0, "horizon_height_deg": 1.0},
            {"azimuth_deg": 270.0, "horizon_height_deg": 0.0},
        ],
    )

    captured = {}

    def fake_get_total_irradiance(**kwargs):
        captured.update(kwargs)
        dni = np.asarray(kwargs["dni"], dtype=float)
        ghi = np.asarray(kwargs["ghi"], dtype=float)
        dhi = np.asarray(kwargs["dhi"], dtype=float)
        return pd.DataFrame(
            {
                "poa_global": ghi + dhi + np.nan_to_num(dni, nan=0.0),
                "poa_direct": dni,
                "poa_diffuse": dhi,
                "poa_sky_diffuse": dhi,
                "poa_ground_diffuse": ghi * 0.01,
            }
        )

    monkeypatch.setattr(compute_poa.pvlib.irradiance, "get_total_irradiance", fake_get_total_irradiance)

    compute_poa.main()

    assert captured["surface_tilt"] == 20
    assert captured["surface_azimuth"] == 180
    assert captured["albedo"] == pytest.approx(0.20)
    assert captured["model"] == "perez-driesse"
    np.testing.assert_allclose(
        np.asarray(captured["dni"], dtype=float),
        np.array([800.0, 0.0, 0.0, np.nan, 0.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(np.asarray(captured["ghi"], dtype=float), np.array([100.0, 100.0, 100.0, 100.0, 0.0]))
    np.testing.assert_allclose(np.asarray(captured["dhi"], dtype=float), np.array([50.0, 50.0, 50.0, 50.0, 0.0]))

    output = pd.read_csv(pv_test_repo.path("data/poa.csv"), parse_dates=["timestamp_utc"])
    assert output["timestamp_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2020-01-01 00:10:00+00:00",
                "2020-01-01 00:20:00+00:00",
                "2020-01-01 00:30:00+00:00",
                "2020-01-01 00:40:00+00:00",
                "2020-01-01 00:50:00+00:00",
            ],
            utc=True,
        )
    )


def test_main_rejects_duplicate_timestamps_in_poa_merge(pv_test_repo, patch_repo_root):
    patch_repo_root(compute_poa)
    pv_test_repo.write_csv(
        "data/dni.csv",
        [{"timestamp_utc": "2020-01-01 00:10:00+00:00", "ghi_wm2": 100.0, "dhi_wm2": 50.0, "dni_wm2": 800.0}],
    )
    pv_test_repo.write_csv(
        "data/apparent.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "apparent_zenith_deg": 80.0, "apparent_elevation_deg": 10.0, "apparent_azimuth_deg": 180.0},
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "apparent_zenith_deg": 81.0, "apparent_elevation_deg": 9.0, "apparent_azimuth_deg": 181.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/horizon.csv",
        [{"azimuth_deg": 0.0, "horizon_height_deg": 0.0}, {"azimuth_deg": 180.0, "horizon_height_deg": 0.0}],
    )

    with pytest.raises(pd.errors.MergeError):
        compute_poa.main()
