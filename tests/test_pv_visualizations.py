from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_sim.visualization import energy_prod_visual, horizon_visual


def test_horizon_coordinates_close_curve_and_map_axes():
    df = pd.DataFrame(
        {
            "azimuth_deg": [0.0, 90.0],
            "horizon_height_deg": [2.0, 3.0],
        }
    )

    x, y, z = horizon_visual._horizon_coordinates(df)

    np.testing.assert_allclose(x, np.array([0.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(y, np.array([1.0, 0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(z, np.array([2.0, 3.0, 2.0]))


def test_horizon_visual_main_writes_png(pv_test_repo, patch_repo_root, monkeypatch):
    patch_repo_root(horizon_visual)
    pv_test_repo.write_csv(
        "data/horizon.csv",
        [
            {"azimuth_deg": 0.0, "horizon_height_deg": 1.0},
            {"azimuth_deg": 90.0, "horizon_height_deg": 2.0},
            {"azimuth_deg": 180.0, "horizon_height_deg": 1.0},
        ],
    )
    monkeypatch.setattr(horizon_visual.plt, "show", lambda: None)

    horizon_visual.main()

    png = pv_test_repo.path("results/horizon_profile.png")
    assert png.exists()
    assert png.stat().st_size > 0


def test_daily_summary_aggregates_energy_power_and_temperatures():
    df = pd.DataFrame(
        [
            {
                "timestamp_utc": "2020-01-01 00:10:00+00:00",
                "e_net_ac_kwh": 1.0,
                "poa_global": 100.0,
                "p_ac_w": 1000.0,
                "TT_10": 10.0,
                "t_module_faiman_c": 20.0,
            },
            {
                "timestamp_utc": "2020-01-01 00:20:00+00:00",
                "e_net_ac_kwh": 2.0,
                "poa_global": 200.0,
                "p_ac_w": 2000.0,
                "TT_10": 12.0,
                "t_module_faiman_c": 22.0,
            },
            {
                "timestamp_utc": "2020-01-02 00:10:00+00:00",
                "e_net_ac_kwh": 4.0,
                "poa_global": 300.0,
                "p_ac_w": 1500.0,
                "TT_10": 14.0,
                "t_module_faiman_c": 24.0,
            },
        ]
    )

    daily = energy_prod_visual._daily_summary(df)

    assert daily.iloc[0]["e_net_ac_kwh"] == pytest.approx(3.0)
    assert daily.iloc[0]["poa_global"] == pytest.approx(150.0)
    assert daily.iloc[0]["p_ac_w"] == pytest.approx(2000.0)
    assert daily.iloc[0]["TT_10"] == pytest.approx(11.0)
    assert daily.iloc[0]["t_module_faiman_c"] == pytest.approx(21.0)
    assert daily.iloc[0]["e_net_ac_kwh_30d"] == pytest.approx(3.0)
    assert daily.iloc[1]["e_net_ac_kwh_30d"] == pytest.approx(3.5)


def test_energy_visual_main_writes_png(pv_test_repo, patch_repo_root, monkeypatch):
    patch_repo_root(energy_prod_visual)
    pv_test_repo.write_csv(
        "results/energy_curve.csv",
        [
            {
                "timestamp_utc": "2020-01-01 00:10:00+00:00",
                "e_net_ac_kwh": 1.0,
                "poa_global": 100.0,
                "p_ac_w": 1000.0,
                "TT_10": 10.0,
                "t_module_faiman_c": 20.0,
            },
            {
                "timestamp_utc": "2020-01-02 00:10:00+00:00",
                "e_net_ac_kwh": 2.0,
                "poa_global": 200.0,
                "p_ac_w": 2000.0,
                "TT_10": 12.0,
                "t_module_faiman_c": 22.0,
            },
        ],
    )
    monkeypatch.setattr(energy_prod_visual.plt, "show", lambda: None)

    energy_prod_visual.main()

    png = pv_test_repo.path("results/energy_overview.png")
    assert png.exists()
    assert png.stat().st_size > 0
