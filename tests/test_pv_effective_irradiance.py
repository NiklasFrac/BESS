from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_sim import compute_effective_irradiance


def test_main_computes_aoi_airmass_iam_and_effective_irradiance(
    pv_test_repo, patch_repo_root, monkeypatch
):
    patch_repo_root(compute_effective_irradiance)
    pv_test_repo.write_csv(
        "data/true_sun.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "solar_zenith_deg": 40.0, "solar_azimuth_deg": 180.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "solar_zenith_deg": 50.0, "solar_azimuth_deg": 190.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "solar_zenith_deg": 60.0, "solar_azimuth_deg": 200.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/apparent.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "apparent_zenith_deg": 39.5},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "apparent_zenith_deg": 49.5},
        ],
    )
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "PP_10": 1000.0},
            {"timestamp_utc": "2020-01-01 00:20:00+00:00", "PP_10": 900.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/poa.csv",
        [
            {"timestamp_utc": "2020-01-01 00:10:00+00:00", "poa_direct": 100.0, "poa_diffuse": 20.0},
            {"timestamp_utc": "2020-01-01 00:30:00+00:00", "poa_direct": 200.0, "poa_diffuse": 30.0},
        ],
    )

    captured = {}

    def fake_aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
        captured["surface_tilt"] = surface_tilt
        captured["surface_azimuth"] = surface_azimuth
        captured["solar_zenith"] = list(solar_zenith)
        captured["solar_azimuth"] = list(solar_azimuth)
        return pd.Series([10.0, 70.0, 100.0], index=solar_zenith.index)

    def fake_relative_airmass(apparent_zenith, model):
        captured["apparent_zenith"] = list(apparent_zenith)
        captured["airmass_model"] = model
        return pd.Series([1.0, 2.0, np.nan], index=apparent_zenith.index)

    def fake_absolute_airmass(airmass_relative, pressure):
        captured["pressure"] = list(pressure)
        return airmass_relative * pressure / 100000.0

    def fake_iam(aoi):
        captured["aoi_for_iam"] = list(aoi)
        return pd.Series([0.8, 0.2, 0.0], index=aoi.index)

    monkeypatch.setattr(compute_effective_irradiance.pvlib.irradiance, "aoi", fake_aoi)
    monkeypatch.setattr(compute_effective_irradiance.pvlib.atmosphere, "get_relative_airmass", fake_relative_airmass)
    monkeypatch.setattr(compute_effective_irradiance.pvlib.atmosphere, "get_absolute_airmass", fake_absolute_airmass)
    monkeypatch.setattr(compute_effective_irradiance.pvlib.iam, "physical", fake_iam)

    compute_effective_irradiance.main()

    output = pd.read_csv(pv_test_repo.path("data/effective_irradiance.csv"), parse_dates=["timestamp_utc"])
    assert captured["surface_tilt"] == 20
    assert captured["surface_azimuth"] == 180
    assert captured["solar_zenith"] == [40.0, 50.0, 60.0]
    assert captured["solar_azimuth"] == [180.0, 190.0, 200.0]
    assert captured["apparent_zenith"][:2] == [39.5, 49.5]
    assert np.isnan(captured["apparent_zenith"][2])
    assert captured["airmass_model"] == "kastenyoung1989"
    assert captured["pressure"][:2] == [100000.0, 90000.0]
    assert np.isnan(captured["pressure"][2])
    assert captured["aoi_for_iam"] == [10.0, 70.0, 100.0]

    first = output.iloc[0]
    assert first["aoi_deg"] == pytest.approx(10.0)
    assert first["pressure_pa"] == pytest.approx(100000.0)
    assert first["airmass_relative"] == pytest.approx(1.0)
    assert first["airmass_absolute"] == pytest.approx(1.0)
    assert first["iam"] == pytest.approx(0.8)
    assert first["effective_irradiance"] == pytest.approx(100.0 * 0.8 + 20.0)

    missing_poa = output.iloc[1]
    assert np.isnan(missing_poa["poa_direct"])
    assert np.isnan(missing_poa["effective_irradiance"])

    missing_meteo = output.iloc[2]
    assert np.isnan(missing_meteo["pressure_pa"])
    assert np.isnan(missing_meteo["airmass_relative"])
    assert missing_meteo["effective_irradiance"] == pytest.approx(30.0)
