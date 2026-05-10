import pandas as pd
import pytest

import pv_sim.compute_effective_irradiance as effective_module
from pv_sim.compute_effective_irradiance import compute_effective_irradiance


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_effective_irradiance_merges_inputs_and_applies_iam_formula(
    tmp_path,
    monkeypatch,
):
    true_path = tmp_path / "true.csv"
    apparent_path = tmp_path / "apparent.csv"
    meteo_path = tmp_path / "meteo.csv"
    poa_path = tmp_path / "poa.csv"
    out_path = tmp_path / "out" / "effective.csv"
    write_csv(
        true_path,
        [
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "solar_zenith_deg": 50.0, "solar_azimuth_deg": 180.0},
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "solar_zenith_deg": 40.0, "solar_azimuth_deg": 170.0},
        ],
    )
    write_csv(
        apparent_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "apparent_zenith_deg": 42.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "apparent_zenith_deg": 52.0},
        ],
    )
    write_csv(
        meteo_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "PP_10": 1000.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "PP_10": 990.0},
        ],
    )
    write_csv(
        poa_path,
        [
            {"timestamp_utc": "2024-01-01 00:00:00+00:00", "poa_direct": 100.0, "poa_diffuse": 20.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "poa_direct": 200.0, "poa_diffuse": 30.0},
        ],
    )

    monkeypatch.setattr(effective_module.pvlib.irradiance, "aoi", lambda **kwargs: pd.Series([10.0, 20.0]))
    monkeypatch.setattr(effective_module.pvlib.atmosphere, "get_relative_airmass", lambda zenith, model: pd.Series([1.1, 1.2]))
    monkeypatch.setattr(effective_module.pvlib.atmosphere, "get_absolute_airmass", lambda rel, pressure: rel * pressure / 100000.0)
    monkeypatch.setattr(effective_module.pvlib.iam, "physical", lambda aoi: pd.Series([0.9, 0.8]))

    compute_effective_irradiance(
        true_sun_path=true_path,
        apparent_path=apparent_path,
        meteo_path=meteo_path,
        poa_path=poa_path,
        out_path=out_path,
        surface_tilt=30.0,
        surface_azimuth=180.0,
    )

    out = pd.read_csv(out_path)
    assert out["timestamp_utc"].tolist() == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 01:00:00+00:00",
    ]
    assert out["pressure_pa"].tolist() == pytest.approx([100000.0, 99000.0])
    assert out["iam"].tolist() == pytest.approx([0.9, 0.8])
    assert out["effective_irradiance"].tolist() == pytest.approx([110.0, 190.0])
