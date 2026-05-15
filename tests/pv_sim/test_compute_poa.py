import pandas as pd
import pytest

import pv_sim.compute_poa as compute_poa_module
from pv_sim.compute_poa import compute_poa


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_poa_applies_horizon_shading_and_clips_negative_irradiance(
    tmp_path,
    monkeypatch,
):
    dni_path = tmp_path / "dni.csv"
    apparent_path = tmp_path / "apparent.csv"
    horizon_path = tmp_path / "horizon.csv"
    out_path = tmp_path / "out" / "poa.csv"
    write_csv(
        dni_path,
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "ghi_wm2": -5.0,
                "dhi_wm2": -1.0,
                "dni_wm2": 100.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "ghi_wm2": 50.0,
                "dhi_wm2": 10.0,
                "dni_wm2": 120.0,
            },
        ],
    )
    write_csv(
        apparent_path,
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "apparent_zenith_deg": 70.0,
                "apparent_elevation_deg": 5.0,
                "apparent_azimuth_deg": 0.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "apparent_zenith_deg": 60.0,
                "apparent_elevation_deg": 20.0,
                "apparent_azimuth_deg": 90.0,
            },
        ],
    )
    write_csv(
        horizon_path,
        [
            {"azimuth_deg": 0.0, "horizon_height_deg": 10.0},
            {"azimuth_deg": 90.0, "horizon_height_deg": 10.0},
            {"azimuth_deg": 180.0, "horizon_height_deg": 10.0},
            {"azimuth_deg": 270.0, "horizon_height_deg": 10.0},
        ],
    )

    captured = {}

    def fake_airmass(zenith):
        return pd.Series([1.0] * len(zenith), index=zenith.index)

    def fake_extra_radiation(index):
        return pd.Series([1367.0] * len(index), index=index)

    def fake_total_irradiance(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "poa_global": kwargs["dni"].to_numpy()
                + kwargs["ghi"].to_numpy()
                + kwargs["dhi"].to_numpy(),
                "poa_direct": kwargs["dni"].to_numpy(),
                "poa_diffuse": kwargs["dhi"].to_numpy(),
            }
        )

    monkeypatch.setattr(
        compute_poa_module.pvlib.atmosphere, "get_relative_airmass", fake_airmass
    )
    monkeypatch.setattr(
        compute_poa_module.pvlib.irradiance, "get_extra_radiation", fake_extra_radiation
    )
    monkeypatch.setattr(
        compute_poa_module.pvlib.irradiance,
        "get_total_irradiance",
        fake_total_irradiance,
    )

    compute_poa(
        dni_path=dni_path,
        apparent_path=apparent_path,
        horizon_path=horizon_path,
        out_path=out_path,
        surface_tilt=30.0,
        surface_azimuth=180.0,
        albedo=0.2,
    )

    out = pd.read_csv(out_path)
    assert pd.to_datetime(out["timestamp_utc"], utc=True).tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
    ]
    assert captured["dni"].tolist() == pytest.approx([0.0, 120.0])
    assert captured["ghi"].tolist() == pytest.approx([0.0, 50.0])
    assert captured["dhi"].tolist() == pytest.approx([0.0, 10.0])
    assert out["poa_direct"].tolist() == pytest.approx([0.0, 120.0])
    assert out["poa_global"].tolist() == pytest.approx([0.0, 180.0])


def test_compute_poa_uses_inner_timestamp_join(tmp_path, monkeypatch):
    dni_path = tmp_path / "dni.csv"
    apparent_path = tmp_path / "apparent.csv"
    horizon_path = tmp_path / "horizon.csv"
    out_path = tmp_path / "poa.csv"
    write_csv(
        dni_path,
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "ghi_wm2": 10.0,
                "dhi_wm2": 5.0,
                "dni_wm2": 20.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "ghi_wm2": 10.0,
                "dhi_wm2": 5.0,
                "dni_wm2": 20.0,
            },
        ],
    )
    write_csv(
        apparent_path,
        [
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "apparent_zenith_deg": 60.0,
                "apparent_elevation_deg": 20.0,
                "apparent_azimuth_deg": 90.0,
            }
        ],
    )
    write_csv(
        horizon_path,
        [
            {"azimuth_deg": 0.0, "horizon_height_deg": 0.0},
            {"azimuth_deg": 180.0, "horizon_height_deg": 0.0},
        ],
    )

    monkeypatch.setattr(
        compute_poa_module.pvlib.atmosphere,
        "get_relative_airmass",
        lambda zenith: pd.Series([1.0] * len(zenith)),
    )
    monkeypatch.setattr(
        compute_poa_module.pvlib.irradiance,
        "get_extra_radiation",
        lambda index: pd.Series([1367.0] * len(index)),
    )
    monkeypatch.setattr(
        compute_poa_module.pvlib.irradiance,
        "get_total_irradiance",
        lambda **kwargs: pd.DataFrame({"poa_global": kwargs["dni"].to_numpy()}),
    )

    compute_poa(dni_path, apparent_path, horizon_path, out_path, 30.0, 180.0, 0.2)

    out = pd.read_csv(out_path)
    assert len(out) == 1
