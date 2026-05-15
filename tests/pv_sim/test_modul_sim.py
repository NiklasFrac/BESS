import pandas as pd
import pytest

import pv_sim.modul_sim as modul_sim_module
from pv_sim.modul_sim import compute_energy


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_energy_builds_energy_and_optimizer_output(
    tmp_path,
    monkeypatch,
):
    meteo_path = tmp_path / "meteo.csv"
    poa_path = tmp_path / "poa.csv"
    effective_path = tmp_path / "effective.csv"
    out_path = tmp_path / "out" / "energy.csv"
    pv_output_path = tmp_path / "out" / "pv_output.csv"
    write_csv(
        meteo_path,
        [
            {"timestamp_utc": "2025-01-01 01:00:00+00:00", "TT_10": 20.0, "FF_10": 1.0},
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "TT_10": 10.0, "FF_10": 2.0},
        ],
    )
    write_csv(
        poa_path,
        [
            {"timestamp_utc": "2024-01-01 01:00:00+00:00", "poa_global": -50.0},
            {"timestamp_utc": "2025-01-01 01:00:00+00:00", "poa_global": 100.0},
        ],
    )
    write_csv(
        effective_path,
        [
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "effective_irradiance": -10.0,
            },
            {
                "timestamp_utc": "2025-01-01 01:00:00+00:00",
                "effective_irradiance": 500.0,
            },
        ],
    )

    captured = {}
    monkeypatch.setattr(
        modul_sim_module.pvlib.temperature,
        "faiman",
        lambda poa_global, temp_air, wind_speed, u0, u1: temp_air + poa_global / 100.0,
    )
    monkeypatch.setattr(
        modul_sim_module.pvlib.pvsystem,
        "pvwatts_dc",
        lambda effective_irradiance, temp_cell, pdc0, gamma_pdc: (
            effective_irradiance / 1000.0 * pdc0
        ),
    )

    def fake_losses(**kwargs):
        captured["age"] = kwargs["age"].copy()
        return pd.Series([10.0, 20.0])

    monkeypatch.setattr(modul_sim_module.pvlib.pvsystem, "pvwatts_losses", fake_losses)
    monkeypatch.setattr(
        modul_sim_module.pvlib.inverter,
        "pvwatts",
        lambda pdc, pdc0, eta_inv_nom: pdc * eta_inv_nom,
    )

    compute_energy(
        meteo_path=meteo_path,
        poa_path=poa_path,
        effective_irradiance_path=effective_path,
        out_path=out_path,
        pv_output_path=pv_output_path,
        module_pdc0=400.0,
        module_count=10,
        gamma_pdc=-0.003,
        annual_age_loss_pct=1.0,
        pac0_each=3000.0,
        inverter_count=1,
        eta_inv_nom=0.96,
        freq="30min",
    )

    out = pd.read_csv(out_path)
    pv_out = pd.read_csv(pv_output_path)
    assert out["timestamp_utc"].tolist() == [
        "2024-01-01 01:00:00+00:00",
        "2025-01-01 01:00:00+00:00",
    ]
    assert out["t_module_faiman_c"].tolist() == pytest.approx([10.0, 21.0])
    assert out["p_dc_gross_w"].tolist() == pytest.approx([0.0, 2000.0])
    assert out["p_dc_net_w"].tolist() == pytest.approx([0.0, 1600.0])
    assert out["p_ac_w"].tolist() == pytest.approx([0.0, 1536.0])
    assert out["e_net_ac_kwh"].tolist() == pytest.approx([0.0, 0.768])
    assert captured["age"].iloc[0] == pytest.approx(0.0)
    assert captured["age"].iloc[1] > 0.99
    assert pv_out.columns.tolist() == ["timestamp_utc", "pv_kw", "ambient_temp_degC"]
    assert pv_out["pv_kw"].tolist() == pytest.approx([0.0, 1.536])
