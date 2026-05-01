from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_sim import modul_sim


def test_main_computes_temperature_losses_inverter_and_energy(
    pv_test_repo, patch_repo_root, monkeypatch
):
    patch_repo_root(modul_sim)
    start = "2020-01-01 00:00:00+00:00"
    one_year = "2020-12-31 06:00:00+00:00"
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [
            {"timestamp_utc": start, "TT_10": 20.0, "FF_10": 1.0},
            {"timestamp_utc": one_year, "TT_10": 20.0, "FF_10": 2.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/poa.csv",
        [
            {"timestamp_utc": start, "poa_global": -50.0},
            {"timestamp_utc": one_year, "poa_global": 1000.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/effective_irradiance.csv",
        [
            {"timestamp_utc": start, "effective_irradiance": -10.0},
            {"timestamp_utc": one_year, "effective_irradiance": 800.0},
        ],
    )

    captured = {}

    def fake_faiman(poa_global, temp_air, wind_speed, u0, u1):
        captured["faiman_poa"] = list(poa_global)
        captured["faiman_temp"] = list(temp_air)
        captured["faiman_wind"] = list(wind_speed)
        captured["faiman_u0"] = u0
        captured["faiman_u1"] = u1
        return temp_air + poa_global / 100.0

    def fake_pvwatts_dc(effective_irradiance, temp_cell, pdc0, gamma_pdc):
        captured["dc_effective"] = list(effective_irradiance)
        captured["dc_temp_cell"] = list(temp_cell)
        captured["dc_pdc0"] = pdc0
        captured["dc_gamma"] = gamma_pdc
        return effective_irradiance / 1000.0 * pdc0 * (1 + gamma_pdc * (temp_cell - 25.0))

    def fake_pvwatts_losses(**kwargs):
        captured["losses"] = kwargs
        return kwargs["age"] + 10.0

    def fake_inverter_pvwatts(pdc, pdc0, eta_inv_nom):
        captured["inverter_pdc"] = list(pdc)
        captured["inverter_pdc0"] = pdc0
        captured["eta_inv_nom"] = eta_inv_nom
        return pdc * eta_inv_nom

    monkeypatch.setattr(modul_sim.pvlib.temperature, "faiman", fake_faiman)
    monkeypatch.setattr(modul_sim.pvlib.pvsystem, "pvwatts_dc", fake_pvwatts_dc)
    monkeypatch.setattr(modul_sim.pvlib.pvsystem, "pvwatts_losses", fake_pvwatts_losses)
    monkeypatch.setattr(modul_sim.pvlib.inverter, "pvwatts", fake_inverter_pvwatts)

    modul_sim.main()

    output = pd.read_csv(pv_test_repo.path("results/energy_curve.csv"), parse_dates=["timestamp_utc"])
    assert captured["faiman_poa"] == [0.0, 1000.0]
    assert captured["faiman_u0"] == pytest.approx(20.0)
    assert captured["faiman_u1"] == pytest.approx(5.0)
    assert captured["dc_effective"] == [0.0, 800.0]
    assert captured["dc_pdc0"] == pytest.approx(1000.0)
    assert captured["dc_gamma"] == pytest.approx(-0.003)
    assert captured["losses"]["soiling"] == 2
    assert captured["losses"]["shading"] == 3
    assert captured["losses"]["availability"] == 3
    assert list(captured["losses"]["age"]) == [pytest.approx(0.0), pytest.approx(0.5)]
    assert captured["inverter_pdc0"] == pytest.approx(800.0 / 0.96)
    assert captured["eta_inv_nom"] == pytest.approx(0.96)

    assert output.iloc[0]["age_loss_pct"] == pytest.approx(0.0)
    assert output.iloc[0]["p_dc_gross_w"] == pytest.approx(0.0)
    assert output.iloc[0]["p_ac_w"] == pytest.approx(0.0)
    assert output.iloc[1]["age_loss_pct"] == pytest.approx(0.5)

    expected_gross = 800.0 / 1000.0 * 1000.0 * (1 + (-0.003) * (30.0 - 25.0))
    expected_net = expected_gross * (1 - (10.0 + 0.5) / 100.0)
    expected_ac = expected_net * 0.96
    assert output.iloc[1]["p_dc_gross_w"] == pytest.approx(expected_gross)
    assert output.iloc[1]["p_dc_net_w"] == pytest.approx(expected_net)
    assert output.iloc[1]["p_ac_w"] == pytest.approx(expected_ac)
    assert output.iloc[1]["e_net_ac_kwh"] == pytest.approx(expected_ac / 1000.0 * (10.0 / 60.0))


def test_main_rejects_duplicate_timestamps_in_modul_merge(pv_test_repo, patch_repo_root):
    patch_repo_root(modul_sim)
    timestamp = "2020-01-01 00:00:00+00:00"
    pv_test_repo.write_csv("data/poa.csv", [{"timestamp_utc": timestamp, "poa_global": 100.0}])
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [
            {"timestamp_utc": timestamp, "TT_10": 20.0, "FF_10": 1.0},
            {"timestamp_utc": timestamp, "TT_10": 21.0, "FF_10": 1.1},
        ],
    )
    pv_test_repo.write_csv(
        "data/effective_irradiance.csv",
        [{"timestamp_utc": timestamp, "effective_irradiance": 80.0}],
    )

    with pytest.raises(pd.errors.MergeError):
        modul_sim.main()
