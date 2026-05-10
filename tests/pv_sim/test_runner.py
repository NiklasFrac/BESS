from pathlib import Path

import pv_sim.runner as runner_module
from pv_sim.runner import PvSimParams, PvSimPaths, run_pv_sim


def paths() -> PvSimPaths:
    base = Path("data")
    return PvSimPaths(
        metadata=base / "metadata.csv",
        meteo=base / "meteo.csv",
        solar=base / "solar.csv",
        horizon=base / "horizon.csv",
        true_sun_position=base / "true.csv",
        apparent=base / "apparent.csv",
        dni=base / "dni.csv",
        poa=base / "poa.csv",
        effective_irradiance=base / "effective.csv",
        energy=base / "energy.csv",
        pv_output=base / "pv_output.csv",
        energy_plot=base / "energy.png",
        horizon_plot=base / "horizon.png",
    )


def params() -> PvSimParams:
    return PvSimParams(
        station_name="Augsburg",
        start_utc="2024-01-01",
        end_utc="2024-01-02",
        freq="1h",
        timestamp_col="timestamp_utc",
        missing_value=-999.0,
        solar_unit="wm2",
        surface_tilt=30.0,
        surface_azimuth=180.0,
        albedo=0.2,
        module_pdc0=400.0,
        module_count=10,
        gamma_pdc=-0.003,
        annual_age_loss_pct=1.0,
        pac0_each=3000.0,
        inverter_count=1,
        eta_inv_nom=0.96,
    )


def test_run_pv_sim_calls_pipeline_steps_in_order(monkeypatch):
    calls = []

    def recorder(name):
        def record(**kwargs):
            calls.append((name, kwargs))

        return record

    monkeypatch.setattr(runner_module, "compute_true_sun_position", recorder("true"))
    monkeypatch.setattr(runner_module, "compute_apparent_sun_position", recorder("apparent"))
    monkeypatch.setattr(runner_module, "compute_dni", recorder("dni"))
    monkeypatch.setattr(runner_module, "compute_poa", recorder("poa"))
    monkeypatch.setattr(runner_module, "compute_effective_irradiance", recorder("effective"))
    monkeypatch.setattr(runner_module, "compute_energy", recorder("energy"))
    monkeypatch.setattr(runner_module, "plot_energy_overview", lambda *args: calls.append(("energy_plot", args)))
    monkeypatch.setattr(runner_module, "plot_horizon_profile", lambda *args: calls.append(("horizon_plot", args)))

    p = paths()
    cfg = params()

    run_pv_sim(p, cfg)

    assert [call[0] for call in calls] == [
        "true",
        "apparent",
        "dni",
        "poa",
        "effective",
        "energy",
        "energy_plot",
        "horizon_plot",
    ]
    assert calls[0][1]["metadata_path"] == p.metadata
    assert calls[2][1]["solar_unit"] == "wm2"
    assert calls[5][1]["pv_output_path"] == p.pv_output
    assert calls[-1][1][-1] == "Augsburg"
