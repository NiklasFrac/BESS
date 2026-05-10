import matplotlib
import pandas as pd

matplotlib.use("Agg")

from pv_sim.runner import PvSimParams, PvSimPaths, run_pv_sim


def test_pv_sim_e2e_smoke_runs_full_pipeline_with_mini_csvs(tmp_path):
    paths = PvSimPaths(
        metadata=tmp_path / "metadata.csv",
        meteo=tmp_path / "meteo.csv",
        solar=tmp_path / "solar.csv",
        horizon=tmp_path / "horizon.csv",
        true_sun_position=tmp_path / "debug" / "true_sun.csv",
        apparent=tmp_path / "debug" / "apparent.csv",
        dni=tmp_path / "debug" / "dni.csv",
        poa=tmp_path / "debug" / "poa.csv",
        effective_irradiance=tmp_path / "debug" / "effective.csv",
        energy=tmp_path / "energy.csv",
        pv_output=tmp_path / "pv_output.csv",
        energy_plot=tmp_path / "plots" / "energy.png",
        horizon_plot=tmp_path / "plots" / "horizon.png",
    )
    params = PvSimParams(
        station_name="Augsburg",
        start_utc="2024-06-21 09:00:00+00:00",
        end_utc="2024-06-21 12:00:00+00:00",
        freq="1h",
        timestamp_col="timestamp_utc",
        missing_value=-999.0,
        solar_unit="wm2",
        surface_tilt=30.0,
        surface_azimuth=180.0,
        albedo=0.2,
        module_pdc0=400.0,
        module_count=12,
        gamma_pdc=-0.0035,
        annual_age_loss_pct=0.5,
        pac0_each=4000.0,
        inverter_count=1,
        eta_inv_nom=0.96,
    )

    pd.DataFrame(
        [
            {
                "station_id": "0232",
                "station_name": "Augsburg",
                "latitude": 48.425,
                "longitude": 10.942,
                "height_m_amsl": 461.0,
            }
        ]
    ).to_csv(paths.metadata, index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-06-21 09:00:00+00:00",
                periods=4,
                freq="1h",
            ),
            "TT_10": [20.0, 22.0, 24.0, 25.0],
            "PP_10": [1000.0, 1000.0, 999.0, 999.0],
            "FF_10": [2.0, 2.5, 3.0, 3.0],
        }
    ).to_csv(paths.meteo, index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-06-21 10:00:00+00:00",
                periods=3,
                freq="1h",
            ),
            "GS_10": [700.0, 820.0, 760.0],
            "DS_10": [120.0, 130.0, 140.0],
        }
    ).to_csv(paths.solar, index=False)
    pd.DataFrame(
        {
            "azimuth_deg": [0.0, 90.0, 180.0, 270.0],
            "horizon_height_deg": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(paths.horizon, index=False)

    run_pv_sim(paths, params)

    expected_rows = 3
    for output_path in (
        paths.true_sun_position,
        paths.apparent,
        paths.dni,
        paths.poa,
        paths.effective_irradiance,
        paths.energy,
        paths.pv_output,
    ):
        assert output_path.exists()
        assert len(pd.read_csv(output_path)) == expected_rows

    energy = pd.read_csv(paths.energy)
    pv_output = pd.read_csv(paths.pv_output)
    assert energy["p_ac_w"].notna().all()
    assert energy["e_net_ac_kwh"].notna().all()
    assert energy["e_net_ac_kwh"].sum() > 0.0
    assert pv_output.columns.tolist() == ["timestamp_utc", "pv_kw", "ambient_temp_degC"]
    assert pv_output["pv_kw"].max() > 0.0
    assert paths.energy_plot.exists() and paths.energy_plot.stat().st_size > 0
    assert paths.horizon_plot.exists() and paths.horizon_plot.stat().st_size > 0
