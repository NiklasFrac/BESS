from __future__ import annotations

import pandas as pd

from pv_sim import (
    compute_dni,
    compute_effective_irradiance,
    compute_poa,
    modul_sim,
    seen_pos,
    true_pos,
)


def test_recomputes_mini_pipeline_from_inputs_to_energy_curve(pv_test_repo, patch_repo_root):
    for module in [
        true_pos,
        seen_pos,
        compute_dni,
        compute_poa,
        compute_effective_irradiance,
        modul_sim,
    ]:
        patch_repo_root(module)

    pv_test_repo.config["time"]["start_utc"] = "2020-06-21 10:00:00"
    pv_test_repo.config["time"]["end_utc"] = "2020-06-21 10:30:00"
    pv_test_repo.write_config()
    pv_test_repo.write_csv(
        "data/metadata_stations.csv",
        [
            {
                "station_id": "00232",
                "station_name": "Augsburg",
                "latitude": 48.4253,
                "longitude": 10.9417,
                "height_m_amsl": 462.0,
            }
        ],
    )
    pv_test_repo.write_csv(
        "data/meteo.csv",
        [
            {"timestamp_utc": "2020-06-21 10:00:00+00:00", "TT_10": 24.0, "PP_10": 980.0, "FF_10": 2.0},
            {"timestamp_utc": "2020-06-21 10:10:00+00:00", "TT_10": 25.0, "PP_10": 980.0, "FF_10": 2.2},
            {"timestamp_utc": "2020-06-21 10:20:00+00:00", "TT_10": 26.0, "PP_10": 980.0, "FF_10": 2.4},
            {"timestamp_utc": "2020-06-21 10:30:00+00:00", "TT_10": 27.0, "PP_10": 980.0, "FF_10": 2.6},
        ],
    )
    pv_test_repo.write_csv(
        "data/solar.csv",
        [
            {"timestamp_utc": "2020-06-21 10:10:00+00:00", "GS_10": 54.0, "DS_10": 12.0},
            {"timestamp_utc": "2020-06-21 10:20:00+00:00", "GS_10": 57.0, "DS_10": 13.0},
            {"timestamp_utc": "2020-06-21 10:30:00+00:00", "GS_10": 60.0, "DS_10": 14.0},
        ],
    )
    pv_test_repo.write_csv(
        "data/horizon.csv",
        [
            {"azimuth_deg": 0.0, "horizon_height_deg": 0.0},
            {"azimuth_deg": 90.0, "horizon_height_deg": 0.0},
            {"azimuth_deg": 180.0, "horizon_height_deg": 0.0},
            {"azimuth_deg": 270.0, "horizon_height_deg": 0.0},
        ],
    )

    true_pos.main()
    seen_pos.main()
    compute_dni.main()
    compute_poa.main()
    compute_effective_irradiance.main()
    modul_sim.main()

    energy = pd.read_csv(pv_test_repo.path("results/energy_curve.csv"), parse_dates=["timestamp_utc"])
    assert len(energy) == 3
    assert energy["timestamp_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2020-06-21 10:10:00+00:00",
                "2020-06-21 10:20:00+00:00",
                "2020-06-21 10:30:00+00:00",
            ],
            utc=True,
        )
    )
    assert (energy["poa_global"] > 0).all()
    assert (energy["effective_irradiance"] > 0).all()
    assert (energy["p_dc_gross_w"] >= energy["p_dc_net_w"]).all()
    assert (energy["p_ac_w"] >= 0).all()
    assert (energy["e_net_ac_kwh"] >= 0).all()
    assert energy["e_net_ac_kwh"].sum() > 0
