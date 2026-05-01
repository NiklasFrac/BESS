from __future__ import annotations

import pandas as pd
import pytest

from pv_sim import true_pos


def test_load_station_row_accepts_station_id(tmp_path):
    metadata_path = tmp_path / "metadata_stations.csv"
    pd.DataFrame(
        {
            "station_id": ["00232", "00233"],
            "station_name": ["Augsburg", "Munich"],
            "latitude": [48.4253, 48.1351],
            "longitude": [10.9417, 11.5820],
            "height_m_amsl": [462.0, 520.0],
        }
    ).to_csv(metadata_path, index=False)

    row = true_pos._load_station_row(metadata_path, "00232")

    assert row["station_name"] == "Augsburg"
    assert float(row["latitude"]) == pytest.approx(48.4253)


def test_main_writes_interval_midpoint_and_end_times(pv_test_repo, patch_repo_root, monkeypatch):
    patch_repo_root(true_pos)
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

    def fake_get_solarposition(time, latitude, longitude, altitude):
        assert latitude == pytest.approx(48.4253)
        assert longitude == pytest.approx(10.9417)
        assert altitude == pytest.approx(462.0)
        return pd.DataFrame(
            {
                "zenith": [80.0, 70.0, 60.0],
                "elevation": [10.0, 20.0, 30.0],
                "azimuth": [120.0, 130.0, 140.0],
            },
            index=time,
        )

    monkeypatch.setattr(true_pos.pvlib.solarposition, "get_solarposition", fake_get_solarposition)

    true_pos.main()

    output = pd.read_csv(
        pv_test_repo.path("data/true_sun.csv"),
        parse_dates=["solar_position_reference_utc", "timestamp_utc"],
        dtype={"station_id": str},
    )
    assert output["solar_position_reference_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2020-01-01 00:05:00+00:00",
                "2020-01-01 00:15:00+00:00",
                "2020-01-01 00:25:00+00:00",
            ],
            utc=True,
        )
    )
    assert output["timestamp_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2020-01-01 00:10:00+00:00",
                "2020-01-01 00:20:00+00:00",
                "2020-01-01 00:30:00+00:00",
            ],
            utc=True,
        )
    )
    assert output["station_id"].astype(str).tolist() == ["00232", "00232", "00232"]
    assert output["solar_zenith_deg"].tolist() == [80.0, 70.0, 60.0]
    assert output["solar_elevation_deg"].tolist() == [10.0, 20.0, 30.0]
    assert output["solar_azimuth_deg"].tolist() == [120.0, 130.0, 140.0]


def test_main_rejects_non_positive_frequency(pv_test_repo, patch_repo_root):
    patch_repo_root(true_pos)
    pv_test_repo.config["time"]["freq"] = "0min"
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

    with pytest.raises(ValueError, match="Invalid freq"):
        true_pos.main()
