import pandas as pd
import pytest

import pv_sim.true_pos as true_pos_module
from pv_sim.true_pos import _load_station_row, compute_true_sun_position


def write_metadata(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def station_rows():
    return [
        {
            "station_id": "0232",
            "station_name": "Augsburg",
            "latitude": 48.425,
            "longitude": 10.942,
            "height_m_amsl": 461.0,
        }
    ]


def test_load_station_row_matches_case_insensitive_name_and_station_id(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, station_rows())

    by_name = _load_station_row(metadata_path, "  augsburg ")
    by_id = _load_station_row(metadata_path, "0232")

    assert by_name["station_id"] == "0232"
    assert by_id["station_name"] == "Augsburg"


def test_load_station_row_rejects_missing_or_ambiguous_station(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, station_rows() + station_rows())

    with pytest.raises(ValueError, match="not unique"):
        _load_station_row(metadata_path, "Augsburg")

    write_metadata(metadata_path, station_rows())
    with pytest.raises(ValueError, match="not found"):
        _load_station_row(metadata_path, "Munich")


def test_compute_true_sun_position_uses_interval_midpoints_and_end_timestamps(
    tmp_path,
    monkeypatch,
):
    metadata_path = tmp_path / "metadata.csv"
    out_path = tmp_path / "out" / "true.csv"
    write_metadata(metadata_path, station_rows())
    captured = {}

    def fake_get_solarposition(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "zenith": [50.0, 51.0],
                "elevation": [40.0, 39.0],
                "azimuth": [170.0, 171.0],
            },
            index=kwargs["time"],
        )

    monkeypatch.setattr(
        true_pos_module.pvlib.solarposition,
        "get_solarposition",
        fake_get_solarposition,
    )

    compute_true_sun_position(
        metadata_path=metadata_path,
        out_path=out_path,
        station_name="Augsburg",
        start_utc="2024-01-01 00:00:00+00:00",
        end_utc="2024-01-01 02:00:00+00:00",
        freq="1h",
    )

    out = pd.read_csv(out_path)
    assert captured["latitude"] == pytest.approx(48.425)
    assert captured["longitude"] == pytest.approx(10.942)
    assert list(captured["time"]) == [
        pd.Timestamp("2024-01-01 00:30:00+00:00"),
        pd.Timestamp("2024-01-01 01:30:00+00:00"),
    ]
    assert out["timestamp_utc"].tolist() == [
        "2024-01-01 01:00:00+00:00",
        "2024-01-01 02:00:00+00:00",
    ]
    assert out["solar_zenith_deg"].tolist() == pytest.approx([50.0, 51.0])


def test_compute_true_sun_position_rejects_non_positive_frequency(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, station_rows())

    with pytest.raises(ValueError, match="Invalid freq"):
        compute_true_sun_position(
            metadata_path,
            tmp_path / "true.csv",
            "Augsburg",
            "2024-01-01",
            "2024-01-02",
            "0h",
        )
