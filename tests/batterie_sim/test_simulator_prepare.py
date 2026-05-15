import pandas as pd
import pytest

from battery_sim.simulator import _prepare_action_df


def test_prepare_action_df_selects_sorts_converts_and_resets_index():
    raw = pd.DataFrame(
        {
            "timestamp_utc": [
                "2024-01-01 03:00:00+00:00",
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
            ],
            "action_kw": ["2.5", "0", "1.5"],
            "ambient_temp_degC": ["13.0", "10.0", "11.0"],
            "ignored": ["x", "y", "z"],
        },
        index=[10, 20, 30],
    )

    prepared = _prepare_action_df(raw)

    assert list(prepared.columns) == [
        "timestamp_utc",
        "action_kw",
        "ambient_temp_degC",
    ]
    assert list(prepared.index) == [0, 1, 2]
    assert prepared["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
        pd.Timestamp("2024-01-01 03:00:00+00:00"),
    ]
    assert prepared["action_kw"].tolist() == pytest.approx([0.0, 1.5, 2.5])
    assert prepared["ambient_temp_degC"].tolist() == pytest.approx([10.0, 11.0, 13.0])


def test_prepare_action_df_converts_aware_timestamps_to_utc():
    raw = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2024-01-01 01:00:00", tz="Europe/Berlin"),
                pd.Timestamp("2024-01-01 02:00:00", tz="Europe/Berlin"),
            ],
            "action_kw": [0.0, 1.0],
            "ambient_temp_degC": [20.0, 21.0],
        }
    )

    prepared = _prepare_action_df(raw)

    assert prepared["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
    ]


def test_prepare_action_df_does_not_mutate_input_dataframe():
    raw = pd.DataFrame(
        {
            "timestamp_utc": ["2024-01-01 00:00:00+00:00"],
            "action_kw": ["1.0"],
            "ambient_temp_degC": ["20.0"],
            "ignored": ["keep"],
        }
    )
    before = raw.copy(deep=True)

    _prepare_action_df(raw)

    pd.testing.assert_frame_equal(raw, before)


def test_prepare_action_df_rejects_bad_shape_or_values():
    with pytest.raises(ValueError, match="action_df missing columns"):
        _prepare_action_df(pd.DataFrame({"timestamp_utc": ["2024-01-01"]}))

    with pytest.raises(ValueError, match="action_df must not be empty"):
        _prepare_action_df(
            pd.DataFrame(columns=["timestamp_utc", "action_kw", "ambient_temp_degC"])
        )

    with pytest.raises((TypeError, ValueError), match="timestamp|Unknown datetime"):
        _prepare_action_df(
            pd.DataFrame(
                {
                    "timestamp_utc": ["not-a-timestamp"],
                    "action_kw": [0.0],
                    "ambient_temp_degC": [20.0],
                }
            )
        )

    with pytest.raises((TypeError, ValueError), match="action_kw|Unable to parse"):
        _prepare_action_df(
            pd.DataFrame(
                {
                    "timestamp_utc": ["2024-01-01 00:00:00+00:00"],
                    "action_kw": ["bad"],
                    "ambient_temp_degC": [20.0],
                }
            )
        )
