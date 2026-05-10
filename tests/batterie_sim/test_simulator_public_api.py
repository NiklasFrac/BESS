import pandas as pd

from battery_sim import simulator


def action_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="h",
                tz="UTC",
            ),
            "action_kw": [0.0, 10.0, -5.0],
            "ambient_temp_degC": [20.0, 21.0, 22.0],
        }
    )


def test_simulate_returns_three_dataframes_and_finalizes_last_period(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    result = simulator.simulate(
        action_df(),
        valid_battery_spec,
        valid_thermal_spec,
        valid_degradation_spec,
        dt_h=1.0,
    )

    assert len(result) == 3
    battery_df, temperature_df, degradation_df = result
    assert isinstance(battery_df, pd.DataFrame)
    assert isinstance(temperature_df, pd.DataFrame)
    assert isinstance(degradation_df, pd.DataFrame)
    assert len(battery_df) == 3
    assert len(temperature_df) == 3
    assert len(degradation_df) == 1
    assert degradation_df.iloc[0]["timestamp_utc"] == pd.Timestamp(
        "2024-01-01 03:00:00+00:00"
    )


def test_simulate_matches_manual_initial_state_and_finalized_period(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    actions = action_df()

    batch_battery, batch_temperature, batch_degradation = simulator.simulate(
        actions,
        valid_battery_spec,
        valid_thermal_spec,
        valid_degradation_spec,
        dt_h=1.0,
    )

    state = simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
    )
    period_battery, period_temperature, period_degradation, _state = (
        simulator.simulate_period(
            action_df=actions,
            battery_spec=valid_battery_spec,
            thermal_spec=valid_thermal_spec,
            degradation_spec=valid_degradation_spec,
            dt_h=1.0,
            state=state,
            finalize_period=True,
        )
    )

    pd.testing.assert_frame_equal(batch_battery, period_battery)
    pd.testing.assert_frame_equal(batch_temperature, period_temperature)
    pd.testing.assert_frame_equal(batch_degradation, period_degradation)
