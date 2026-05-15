import pandas as pd

from battery_sim.simulator import simulate


def test_battery_sim_e2e_smoke_across_month_boundary(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    actions = pd.DataFrame(
        {
            "timestamp_utc": [
                "2024-01-31 23:00:00+00:00",
                "2024-02-01 00:00:00+00:00",
                "2024-02-01 01:00:00+00:00",
                "2024-02-01 02:00:00+00:00",
            ],
            "action_kw": [30.0, -10.0, 0.0, 20.0],
            "ambient_temp_degC": [10.0, 12.0, 14.0, 16.0],
        }
    )

    battery_df, temperature_df, degradation_df = simulate(
        action_df=actions,
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
    )

    assert len(battery_df) == len(actions)
    assert len(temperature_df) == len(actions)
    assert len(degradation_df) == 2
    assert battery_df["timestamp_utc"].is_monotonic_increasing
    assert (
        temperature_df["timestamp_utc"].tolist() == battery_df["timestamp_utc"].tolist()
    )

    assert (
        battery_df["soc_fraction"]
        .between(
            valid_battery_spec["soc_min"],
            valid_battery_spec["soc_max"],
        )
        .all()
    )
    assert (battery_df["capacity_kwh"] > 0).all()
    assert (battery_df["loss_kwh"] >= 0).all()
    assert battery_df.iloc[0]["actual_kw"] > 0.0
    assert battery_df.iloc[1]["actual_kw"] < 0.0
    assert battery_df.iloc[2]["actual_kw"] == 0.0

    assert degradation_df["period_month"].tolist() == [1, 2]
    assert degradation_df["capacity_factor"].between(0.0, 1.0).all()
    assert degradation_df["calendar_days_elapsed"].is_monotonic_increasing
