import math

import pandas as pd
import pytest

from battery_sim import simulator


def battery_spec(**overrides) -> dict:
    spec = {
        "capacity_kwh": 100.0,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "charge": {
            "max_kw": 50.0,
            "eta_nominal": 0.96,
            "loss_factor_cold": 1.5,
            "loss_factor_hot": 1.3,
            "hard_min": 0.0,
            "optimal_min_temp": 5.0,
            "optimal_max_temp": 40.0,
            "hard_max": 45.0,
        },
        "discharge": {
            "max_kw": 50.0,
            "eta_nominal": 0.96,
            "loss_factor_cold": 1.5,
            "loss_factor_hot": 1.3,
            "hard_min": -20.0,
            "optimal_min_temp": -10.0,
            "optimal_max_temp": 45.0,
            "hard_max": 55.0,
        },
    }
    spec.update(overrides)
    return spec


def thermal_spec(**overrides) -> dict:
    spec = {
        "initial_temp_degC": 20.0,
        "thermal_time_constant_h": 6.0,
        "heat_capacity_kwh_per_degC": 50.0,
        "heat_to_battery_fraction": 1.0,
    }
    spec.update(overrides)
    return spec


def action_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def standard_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="h",
                tz="UTC",
            ),
            "action_kw": [10.0, -5.0, 0.0, 20.0],
            "ambient_temp_degC": [20.0, 21.0, 22.0, 23.0],
        }
    )


def expected_temperature(
    *,
    temp_before: float,
    spec: dict,
    ambient_temp_degC: float,
    heat_loss_kwh: float,
    dt_h: float,
) -> float:
    heat_to_battery_kw = heat_loss_kwh * spec["heat_to_battery_fraction"] / dt_h
    thermal_resistance = (
        spec["thermal_time_constant_h"] / spec["heat_capacity_kwh_per_degC"]
    )
    equilibrium_temp = ambient_temp_degC + thermal_resistance * heat_to_battery_kw
    decay = math.exp(-dt_h / spec["thermal_time_constant_h"])
    return equilibrium_temp + (temp_before - equilibrium_temp) * decay


def initial_state(start_soc_kwh: float = 50.0, **thermal_overrides):
    return simulator.initial_simulation_state(
        battery_spec(),
        thermal_spec(**thermal_overrides),
        start_soc_kwh=start_soc_kwh,
    )


def test_simulate_period_without_close_keeps_month_degradation_buffers(
    valid_degradation_spec: dict[str, float],
):
    actions = action_df(
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
        ]
    )

    _battery_df, _temp_df, degradation_df, state = simulator.simulate_period(
        action_df=actions,
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(),
    )

    assert degradation_df.empty
    assert state.current_month == (2024, 1)
    assert state.month_soc == pytest.approx([0.5, 0.5])
    assert state.month_temp == pytest.approx([20.0, 20.0])
    assert state.month_power == pytest.approx([0.0, 0.0])


def test_simulate_period_finalize_closes_month_and_writes_stable_columns(
    valid_degradation_spec: dict[str, float],
):
    actions = action_df(
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
            {
                "timestamp_utc": "2024-01-01 01:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
        ]
    )

    _battery_df, _temp_df, degradation_df, state = simulator.simulate_period(
        action_df=actions,
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(),
        finalize_period=True,
    )

    assert len(degradation_df) == 1
    assert list(degradation_df.columns) == [
        "timestamp_utc",
        "period_year",
        "period_month",
        "efc",
        "cycle_fade",
        "calendar_fade",
        "mean_calendar_stress_factor",
        "capacity_factor_before",
        "capacity_factor_after",
        "capacity_factor",
        "cumulative_efc",
        "calendar_days_elapsed",
    ]
    assert degradation_df.iloc[0]["timestamp_utc"] == pd.Timestamp(
        "2024-01-01 02:00:00+00:00"
    )
    assert degradation_df.iloc[0]["period_year"] == 2024
    assert degradation_df.iloc[0]["period_month"] == 1
    assert state.month_soc == []
    assert state.month_temp == []
    assert state.month_power == []


def test_simulate_period_month_change_closes_previous_month(
    valid_degradation_spec: dict[str, float],
):
    actions = action_df(
        [
            {
                "timestamp_utc": "2024-01-31 23:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
            {
                "timestamp_utc": "2024-02-01 00:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
        ]
    )

    _battery_df, _temp_df, degradation_df, state = simulator.simulate_period(
        action_df=actions,
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(),
    )

    assert len(degradation_df) == 1
    row = degradation_df.iloc[0]
    assert row["timestamp_utc"] == pd.Timestamp("2024-02-01 00:00:00+00:00")
    assert row["period_year"] == 2024
    assert row["period_month"] == 1
    assert state.current_month == (2024, 2)
    assert len(state.month_soc) == 1
    assert state.month_temp == pytest.approx([20.0])
    assert state.month_power == pytest.approx([0.0])


def test_close_period_passes_collected_series_to_degradation_update(
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}
    thermal = thermal_spec()

    def fake_update_degradation_for_period(**kwargs):
        captured.update(kwargs)
        new_state = kwargs["state"].copy()
        new_state["capacity_factor"] = 0.9
        new_state["calendar_days_elapsed"] += kwargs["period_days"]
        return new_state, {
            "efc": 0.0,
            "cycle_fade": 0.0,
            "calendar_fade": 0.0,
            "mean_calendar_stress_factor": 1.0,
            "capacity_factor_before": kwargs["state"]["capacity_factor"],
            "capacity_factor_after": new_state["capacity_factor"],
        }

    monkeypatch.setattr(
        simulator,
        "update_degradation_for_period",
        fake_update_degradation_for_period,
    )

    _battery_df, _temp_df, _degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 30.0,
                },
                {
                    "timestamp_utc": "2024-01-01 01:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 30.0,
                },
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=thermal,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(),
        finalize_period=True,
    )

    expected_after_first = expected_temperature(
        temp_before=20.0,
        spec=thermal,
        ambient_temp_degC=30.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )
    assert captured["state"]["capacity_factor"] == pytest.approx(1.0)
    assert captured["spec"] is valid_degradation_spec
    assert captured["soc_fraction_series"] == pytest.approx([0.5, 0.5])
    assert captured["battery_temp_degC_series"] == pytest.approx(
        [20.0, expected_after_first]
    )
    assert captured["power_kW_series"] == pytest.approx([0.0, 0.0])
    assert captured["nominal_capacity_kWh"] == pytest.approx(100.0)
    assert captured["period_days"] == pytest.approx(2.0 / 24.0)
    assert state.capacity_kwh == pytest.approx(90.0)


def test_close_period_updates_capacity_from_capacity_factor(
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_update_degradation_for_period(**kwargs):
        new_state = kwargs["state"].copy()
        new_state["capacity_factor"] = 0.75
        return new_state, {
            "efc": 0.0,
            "cycle_fade": 0.0,
            "calendar_fade": 0.0,
            "mean_calendar_stress_factor": 1.0,
            "capacity_factor_before": kwargs["state"]["capacity_factor"],
            "capacity_factor_after": 0.75,
        }

    monkeypatch.setattr(
        simulator,
        "update_degradation_for_period",
        fake_update_degradation_for_period,
    )

    _battery_df, _temp_df, degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(),
        finalize_period=True,
    )

    assert state.degradation_state["capacity_factor"] == pytest.approx(0.75)
    assert state.capacity_kwh == pytest.approx(75.0)
    assert degradation_df.iloc[0]["capacity_factor"] == pytest.approx(0.75)


def test_close_period_clamps_soc_to_new_capacity_limits(
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_update_degradation_for_period(**kwargs):
        new_state = kwargs["state"].copy()
        new_state["capacity_factor"] = 0.5
        return new_state, {
            "efc": 0.0,
            "cycle_fade": 0.0,
            "calendar_fade": 0.0,
            "mean_calendar_stress_factor": 1.0,
            "capacity_factor_before": kwargs["state"]["capacity_factor"],
            "capacity_factor_after": 0.5,
        }

    monkeypatch.setattr(
        simulator,
        "update_degradation_for_period",
        fake_update_degradation_for_period,
    )

    _battery_df, _temp_df, _degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(start_soc_kwh=90.0),
        finalize_period=True,
    )

    assert state.capacity_kwh == pytest.approx(50.0)
    assert state.soc_kwh == pytest.approx(50.0 * 0.95)


def test_chunked_simulation_matches_batch_simulation(
    valid_degradation_spec: dict[str, float],
):
    spec = battery_spec()
    thermal = thermal_spec()
    actions = standard_actions()

    batch_battery, _batch_temp, batch_degradation = simulator.simulate(
        actions,
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
    )

    state = simulator.initial_simulation_state(spec, thermal)
    first_battery, _first_temp, first_degradation, state = simulator.simulate_period(
        actions.iloc[:2],
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
        state=state,
    )
    second_battery, _second_temp, second_degradation, state = simulator.simulate_period(
        actions.iloc[2:],
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
        state=state,
        finalize_period=True,
    )

    period_battery = pd.concat([first_battery, second_battery], ignore_index=True)
    period_degradation = pd.concat(
        [first_degradation, second_degradation],
        ignore_index=True,
    )

    assert period_battery["soc_kwh"].tolist() == pytest.approx(
        batch_battery["soc_kwh"].tolist()
    )
    assert state.soc_kwh == pytest.approx(batch_battery.iloc[-1]["soc_kwh"])
    assert state.month_soc == []
    assert len(first_degradation) == 0
    assert len(period_degradation) == len(batch_degradation) == 1
    assert period_degradation.iloc[0]["capacity_factor"] == pytest.approx(
        batch_degradation.iloc[0]["capacity_factor"]
    )
