import copy
import math

import pandas as pd
import pytest

from battery_sim import simulator


BATTERY_COLUMNS = [
    "timestamp_utc",
    "action_kw",
    "actual_kw",
    "charge_ac_kwh",
    "discharge_ac_kwh",
    "loss_kwh",
    "soc_kwh",
    "soc_fraction",
    "capacity_kwh",
]


def action_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def valid_actions() -> pd.DataFrame:
    return action_df(
        [
            {
                "timestamp_utc": "2024-01-01 00:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            }
        ]
    )


def initial_state(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    *,
    start_soc_kwh: float = 50.0,
):
    return simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
        start_soc_kwh=start_soc_kwh,
    )


def test_simulate_period_rejects_invalid_timestep(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    with pytest.raises(ValueError, match="dt_h"):
        simulator.simulate_period(
            action_df=valid_actions(),
            battery_spec=valid_battery_spec,
            thermal_spec=valid_thermal_spec,
            degradation_spec=valid_degradation_spec,
            dt_h=0.0,
            state=initial_state(valid_battery_spec, valid_thermal_spec),
        )


def test_simulate_period_propagates_prepare_action_errors(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    with pytest.raises(ValueError, match="action_df missing columns"):
        simulator.simulate_period(
            action_df=pd.DataFrame(
                {
                    "timestamp_utc": ["2024-01-01 00:00:00+00:00"],
                    "action_kw": [0.0],
                }
            ),
            battery_spec=valid_battery_spec,
            thermal_spec=valid_thermal_spec,
            degradation_spec=valid_degradation_spec,
            dt_h=1.0,
            state=initial_state(valid_battery_spec, valid_thermal_spec),
        )


def test_simulate_period_propagates_invalid_specs(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    invalid_battery = copy.deepcopy(valid_battery_spec)
    invalid_battery["charge"]["max_kw"] = math.nan
    invalid_thermal = valid_thermal_spec.copy()
    invalid_thermal["heat_capacity_kwh_per_degC"] = 0.0
    invalid_degradation = valid_degradation_spec.copy()
    invalid_degradation["dod_exponent"] = 0.0

    cases = [
        (invalid_battery, valid_thermal_spec, valid_degradation_spec, "charge.max_kw"),
        (
            valid_battery_spec,
            invalid_thermal,
            valid_degradation_spec,
            "heat_capacity_kwh_per_degC",
        ),
        (
            valid_battery_spec,
            valid_thermal_spec,
            invalid_degradation,
            "dod_exponent",
        ),
    ]

    for battery_spec, thermal_spec, degradation_spec, error_match in cases:
        with pytest.raises(ValueError, match=error_match):
            simulator.simulate_period(
                action_df=valid_actions(),
                battery_spec=battery_spec,
                thermal_spec=thermal_spec,
                degradation_spec=degradation_spec,
                dt_h=1.0,
                state=initial_state(valid_battery_spec, valid_thermal_spec),
            )


def test_simulate_period_uses_state_capacity_without_mutating_input_spec(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    captured_capacity = []

    def fake_step_battery(**kwargs):
        captured_capacity.append(kwargs["spec"]["capacity_kwh"])
        return {
            "soc_after_kwh": kwargs["state"]["soc_kwh"],
            "charge_ac_kwh": 0.0,
            "discharge_ac_kwh": 0.0,
            "loss_kwh": 0.0,
        }

    monkeypatch.setattr(simulator, "step_battery", fake_step_battery)
    original_capacity = valid_battery_spec["capacity_kwh"]
    state = initial_state(valid_battery_spec, valid_thermal_spec, start_soc_kwh=40.0)
    state.capacity_kwh = 80.0

    battery_df, _temp_df, _degradation_df, _state = simulator.simulate_period(
        action_df=valid_actions(),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=state,
    )

    assert captured_capacity == pytest.approx([80.0])
    assert battery_df.iloc[0]["capacity_kwh"] == pytest.approx(80.0)
    assert battery_df.iloc[0]["soc_fraction"] == pytest.approx(0.5)
    assert valid_battery_spec["capacity_kwh"] == pytest.approx(original_capacity)


def test_simulate_period_processes_unsorted_actions_in_timestamp_order(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    actions = action_df(
        [
            {
                "timestamp_utc": "2024-01-01 02:00:00+00:00",
                "action_kw": 0.0,
                "ambient_temp_degC": 20.0,
            },
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

    battery_df, temperature_df, degradation_df, state = simulator.simulate_period(
        action_df=actions,
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(valid_battery_spec, valid_thermal_spec),
    )

    assert battery_df["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
        pd.Timestamp("2024-01-01 02:00:00+00:00"),
    ]
    assert (
        temperature_df["timestamp_utc"].tolist() == battery_df["timestamp_utc"].tolist()
    )
    assert degradation_df.empty
    assert state.current_month == (2024, 1)


def test_simulate_period_outputs_stable_frames_and_updates_state_buffers(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    battery_df, temperature_df, degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 10.0,
                    "ambient_temp_degC": 30.0,
                },
                {
                    "timestamp_utc": "2024-01-01 01:00:00+00:00",
                    "action_kw": -5.0,
                    "ambient_temp_degC": 25.0,
                },
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(valid_battery_spec, valid_thermal_spec),
    )

    assert list(battery_df.columns) == BATTERY_COLUMNS
    assert list(temperature_df.columns) == ["timestamp_utc", "battery_temp_degC"]
    assert degradation_df.empty
    assert len(battery_df) == 2
    assert len(temperature_df) == 2
    assert state.soc_kwh == pytest.approx(battery_df.iloc[-1]["soc_kwh"])
    assert state.battery_temp_degC != pytest.approx(20.0)
    assert state.month_soc == pytest.approx(battery_df["soc_fraction"].tolist())
    assert state.month_temp == pytest.approx(
        temperature_df["battery_temp_degC"].tolist()
    )
    assert state.month_power == pytest.approx(battery_df["actual_kw"].tolist())


def test_simulate_period_finalize_closes_last_period(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    _battery_df, _temperature_df, degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=initial_state(valid_battery_spec, valid_thermal_spec),
        finalize_period=True,
    )

    assert len(degradation_df) == 1
    assert degradation_df.iloc[0]["timestamp_utc"] == pd.Timestamp(
        "2024-01-01 01:00:00+00:00"
    )
    assert state.month_soc == []
    assert state.month_temp == []
    assert state.month_power == []
