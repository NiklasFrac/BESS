import copy
import math

import pandas as pd
import pytest

from battery_sim import simulator


def action_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


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


def test_simulate_period_passes_current_battery_temperature_to_core_step(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    captured_temps = []
    original_step_battery = simulator.step_battery

    def recording_step_battery(**kwargs):
        captured_temps.append(kwargs["battery_temp_degC"])
        return original_step_battery(**kwargs)

    monkeypatch.setattr(simulator, "step_battery", recording_step_battery)

    simulator.simulate_period(
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
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=simulator.initial_simulation_state(
            valid_battery_spec,
            valid_thermal_spec,
            start_soc_kwh=50.0,
        ),
    )

    expected_after_first = expected_temperature(
        temp_before=20.0,
        spec=valid_thermal_spec,
        ambient_temp_degC=30.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )
    assert captured_temps == pytest.approx([20.0, expected_after_first])


def test_simulate_period_battery_df_uses_core_step_result_fields(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    captured_kwargs = {}

    def fake_step_battery(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "soc_after_kwh": 60.0,
            "charge_ac_kwh": 3.0,
            "discharge_ac_kwh": 1.0,
            "loss_kwh": 0.5,
        }

    monkeypatch.setattr(simulator, "step_battery", fake_step_battery)

    battery_df, _temp_df, _degradation_df, state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 7.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=2.0,
        state=simulator.initial_simulation_state(
            valid_battery_spec,
            valid_thermal_spec,
            start_soc_kwh=50.0,
        ),
    )

    assert captured_kwargs["state"] == {"soc_kwh": 50.0}
    assert captured_kwargs["action_kw"] == pytest.approx(7.0)
    assert captured_kwargs["dt_h"] == pytest.approx(2.0)
    assert list(battery_df.columns) == [
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
    row = battery_df.iloc[0]
    assert row["actual_kw"] == pytest.approx((3.0 - 1.0) / 2.0)
    assert row["charge_ac_kwh"] == pytest.approx(3.0)
    assert row["discharge_ac_kwh"] == pytest.approx(1.0)
    assert row["loss_kwh"] == pytest.approx(0.5)
    assert row["soc_kwh"] == pytest.approx(60.0)
    assert row["soc_fraction"] == pytest.approx(0.6)
    assert row["capacity_kwh"] == pytest.approx(100.0)
    assert state.soc_kwh == pytest.approx(60.0)


def test_simulate_period_actual_kw_signs_for_charge_discharge_and_idle(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
):
    battery_df, _temp_df, _degradation_df, _state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 10.0,
                    "ambient_temp_degC": 20.0,
                },
                {
                    "timestamp_utc": "2024-01-01 01:00:00+00:00",
                    "action_kw": -5.0,
                    "ambient_temp_degC": 20.0,
                },
                {
                    "timestamp_utc": "2024-01-01 02:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 20.0,
                },
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=simulator.initial_simulation_state(
            valid_battery_spec,
            valid_thermal_spec,
            start_soc_kwh=50.0,
        ),
    )

    assert battery_df.iloc[0]["actual_kw"] > 0.0
    assert battery_df.iloc[1]["actual_kw"] < 0.0
    assert battery_df.iloc[2]["actual_kw"] == pytest.approx(0.0)


def test_simulate_period_actual_kw_is_reduced_by_temperature_limit(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
):
    thermal = copy.deepcopy(valid_thermal_spec)
    thermal["initial_temp_degC"] = 0.0

    battery_df, _temp_df, _degradation_df, _state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 10.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=thermal,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=simulator.initial_simulation_state(
            valid_battery_spec,
            thermal,
            start_soc_kwh=50.0,
        ),
    )

    assert battery_df.iloc[0]["actual_kw"] == pytest.approx(0.0)
    assert battery_df.iloc[0]["actual_kw"] < battery_df.iloc[0]["action_kw"]


def test_simulate_period_actual_kw_is_reduced_by_soc_limit(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
):
    battery_df, _temp_df, _degradation_df, _state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 50.0,
                    "ambient_temp_degC": 20.0,
                }
            ]
        ),
        battery_spec=valid_battery_spec,
        thermal_spec=valid_thermal_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        state=simulator.initial_simulation_state(
            valid_battery_spec,
            valid_thermal_spec,
            start_soc_kwh=94.0,
        ),
    )

    assert 0.0 < battery_df.iloc[0]["actual_kw"] < battery_df.iloc[0]["action_kw"]


def test_initial_simulation_state_propagates_invalid_battery_spec(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
):
    spec = copy.deepcopy(valid_battery_spec)
    spec["capacity_kwh"] = math.nan

    with pytest.raises(ValueError, match="capacity_kwh"):
        simulator.initial_simulation_state(spec, valid_thermal_spec)


def test_simulate_period_propagates_invalid_battery_spec(
    valid_battery_spec: dict,
    valid_thermal_spec: dict,
    valid_degradation_spec: dict[str, float],
):
    invalid_spec = copy.deepcopy(valid_battery_spec)
    invalid_spec["charge"]["max_kw"] = math.nan
    state = simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
        start_soc_kwh=50.0,
    )

    with pytest.raises(ValueError, match="charge.max_kw"):
        simulator.simulate_period(
            action_df=action_df(
                [
                    {
                        "timestamp_utc": "2024-01-01 00:00:00+00:00",
                        "action_kw": 0.0,
                        "ambient_temp_degC": 20.0,
                    }
                ]
            ),
            battery_spec=invalid_spec,
            thermal_spec=valid_thermal_spec,
            degradation_spec=valid_degradation_spec,
            dt_h=1.0,
            state=state,
        )
