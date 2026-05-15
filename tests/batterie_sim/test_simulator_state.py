import copy
import math

import pandas as pd
import pytest

from battery_sim import simulator


def test_initial_simulation_state_defaults_to_soc_min(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    state = simulator.initial_simulation_state(valid_battery_spec, valid_thermal_spec)

    assert state.soc_kwh == pytest.approx(5.0)
    assert state.battery_temp_degC == pytest.approx(20.0)
    assert state.nominal_capacity_kwh == pytest.approx(100.0)
    assert state.capacity_kwh == pytest.approx(100.0)
    assert state.degradation_state == {
        "capacity_factor": 1.0,
        "cumulative_efc": 0.0,
        "cycle_fade": 0.0,
        "calendar_fade": 0.0,
        "calendar_days_elapsed": 0.0,
    }
    assert state.current_month is None
    assert state.month_soc == []
    assert state.month_temp == []
    assert state.month_power == []


def test_initial_simulation_state_accepts_explicit_start_soc(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    state = simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
        start_soc_kwh=50.0,
    )

    assert state.soc_kwh == pytest.approx(50.0)


def test_initial_simulation_state_rejects_invalid_start_soc(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    with pytest.raises(ValueError, match="start_soc_kwh"):
        simulator.initial_simulation_state(
            valid_battery_spec,
            valid_thermal_spec,
            start_soc_kwh=95.001,
        )


def test_initial_simulation_state_propagates_invalid_battery_spec(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    invalid_spec = copy.deepcopy(valid_battery_spec)
    invalid_spec["capacity_kwh"] = math.nan

    with pytest.raises(ValueError, match="capacity_kwh"):
        simulator.initial_simulation_state(invalid_spec, valid_thermal_spec)


def test_initial_simulation_state_propagates_invalid_thermal_spec(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    invalid_spec = valid_thermal_spec.copy()
    invalid_spec["initial_temp_degC"] = math.nan

    with pytest.raises(ValueError, match="initial_temp_degC"):
        simulator.initial_simulation_state(valid_battery_spec, invalid_spec)


def test_initial_simulation_state_uses_independent_mutable_defaults(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
):
    first = simulator.initial_simulation_state(valid_battery_spec, valid_thermal_spec)
    second = simulator.initial_simulation_state(valid_battery_spec, valid_thermal_spec)

    first.month_soc.append(0.5)
    first.month_temp.append(20.0)
    first.month_power.append(1.0)
    first.degradation_state["capacity_factor"] = 0.5

    assert second.month_soc == []
    assert second.month_temp == []
    assert second.month_power == []
    assert second.degradation_state["capacity_factor"] == pytest.approx(1.0)


def test_close_degradation_period_without_month_data_returns_none_and_keeps_state(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    state = simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
        start_soc_kwh=50.0,
    )
    before = copy.deepcopy(state)

    row = simulator._close_degradation_period(
        state=state,
        battery_spec=valid_battery_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=1.0,
        timestamp_utc=pd.Timestamp("2024-01-01 00:00:00+00:00"),
    )

    assert row is None
    assert state == before


def test_close_degradation_period_updates_state_and_returns_row(
    valid_battery_spec: dict,
    valid_thermal_spec: dict[str, float],
    valid_degradation_spec: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_update_degradation_for_period(**kwargs):
        captured.update(kwargs)
        new_state = kwargs["state"].copy()
        new_state["capacity_factor"] = 0.5
        new_state["calendar_days_elapsed"] += kwargs["period_days"]
        new_state["calendar_fade"] = 0.1
        return new_state, {
            "efc": 1.25,
            "cycle_fade": 0.02,
            "calendar_fade": 0.1,
            "mean_calendar_stress_factor": 1.5,
            "capacity_factor_before": kwargs["state"]["capacity_factor"],
            "capacity_factor_after": 0.5,
        }

    monkeypatch.setattr(
        simulator,
        "update_degradation_for_period",
        fake_update_degradation_for_period,
    )
    state = simulator.initial_simulation_state(
        valid_battery_spec,
        valid_thermal_spec,
        start_soc_kwh=90.0,
    )
    state.current_month = (2024, 1)
    state.month_soc = [0.9, 0.8]
    state.month_temp = [20.0, 21.0]
    state.month_power = [10.0, -5.0]

    row = simulator._close_degradation_period(
        state=state,
        battery_spec=valid_battery_spec,
        degradation_spec=valid_degradation_spec,
        dt_h=0.5,
        timestamp_utc="2024-02-01 00:00:00+00:00",
    )

    assert row is not None
    assert captured["soc_fraction_series"] == pytest.approx([0.9, 0.8])
    assert captured["battery_temp_degC_series"] == pytest.approx([20.0, 21.0])
    assert captured["power_kW_series"] == pytest.approx([10.0, -5.0])
    assert captured["nominal_capacity_kWh"] == pytest.approx(100.0)
    assert captured["period_days"] == pytest.approx(2 * 0.5 / 24.0)
    assert state.degradation_state["capacity_factor"] == pytest.approx(0.5)
    assert state.capacity_kwh == pytest.approx(50.0)
    assert state.soc_kwh == pytest.approx(47.5)
    assert state.month_soc == []
    assert state.month_temp == []
    assert state.month_power == []
    assert row["timestamp_utc"] == pd.Timestamp("2024-02-01 00:00:00+00:00")
    assert row["period_year"] == 2024
    assert row["period_month"] == 1
    assert row["efc"] == pytest.approx(1.25)
    assert row["capacity_factor"] == pytest.approx(0.5)
    assert row["capacity_factor_after"] == pytest.approx(0.5)
