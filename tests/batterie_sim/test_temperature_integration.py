import math

import pandas as pd
import pytest

from battery_sim import simulator
from battery_sim.temp import ABSOLUTE_ZERO_DEGC


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


def degradation_spec(**overrides) -> dict:
    spec = {
        "cycle_fade_per_efc_at_100dod": 0.0001,
        "dod_exponent": 2.0,
        "cycle_reference_temp_degC": 25.0,
        "cycle_activation_energy_over_R_K": 3500.0,
        "calendar_fade_at_1yr": 0.03,
        "calendar_reference_temp_degC": 25.0,
        "calendar_activation_energy_over_R_K": 6000.0,
        "calendar_low_soc_reference": 0.2,
        "calendar_low_soc_factor": 1.5,
        "calendar_high_soc_reference": 0.8,
        "calendar_high_soc_factor": 2.5,
        "c_rate_reference": 0.5,
        "c_rate_exponent": 1.0,
    }
    spec.update(overrides)
    return spec


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


def initial_state(start_soc_kwh: float = 50.0, **thermal_overrides):
    return simulator.initial_simulation_state(
        battery_spec(),
        thermal_spec(**thermal_overrides),
        start_soc_kwh=start_soc_kwh,
    )


def test_initial_simulation_state_uses_valid_initial_temperature():
    state = initial_state(initial_temp_degC=12.5)

    assert state.battery_temp_degC == pytest.approx(12.5)
    assert state.capacity_kwh == pytest.approx(100.0)
    assert state.nominal_capacity_kwh == pytest.approx(100.0)


def test_initial_simulation_state_rejects_absolute_zero_initial_temperature():
    with pytest.raises(ValueError, match="initial_temp_degC"):
        simulator.initial_simulation_state(
            battery_spec(),
            thermal_spec(initial_temp_degC=ABSOLUTE_ZERO_DEGC),
            start_soc_kwh=50.0,
        )


def test_simulate_period_forwards_core_losses_to_temperature_step(monkeypatch):
    captured_heat_losses = []
    original_step_temperature = simulator.step_temperature

    def recording_step_temperature(**kwargs):
        captured_heat_losses.append(kwargs["heat_loss_kwh"])
        return original_step_temperature(**kwargs)

    monkeypatch.setattr(simulator, "step_temperature", recording_step_temperature)

    battery_df, _temperature_df, _degradation_df, _state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 10.0,
                    "ambient_temp_degC": 20.0,
                },
                {
                    "timestamp_utc": "2024-01-01 01:00:00+00:00",
                    "action_kw": -10.0,
                    "ambient_temp_degC": 20.0,
                },
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=thermal_spec(),
        degradation_spec=degradation_spec(),
        dt_h=1.0,
        state=initial_state(),
    )

    assert captured_heat_losses == pytest.approx(battery_df["loss_kwh"].tolist())


def test_simulate_period_uses_ambient_temperature_and_updates_state():
    spec = thermal_spec()

    _battery_df, temperature_df, _degradation_df, state = simulator.simulate_period(
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
                    "ambient_temp_degC": 10.0,
                },
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=spec,
        degradation_spec=degradation_spec(),
        dt_h=1.0,
        state=initial_state(),
    )

    expected_after_first = expected_temperature(
        temp_before=20.0,
        spec=spec,
        ambient_temp_degC=30.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )
    expected_after_second = expected_temperature(
        temp_before=expected_after_first,
        spec=spec,
        ambient_temp_degC=10.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )

    assert temperature_df["battery_temp_degC"].tolist() == pytest.approx(
        [20.0, expected_after_first]
    )
    assert state.battery_temp_degC == pytest.approx(expected_after_second)


def test_simulate_period_temperature_df_records_timestamp_and_pre_step_temperature():
    spec = thermal_spec(initial_temp_degC=18.0)
    state = initial_state(initial_temp_degC=18.0)

    _battery_df, temperature_df, _degradation_df, _state = simulator.simulate_period(
        action_df=action_df(
            [
                {
                    "timestamp_utc": "2024-01-01 01:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 24.0,
                },
                {
                    "timestamp_utc": "2024-01-01 00:00:00+00:00",
                    "action_kw": 0.0,
                    "ambient_temp_degC": 24.0,
                },
            ]
        ),
        battery_spec=battery_spec(),
        thermal_spec=spec,
        degradation_spec=degradation_spec(),
        dt_h=1.0,
        state=state,
    )

    expected_after_first = expected_temperature(
        temp_before=18.0,
        spec=spec,
        ambient_temp_degC=24.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )

    assert list(temperature_df.columns) == ["timestamp_utc", "battery_temp_degC"]
    assert temperature_df["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 01:00:00+00:00"),
    ]
    assert temperature_df["battery_temp_degC"].tolist() == pytest.approx(
        [18.0, expected_after_first]
    )


@pytest.mark.parametrize(
    ("ambient_temp_degC", "error_match"),
    [
        (math.nan, "ambient_temp_degC contains"),
        (math.inf, "ambient_temp_degC contains"),
        (ABSOLUTE_ZERO_DEGC, "ambient_temp_degC"),
        (ABSOLUTE_ZERO_DEGC - 1.0, "ambient_temp_degC"),
    ],
)
def test_simulate_period_rejects_invalid_ambient_temperatures(
    ambient_temp_degC,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        simulator.simulate_period(
            action_df=action_df(
                [
                    {
                        "timestamp_utc": "2024-01-01 00:00:00+00:00",
                        "action_kw": 0.0,
                        "ambient_temp_degC": ambient_temp_degC,
                    }
                ]
            ),
            battery_spec=battery_spec(),
            thermal_spec=thermal_spec(),
            degradation_spec=degradation_spec(),
            dt_h=1.0,
            state=initial_state(),
        )
