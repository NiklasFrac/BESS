import math

import pytest

from battery_sim.temp import step_temperature, validate_thermal_spec


def base_spec(**overrides) -> dict:
    spec = {
        "initial_temp_degC": 20.0,
        "thermal_time_constant_h": 6.0,
        "heat_capacity_kwh_per_degC": 50.0,
        "heat_to_battery_fraction": 1.0,
    }
    spec.update(overrides)
    return spec


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


def test_validate_thermal_spec_rejects_invalid_heat_fraction():
    with pytest.raises(ValueError, match="heat_to_battery_fraction"):
        validate_thermal_spec(base_spec(heat_to_battery_fraction=1.01))


def test_step_temperature_matches_rc_formula():
    spec = base_spec(heat_to_battery_fraction=0.5)

    result = step_temperature(
        state={"battery_temp_degC": 40.0},
        spec=spec,
        ambient_temp_degC=5.0,
        heat_loss_kwh=1.2,
        dt_h=2.0,
    )

    assert result["battery_temp_degC"] == pytest.approx(
        expected_temperature(
            temp_before=40.0,
            spec=spec,
            ambient_temp_degC=5.0,
            heat_loss_kwh=1.2,
            dt_h=2.0,
        )
    )


def test_step_temperature_moves_toward_ambient_without_losses():
    result = step_temperature(
        state={"battery_temp_degC": 10.0},
        spec=base_spec(),
        ambient_temp_degC=20.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )

    assert 10.0 < result["battery_temp_degC"] < 20.0


def test_step_temperature_same_heat_loss_in_shorter_step_has_higher_heat_power_effect():
    spec = base_spec()
    one_hour = step_temperature(
        state={"battery_temp_degC": 20.0},
        spec=spec,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.0,
        dt_h=1.0,
    )
    half_hour = step_temperature(
        state={"battery_temp_degC": 20.0},
        spec=spec,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.0,
        dt_h=0.5,
    )

    assert half_hour["battery_temp_degC"] > one_hour["battery_temp_degC"]


def test_step_temperature_returns_audit_fields_and_does_not_mutate_input_state():
    state = {"battery_temp_degC": 20.0}

    result = step_temperature(
        state=state,
        spec=base_spec(heat_to_battery_fraction=0.25),
        ambient_temp_degC=10.0,
        heat_loss_kwh=4.0,
        dt_h=1.0,
    )

    assert state == {"battery_temp_degC": 20.0}
    assert set(result) == {
        "battery_temp_degC",
        "battery_temp_before_degC",
        "ambient_temp_degC",
        "heat_loss_kwh",
        "heat_to_battery_kwh",
    }
    assert result["heat_to_battery_kwh"] == pytest.approx(1.0)


def test_step_temperature_rejects_invalid_step_inputs():
    with pytest.raises(ValueError, match="ambient_temp_degC"):
        step_temperature({"battery_temp_degC": 20.0}, base_spec(), -273.15, 0.0, 1.0)
    with pytest.raises(ValueError, match="heat_loss_kwh"):
        step_temperature({"battery_temp_degC": 20.0}, base_spec(), 20.0, -0.01, 1.0)
    with pytest.raises(ValueError, match="dt_h"):
        step_temperature({"battery_temp_degC": 20.0}, base_spec(), 20.0, 0.0, 0.0)
