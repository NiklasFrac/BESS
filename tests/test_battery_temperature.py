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
    thermal_resistance = spec["thermal_time_constant_h"] / spec["heat_capacity_kwh_per_degC"]
    equilibrium_temp = ambient_temp_degC + thermal_resistance * heat_to_battery_kw
    decay = math.exp(-dt_h / spec["thermal_time_constant_h"])
    return equilibrium_temp + (temp_before - equilibrium_temp) * decay


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_validate_thermal_spec_accepts_valid_spec_and_fraction_limits(fraction):
    validate_thermal_spec(base_spec(heat_to_battery_fraction=fraction))


@pytest.mark.parametrize("thermal_time_constant_h", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_validate_thermal_spec_rejects_invalid_time_constant(thermal_time_constant_h):
    with pytest.raises(ValueError, match="thermal_time_constant_h"):
        validate_thermal_spec(base_spec(thermal_time_constant_h=thermal_time_constant_h))


@pytest.mark.parametrize("heat_capacity_kwh_per_degC", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_validate_thermal_spec_rejects_invalid_heat_capacity(heat_capacity_kwh_per_degC):
    with pytest.raises(ValueError, match="heat_capacity_kwh_per_degC"):
        validate_thermal_spec(base_spec(heat_capacity_kwh_per_degC=heat_capacity_kwh_per_degC))


@pytest.mark.parametrize("heat_to_battery_fraction", [-0.01, 1.01, math.inf, -math.inf, math.nan])
def test_validate_thermal_spec_rejects_invalid_heat_fraction(heat_to_battery_fraction):
    with pytest.raises(ValueError, match="heat_to_battery_fraction"):
        validate_thermal_spec(base_spec(heat_to_battery_fraction=heat_to_battery_fraction))


@pytest.mark.parametrize("initial_temp_degC", [math.inf, -math.inf, math.nan])
def test_validate_thermal_spec_rejects_non_finite_initial_temperature(initial_temp_degC):
    with pytest.raises(ValueError, match="initial_temp_degC"):
        validate_thermal_spec(base_spec(initial_temp_degC=initial_temp_degC))


@pytest.mark.parametrize(
    ("temp_before", "ambient_temp_degC", "heat_loss_kwh", "fraction", "dt_h"),
    [
        (20.0, 20.0, 0.0, 1.0, 1.0),
        (10.0, 25.0, 0.0, 1.0, 0.5),
        (40.0, 5.0, 1.2, 0.5, 2.0),
        (-5.0, -10.0, 0.3, 1.0, 1 / 6),
    ],
)
def test_step_temperature_matches_rc_formula(
    temp_before,
    ambient_temp_degC,
    heat_loss_kwh,
    fraction,
    dt_h,
):
    spec = base_spec(heat_to_battery_fraction=fraction)
    result = step_temperature(
        state={"battery_temp_degC": temp_before},
        spec=spec,
        ambient_temp_degC=ambient_temp_degC,
        heat_loss_kwh=heat_loss_kwh,
        dt_h=dt_h,
    )

    expected = expected_temperature(
        temp_before=temp_before,
        spec=spec,
        ambient_temp_degC=ambient_temp_degC,
        heat_loss_kwh=heat_loss_kwh,
        dt_h=dt_h,
    )
    assert result["battery_temp_degC"] == pytest.approx(expected)


def test_step_temperature_stays_constant_without_losses_at_ambient_temperature():
    result = step_temperature(
        state={"battery_temp_degC": 17.0},
        spec=base_spec(),
        ambient_temp_degC=17.0,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )
    assert result["battery_temp_degC"] == pytest.approx(17.0)


@pytest.mark.parametrize(
    ("temp_before", "ambient_temp_degC"),
    [
        (10.0, 20.0),
        (30.0, 20.0),
    ],
)
def test_step_temperature_moves_monotonically_toward_ambient_without_losses(
    temp_before,
    ambient_temp_degC,
):
    result = step_temperature(
        state={"battery_temp_degC": temp_before},
        spec=base_spec(),
        ambient_temp_degC=ambient_temp_degC,
        heat_loss_kwh=0.0,
        dt_h=1.0,
    )

    temp_after = result["battery_temp_degC"]
    assert min(temp_before, ambient_temp_degC) <= temp_after <= max(temp_before, ambient_temp_degC)
    assert abs(temp_after - ambient_temp_degC) < abs(temp_before - ambient_temp_degC)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_step_temperature_heat_input_scales_with_fraction(fraction):
    spec = base_spec(heat_to_battery_fraction=fraction)
    result = step_temperature(
        state={"battery_temp_degC": 20.0},
        spec=spec,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.0,
        dt_h=1.0,
    )
    expected = expected_temperature(
        temp_before=20.0,
        spec=spec,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.0,
        dt_h=1.0,
    )
    assert result["battery_temp_degC"] == pytest.approx(expected)


def test_step_temperature_large_timestep_converges_to_equilibrium_temperature():
    spec = base_spec(thermal_time_constant_h=1.0, heat_capacity_kwh_per_degC=10.0)
    heat_loss_kwh = 2.0
    dt_h = 20.0
    thermal_resistance = spec["thermal_time_constant_h"] / spec["heat_capacity_kwh_per_degC"]
    equilibrium_temp = 5.0 + thermal_resistance * (heat_loss_kwh / dt_h)

    result = step_temperature(
        state={"battery_temp_degC": 80.0},
        spec=spec,
        ambient_temp_degC=5.0,
        heat_loss_kwh=heat_loss_kwh,
        dt_h=dt_h,
    )

    assert result["battery_temp_degC"] == pytest.approx(equilibrium_temp, abs=1e-6)


def test_step_temperature_tiny_timestep_without_losses_changes_only_tiny_amount():
    result = step_temperature(
        state={"battery_temp_degC": 20.0},
        spec=base_spec(),
        ambient_temp_degC=40.0,
        heat_loss_kwh=0.0,
        dt_h=1e-6,
    )
    assert result["battery_temp_degC"] - 20.0 == pytest.approx(20.0 * (1 - math.exp(-1e-6 / 6.0)))


def test_step_temperature_many_small_steps_match_one_large_step_for_same_heat_power():
    spec = base_spec()
    heat_power_kw = 2.0
    one_step = step_temperature(
        state={"battery_temp_degC": 20.0},
        spec=spec,
        ambient_temp_degC=15.0,
        heat_loss_kwh=heat_power_kw * 1.0,
        dt_h=1.0,
    )

    state = {"battery_temp_degC": 20.0}
    for _ in range(6):
        state = step_temperature(
            state=state,
            spec=spec,
            ambient_temp_degC=15.0,
            heat_loss_kwh=heat_power_kw * (1 / 6),
            dt_h=1 / 6,
        )

    assert state["battery_temp_degC"] == pytest.approx(one_step["battery_temp_degC"])


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
    assert result["battery_temp_before_degC"] == pytest.approx(20.0)
    assert result["ambient_temp_degC"] == pytest.approx(10.0)
    assert result["heat_loss_kwh"] == pytest.approx(4.0)
    assert result["heat_to_battery_kwh"] == pytest.approx(1.0)


@pytest.mark.parametrize("ambient_temp_degC", [math.inf, -math.inf, math.nan])
def test_step_temperature_rejects_non_finite_ambient_temperature(ambient_temp_degC):
    with pytest.raises(ValueError, match="ambient_temp_degC"):
        step_temperature(
            state={"battery_temp_degC": 20.0},
            spec=base_spec(),
            ambient_temp_degC=ambient_temp_degC,
            heat_loss_kwh=0.0,
            dt_h=1.0,
        )


@pytest.mark.parametrize("heat_loss_kwh", [-0.01, math.inf, -math.inf, math.nan])
def test_step_temperature_rejects_invalid_heat_loss(heat_loss_kwh):
    with pytest.raises(ValueError, match="heat_loss_kwh"):
        step_temperature(
            state={"battery_temp_degC": 20.0},
            spec=base_spec(),
            ambient_temp_degC=20.0,
            heat_loss_kwh=heat_loss_kwh,
            dt_h=1.0,
        )


@pytest.mark.parametrize("dt_h", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_step_temperature_rejects_invalid_timestep(dt_h):
    with pytest.raises(ValueError, match="dt_h"):
        step_temperature(
            state={"battery_temp_degC": 20.0},
            spec=base_spec(),
            ambient_temp_degC=20.0,
            heat_loss_kwh=0.0,
            dt_h=dt_h,
        )


@pytest.mark.parametrize("battery_temp_degC", [math.inf, -math.inf, math.nan])
def test_step_temperature_rejects_non_finite_state_temperature(battery_temp_degC):
    with pytest.raises(ValueError, match="battery_temp_degC"):
        step_temperature(
            state={"battery_temp_degC": battery_temp_degC},
            spec=base_spec(),
            ambient_temp_degC=20.0,
            heat_loss_kwh=0.0,
            dt_h=1.0,
        )
