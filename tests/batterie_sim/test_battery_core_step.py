import pytest

from battery_sim.battery_core import step


EXPECTED_RESULT_KEYS = {
    "soc_before_kwh",
    "soc_after_kwh",
    "charge_ac_kwh",
    "discharge_ac_kwh",
    "charge_power_limited_ac_kwh",
    "discharge_power_limited_ac_kwh",
    "loss_kwh",
    "charge_allowed",
    "discharge_allowed",
    "charge_temp_limited_ac_kwh",
    "discharge_temp_limited_ac_kwh",
    "eta_charge_effective",
    "eta_discharge_effective",
    "charge_power_factor",
    "discharge_power_factor",
    "max_charge_kw_effective",
    "max_discharge_kw_effective",
    "charge_soc_limited_ac_kwh",
    "discharge_soc_limited_ac_kwh",
}


def test_step_idle_keeps_soc_and_returns_stable_audit_fields(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=0.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["soc_before_kwh"] == pytest.approx(50.0)
    assert result["soc_after_kwh"] == pytest.approx(50.0)
    assert state_at_soc_mid["soc_kwh"] == pytest.approx(50.0)
    assert result["loss_kwh"] == pytest.approx(0.0)
    assert result["eta_charge_effective"] is None
    assert result["eta_discharge_effective"] is None


def test_step_charge_applies_eta_and_mutates_state(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=10.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    expected_delta_soc = 10.0 * valid_battery_spec["charge"]["eta_nominal"]
    assert result["eta_charge_effective"] == pytest.approx(0.96)
    assert result["soc_after_kwh"] == pytest.approx(50.0 + expected_delta_soc)
    assert state_at_soc_mid["soc_kwh"] == pytest.approx(result["soc_after_kwh"])
    assert result["charge_ac_kwh"] == pytest.approx(10.0)
    assert result["loss_kwh"] == pytest.approx(10.0 - expected_delta_soc)


def test_step_charge_reports_power_limit(
    valid_battery_spec: dict,
    state_at_soc_min: dict[str, float],
):
    result = step(
        state=state_at_soc_min,
        spec=valid_battery_spec,
        action_kw=100.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    assert result["charge_ac_kwh"] == pytest.approx(50.0)
    assert result["charge_power_limited_ac_kwh"] == pytest.approx(50.0)
    assert result["charge_temp_limited_ac_kwh"] == pytest.approx(0.0)


def test_step_charge_reports_temperature_limit(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=50.0,
        dt_h=1.0,
        battery_temp_degC=2.5,
    )

    assert result["charge_power_factor"] == pytest.approx(0.5)
    assert result["max_charge_kw_effective"] == pytest.approx(25.0)
    assert result["charge_temp_limited_ac_kwh"] == pytest.approx(25.0)
    assert result["eta_charge_effective"] == pytest.approx(0.95)
    assert result["soc_after_kwh"] - result["soc_before_kwh"] == pytest.approx(
        25.0 * 0.95
    )


def test_step_charge_hard_temperature_boundary_disallows_charging(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=10.0,
        dt_h=1.0,
        battery_temp_degC=0.0,
    )

    assert result["charge_allowed"] is False
    assert result["charge_ac_kwh"] == pytest.approx(0.0)
    assert result["eta_charge_effective"] is None
    assert result["soc_after_kwh"] == pytest.approx(result["soc_before_kwh"])


def test_step_charge_soc_max_limit_caps_energy(valid_battery_spec: dict):
    state = {"soc_kwh": 94.0}

    result = step(
        state=state,
        spec=valid_battery_spec,
        action_kw=50.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    assert result["soc_after_kwh"] == pytest.approx(95.0)
    assert state["soc_kwh"] == pytest.approx(95.0)
    assert result["charge_ac_kwh"] == pytest.approx(1.0 / 0.96)
    assert result["charge_soc_limited_ac_kwh"] == pytest.approx(50.0 - 1.0 / 0.96)


def test_step_discharge_applies_eta_and_mutates_state(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=-10.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    expected_delta_soc = 10.0 / valid_battery_spec["discharge"]["eta_nominal"]
    assert result["eta_discharge_effective"] == pytest.approx(0.96)
    assert result["soc_after_kwh"] == pytest.approx(50.0 - expected_delta_soc)
    assert state_at_soc_mid["soc_kwh"] == pytest.approx(result["soc_after_kwh"])
    assert result["discharge_ac_kwh"] == pytest.approx(10.0)
    assert result["loss_kwh"] == pytest.approx(expected_delta_soc - 10.0)


def test_step_discharge_reports_power_limit(
    valid_battery_spec: dict,
    state_at_soc_max: dict[str, float],
):
    result = step(
        state=state_at_soc_max,
        spec=valid_battery_spec,
        action_kw=-100.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    assert result["discharge_ac_kwh"] == pytest.approx(50.0)
    assert result["discharge_power_limited_ac_kwh"] == pytest.approx(50.0)
    assert result["discharge_temp_limited_ac_kwh"] == pytest.approx(0.0)


def test_step_discharge_reports_temperature_limit(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=-50.0,
        dt_h=1.0,
        battery_temp_degC=50.0,
    )

    assert result["discharge_power_factor"] == pytest.approx(0.5)
    assert result["max_discharge_kw_effective"] == pytest.approx(25.0)
    assert result["discharge_temp_limited_ac_kwh"] == pytest.approx(25.0)
    assert result["eta_discharge_effective"] == pytest.approx(0.954)
    assert result["soc_before_kwh"] - result["soc_after_kwh"] == pytest.approx(
        25.0 / 0.954
    )


def test_step_discharge_hard_temperature_boundary_disallows_discharging(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=-10.0,
        dt_h=1.0,
        battery_temp_degC=55.0,
    )

    assert result["discharge_allowed"] is False
    assert result["discharge_ac_kwh"] == pytest.approx(0.0)
    assert result["eta_discharge_effective"] is None
    assert result["soc_after_kwh"] == pytest.approx(result["soc_before_kwh"])


def test_step_discharge_soc_min_limit_caps_energy(valid_battery_spec: dict):
    state = {"soc_kwh": 6.0}

    result = step(
        state=state,
        spec=valid_battery_spec,
        action_kw=-50.0,
        dt_h=1.0,
        battery_temp_degC=20.0,
    )

    assert result["soc_after_kwh"] == pytest.approx(5.0)
    assert state["soc_kwh"] == pytest.approx(5.0)
    assert result["discharge_ac_kwh"] == pytest.approx(0.96)
    assert result["discharge_soc_limited_ac_kwh"] == pytest.approx(50.0 - 0.96)


def test_step_allowed_flags_depend_on_mode_temperature_windows(
    valid_battery_spec: dict,
    state_at_soc_mid: dict[str, float],
):
    result = step(
        state=state_at_soc_mid,
        spec=valid_battery_spec,
        action_kw=0.0,
        dt_h=1.0,
        battery_temp_degC=-15.0,
    )

    assert result["charge_allowed"] is False
    assert result["discharge_allowed"] is True
    assert result["charge_power_factor"] == pytest.approx(0.0)
    assert result["discharge_power_factor"] == pytest.approx(0.5)
