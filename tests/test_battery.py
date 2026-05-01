import pytest

from battery_sim.battery_core import step, validate_spec


DT_H = 10 / 60  # 10-minute timestep


@pytest.fixture()
def spec() -> dict:
    return {
        "capacity_kwh": 100.0,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "max_charge_kw": 50.0,
        "max_discharge_kw": 50.0,
        "eta_charge": 0.96,
        "eta_discharge": 0.96,
    }


@pytest.fixture()
def state_min(spec: dict) -> dict:
    return {"soc_kwh": spec["capacity_kwh"] * spec["soc_min"]}


@pytest.fixture()
def state_max(spec: dict) -> dict:
    return {"soc_kwh": spec["capacity_kwh"] * spec["soc_max"]}


def test_validate_spec_accepts_valid_spec(spec: dict):
    validate_spec(spec)


def test_usable_capacity(spec: dict):
    usable_capacity = spec["capacity_kwh"] * (spec["soc_max"] - spec["soc_min"])
    assert usable_capacity == pytest.approx(90.0)


def test_soc_limits_kwh(spec: dict):
    assert spec["capacity_kwh"] * spec["soc_min"] == pytest.approx(5.0)
    assert spec["capacity_kwh"] * spec["soc_max"] == pytest.approx(95.0)


def test_idle_no_change(spec: dict, state_min: dict):
    result = step(state_min, spec, action_kw=0.0, dt_h=DT_H)
    assert result["soc_after_kwh"] == pytest.approx(result["soc_before_kwh"])
    assert result["charge_ac_kwh"] == 0.0
    assert result["discharge_ac_kwh"] == 0.0
    assert result["loss_kwh"] == 0.0


def test_charge_soc_increases(spec: dict, state_min: dict):
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    assert result["soc_after_kwh"] > result["soc_before_kwh"]


def test_charge_eta_applied(spec: dict, state_min: dict):
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    expected_delta_soc = 50.0 * DT_H * spec["eta_charge"]
    assert result["soc_after_kwh"] == pytest.approx(result["soc_before_kwh"] + expected_delta_soc)


def test_charge_ac_is_grid_energy(spec: dict, state_min: dict):
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    delta_soc = result["soc_after_kwh"] - result["soc_before_kwh"]
    assert result["charge_ac_kwh"] == pytest.approx(delta_soc / spec["eta_charge"])


def test_charge_capped_at_max_power(spec: dict, state_min: dict):
    result_max = step({"soc_kwh": state_min["soc_kwh"]}, spec, action_kw=spec["max_charge_kw"], dt_h=DT_H)
    result_over = step({"soc_kwh": state_min["soc_kwh"]}, spec, action_kw=200.0, dt_h=DT_H)
    assert result_over["soc_after_kwh"] == pytest.approx(result_max["soc_after_kwh"])
    assert result_over["charge_power_limited_ac_kwh"] > 0.0


def test_charge_capped_at_soc_max(spec: dict):
    soc_max_kwh = spec["capacity_kwh"] * spec["soc_max"]
    state = {"soc_kwh": soc_max_kwh - 0.1}
    result = step(state, spec, action_kw=50.0, dt_h=DT_H)
    assert result["soc_after_kwh"] <= soc_max_kwh + 1e-9


def test_discharge_soc_decreases(spec: dict, state_max: dict):
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    assert result["soc_after_kwh"] < result["soc_before_kwh"]


def test_discharge_eta_applied(spec: dict, state_max: dict):
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    expected_delta_soc = 50.0 * DT_H / spec["eta_discharge"]
    assert result["soc_before_kwh"] - result["soc_after_kwh"] == pytest.approx(expected_delta_soc)


def test_discharge_ac_is_net_energy(spec: dict, state_max: dict):
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    delta_soc = result["soc_before_kwh"] - result["soc_after_kwh"]
    assert result["discharge_ac_kwh"] == pytest.approx(delta_soc * spec["eta_discharge"])


def test_discharge_capped_at_max_power(spec: dict, state_max: dict):
    result_max = step({"soc_kwh": state_max["soc_kwh"]}, spec, action_kw=-spec["max_discharge_kw"], dt_h=DT_H)
    result_over = step({"soc_kwh": state_max["soc_kwh"]}, spec, action_kw=-200.0, dt_h=DT_H)
    assert result_over["soc_after_kwh"] == pytest.approx(result_max["soc_after_kwh"])
    assert result_over["discharge_power_limited_ac_kwh"] > 0.0


def test_discharge_capped_at_soc_min(spec: dict):
    soc_min_kwh = spec["capacity_kwh"] * spec["soc_min"]
    state = {"soc_kwh": soc_min_kwh + 0.1}
    result = step(state, spec, action_kw=-50.0, dt_h=DT_H)
    assert result["soc_after_kwh"] >= soc_min_kwh - 1e-9


def test_energy_conservation_charge(spec: dict, state_min: dict):
    result = step(state_min, spec, action_kw=30.0, dt_h=DT_H)
    delta_soc = result["soc_after_kwh"] - result["soc_before_kwh"]
    losses = result["charge_ac_kwh"] - delta_soc
    assert losses == pytest.approx(result["loss_kwh"])
    assert losses >= -1e-9


def test_energy_conservation_discharge(spec: dict, state_max: dict):
    result = step(state_max, spec, action_kw=-30.0, dt_h=DT_H)
    delta_soc = result["soc_before_kwh"] - result["soc_after_kwh"]
    losses = delta_soc - result["discharge_ac_kwh"]
    assert losses == pytest.approx(result["loss_kwh"])
    assert losses >= -1e-9
