import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from battery import BatterySpec, BatteryState, step

CONFIG = Path(__file__).parent.parent / "configs" / "config.yaml"
DT_H = 10 / 60  # 10-Minuten Timestep


@pytest.fixture
def spec() -> BatterySpec:
    return BatterySpec.from_yaml(CONFIG)


@pytest.fixture
def state_min(spec) -> BatteryState:
    return BatteryState.initial(spec)  # SoC = soc_min


@pytest.fixture
def state_max(spec) -> BatteryState:
    return BatteryState(soc_kwh=spec.soc_max_kwh)  # SoC = soc_max


# --- Spec ---

def test_usable_capacity(spec):
    assert spec.usable_capacity_kwh == pytest.approx(90.0)


def test_soc_limits_kwh(spec):
    assert spec.soc_min_kwh == pytest.approx(5.0)
    assert spec.soc_max_kwh == pytest.approx(95.0)


# --- Idle ---

def test_idle_no_change(spec, state_min):
    result = step(state_min, spec, action_kw=0.0, dt_h=DT_H)
    assert result.soc_after_kwh == pytest.approx(state_min.soc_kwh)
    assert result.charge_ac_kwh == 0.0
    assert result.discharge_ac_kwh == 0.0


# --- Laden ---

def test_charge_soc_increases(spec, state_min):
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    assert result.soc_after_kwh > result.soc_before_kwh


def test_charge_eta_applied(spec, state_min):
    # 50 kW * (10/60) h = 8.333 kWh vom Netz → 8.333 * 0.96 ins Akku
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    expected_delta_soc = 50.0 * DT_H * spec.eta_charge
    assert result.soc_after_kwh == pytest.approx(result.soc_before_kwh + expected_delta_soc)


def test_charge_ac_is_grid_energy(spec, state_min):
    # charge_ac_kwh muss delta_soc / eta ergeben
    result = step(state_min, spec, action_kw=50.0, dt_h=DT_H)
    delta_soc = result.soc_after_kwh - result.soc_before_kwh
    assert result.charge_ac_kwh == pytest.approx(delta_soc / spec.eta_charge)


def test_charge_capped_at_max_power(spec, state_min):
    # Anforderung > max_charge_kw → wird auf max_charge_kw geclippt
    result_max = step(BatteryState(soc_kwh=state_min.soc_kwh), spec, action_kw=spec.max_charge_kw, dt_h=DT_H)
    result_over = step(BatteryState(soc_kwh=state_min.soc_kwh), spec, action_kw=200.0, dt_h=DT_H)
    assert result_over.soc_after_kwh == pytest.approx(result_max.soc_after_kwh)


def test_charge_capped_at_soc_max(spec):
    # Fast voll → darf soc_max nicht überschreiten
    state = BatteryState(soc_kwh=spec.soc_max_kwh - 0.1)
    result = step(state, spec, action_kw=50.0, dt_h=DT_H)
    assert result.soc_after_kwh <= spec.soc_max_kwh + 1e-9


# --- Entladen ---

def test_discharge_soc_decreases(spec, state_max):
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    assert result.soc_after_kwh < result.soc_before_kwh


def test_discharge_eta_applied(spec, state_max):
    # 50 kW angefordert → Batterie gibt 50 * dt / eta ab, Netz bekommt 50 * dt
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    expected_delta_soc = 50.0 * DT_H / spec.eta_discharge
    assert result.soc_before_kwh - result.soc_after_kwh == pytest.approx(expected_delta_soc)


def test_discharge_ac_is_net_energy(spec, state_max):
    # discharge_ac_kwh = delta_soc * eta
    result = step(state_max, spec, action_kw=-50.0, dt_h=DT_H)
    delta_soc = result.soc_before_kwh - result.soc_after_kwh
    assert result.discharge_ac_kwh == pytest.approx(delta_soc * spec.eta_discharge)


def test_discharge_capped_at_max_power(spec, state_max):
    result_max = step(BatteryState(soc_kwh=state_max.soc_kwh), spec, action_kw=-spec.max_discharge_kw, dt_h=DT_H)
    result_over = step(BatteryState(soc_kwh=state_max.soc_kwh), spec, action_kw=-200.0, dt_h=DT_H)
    assert result_over.soc_after_kwh == pytest.approx(result_max.soc_after_kwh)


def test_discharge_capped_at_soc_min(spec):
    # Fast leer → darf soc_min nicht unterschreiten
    state = BatteryState(soc_kwh=spec.soc_min_kwh + 0.1)
    result = step(state, spec, action_kw=-50.0, dt_h=DT_H)
    assert result.soc_after_kwh >= spec.soc_min_kwh - 1e-9


# --- Energieerhaltung ---

def test_energy_conservation_charge(spec, state_min):
    result = step(state_min, spec, action_kw=30.0, dt_h=DT_H)
    delta_soc = result.soc_after_kwh - result.soc_before_kwh
    losses = result.charge_ac_kwh - delta_soc
    assert losses >= -1e-9  # keine negativen Verluste


def test_energy_conservation_discharge(spec, state_max):
    result = step(state_max, spec, action_kw=-30.0, dt_h=DT_H)
    delta_soc = result.soc_before_kwh - result.soc_after_kwh
    losses = delta_soc - result.discharge_ac_kwh
    assert losses >= -1e-9
