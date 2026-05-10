import pytest

from battery_sim.degradation import initial_degradation_state


REQUIRED_DEGRADATION_KEYS = (
    "cycle_fade_per_efc_at_100dod",
    "dod_exponent",
    "cycle_reference_temp_degC",
    "cycle_activation_energy_over_R_K",
    "calendar_fade_at_1yr",
    "calendar_reference_temp_degC",
    "calendar_activation_energy_over_R_K",
    "calendar_low_soc_reference",
    "calendar_low_soc_factor",
    "calendar_high_soc_reference",
    "calendar_high_soc_factor",
    "c_rate_exponent",
    "c_rate_reference",
)


@pytest.fixture()
def valid_degradation_spec() -> dict[str, float]:
    return {
        "cycle_fade_per_efc_at_100dod": 0.001,
        "dod_exponent": 1.5,
        "cycle_reference_temp_degC": 25.0,
        "cycle_activation_energy_over_R_K": 4000.0,
        "calendar_fade_at_1yr": 0.02,
        "calendar_reference_temp_degC": 25.0,
        "calendar_activation_energy_over_R_K": 4000.0,
        "calendar_low_soc_reference": 0.2,
        "calendar_low_soc_factor": 1.5,
        "calendar_high_soc_reference": 0.8,
        "calendar_high_soc_factor": 2.0,
        "c_rate_exponent": 1.0,
        "c_rate_reference": 0.5,
    }


@pytest.fixture()
def fresh_degradation_state() -> dict[str, float]:
    return initial_degradation_state()


@pytest.fixture()
def degradation_standard_soc() -> list[float]:
    return [0.5, 0.5, 0.5]


@pytest.fixture()
def degradation_standard_temp() -> list[float]:
    return [25.0, 25.0, 25.0]


@pytest.fixture()
def degradation_standard_power() -> list[float]:
    return [0.0, 0.0, 0.0]


@pytest.fixture()
def nominal_capacity_kwh() -> float:
    return 100.0


@pytest.fixture()
def valid_battery_spec() -> dict:
    return {
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


@pytest.fixture()
def valid_thermal_spec() -> dict[str, float]:
    return {
        "initial_temp_degC": 20.0,
        "thermal_time_constant_h": 6.0,
        "heat_capacity_kwh_per_degC": 50.0,
        "heat_to_battery_fraction": 1.0,
    }


@pytest.fixture()
def state_at_soc_mid(valid_battery_spec: dict) -> dict[str, float]:
    return {"soc_kwh": valid_battery_spec["capacity_kwh"] * 0.5}


@pytest.fixture()
def state_at_soc_min(valid_battery_spec: dict) -> dict[str, float]:
    return {
        "soc_kwh": valid_battery_spec["capacity_kwh"] * valid_battery_spec["soc_min"]
    }


@pytest.fixture()
def state_at_soc_max(valid_battery_spec: dict) -> dict[str, float]:
    return {
        "soc_kwh": valid_battery_spec["capacity_kwh"] * valid_battery_spec["soc_max"]
    }
