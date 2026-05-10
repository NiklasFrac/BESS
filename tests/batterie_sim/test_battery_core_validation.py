import copy
import math

import pytest

from battery_sim.battery_core import ABSOLUTE_ZERO_DEGC, validate_spec


MODE_FIELDS = (
    "max_kw",
    "eta_nominal",
    "loss_factor_cold",
    "loss_factor_hot",
    "hard_min",
    "optimal_min_temp",
    "optimal_max_temp",
    "hard_max",
)


def clone_spec(spec: dict) -> dict:
    return copy.deepcopy(spec)


def test_validate_spec_accepts_valid_battery_spec(valid_battery_spec: dict):
    validate_spec(valid_battery_spec)


@pytest.mark.parametrize("value", [1.0, 100.0])
def test_validate_spec_accepts_positive_finite_capacity(
    valid_battery_spec: dict,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec["capacity_kwh"] = value

    validate_spec(spec)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_validate_spec_rejects_invalid_capacity(
    valid_battery_spec: dict,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec["capacity_kwh"] = value

    with pytest.raises(ValueError, match="capacity_kwh"):
        validate_spec(spec)


@pytest.mark.parametrize(
    ("soc_min", "soc_max"),
    [
        (0.0, 1.0),
        (0.05, 0.95),
    ],
)
def test_validate_spec_accepts_valid_soc_window(
    valid_battery_spec: dict,
    soc_min: float,
    soc_max: float,
):
    spec = clone_spec(valid_battery_spec)
    spec["soc_min"] = soc_min
    spec["soc_max"] = soc_max

    validate_spec(spec)


@pytest.mark.parametrize(
    ("soc_min", "soc_max"),
    [
        (-0.001, 0.95),
        (0.05, 1.001),
        (0.95, 0.95),
        (0.96, 0.95),
        (math.nan, 0.95),
        (0.05, math.nan),
        (math.inf, 0.95),
        (0.05, -math.inf),
    ],
)
def test_validate_spec_rejects_invalid_soc_window(
    valid_battery_spec: dict,
    soc_min: float,
    soc_max: float,
):
    spec = clone_spec(valid_battery_spec)
    spec["soc_min"] = soc_min
    spec["soc_max"] = soc_max

    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("value", [0.0, 50.0])
def test_validate_spec_accepts_non_negative_finite_max_kw(
    valid_battery_spec: dict,
    mode: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["max_kw"] = value

    validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("value", [-0.001, math.nan, math.inf, -math.inf])
def test_validate_spec_rejects_invalid_max_kw(
    valid_battery_spec: dict,
    mode: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["max_kw"] = value

    with pytest.raises(ValueError, match=f"{mode}.max_kw"):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("value", [0.5, 0.96, 1.0])
def test_validate_spec_accepts_eta_nominal_in_open_closed_range(
    valid_battery_spec: dict,
    mode: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["eta_nominal"] = value

    validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("value", [0.0, -0.1, 1.001, math.nan, math.inf, -math.inf])
def test_validate_spec_rejects_invalid_eta_nominal(
    valid_battery_spec: dict,
    mode: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["eta_nominal"] = value

    with pytest.raises(ValueError, match=f"{mode}.eta_nominal"):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("field", ["loss_factor_cold", "loss_factor_hot"])
@pytest.mark.parametrize("value", [1.0, 1.5])
def test_validate_spec_accepts_loss_factors_at_or_above_one(
    valid_battery_spec: dict,
    mode: str,
    field: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode][field] = value

    validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("field", ["loss_factor_cold", "loss_factor_hot"])
@pytest.mark.parametrize("value", [0.999, 0.0, -1.0, math.nan, math.inf, -math.inf])
def test_validate_spec_rejects_invalid_loss_factors(
    valid_battery_spec: dict,
    mode: str,
    field: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode][field] = value

    with pytest.raises(ValueError, match=f"{mode}.{field}"):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("field", ["loss_factor_cold", "loss_factor_hot"])
def test_validate_spec_rejects_loss_factor_that_makes_eta_non_positive(
    valid_battery_spec: dict,
    mode: str,
    field: str,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode][field] = 25.1

    with pytest.raises(ValueError, match=f"{mode}.{field} makes eta <= 0"):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize(
    ("hard_min", "optimal_min_temp", "optimal_max_temp", "hard_max"),
    [
        (-20.0, -10.0, 45.0, 55.0),
        (0.0, 5.0, 5.0, 45.0),
    ],
)
def test_validate_spec_accepts_ordered_temperature_window(
    valid_battery_spec: dict,
    mode: str,
    hard_min: float,
    optimal_min_temp: float,
    optimal_max_temp: float,
    hard_max: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["hard_min"] = hard_min
    spec[mode]["optimal_min_temp"] = optimal_min_temp
    spec[mode]["optimal_max_temp"] = optimal_max_temp
    spec[mode]["hard_max"] = hard_max

    validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize("field", ["hard_min", "optimal_min_temp", "optimal_max_temp", "hard_max"])
@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, ABSOLUTE_ZERO_DEGC, ABSOLUTE_ZERO_DEGC - 1.0],
)
def test_validate_spec_rejects_invalid_temperature_values(
    valid_battery_spec: dict,
    mode: str,
    field: str,
    value: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode][field] = value

    with pytest.raises(ValueError, match=f"{mode}.{field}"):
        validate_spec(spec)


@pytest.mark.parametrize("mode", ["charge", "discharge"])
@pytest.mark.parametrize(
    ("hard_min", "optimal_min_temp", "optimal_max_temp", "hard_max"),
    [
        (5.0, 5.0, 40.0, 45.0),
        (6.0, 5.0, 40.0, 45.0),
        (0.0, 41.0, 40.0, 45.0),
        (0.0, 5.0, 45.0, 45.0),
        (0.0, 5.0, 46.0, 45.0),
    ],
)
def test_validate_spec_rejects_unordered_temperature_window(
    valid_battery_spec: dict,
    mode: str,
    hard_min: float,
    optimal_min_temp: float,
    optimal_max_temp: float,
    hard_max: float,
):
    spec = clone_spec(valid_battery_spec)
    spec[mode]["hard_min"] = hard_min
    spec[mode]["optimal_min_temp"] = optimal_min_temp
    spec[mode]["optimal_max_temp"] = optimal_max_temp
    spec[mode]["hard_max"] = hard_max

    with pytest.raises(ValueError, match=f"Require {mode}.hard_min"):
        validate_spec(spec)
