import math

import pytest

from battery_sim.degradation import validate_degradation_spec


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


def test_validate_degradation_spec_accepts_valid_spec(
    valid_degradation_spec: dict[str, float],
):
    validate_degradation_spec(valid_degradation_spec)


@pytest.mark.parametrize("missing_key", REQUIRED_DEGRADATION_KEYS)
def test_validate_degradation_spec_rejects_each_missing_required_key(
    valid_degradation_spec: dict[str, float],
    missing_key: str,
):
    spec = valid_degradation_spec.copy()
    del spec[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("key", REQUIRED_DEGRADATION_KEYS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_validate_degradation_spec_rejects_non_finite_values(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    with pytest.raises(ValueError, match=key):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.0, 0.999999])
def test_validate_degradation_spec_accepts_cycle_fade_boundaries(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["cycle_fade_per_efc_at_100dod"] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [-0.0001, 1.0, 1.1])
def test_validate_degradation_spec_rejects_invalid_cycle_fade(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["cycle_fade_per_efc_at_100dod"] = value

    with pytest.raises(ValueError, match="cycle_fade_per_efc_at_100dod"):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.1, 1.0, 2.0])
def test_validate_degradation_spec_accepts_positive_dod_exponent(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["dod_exponent"] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_validate_degradation_spec_rejects_non_positive_dod_exponent(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["dod_exponent"] = value

    with pytest.raises(ValueError, match="dod_exponent"):
        validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["cycle_reference_temp_degC", "calendar_reference_temp_degC"],
)
@pytest.mark.parametrize("value", [-273.149, 0.0, 25.0])
def test_validate_degradation_spec_accepts_reference_temperatures_above_zero(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["cycle_reference_temp_degC", "calendar_reference_temp_degC"],
)
@pytest.mark.parametrize("value", [-273.15, -300.0])
def test_validate_degradation_spec_rejects_reference_temperatures_at_or_below_zero(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    with pytest.raises(ValueError, match=key):
        validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["cycle_activation_energy_over_R_K", "calendar_activation_energy_over_R_K"],
)
@pytest.mark.parametrize("value", [0.0, 4000.0])
def test_validate_degradation_spec_accepts_non_negative_activation_energy(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["cycle_activation_energy_over_R_K", "calendar_activation_energy_over_R_K"],
)
def test_validate_degradation_spec_rejects_negative_activation_energy(
    valid_degradation_spec: dict[str, float],
    key: str,
):
    spec = valid_degradation_spec.copy()
    spec[key] = -1.0

    with pytest.raises(ValueError, match=key):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.0, 0.02, 0.999999])
def test_validate_degradation_spec_accepts_calendar_fade_boundaries(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_fade_at_1yr"] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [-0.001, 1.0, 1.1])
def test_validate_degradation_spec_rejects_invalid_calendar_fade(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_fade_at_1yr"] = value

    with pytest.raises(ValueError, match="calendar_fade_at_1yr"):
        validate_degradation_spec(spec)


def test_validate_degradation_spec_accepts_strict_inner_soc_reference_order(
    valid_degradation_spec: dict[str, float],
):
    spec = valid_degradation_spec.copy()
    spec["calendar_low_soc_reference"] = 0.2
    spec["calendar_high_soc_reference"] = 0.8

    validate_degradation_spec(spec)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.8, 0.2),
        (0.5, 0.5),
    ],
)
def test_validate_degradation_spec_rejects_unordered_soc_references(
    valid_degradation_spec: dict[str, float],
    low: float,
    high: float,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_low_soc_reference"] = low
    spec["calendar_high_soc_reference"] = high

    with pytest.raises(ValueError, match="calendar_low_soc_reference"):
        validate_degradation_spec(spec)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (-0.1, 0.8),
        (0.2, 1.1),
    ],
)
def test_validate_degradation_spec_rejects_soc_references_outside_closed_range(
    valid_degradation_spec: dict[str, float],
    low: float,
    high: float,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_low_soc_reference"] = low
    spec["calendar_high_soc_reference"] = high

    with pytest.raises(ValueError):
        validate_degradation_spec(spec)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.0, 0.8),
        (0.2, 1.0),
    ],
)
def test_validate_degradation_spec_rejects_soc_references_on_outer_boundaries(
    valid_degradation_spec: dict[str, float],
    low: float,
    high: float,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_low_soc_reference"] = low
    spec["calendar_high_soc_reference"] = high

    with pytest.raises(ValueError, match="Require 0 <"):
        validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["calendar_low_soc_factor", "calendar_high_soc_factor"],
)
@pytest.mark.parametrize("value", [1.0, 1.5, 2.0])
def test_validate_degradation_spec_accepts_soc_stress_factors_at_or_above_one(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize(
    "key",
    ["calendar_low_soc_factor", "calendar_high_soc_factor"],
)
@pytest.mark.parametrize("value", [0.999, 0.0, -1.0])
def test_validate_degradation_spec_rejects_soc_stress_factors_below_one(
    valid_degradation_spec: dict[str, float],
    key: str,
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec[key] = value

    with pytest.raises(ValueError, match=key):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
def test_validate_degradation_spec_accepts_non_negative_c_rate_exponent(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = value

    validate_degradation_spec(spec)


def test_validate_degradation_spec_rejects_negative_c_rate_exponent(
    valid_degradation_spec: dict[str, float],
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = -0.1

    with pytest.raises(ValueError, match="c_rate_exponent"):
        validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.1, 1.0])
def test_validate_degradation_spec_accepts_positive_c_rate_reference(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_reference"] = value

    validate_degradation_spec(spec)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_validate_degradation_spec_rejects_non_positive_c_rate_reference(
    valid_degradation_spec: dict[str, float],
    value: float,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_reference"] = value

    with pytest.raises(ValueError, match="c_rate_reference"):
        validate_degradation_spec(spec)
