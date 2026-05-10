import math

import pytest

from battery_sim.degradation import (
    _arrhenius_factor,
    _c_rate_factor,
    _soc_calendar_factor,
)


def test_arrhenius_factor_is_one_at_reference_temperature():
    assert _arrhenius_factor(25.0, 25.0, 4000.0) == pytest.approx(1.0)


@pytest.mark.parametrize("temp_degC", [0.0, 25.0, 50.0])
def test_arrhenius_factor_activation_energy_zero_disables_temperature_stress(
    temp_degC: float,
):
    assert _arrhenius_factor(temp_degC, 25.0, 0.0) == pytest.approx(1.0)


def test_arrhenius_factor_increases_with_higher_temperature():
    cold = _arrhenius_factor(15.0, 25.0, 4000.0)
    reference = _arrhenius_factor(25.0, 25.0, 4000.0)
    hot = _arrhenius_factor(35.0, 25.0, 4000.0)

    assert hot > reference > cold


def test_arrhenius_factor_is_monotonic_in_temperature():
    factors = [
        _arrhenius_factor(temp, 25.0, 4000.0)
        for temp in [0.0, 10.0, 20.0, 30.0, 40.0]
    ]

    assert factors == sorted(factors)
    assert len(set(factors)) == len(factors)


@pytest.mark.parametrize(
    "temp_degC",
    [math.nan, math.inf, -math.inf, -273.15, -300.0],
)
def test_arrhenius_factor_rejects_invalid_temperatures(temp_degC: float):
    with pytest.raises(ValueError):
        _arrhenius_factor(temp_degC, 25.0, 4000.0)


@pytest.mark.parametrize("soc", [0.2, 0.5, 0.8])
def test_soc_calendar_factor_is_one_in_middle_region(soc: float):
    assert _soc_calendar_factor(soc, 0.2, 0.8, 1.5, 2.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("soc", "expected"),
    [
        (0.0, 1.5),
        (0.1, 1.25),
        (0.2, 1.0),
    ],
)
def test_soc_calendar_factor_interpolates_lower_region(
    soc: float,
    expected: float,
):
    assert _soc_calendar_factor(soc, 0.2, 0.8, 1.5, 2.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("soc", "expected"),
    [
        (0.8, 1.0),
        (0.9, 1.5),
        (1.0, 2.0),
    ],
)
def test_soc_calendar_factor_interpolates_upper_region(
    soc: float,
    expected: float,
):
    assert _soc_calendar_factor(soc, 0.2, 0.8, 1.5, 2.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("soc", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_soc_calendar_factor_one_factors_disable_soc_stress(soc: float):
    assert _soc_calendar_factor(soc, 0.2, 0.8, 1.0, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("soc", [math.nan, math.inf, -math.inf, -0.001, 1.001])
def test_soc_calendar_factor_rejects_invalid_soc(soc: float):
    with pytest.raises(ValueError):
        _soc_calendar_factor(soc, 0.2, 0.8, 1.5, 2.0)


def test_c_rate_factor_returns_one_for_zero_power():
    assert _c_rate_factor(0.0, 100.0, 0.5, 1.0) == pytest.approx(1.0)


def test_c_rate_factor_uses_absolute_power():
    assert _c_rate_factor(50.0, 100.0, 0.5, 1.0) == pytest.approx(
        _c_rate_factor(-50.0, 100.0, 0.5, 1.0)
    )


def test_c_rate_factor_is_one_at_reference_c_rate():
    assert _c_rate_factor(50.0, 100.0, 0.5, 1.0) == pytest.approx(1.0)


def test_c_rate_factor_does_not_reward_below_reference_c_rate():
    assert _c_rate_factor(25.0, 100.0, 0.5, 1.0) == pytest.approx(1.0)


def test_c_rate_factor_increases_above_reference_c_rate():
    assert _c_rate_factor(100.0, 100.0, 0.5, 1.0) == pytest.approx(2.0)
    assert _c_rate_factor(100.0, 100.0, 0.5, 2.0) == pytest.approx(4.0)


def test_c_rate_factor_exponent_zero_disables_c_rate_stress():
    assert _c_rate_factor(100.0, 100.0, 0.5, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("nominal_capacity_kWh", [0.0, -1.0, math.nan, math.inf])
def test_c_rate_factor_rejects_invalid_capacity(nominal_capacity_kWh: float):
    with pytest.raises(ValueError, match="nominal_capacity_kWh"):
        _c_rate_factor(50.0, nominal_capacity_kWh, 0.5, 1.0)
