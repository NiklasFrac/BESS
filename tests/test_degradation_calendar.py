import math

import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import DAYS_PER_YEAR, update_degradation_for_period


@pytest.fixture(autouse=True)
def no_rainflow_cycles(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(degradation.rainflow, "extract_cycles", lambda _series: [])


def test_calendar_aging_one_year_at_reference_conditions(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
):
    result = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    assert result["calendar_fade"] == pytest.approx(valid_degradation_spec["calendar_fade_at_1yr"])
    assert fresh_degradation_state["calendar_fade"] == pytest.approx(0.02)
    assert fresh_degradation_state["capacity_factor"] == pytest.approx(0.98)


def test_calendar_aging_zero_period_days_does_not_advance_calendar(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
):
    result = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        0.0,
    )

    assert result["calendar_fade"] == pytest.approx(0.0)
    assert fresh_degradation_state["calendar_days_elapsed"] == pytest.approx(0.0)
    assert fresh_degradation_state["calendar_fade"] == pytest.approx(0.0)


def test_two_half_year_calendar_updates_equal_one_full_year(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
):
    for _ in range(2):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            degradation_standard_soc,
            degradation_standard_temp,
            degradation_standard_power,
            nominal_capacity_kwh,
            DAYS_PER_YEAR / 2.0,
        )

    assert fresh_degradation_state["calendar_days_elapsed"] == pytest.approx(DAYS_PER_YEAR)
    assert fresh_degradation_state["calendar_fade"] == pytest.approx(
        valid_degradation_spec["calendar_fade_at_1yr"]
    )


def test_calendar_update_from_existing_elapsed_time_uses_sqrt_increment(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
):
    fade_at_1yr = valid_degradation_spec["calendar_fade_at_1yr"]
    fresh_degradation_state["calendar_days_elapsed"] = DAYS_PER_YEAR
    fresh_degradation_state["calendar_fade"] = fade_at_1yr
    fresh_degradation_state["capacity_factor"] = 1.0 - fade_at_1yr

    result = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    expected_increment = fade_at_1yr * (math.sqrt(2.0) - math.sqrt(1.0))
    assert result["calendar_fade"] == pytest.approx(expected_increment)
    assert fresh_degradation_state["calendar_fade"] == pytest.approx(fade_at_1yr + expected_increment)


def test_high_soc_increases_calendar_aging(
    valid_degradation_spec: dict[str, float], nominal_capacity_kwh: float
):
    mid_state = degradation.initial_degradation_state()
    high_state = degradation.initial_degradation_state()

    mid = update_degradation_for_period(
        mid_state, valid_degradation_spec, [0.5, 0.5, 0.5], [25.0, 25.0, 25.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )
    high = update_degradation_for_period(
        high_state, valid_degradation_spec, [1.0, 1.0, 1.0], [25.0, 25.0, 25.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )

    assert high["calendar_fade"] > mid["calendar_fade"]


def test_low_soc_increases_calendar_aging(
    valid_degradation_spec: dict[str, float], nominal_capacity_kwh: float
):
    mid_state = degradation.initial_degradation_state()
    low_state = degradation.initial_degradation_state()

    mid = update_degradation_for_period(
        mid_state, valid_degradation_spec, [0.5, 0.5, 0.5], [25.0, 25.0, 25.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )
    low = update_degradation_for_period(
        low_state, valid_degradation_spec, [0.0, 0.0, 0.0], [25.0, 25.0, 25.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )

    assert low["calendar_fade"] > mid["calendar_fade"]


def test_middle_soc_region_has_unit_calendar_stress(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
):
    result = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        [0.2, 0.5, 0.8],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        1.0,
    )

    assert result["mean_calendar_stress_factor"] == pytest.approx(1.0)


def test_mixed_soc_values_average_calendar_stress(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
):
    result = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        [0.5, 1.0, 0.0],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        1.0,
    )

    assert result["mean_calendar_stress_factor"] == pytest.approx((1.0 + 2.0 + 1.5) / 3.0)


def test_temperature_stress_affects_calendar_aging(
    valid_degradation_spec: dict[str, float], nominal_capacity_kwh: float
):
    cold_state = degradation.initial_degradation_state()
    ref_state = degradation.initial_degradation_state()
    hot_state = degradation.initial_degradation_state()

    cold = update_degradation_for_period(
        cold_state, valid_degradation_spec, [0.5, 0.5, 0.5], [15.0, 15.0, 15.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )
    ref = update_degradation_for_period(
        ref_state, valid_degradation_spec, [0.5, 0.5, 0.5], [25.0, 25.0, 25.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )
    hot = update_degradation_for_period(
        hot_state, valid_degradation_spec, [0.5, 0.5, 0.5], [35.0, 35.0, 35.0], [0.0, 0.0, 0.0],
        nominal_capacity_kwh, 10.0
    )

    assert hot["calendar_fade"] > ref["calendar_fade"] > cold["calendar_fade"]


def test_calendar_aging_increment_is_sublinear_over_time(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
):
    first_year = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )
    second_year = update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    assert second_year["calendar_fade"] < first_year["calendar_fade"]
