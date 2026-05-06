import math

import pytest

from battery_sim.degradation import initial_degradation_state, update_degradation_for_period


def test_constant_soc_series_produces_no_cycles_with_real_rainflow(
    valid_degradation_spec: dict[str, float],
):
    state = initial_degradation_state()

    result = update_degradation_for_period(
        state,
        valid_degradation_spec,
        [0.5, 0.5, 0.5, 0.5],
        [25.0, 25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0, 0.0],
        100.0,
        0.0,
    )

    assert result["efc"] == pytest.approx(0.0)
    assert result["cycle_fade"] == pytest.approx(0.0)


def test_simple_soc_series_produces_positive_cycles_with_real_rainflow(
    valid_degradation_spec: dict[str, float],
):
    state = initial_degradation_state()

    result = update_degradation_for_period(
        state,
        valid_degradation_spec,
        [0.5, 1.0, 0.0, 0.5],
        [25.0, 25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0, 0.0],
        100.0,
        0.0,
    )

    assert result["efc"] > 0.0
    assert result["cycle_fade"] > 0.0


def test_update_return_dict_has_stable_finite_fields(
    valid_degradation_spec: dict[str, float],
):
    state = initial_degradation_state()

    result = update_degradation_for_period(
        state,
        valid_degradation_spec,
        [0.5, 0.5, 0.5],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        100.0,
        1.0,
    )

    assert set(result) == {
        "efc",
        "cycle_fade",
        "calendar_fade",
        "mean_calendar_stress_factor",
        "capacity_factor_before",
        "capacity_factor_after",
    }
    assert all(math.isfinite(value) for value in result.values())
