import math

import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import (
    DAYS_PER_YEAR,
    _arrhenius_factor,
    initial_degradation_state,
    update_degradation_for_period,
)


@pytest.fixture()
def patch_rainflow_cycles(monkeypatch: pytest.MonkeyPatch):
    def patch(cycles: list[tuple[float, float, float, int, int]]) -> None:
        monkeypatch.setattr(
            degradation.rainflow,
            "extract_cycles",
            lambda _series: cycles,
        )

    return patch


def run_update(
    spec: dict[str, float],
    *,
    state: dict[str, float] | None = None,
    soc: list[float] | None = None,
    temp: list[float] | None = None,
    power: list[float] | None = None,
    capacity: float = 100.0,
    period_days: float = 0.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    input_state = initial_degradation_state() if state is None else state
    new_state, info = update_degradation_for_period(
        input_state,
        spec,
        [0.5, 0.9, 0.1, 0.5] if soc is None else soc,
        [25.0, 25.0, 25.0, 25.0] if temp is None else temp,
        [0.0, 0.0, 0.0, 0.0] if power is None else power,
        capacity,
        period_days,
    )
    return input_state, new_state, info


def test_initial_degradation_state_has_exact_initial_values():
    state = initial_degradation_state()

    assert state == {
        "capacity_factor": 1.0,
        "cumulative_efc": 0.0,
        "cycle_fade": 0.0,
        "calendar_fade": 0.0,
        "calendar_days_elapsed": 0.0,
    }


def test_initial_degradation_state_returns_new_dict_each_time():
    first = initial_degradation_state()
    second = initial_degradation_state()

    first["capacity_factor"] = 0.5

    assert second["capacity_factor"] == pytest.approx(1.0)


def test_valid_update_returns_stable_finite_info_fields(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _input_state, _new_state, info = run_update(
        valid_degradation_spec,
        period_days=1.0,
    )

    assert set(info) == {
        "efc",
        "cycle_fade",
        "calendar_fade",
        "mean_calendar_stress_factor",
        "capacity_factor_before",
        "capacity_factor_after",
    }
    assert all(math.isfinite(value) for value in info.values())


def test_update_returns_new_state_without_mutating_input_state(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    state = initial_degradation_state()
    before = state.copy()

    input_state, new_state, info = run_update(
        valid_degradation_spec,
        state=state,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
        period_days=1.0,
    )

    assert input_state == before
    assert new_state is not input_state
    assert new_state["calendar_days_elapsed"] == pytest.approx(1.0)
    assert new_state["calendar_fade"] > 0.0
    assert new_state["cycle_fade"] > 0.0
    assert new_state["cumulative_efc"] > 0.0
    assert info["capacity_factor_before"] == pytest.approx(before["capacity_factor"])
    assert info["capacity_factor_after"] == pytest.approx(new_state["capacity_factor"])


def test_period_days_zero_does_not_create_calendar_aging(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _input_state, new_state, info = run_update(
        valid_degradation_spec,
        period_days=0.0,
    )

    assert info["calendar_fade"] == pytest.approx(0.0)
    assert new_state["calendar_days_elapsed"] == pytest.approx(0.0)
    assert new_state["calendar_fade"] == pytest.approx(0.0)


def test_zero_period_days_can_still_have_cycle_aging(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, new_state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
        period_days=0.0,
    )

    assert info["calendar_fade"] == pytest.approx(0.0)
    assert info["efc"] > 0.0
    assert info["cycle_fade"] > 0.0
    assert new_state["cycle_fade"] == pytest.approx(info["cycle_fade"])


def test_calendar_aging_one_year_at_reference_conditions(
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])
    state = initial_degradation_state()

    new_state, info = update_degradation_for_period(
        state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    assert info["calendar_fade"] == pytest.approx(
        valid_degradation_spec["calendar_fade_at_1yr"]
    )
    assert new_state["calendar_fade"] == pytest.approx(0.02)
    assert new_state["capacity_factor"] == pytest.approx(0.98)


def test_two_half_year_calendar_updates_equal_one_full_year(
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])
    state = initial_degradation_state()

    for _ in range(2):
        state, _info = update_degradation_for_period(
            state,
            valid_degradation_spec,
            degradation_standard_soc,
            degradation_standard_temp,
            degradation_standard_power,
            nominal_capacity_kwh,
            DAYS_PER_YEAR / 2.0,
        )

    assert state["calendar_days_elapsed"] == pytest.approx(DAYS_PER_YEAR)
    assert state["calendar_fade"] == pytest.approx(
        valid_degradation_spec["calendar_fade_at_1yr"]
    )


def test_calendar_update_from_existing_elapsed_time_uses_sqrt_increment(
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])
    fade_at_1yr = valid_degradation_spec["calendar_fade_at_1yr"]
    state = initial_degradation_state()
    state["calendar_days_elapsed"] = DAYS_PER_YEAR
    state["calendar_fade"] = fade_at_1yr
    state["capacity_factor"] = 1.0 - fade_at_1yr

    new_state, info = update_degradation_for_period(
        state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    expected_increment = fade_at_1yr * (math.sqrt(2.0) - math.sqrt(1.0))
    assert info["calendar_fade"] == pytest.approx(expected_increment)
    assert new_state["calendar_fade"] == pytest.approx(
        fade_at_1yr + expected_increment
    )


def test_low_and_high_soc_increase_calendar_aging(
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _mid_state, mid = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.5, 0.5, 0.5],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )
    _low_state, low = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.0, 0.0, 0.0],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )
    _high_state, high = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [1.0, 1.0, 1.0],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )

    assert low["calendar_fade"] > mid["calendar_fade"]
    assert high["calendar_fade"] > mid["calendar_fade"]


def test_temperature_stress_affects_calendar_aging(
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _cold_state, cold = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.5, 0.5, 0.5],
        [15.0, 15.0, 15.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )
    _ref_state, ref = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.5, 0.5, 0.5],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )
    _hot_state, hot = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.5, 0.5, 0.5],
        [35.0, 35.0, 35.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        10.0,
    )

    assert hot["calendar_fade"] > ref["calendar_fade"] > cold["calendar_fade"]


def test_middle_soc_region_has_unit_calendar_stress(
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _new_state, info = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.2, 0.5, 0.8],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        1.0,
    )

    assert info["mean_calendar_stress_factor"] == pytest.approx(1.0)


def test_mixed_soc_values_average_calendar_stress(
    valid_degradation_spec: dict[str, float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _new_state, info = update_degradation_for_period(
        initial_degradation_state(),
        valid_degradation_spec,
        [0.5, 1.0, 0.0],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        nominal_capacity_kwh,
        1.0,
    )

    assert info["mean_calendar_stress_factor"] == pytest.approx(
        (1.0 + 2.0 + 1.5) / 3.0
    )


def test_calendar_aging_increment_is_sublinear_over_time(
    valid_degradation_spec: dict[str, float],
    degradation_standard_soc: list[float],
    degradation_standard_temp: list[float],
    degradation_standard_power: list[float],
    nominal_capacity_kwh: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])
    state = initial_degradation_state()

    state, first_year = update_degradation_for_period(
        state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )
    _state, second_year = update_degradation_for_period(
        state,
        valid_degradation_spec,
        degradation_standard_soc,
        degradation_standard_temp,
        degradation_standard_power,
        nominal_capacity_kwh,
        DAYS_PER_YEAR,
    )

    assert second_year["calendar_fade"] < first_year["calendar_fade"]


def test_cycle_efc_uses_dod_times_count(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 1, 3)])

    _input_state, _new_state, info = run_update(valid_degradation_spec)

    assert info["efc"] == pytest.approx(0.8 * 0.5)


def test_cycle_fade_formula_without_additional_stress(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 1, 3)])

    _input_state, new_state, info = run_update(spec)

    expected = spec["cycle_fade_per_efc_at_100dod"] * 0.5 * 0.8 ** spec[
        "dod_exponent"
    ]
    assert info["cycle_fade"] == pytest.approx(expected)
    assert new_state["cycle_fade"] == pytest.approx(expected)


def test_dod_exponent_makes_larger_dod_more_damaging(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0

    patch_rainflow_cycles([(0.5, 0.0, 1.0, 0, 2)])
    _input_state, _state, low_dod = run_update(spec)
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    _input_state, _state, full_dod = run_update(spec)

    assert full_dod["cycle_fade"] > low_dod["cycle_fade"]


def test_cycle_count_scales_fade_linearly(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0

    patch_rainflow_cycles([(0.8, 0.0, 0.5, 0, 2)])
    _input_state, _state, half_count = run_update(spec)
    patch_rainflow_cycles([(0.8, 0.0, 1.0, 0, 2)])
    _input_state, _state, full_count = run_update(spec)

    assert full_count["cycle_fade"] == pytest.approx(
        2.0 * half_count["cycle_fade"]
    )


def test_cycle_temperature_is_averaged_over_inclusive_index_range(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(0.8, 0.0, 1.0, 1, 3)])

    _input_state, _state, info = run_update(
        spec,
        soc=[0.5, 0.9, 0.1, 0.5, 0.5],
        temp=[0.0, 10.0, 20.0, 30.0, 40.0],
        power=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    expected_temp_factor = _arrhenius_factor(
        20.0,
        spec["cycle_reference_temp_degC"],
        spec["cycle_activation_energy_over_R_K"],
    )
    expected = (
        spec["cycle_fade_per_efc_at_100dod"]
        * 0.8 ** spec["dod_exponent"]
        * expected_temp_factor
    )
    assert info["cycle_fade"] == pytest.approx(expected)


def test_cycle_power_is_averaged_as_absolute_power_over_inclusive_index_range(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_reference"] = 0.2
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 1, 3)])

    _input_state, _state, info = run_update(
        spec,
        soc=[0.5, 0.9, 0.1, 0.5, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0, 25.0],
        power=[0.0, -20.0, 40.0, -60.0, 0.0],
    )

    expected_c_rate_factor = 2.0
    expected = spec["cycle_fade_per_efc_at_100dod"] * expected_c_rate_factor
    assert info["cycle_fade"] == pytest.approx(expected)


def test_c_rate_stress_above_reference_doubles_cycle_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, _state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[100.0, 100.0, 100.0],
    )

    expected = 2.0 * valid_degradation_spec["cycle_fade_per_efc_at_100dod"]
    assert info["cycle_fade"] == pytest.approx(expected)


def test_c_rate_below_reference_does_not_reduce_cycle_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, _state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[25.0, 25.0, 25.0],
    )

    expected = valid_degradation_spec["cycle_fade_per_efc_at_100dod"]
    assert info["cycle_fade"] == pytest.approx(expected)


def test_multiple_cycles_sum_efc_and_cycle_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles(
        [
            (0.5, 0.0, 1.0, 0, 2),
            (0.8, 0.0, 0.5, 2, 4),
        ]
    )

    _input_state, _state, info = run_update(
        spec,
        soc=[0.5, 1.0, 0.5, 0.0, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    expected_efc = 0.5 * 1.0 + 0.8 * 0.5
    expected_fade = spec["cycle_fade_per_efc_at_100dod"] * (
        1.0 * 0.5 ** spec["dod_exponent"]
        + 0.5 * 0.8 ** spec["dod_exponent"]
    )
    assert info["efc"] == pytest.approx(expected_efc)
    assert info["cycle_fade"] == pytest.approx(expected_fade)


def test_cycles_with_non_positive_dod_are_ignored(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles(
        [
            (0.0, 0.0, 1.0, 0, 2),
            (-0.1, 0.0, 1.0, 0, 2),
        ]
    )

    _input_state, _state, info = run_update(valid_degradation_spec)

    assert info["efc"] == pytest.approx(0.0)
    assert info["cycle_fade"] == pytest.approx(0.0)


def test_single_point_soc_series_has_no_cycle_aging(
    monkeypatch: pytest.MonkeyPatch,
    valid_degradation_spec: dict[str, float],
):
    def fail_if_called(_series):
        raise AssertionError("rainflow.extract_cycles should not be called")

    monkeypatch.setattr(degradation.rainflow, "extract_cycles", fail_if_called)

    _input_state, _new_state, info = run_update(
        valid_degradation_spec,
        soc=[0.5],
        temp=[25.0],
        power=[0.0],
        period_days=1.0,
    )

    assert info["efc"] == pytest.approx(0.0)
    assert info["cycle_fade"] == pytest.approx(0.0)
    assert info["calendar_fade"] > 0.0


def test_cycle_temperature_monotonicity(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, _state, cold = run_update(
        spec,
        soc=[0.5, 1.0, 0.0],
        temp=[15.0, 15.0, 15.0],
        power=[0.0, 0.0, 0.0],
    )
    _input_state, _state, ref = run_update(
        spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
    )
    _input_state, _state, hot = run_update(
        spec,
        soc=[0.5, 1.0, 0.0],
        temp=[35.0, 35.0, 35.0],
        power=[0.0, 0.0, 0.0],
    )

    assert hot["cycle_fade"] > ref["cycle_fade"] > cold["cycle_fade"]


def test_cycle_c_rate_monotonicity(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, _state, low_power = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[25.0, 25.0, 25.0],
    )
    _input_state, _state, high_power = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[100.0, 100.0, 100.0],
    )

    assert high_power["cycle_fade"] > low_power["cycle_fade"]


def test_constant_soc_series_produces_no_cycles_with_real_rainflow(
    valid_degradation_spec: dict[str, float],
):
    _input_state, _new_state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 0.5, 0.5, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0, 0.0],
    )

    assert info["efc"] == pytest.approx(0.0)
    assert info["cycle_fade"] == pytest.approx(0.0)


def test_simple_soc_series_produces_positive_cycles_with_real_rainflow(
    valid_degradation_spec: dict[str, float],
):
    _input_state, _new_state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0, 0.0],
    )

    assert info["efc"] > 0.0
    assert info["cycle_fade"] > 0.0


def test_capacity_factor_formula_uses_total_cycle_and_calendar_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, new_state, _info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
        period_days=DAYS_PER_YEAR,
    )

    expected = max(
        0.0,
        (1.0 - new_state["cycle_fade"]) * (1.0 - new_state["calendar_fade"]),
    )
    assert new_state["capacity_factor"] == pytest.approx(expected)


def test_capacity_factor_is_floored_at_zero(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 2000.0, 0, 2)])

    _input_state, new_state, _info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
    )

    assert new_state["cycle_fade"] > 1.0
    assert new_state["capacity_factor"] == pytest.approx(0.0)


def test_zero_degradation_parameters_disable_fade_but_not_efc(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["cycle_fade_per_efc_at_100dod"] = 0.0
    spec["calendar_fade_at_1yr"] = 0.0
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _input_state, new_state, info = run_update(
        spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
        period_days=DAYS_PER_YEAR,
    )

    assert info["efc"] > 0.0
    assert new_state["cycle_fade"] == pytest.approx(0.0)
    assert new_state["calendar_fade"] == pytest.approx(0.0)
    assert new_state["capacity_factor"] == pytest.approx(1.0)


@pytest.mark.parametrize("period_days", [0.0, 1.0])
def test_update_accepts_non_negative_finite_period_days(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    period_days: float,
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        [0.5],
        [25.0],
        [0.0],
        100.0,
        period_days,
    )


@pytest.mark.parametrize("period_days", [-0.001, math.nan, math.inf, -math.inf])
def test_update_rejects_invalid_period_days(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    period_days: float,
):
    with pytest.raises(ValueError, match="period_days"):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            [0.5],
            [25.0],
            [0.0],
            100.0,
            period_days,
        )


@pytest.mark.parametrize(
    ("soc", "temp", "power", "error_match"),
    [
        ([], [25.0], [0.0], "soc_fraction_series must not be empty"),
        ([0.5], [], [0.0], "battery_temp_degC_series must not be empty"),
        ([0.5], [25.0], [], "power_kW_series"),
    ],
)
def test_update_rejects_empty_series(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    soc: list[float],
    temp: list[float],
    power: list[float],
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            soc,
            temp,
            power,
            100.0,
            1.0,
        )


@pytest.mark.parametrize(
    ("soc", "temp", "power", "error_match"),
    [
        ([0.5, 0.5, 0.5], [25.0, 25.0], [0.0, 0.0, 0.0], "same length"),
        ([0.5, 0.5], [25.0, 25.0], [0.0, 0.0, 0.0], "power_kW_series"),
    ],
)
def test_update_rejects_series_length_mismatch(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    soc: list[float],
    temp: list[float],
    power: list[float],
    error_match: str,
):
    with pytest.raises(ValueError, match=error_match):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            soc,
            temp,
            power,
            100.0,
            1.0,
        )


@pytest.mark.parametrize("bad_soc", [math.nan, math.inf, -math.inf, -0.001, 1.001])
def test_update_rejects_invalid_soc_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_soc: float,
):
    with pytest.raises(ValueError, match="soc_fraction_series"):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            [0.5, bad_soc, 0.5],
            [25.0, 25.0, 25.0],
            [0.0, 0.0, 0.0],
            100.0,
            1.0,
        )


def test_update_accepts_soc_boundary_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    update_degradation_for_period(
        fresh_degradation_state,
        valid_degradation_spec,
        [0.0, 1.0],
        [25.0, 25.0],
        [0.0, 0.0],
        100.0,
        1.0,
    )


@pytest.mark.parametrize(
    "bad_temp",
    [math.nan, math.inf, -math.inf, -273.15, -300.0],
)
def test_update_rejects_invalid_temperature_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_temp: float,
):
    with pytest.raises(ValueError, match="battery_temp_degC_series"):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            [0.5, 0.5, 0.5],
            [25.0, bad_temp, 25.0],
            [0.0, 0.0, 0.0],
            100.0,
            1.0,
        )


def test_update_accepts_temperature_just_above_absolute_zero(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["calendar_activation_energy_over_R_K"] = 0.0
    patch_rainflow_cycles([])

    update_degradation_for_period(
        fresh_degradation_state,
        spec,
        [0.5, 0.5],
        [-273.149, -273.149],
        [0.0, 0.0],
        100.0,
        1.0,
    )


@pytest.mark.parametrize("bad_power", [math.nan, math.inf, -math.inf])
def test_update_rejects_invalid_power_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_power: float,
):
    with pytest.raises(ValueError, match="power_kW_series"):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            [0.5, 0.5, 0.5],
            [25.0, 25.0, 25.0],
            [0.0, bad_power, 0.0],
            100.0,
            1.0,
        )


@pytest.mark.parametrize("capacity", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_update_rejects_invalid_nominal_capacity(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    capacity: float,
):
    with pytest.raises(ValueError, match="nominal_capacity_kWh"):
        update_degradation_for_period(
            fresh_degradation_state,
            valid_degradation_spec,
            [0.5, 0.5],
            [25.0, 25.0],
            [0.0, 0.0],
            capacity,
            1.0,
        )


def test_update_forwards_invalid_spec_errors(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    spec = valid_degradation_spec.copy()
    spec["dod_exponent"] = 0.0

    with pytest.raises(ValueError, match="dod_exponent"):
        update_degradation_for_period(
            fresh_degradation_state,
            spec,
            [0.5, 0.5],
            [25.0, 25.0],
            [0.0, 0.0],
            100.0,
            1.0,
        )
