import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import _arrhenius_factor, update_degradation_for_period


@pytest.fixture()
def patch_rainflow_cycles(monkeypatch: pytest.MonkeyPatch):
    def patch(cycles: list[tuple[float, float, float, int, int]]) -> None:
        monkeypatch.setattr(degradation.rainflow, "extract_cycles", lambda _series: cycles)

    return patch


def _run_update(
    spec: dict[str, float],
    *,
    soc: list[float] | None = None,
    temp: list[float] | None = None,
    power: list[float] | None = None,
    capacity: float = 100.0,
    period_days: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    state = degradation.initial_degradation_state()
    result = update_degradation_for_period(
        state,
        spec,
        [0.5, 0.9, 0.1, 0.5] if soc is None else soc,
        [25.0, 25.0, 25.0, 25.0] if temp is None else temp,
        [0.0, 0.0, 0.0, 0.0] if power is None else power,
        capacity,
        period_days,
    )
    return result, state


def test_cycle_efc_uses_dod_times_count(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 1, 3)])

    result, _state = _run_update(valid_degradation_spec)

    assert result["efc"] == pytest.approx(0.8 * 0.5)


def test_cycle_fade_formula_without_additional_stress(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 1, 3)])

    result, state = _run_update(spec)

    expected = spec["cycle_fade_per_efc_at_100dod"] * 0.5 * 0.8 ** spec["dod_exponent"]
    assert result["cycle_fade"] == pytest.approx(expected)
    assert state["cycle_fade"] == pytest.approx(expected)


def test_dod_exponent_makes_larger_dod_more_damaging(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0

    patch_rainflow_cycles([(0.5, 0.0, 1.0, 0, 2)])
    low_dod, _state = _run_update(spec)
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    full_dod, _state = _run_update(spec)

    assert full_dod["cycle_fade"] > low_dod["cycle_fade"]


def test_cycle_count_scales_fade_linearly(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0

    patch_rainflow_cycles([(0.8, 0.0, 0.5, 0, 2)])
    half_count, _state = _run_update(spec)
    patch_rainflow_cycles([(0.8, 0.0, 1.0, 0, 2)])
    full_count, _state = _run_update(spec)

    assert full_count["cycle_fade"] == pytest.approx(2.0 * half_count["cycle_fade"])


def test_cycle_temperature_is_averaged_over_inclusive_index_range(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(0.8, 0.0, 1.0, 1, 3)])

    result, _state = _run_update(
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
    expected = spec["cycle_fade_per_efc_at_100dod"] * 0.8 ** spec["dod_exponent"] * expected_temp_factor
    assert result["cycle_fade"] == pytest.approx(expected)


def test_cycle_power_is_averaged_as_absolute_power_over_inclusive_index_range(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_reference"] = 0.2
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 1, 3)])

    result, _state = _run_update(
        spec,
        soc=[0.5, 0.9, 0.1, 0.5, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0, 25.0],
        power=[0.0, -20.0, 40.0, -60.0, 0.0],
    )

    expected_c_rate_factor = 2.0
    expected = spec["cycle_fade_per_efc_at_100dod"] * expected_c_rate_factor
    assert result["cycle_fade"] == pytest.approx(expected)


def test_c_rate_stress_above_reference_doubles_cycle_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    result, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[100.0, 100.0, 100.0],
    )

    expected = 2.0 * valid_degradation_spec["cycle_fade_per_efc_at_100dod"]
    assert result["cycle_fade"] == pytest.approx(expected)


def test_c_rate_below_reference_does_not_reduce_cycle_fade(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    result, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[25.0, 25.0, 25.0],
    )

    expected = valid_degradation_spec["cycle_fade_per_efc_at_100dod"]
    assert result["cycle_fade"] == pytest.approx(expected)


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

    result, _state = _run_update(
        spec,
        soc=[0.5, 1.0, 0.5, 0.0, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    expected_efc = 0.5 * 1.0 + 0.8 * 0.5
    expected_fade = spec["cycle_fade_per_efc_at_100dod"] * (
        1.0 * 0.5 ** spec["dod_exponent"] + 0.5 * 0.8 ** spec["dod_exponent"]
    )
    assert result["efc"] == pytest.approx(expected_efc)
    assert result["cycle_fade"] == pytest.approx(expected_fade)


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

    result, _state = _run_update(valid_degradation_spec)

    assert result["efc"] == pytest.approx(0.0)
    assert result["cycle_fade"] == pytest.approx(0.0)


def test_single_point_soc_series_has_no_cycle_aging(
    monkeypatch: pytest.MonkeyPatch,
    valid_degradation_spec: dict[str, float],
):
    def fail_if_called(_series):
        raise AssertionError("rainflow.extract_cycles should not be called for a single-point SOC series")

    monkeypatch.setattr(degradation.rainflow, "extract_cycles", fail_if_called)

    result, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5],
        temp=[25.0],
        power=[0.0],
        period_days=1.0,
    )

    assert result["efc"] == pytest.approx(0.0)
    assert result["cycle_fade"] == pytest.approx(0.0)
    assert result["calendar_fade"] > 0.0


def test_cycle_temperature_monotonicity(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    cold, _state = _run_update(spec, soc=[0.5, 1.0, 0.0], temp=[15.0, 15.0, 15.0], power=[0.0, 0.0, 0.0])
    ref, _state = _run_update(spec, soc=[0.5, 1.0, 0.0], temp=[25.0, 25.0, 25.0], power=[0.0, 0.0, 0.0])
    hot, _state = _run_update(spec, soc=[0.5, 1.0, 0.0], temp=[35.0, 35.0, 35.0], power=[0.0, 0.0, 0.0])

    assert hot["cycle_fade"] > ref["cycle_fade"] > cold["cycle_fade"]


def test_cycle_c_rate_monotonicity(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    low_power, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[25.0, 25.0, 25.0],
    )
    high_power, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[100.0, 100.0, 100.0],
    )

    assert high_power["cycle_fade"] > low_power["cycle_fade"]


def test_zero_period_days_can_still_have_cycle_aging(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    result, _state = _run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0],
        temp=[25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0],
        period_days=0.0,
    )

    assert result["calendar_fade"] == pytest.approx(0.0)
    assert result["efc"] > 0.0
    assert result["cycle_fade"] > 0.0
