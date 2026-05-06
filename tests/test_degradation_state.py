import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import DAYS_PER_YEAR, initial_degradation_state, update_degradation_for_period


@pytest.fixture()
def patch_rainflow_cycles(monkeypatch: pytest.MonkeyPatch):
    def patch(cycles: list[tuple[float, float, float, int, int]]) -> None:
        monkeypatch.setattr(degradation.rainflow, "extract_cycles", lambda _series: cycles)

    return patch


def _update(
    state: dict[str, float],
    spec: dict[str, float],
    *,
    period_days: float = 1.0,
) -> dict[str, float]:
    return update_degradation_for_period(
        state,
        spec,
        [0.5, 1.0, 0.0],
        [25.0, 25.0, 25.0],
        [0.0, 0.0, 0.0],
        100.0,
        period_days,
    )


def test_initial_degradation_state_has_exact_initial_values():
    state = initial_degradation_state()

    assert state["capacity_factor"] == pytest.approx(1.0)
    assert state["cumulative_efc"] == pytest.approx(0.0)
    assert state["cycle_fade"] == pytest.approx(0.0)
    assert state["calendar_fade"] == pytest.approx(0.0)
    assert state["calendar_days_elapsed"] == pytest.approx(0.0)


def test_initial_degradation_state_returns_new_dict_each_time():
    first = initial_degradation_state()
    second = initial_degradation_state()

    first["capacity_factor"] = 0.5

    assert second["capacity_factor"] == pytest.approx(1.0)


def test_update_mutates_state(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _update(fresh_degradation_state, valid_degradation_spec)

    assert fresh_degradation_state["calendar_days_elapsed"] == pytest.approx(1.0)
    assert fresh_degradation_state["calendar_fade"] > 0.0
    assert fresh_degradation_state["cycle_fade"] > 0.0
    assert fresh_degradation_state["cumulative_efc"] > 0.0
    assert fresh_degradation_state["capacity_factor"] < 1.0


def test_return_dict_and_state_are_consistent(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    before = fresh_degradation_state["capacity_factor"]

    result = _update(fresh_degradation_state, valid_degradation_spec)

    assert result["capacity_factor_before"] == pytest.approx(before)
    assert result["capacity_factor_after"] == pytest.approx(fresh_degradation_state["capacity_factor"])


def test_cumulative_values_add_across_updates(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 0, 2)])

    first = _update(fresh_degradation_state, valid_degradation_spec, period_days=0.0)
    second = _update(fresh_degradation_state, valid_degradation_spec, period_days=0.0)

    assert fresh_degradation_state["cumulative_efc"] == pytest.approx(first["efc"] + second["efc"])
    assert fresh_degradation_state["cycle_fade"] == pytest.approx(
        first["cycle_fade"] + second["cycle_fade"]
    )


def test_capacity_factor_formula_uses_total_cycle_and_calendar_fade(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _update(fresh_degradation_state, valid_degradation_spec, period_days=DAYS_PER_YEAR)

    expected = max(
        0.0,
        (1.0 - fresh_degradation_state["cycle_fade"])
        * (1.0 - fresh_degradation_state["calendar_fade"]),
    )
    assert fresh_degradation_state["capacity_factor"] == pytest.approx(expected)


def test_capacity_factor_is_floored_at_zero(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 2000.0, 0, 2)])

    _update(fresh_degradation_state, valid_degradation_spec, period_days=0.0)

    assert fresh_degradation_state["cycle_fade"] > 1.0
    assert fresh_degradation_state["capacity_factor"] == pytest.approx(0.0)


def test_capacity_factor_does_not_increase_for_positive_degradation(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    before = fresh_degradation_state["capacity_factor"]

    result = _update(fresh_degradation_state, valid_degradation_spec)

    assert result["capacity_factor_after"] <= before


def test_zero_degradation_parameters_disable_fade_but_not_efc(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["cycle_fade_per_efc_at_100dod"] = 0.0
    spec["calendar_fade_at_1yr"] = 0.0
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    result = _update(fresh_degradation_state, spec, period_days=DAYS_PER_YEAR)

    assert result["efc"] > 0.0
    assert fresh_degradation_state["cycle_fade"] == pytest.approx(0.0)
    assert fresh_degradation_state["calendar_fade"] == pytest.approx(0.0)
    assert fresh_degradation_state["capacity_factor"] == pytest.approx(1.0)
