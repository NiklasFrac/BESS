import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import (
    DAYS_PER_YEAR,
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
) -> tuple[dict[str, float], dict[str, float]]:
    return update_degradation_for_period(
        initial_degradation_state() if state is None else state,
        spec,
        [0.5, 1.0, 0.0] if soc is None else soc,
        [25.0, 25.0, 25.0] if temp is None else temp,
        [0.0, 0.0, 0.0] if power is None else power,
        capacity,
        period_days,
    )


def test_initial_degradation_state_has_exact_initial_values():
    assert initial_degradation_state() == {
        "capacity_factor": 1.0,
        "cumulative_efc": 0.0,
        "cycle_fade": 0.0,
        "calendar_fade": 0.0,
        "calendar_days_elapsed": 0.0,
    }


def test_update_returns_new_state_without_mutating_input_state(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])
    state = initial_degradation_state()
    before = state.copy()

    new_state, info = run_update(
        valid_degradation_spec,
        state=state,
        period_days=1.0,
    )

    assert state == before
    assert new_state is not state
    assert new_state["calendar_days_elapsed"] == pytest.approx(1.0)
    assert new_state["calendar_fade"] > 0.0
    assert new_state["cycle_fade"] > 0.0
    assert new_state["cumulative_efc"] > 0.0
    assert info["capacity_factor_before"] == pytest.approx(1.0)
    assert info["capacity_factor_after"] == pytest.approx(new_state["capacity_factor"])


def test_calendar_aging_uses_reference_fade_and_sqrt_time(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    state, first_year = run_update(
        valid_degradation_spec,
        soc=[0.5, 0.5, 0.5],
        period_days=DAYS_PER_YEAR,
    )
    state, second_year = run_update(
        valid_degradation_spec,
        state=state,
        soc=[0.5, 0.5, 0.5],
        period_days=DAYS_PER_YEAR,
    )

    assert first_year["calendar_fade"] == pytest.approx(
        valid_degradation_spec["calendar_fade_at_1yr"]
    )
    assert second_year["calendar_fade"] < first_year["calendar_fade"]
    assert state["calendar_fade"] == pytest.approx(
        valid_degradation_spec["calendar_fade_at_1yr"] * 2.0**0.5
    )


def test_calendar_stress_responds_to_soc_and_temperature(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([])

    _state, reference = run_update(
        valid_degradation_spec,
        soc=[0.5, 0.5, 0.5],
        temp=[25.0, 25.0, 25.0],
        period_days=10.0,
    )
    _state, high_soc = run_update(
        valid_degradation_spec,
        soc=[1.0, 1.0, 1.0],
        temp=[25.0, 25.0, 25.0],
        period_days=10.0,
    )
    _state, hot = run_update(
        valid_degradation_spec,
        soc=[0.5, 0.5, 0.5],
        temp=[35.0, 35.0, 35.0],
        period_days=10.0,
    )

    assert high_soc["calendar_fade"] > reference["calendar_fade"]
    assert hot["calendar_fade"] > reference["calendar_fade"]


def test_cycle_fade_formula_uses_rainflow_dod_and_count(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    spec = valid_degradation_spec.copy()
    spec["c_rate_exponent"] = 0.0
    patch_rainflow_cycles([(0.8, 0.0, 0.5, 0, 2)])

    _state, info = run_update(spec)

    expected = spec["cycle_fade_per_efc_at_100dod"] * 0.5 * 0.8 ** spec["dod_exponent"]
    assert info["efc"] == pytest.approx(0.8 * 0.5)
    assert info["cycle_fade"] == pytest.approx(expected)


def test_cycle_stress_responds_to_temperature_and_c_rate(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    _state, reference = run_update(
        valid_degradation_spec,
        temp=[25.0, 25.0, 25.0],
        power=[25.0, 25.0, 25.0],
    )
    _state, hot = run_update(
        valid_degradation_spec,
        temp=[35.0, 35.0, 35.0],
        power=[25.0, 25.0, 25.0],
    )
    _state, high_c_rate = run_update(
        valid_degradation_spec,
        temp=[25.0, 25.0, 25.0],
        power=[100.0, 100.0, 100.0],
    )

    assert hot["cycle_fade"] > reference["cycle_fade"]
    assert high_c_rate["cycle_fade"] > reference["cycle_fade"]


def test_real_rainflow_simple_soc_series_produces_cycle_aging(
    valid_degradation_spec: dict[str, float],
):
    _state, info = run_update(
        valid_degradation_spec,
        soc=[0.5, 1.0, 0.0, 0.5],
        temp=[25.0, 25.0, 25.0, 25.0],
        power=[0.0, 0.0, 0.0, 0.0],
    )

    assert info["efc"] > 0.0
    assert info["cycle_fade"] > 0.0


def test_capacity_factor_combines_fade_and_is_floored_at_zero(
    valid_degradation_spec: dict[str, float],
    patch_rainflow_cycles,
):
    patch_rainflow_cycles([(1.0, 0.0, 1.0, 0, 2)])

    state, _info = run_update(
        valid_degradation_spec,
        period_days=DAYS_PER_YEAR,
    )

    assert state["capacity_factor"] == pytest.approx(
        (1.0 - state["cycle_fade"]) * (1.0 - state["calendar_fade"])
    )

    patch_rainflow_cycles([(1.0, 0.0, 2000.0, 0, 2)])
    state, _info = run_update(valid_degradation_spec)

    assert state["cycle_fade"] > 1.0
    assert state["capacity_factor"] == pytest.approx(0.0)


def test_update_rejects_bad_series_contract(
    valid_degradation_spec: dict[str, float],
):
    state = initial_degradation_state()

    with pytest.raises(ValueError, match="period_days"):
        update_degradation_for_period(
            state,
            valid_degradation_spec,
            [0.5],
            [25.0],
            [0.0],
            100.0,
            -0.001,
        )

    with pytest.raises(ValueError, match="same length"):
        update_degradation_for_period(
            state,
            valid_degradation_spec,
            [0.5, 0.5],
            [25.0],
            [0.0, 0.0],
            100.0,
            1.0,
        )

    with pytest.raises(ValueError, match="soc_fraction_series"):
        update_degradation_for_period(
            state,
            valid_degradation_spec,
            [0.5, 1.001],
            [25.0, 25.0],
            [0.0, 0.0],
            100.0,
            1.0,
        )
