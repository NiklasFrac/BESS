import math

import pytest

import battery_sim.degradation as degradation
from battery_sim.degradation import update_degradation_for_period


@pytest.fixture(autouse=True)
def no_rainflow_cycles(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(degradation.rainflow, "extract_cycles", lambda _series: [])


def _update(
    *,
    state: dict[str, float],
    spec: dict[str, float],
    soc: list[float] | None = None,
    temp: list[float] | None = None,
    power: list[float] | None = None,
    capacity: float = 100.0,
    period_days: float = 1.0,
) -> dict[str, float]:
    return update_degradation_for_period(
        state,
        spec,
        [0.5, 0.5, 0.5] if soc is None else soc,
        [25.0, 25.0, 25.0] if temp is None else temp,
        [0.0, 0.0, 0.0] if power is None else power,
        capacity,
        period_days,
    )


@pytest.mark.parametrize("period_days", [0.0, 1.0])
def test_update_accepts_non_negative_finite_period_days(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    period_days: float,
):
    _update(state=fresh_degradation_state, spec=valid_degradation_spec, period_days=period_days)


@pytest.mark.parametrize("period_days", [-0.001, math.nan, math.inf])
def test_update_rejects_invalid_period_days(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    period_days: float,
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, period_days=period_days)


@pytest.mark.parametrize(
    ("soc", "temp", "power"),
    [
        ([], [25.0], [0.0]),
        ([0.5], [], [0.0]),
        ([0.5], [25.0], []),
    ],
)
def test_update_rejects_empty_series(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    soc: list[float],
    temp: list[float],
    power: list[float],
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, soc=soc, temp=temp, power=power)


@pytest.mark.parametrize(
    ("soc", "temp", "power"),
    [
        ([0.5, 0.5, 0.5], [25.0, 25.0], [0.0, 0.0, 0.0]),
        ([0.5, 0.5], [25.0, 25.0, 25.0], [0.0, 0.0]),
        ([0.5, 0.5], [25.0, 25.0], [0.0, 0.0, 0.0]),
        ([0.5, 0.5, 0.5], [25.0, 25.0, 25.0], [0.0, 0.0]),
    ],
)
def test_update_rejects_series_length_mismatch(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    soc: list[float],
    temp: list[float],
    power: list[float],
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, soc=soc, temp=temp, power=power)


@pytest.mark.parametrize("bad_soc", [math.nan, math.inf, -0.001, 1.001])
def test_update_rejects_invalid_soc_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_soc: float,
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, soc=[0.5, bad_soc, 0.5])


def test_update_accepts_soc_boundary_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    _update(
        state=fresh_degradation_state,
        spec=valid_degradation_spec,
        soc=[0.0, 1.0],
        temp=[25.0, 25.0],
        power=[0.0, 0.0],
    )


@pytest.mark.parametrize("bad_temp", [math.nan, math.inf, -math.inf, -273.15, -300.0])
def test_update_rejects_invalid_temperature_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_temp: float,
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, temp=[25.0, bad_temp, 25.0])


def test_update_accepts_temperature_just_above_absolute_zero(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    _update(state=fresh_degradation_state, spec=valid_degradation_spec, temp=[-273.149, -273.149, -273.149])


@pytest.mark.parametrize("bad_power", [math.nan, math.inf, -math.inf])
def test_update_rejects_invalid_power_values(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    bad_power: float,
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, power=[0.0, bad_power, 0.0])


@pytest.mark.parametrize("capacity", [0.0, -1.0, math.nan, math.inf])
def test_update_rejects_invalid_nominal_capacity(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
    capacity: float,
):
    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=valid_degradation_spec, capacity=capacity)


def test_update_forwards_invalid_spec_errors(
    fresh_degradation_state: dict[str, float],
    valid_degradation_spec: dict[str, float],
):
    spec = valid_degradation_spec.copy()
    spec["dod_exponent"] = 0.0

    with pytest.raises(ValueError):
        _update(state=fresh_degradation_state, spec=spec)
