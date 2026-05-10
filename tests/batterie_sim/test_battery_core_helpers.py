import pytest

from battery_sim.battery_core import _derating_factor, _eta_T


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (-5.0, 0.0),
        (0.0, 0.0),
        (2.5, 0.5),
        (5.0, 1.0),
        (20.0, 1.0),
        (40.0, 1.0),
        (42.5, 0.5),
        (45.0, 0.0),
        (50.0, 0.0),
    ],
)
def test_derating_factor_piecewise_temperature_curve(
    temperature: float,
    expected: float,
):
    assert _derating_factor(temperature, 0.0, 5.0, 40.0, 45.0) == pytest.approx(
        expected
    )


def test_eta_t_uses_nominal_efficiency_in_optimal_window():
    assert _eta_T(20.0, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3) == pytest.approx(
        0.96
    )


@pytest.mark.parametrize(
    ("temperature", "expected_factor"),
    [
        (0.0, 1.5),
        (2.5, 1.25),
        (5.0, 1.0),
    ],
)
def test_eta_t_interpolates_cold_loss_factor(
    temperature: float,
    expected_factor: float,
):
    eta = _eta_T(temperature, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3)

    assert eta == pytest.approx(1.0 - (1.0 - 0.96) * expected_factor)


@pytest.mark.parametrize(
    ("temperature", "expected_factor"),
    [
        (40.0, 1.0),
        (42.5, 1.15),
        (45.0, 1.3),
    ],
)
def test_eta_t_interpolates_hot_loss_factor(
    temperature: float,
    expected_factor: float,
):
    eta = _eta_T(temperature, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3)

    assert eta == pytest.approx(1.0 - (1.0 - 0.96) * expected_factor)


def test_eta_t_loss_factors_lower_efficiency_outside_optimal_window():
    cold = _eta_T(0.0, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3)
    nominal = _eta_T(20.0, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3)
    hot = _eta_T(45.0, 0.96, 0.0, 5.0, 40.0, 45.0, 1.5, 1.3)

    assert cold < nominal
    assert hot < nominal
