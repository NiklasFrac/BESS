import pandas as pd
import pytest

from battery_sim.simulator import initial_simulation_state, simulate, simulate_period


def battery_spec() -> dict:
    return {
        "capacity_kwh": 100.0,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "charge": {
            "max_kw": 50.0,
            "eta_nominal": 0.96,
            "loss_factor_cold": 1.5,
            "loss_factor_hot": 1.3,
            "hard_min": 0.0,
            "optimal_min_temp": 5.0,
            "optimal_max_temp": 40.0,
            "hard_max": 45.0,
        },
        "discharge": {
            "max_kw": 50.0,
            "eta_nominal": 0.96,
            "loss_factor_cold": 1.5,
            "loss_factor_hot": 1.3,
            "hard_min": -20.0,
            "optimal_min_temp": -10.0,
            "optimal_max_temp": 45.0,
            "hard_max": 55.0,
        },
    }


def thermal_spec() -> dict:
    return {
        "initial_temp_degC": 20.0,
        "thermal_time_constant_h": 6.0,
        "heat_capacity_kwh_per_degC": 50.0,
        "heat_to_battery_fraction": 1.0,
    }


def action_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01",
                periods=4,
                freq="h",
                tz="UTC",
            ),
            "action_kw": [10.0, -5.0, 0.0, 20.0],
            "ambient_temp_degC": [20.0, 21.0, 22.0, 23.0],
        }
    )


def test_simulate_period_carries_state_and_matches_batch_simulation(
    valid_degradation_spec: dict[str, float],
):
    spec = battery_spec()
    thermal = thermal_spec()
    actions = action_df()

    batch_battery, _batch_temp, batch_degradation = simulate(
        actions,
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
    )

    state = initial_simulation_state(spec, thermal)
    first_battery, _first_temp, first_degradation, state = simulate_period(
        actions.iloc[:2],
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
        state=state,
    )
    second_battery, _second_temp, second_degradation, state = simulate_period(
        actions.iloc[2:],
        spec,
        thermal,
        valid_degradation_spec,
        dt_h=1.0,
        state=state,
        finalize_period=True,
    )

    period_battery = pd.concat([first_battery, second_battery], ignore_index=True)
    period_degradation = pd.concat(
        [first_degradation, second_degradation],
        ignore_index=True,
    )

    assert period_battery["soc_kwh"].tolist() == pytest.approx(
        batch_battery["soc_kwh"].tolist()
    )
    assert state.soc_kwh == pytest.approx(batch_battery.iloc[-1]["soc_kwh"])
    assert state.month_soc == []
    assert len(first_degradation) == 0
    assert len(period_degradation) == len(batch_degradation) == 1


def test_initial_simulation_state_accepts_optimizer_start_soc():
    state = initial_simulation_state(
        battery_spec(),
        thermal_spec(),
        start_soc_kwh=50.0,
    )

    assert state.soc_kwh == pytest.approx(50.0)
    assert state.capacity_kwh == pytest.approx(100.0)
