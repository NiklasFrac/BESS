from types import SimpleNamespace

import pandas as pd
import pytest

import optimizer.optimizer as optimizer_module
from optimizer.optimizer import (
    OptimizerEconomicParams,
    OptimizerInitialStates,
    OptimizerSystemParams,
    optimize_energy_system,
)


DISPATCH_COLUMNS = [
    "timestamp_utc",
    "pv_kw",
    "load_kw",
    "planned_grid_import_kw",
    "planned_battery_charge_kw",
    "planned_battery_discharge_kw",
    "planned_battery_action_kw",
    "planned_pv_curtailment_kw",
    "planned_soc_kwh",
    "planned_soc_next_kwh",
]


def system_params(**overrides) -> OptimizerSystemParams:
    values = {
        "dt_h": 1.0,
        "e_nom_kwh": 10.0,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "p_grid_max_kw": 100.0,
        "p_charge_max_kw": 5.0,
        "p_discharge_max_kw": 5.0,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
    }
    values.update(overrides)
    return OptimizerSystemParams(**values)


def economic_params(**overrides) -> OptimizerEconomicParams:
    values = {
        "energy_price_eur_per_kwh": 0.30,
        "demand_charge_eur_per_kw_year": 0.0,
        "battery_replacement_cost_eur": 1000.0,
        "expected_efc": 1000.0,
    }
    values.update(overrides)
    return OptimizerEconomicParams(**values)


def initial_states(**overrides) -> OptimizerInitialStates:
    values = {
        "e_start_kwh": 5.0,
        "p_peak_year_before_kw": 0.0,
    }
    values.update(overrides)
    return OptimizerInitialStates(**values)


def forecast_df(*, pv_kw: list[float], load_kw: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01",
                periods=len(pv_kw),
                freq="h",
                tz="UTC",
            ),
            "pv_kw": pv_kw,
            "load_kw": load_kw,
        }
    )


def solve(
    *,
    system: OptimizerSystemParams | None = None,
    economics: OptimizerEconomicParams | None = None,
    initial: OptimizerInitialStates | None = None,
    forecast: pd.DataFrame | None = None,
) -> dict:
    if forecast is None:
        forecast = forecast_df(pv_kw=[0.0, 5.0, 0.0], load_kw=[2.0, 1.0, 2.0])

    return optimize_energy_system(
        system_params=system or system_params(),
        economic_params=economics or economic_params(),
        initial_states=initial or initial_states(),
        forecast_df=forecast,
    )


def assert_dispatch_invariants(
    result: dict,
    system: OptimizerSystemParams,
    initial: OptimizerInitialStates,
) -> None:
    dispatch = result["dispatch"]
    balance_lhs = (
        dispatch["pv_kw"]
        + dispatch["planned_grid_import_kw"]
        + dispatch["planned_battery_discharge_kw"]
    )
    balance_rhs = (
        dispatch["load_kw"]
        + dispatch["planned_battery_charge_kw"]
        + dispatch["planned_pv_curtailment_kw"]
    )
    expected_soc_next = (
        dispatch["planned_soc_kwh"]
        + system.eta_charge * dispatch["planned_battery_charge_kw"] * system.dt_h
        - dispatch["planned_battery_discharge_kw"] * system.dt_h / system.eta_discharge
    )

    assert balance_lhs.tolist() == pytest.approx(balance_rhs.tolist())
    assert dispatch["planned_soc_next_kwh"].tolist() == pytest.approx(
        expected_soc_next.tolist()
    )
    assert dispatch["planned_soc_kwh"].between(
        system.e_nom_kwh * system.soc_min,
        system.e_nom_kwh * system.soc_max,
    ).all()
    assert dispatch["planned_soc_next_kwh"].between(
        system.e_nom_kwh * system.soc_min,
        system.e_nom_kwh * system.soc_max,
    ).all()
    assert dispatch["planned_grid_import_kw"].between(0.0, system.p_grid_max_kw).all()
    assert dispatch["planned_battery_charge_kw"].between(
        0.0, system.p_charge_max_kw
    ).all()
    assert dispatch["planned_battery_discharge_kw"].between(
        0.0, system.p_discharge_max_kw
    ).all()
    assert dispatch["planned_pv_curtailment_kw"].ge(0.0).all()
    assert (dispatch["planned_pv_curtailment_kw"] <= dispatch["pv_kw"]).all()
    assert result["p_peak_new_kw"] >= initial.p_peak_year_before_kw
    assert result["p_peak_new_kw"] >= dispatch["planned_grid_import_kw"].max()


def test_optimize_returns_expected_frames_and_respects_core_invariants():
    system = system_params()
    initial = initial_states()

    result = solve(system=system, initial=initial)

    assert set(result) == {"action", "dispatch", "p_peak_new_kw", "e_end_kwh"}
    assert result["dispatch"].columns.tolist() == DISPATCH_COLUMNS
    assert result["action"].columns.tolist() == ["timestamp_utc", "action_kw"]
    assert result["action"]["timestamp_utc"].tolist() == result["dispatch"][
        "timestamp_utc"
    ].tolist()
    assert result["action"]["action_kw"].tolist() == pytest.approx(
        result["dispatch"]["planned_battery_action_kw"].tolist()
    )
    assert result["e_end_kwh"] == pytest.approx(
        result["dispatch"].iloc[-1]["planned_soc_next_kwh"]
    )
    assert_dispatch_invariants(result, system, initial)


def test_optimizer_uses_battery_for_peak_shaving_when_demand_charge_is_high():
    system = system_params(eta_charge=1.0, eta_discharge=1.0)
    initial = initial_states(e_start_kwh=6.0, p_peak_year_before_kw=2.0)
    result = solve(
        system=system,
        economics=economic_params(
            energy_price_eur_per_kwh=0.0,
            demand_charge_eur_per_kw_year=100.0,
            battery_replacement_cost_eur=0.0,
        ),
        initial=initial,
        forecast=forecast_df(pv_kw=[0.0, 0.0, 0.0], load_kw=[4.0, 4.0, 4.0]),
    )

    dispatch = result["dispatch"]
    assert dispatch["planned_battery_discharge_kw"].sum() > 0.0
    assert result["p_peak_new_kw"] < dispatch["load_kw"].max()
    assert dispatch["planned_grid_import_kw"].max() == pytest.approx(
        result["p_peak_new_kw"]
    )
    assert_dispatch_invariants(result, system, initial)


def test_optimizer_charges_from_pv_surplus_and_curtails_only_excess():
    system = system_params(eta_charge=1.0, eta_discharge=1.0, p_charge_max_kw=5.0)
    initial = initial_states(e_start_kwh=1.0)

    result = solve(
        system=system,
        economics=economic_params(
            energy_price_eur_per_kwh=0.30,
            demand_charge_eur_per_kw_year=0.0,
            battery_replacement_cost_eur=0.0,
        ),
        initial=initial,
        forecast=forecast_df(pv_kw=[8.0], load_kw=[1.0]),
    )

    row = result["dispatch"].iloc[0]
    assert row["planned_grid_import_kw"] == pytest.approx(0.0)
    assert row["planned_battery_charge_kw"] == pytest.approx(5.0)
    assert row["planned_battery_discharge_kw"] == pytest.approx(0.0)
    assert row["planned_pv_curtailment_kw"] == pytest.approx(2.0)
    assert row["planned_soc_next_kwh"] == pytest.approx(6.0)
    assert_dispatch_invariants(result, system, initial)


def test_optimize_rejects_bad_forecast_and_start_state():
    with pytest.raises(ValueError, match="forecast_df missing columns"):
        solve(forecast=pd.DataFrame({"timestamp_utc": ["2024-01-01"], "pv_kw": [0.0]}))

    with pytest.raises(ValueError, match="forecast_df must not be empty"):
        solve(forecast=pd.DataFrame(columns=["timestamp_utc", "pv_kw", "load_kw"]))

    with pytest.raises(ValueError, match="e_start_kwh"):
        solve(initial=initial_states(e_start_kwh=99.0))


def test_optimize_rejects_invalid_scalar_parameters_before_solving():
    invalid_cases = [
        (system_params(dt_h=0.0), economic_params(), "dt_h"),
        (system_params(e_nom_kwh=0.0), economic_params(), "e_nom_kwh"),
        (system_params(soc_min=0.9, soc_max=0.1), economic_params(), "soc_min"),
        (system_params(eta_charge=1.1), economic_params(), "eta_charge"),
        (system_params(), economic_params(expected_efc=0.0), "expected_efc"),
    ]

    for system, economics, error_match in invalid_cases:
        with pytest.raises(ValueError, match=error_match):
            solve(system=system, economics=economics)


def test_optimize_raises_runtime_error_when_solver_is_not_optimal(monkeypatch):
    class FakeSolver:
        def solve(self, model, tee=False):
            return SimpleNamespace(
                solver=SimpleNamespace(
                    termination_condition=optimizer_module.pyo.TerminationCondition.infeasible
                )
            )

    monkeypatch.setattr(optimizer_module.pyo, "SolverFactory", lambda _name: FakeSolver())

    with pytest.raises(RuntimeError, match="Optimization failed: infeasible"):
        solve()
