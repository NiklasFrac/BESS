import math
from typing import Dict, Any
from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo

REQUIRED_FORECAST_COLUMNS = {"timestamp_utc", "pv_kw", "load_kw"}


@dataclass(frozen=True)
class OptimizerSystemParams:
    dt_h: float
    e_nom_kwh: float
    soc_min: float
    soc_max: float
    p_grid_max_kw: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    eta_charge: float
    eta_discharge: float


@dataclass(frozen=True)
class OptimizerEconomicParams:
    energy_price_eur_per_kwh: float
    demand_charge_eur_per_kw_year: float
    battery_replacement_cost_eur: float
    expected_efc: float


@dataclass(frozen=True)
class OptimizerInitialStates:
    e_start_kwh: float
    p_peak_year_before_kw: float


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_scalar_params(
    *,
    dt_h: float,
    e_nom_kwh: float,
    soc_min: float,
    soc_max: float,
    p_grid_max_kw: float,
    p_charge_max_kw: float,
    p_discharge_max_kw: float,
    eta_charge: float,
    eta_discharge: float,
    energy_price: float,
    demand_charge: float,
    battery_replacement_cost: float,
    expected_efc: float,
    e_start_kwh: float,
    p_peak_before_kw: float,
) -> None:
    for name, value in (
        ("dt_h", dt_h),
        ("e_nom_kwh", e_nom_kwh),
        ("soc_min", soc_min),
        ("soc_max", soc_max),
        ("p_grid_max_kw", p_grid_max_kw),
        ("p_charge_max_kw", p_charge_max_kw),
        ("p_discharge_max_kw", p_discharge_max_kw),
        ("eta_charge", eta_charge),
        ("eta_discharge", eta_discharge),
        ("energy_price_eur_per_kwh", energy_price),
        ("demand_charge_eur_per_kw_year", demand_charge),
        ("battery_replacement_cost_eur", battery_replacement_cost),
        ("expected_efc", expected_efc),
        ("e_start_kwh", e_start_kwh),
        ("p_peak_year_before_kw", p_peak_before_kw),
    ):
        _require_finite(value, name)

    if dt_h <= 0:
        raise ValueError("dt_h must be positive.")
    if e_nom_kwh <= 0:
        raise ValueError("e_nom_kwh must be positive.")
    if not (0 <= soc_min < soc_max <= 1):
        raise ValueError("Require 0 <= soc_min < soc_max <= 1.")
    if p_grid_max_kw < 0:
        raise ValueError("p_grid_max_kw must be non-negative.")
    if p_charge_max_kw < 0:
        raise ValueError("p_charge_max_kw must be non-negative.")
    if p_discharge_max_kw < 0:
        raise ValueError("p_discharge_max_kw must be non-negative.")
    if not (0 < eta_charge <= 1):
        raise ValueError("eta_charge must be in (0, 1].")
    if not (0 < eta_discharge <= 1):
        raise ValueError("eta_discharge must be in (0, 1].")
    if expected_efc <= 0:
        raise ValueError("expected_efc must be positive.")
    if p_peak_before_kw < 0:
        raise ValueError("p_peak_year_before_kw must be non-negative.")


def optimize_energy_system(
    system_params: OptimizerSystemParams,
    economic_params: OptimizerEconomicParams,
    initial_states: OptimizerInitialStates,
    forecast_df: pd.DataFrame,
) -> Dict[str, Any]:
    df = forecast_df.copy()
    missing = REQUIRED_FORECAST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"forecast_df missing columns: {sorted(missing)}")

    n_steps = len(df)
    if n_steps == 0:
        raise ValueError("forecast_df must not be empty.")

    model = pyo.ConcreteModel()

    model.T = pyo.RangeSet(0, n_steps - 1)
    model.T_plus = pyo.RangeSet(0, n_steps)

    df = df.reset_index(drop=True)

    # --- scalar parameters ---
    dt_h = float(system_params.dt_h)

    e_nom_kwh = float(system_params.e_nom_kwh)
    soc_min = float(system_params.soc_min)
    soc_max = float(system_params.soc_max)

    p_grid_max_kw = float(system_params.p_grid_max_kw)
    p_charge_max_kw = float(system_params.p_charge_max_kw)
    p_discharge_max_kw = float(system_params.p_discharge_max_kw)

    eta_charge = float(system_params.eta_charge)
    eta_discharge = float(system_params.eta_discharge)

    energy_price = float(economic_params.energy_price_eur_per_kwh)
    demand_charge = float(economic_params.demand_charge_eur_per_kw_year)
    battery_replacement_cost = float(economic_params.battery_replacement_cost_eur)
    expected_efc = float(economic_params.expected_efc)

    e_start_kwh = float(initial_states.e_start_kwh)

    p_peak_before_kw = float(initial_states.p_peak_year_before_kw)
    _validate_scalar_params(
        dt_h=dt_h,
        e_nom_kwh=e_nom_kwh,
        soc_min=soc_min,
        soc_max=soc_max,
        p_grid_max_kw=p_grid_max_kw,
        p_charge_max_kw=p_charge_max_kw,
        p_discharge_max_kw=p_discharge_max_kw,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
        energy_price=energy_price,
        demand_charge=demand_charge,
        battery_replacement_cost=battery_replacement_cost,
        expected_efc=expected_efc,
        e_start_kwh=e_start_kwh,
        p_peak_before_kw=p_peak_before_kw,
    )

    # --- derived parameters ---
    e_min_kwh = e_nom_kwh * soc_min
    e_max_kwh = e_nom_kwh * soc_max

    e_usable_kwh = e_nom_kwh * (soc_max - soc_min)
    if not e_min_kwh <= e_start_kwh <= e_max_kwh:
        raise ValueError(
            "e_start_kwh must satisfy e_min_kwh <= e_start_kwh <= e_max_kwh."
        )

    throughput_cost_eur_per_kwh = battery_replacement_cost / (
        2.0 * e_usable_kwh * expected_efc
    )
    terminal_value_eur_per_kwh = 0.5 * eta_discharge * energy_price
    # --- Pyomo scalar parameters ---
    model.dt_h = pyo.Param(initialize=dt_h)

    model.e_nom_kwh = pyo.Param(initialize=e_nom_kwh)
    model.e_min_kwh = pyo.Param(initialize=e_min_kwh)
    model.e_max_kwh = pyo.Param(initialize=e_max_kwh)

    model.e_start_kwh = pyo.Param(initialize=e_start_kwh)

    model.p_grid_max_kw = pyo.Param(initialize=p_grid_max_kw)
    model.p_charge_max_kw = pyo.Param(initialize=p_charge_max_kw)
    model.p_discharge_max_kw = pyo.Param(initialize=p_discharge_max_kw)

    model.eta_charge = pyo.Param(initialize=eta_charge)
    model.eta_discharge = pyo.Param(initialize=eta_discharge)

    model.energy_price = pyo.Param(initialize=energy_price)
    model.demand_charge = pyo.Param(initialize=demand_charge)
    model.p_peak_before_kw = pyo.Param(initialize=p_peak_before_kw)

    model.throughput_cost = pyo.Param(initialize=throughput_cost_eur_per_kwh)
    model.terminal_value = pyo.Param(initialize=terminal_value_eur_per_kwh)
    # --- forecasts ---
    model.pv_kw = pyo.Param(
        model.T,
        initialize=df["pv_kw"].to_dict(),
        within=pyo.NonNegativeReals,
    )

    model.load_kw = pyo.Param(
        model.T,
        initialize=df["load_kw"].to_dict(),
        within=pyo.NonNegativeReals,
    )

    # --- decision variables ---
    model.p_grid_kw = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0, model.p_grid_max_kw),
    )

    model.p_charge_kw = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0, model.p_charge_max_kw),
    )

    model.p_discharge_kw = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0, model.p_discharge_max_kw),
    )

    model.p_curt_kw = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.e_bat_kwh = pyo.Var(
        model.T_plus,
        domain=pyo.NonNegativeReals,
        bounds=(model.e_min_kwh, model.e_max_kwh),
    )

    model.p_peak_new_kw = pyo.Var(
        domain=pyo.NonNegativeReals,
    )
    # --- constraints ---

    # Initial battery state
    model.initial_energy = pyo.Constraint(expr=model.e_bat_kwh[0] == model.e_start_kwh)

    # PV curtailment cannot exceed available PV
    def curtailment_limit_rule(model, t):
        return model.p_curt_kw[t] <= model.pv_kw[t]

    model.curtailment_limit = pyo.Constraint(
        model.T,
        rule=curtailment_limit_rule,
    )

    # Power balance at grid connection point
    def energy_balance_rule(model, t):
        return (
            model.pv_kw[t] + model.p_grid_kw[t] + model.p_discharge_kw[t]
            == model.load_kw[t] + model.p_charge_kw[t] + model.p_curt_kw[t]
        )

    model.energy_balance = pyo.Constraint(
        model.T,
        rule=energy_balance_rule,
    )

    # New annual peak must be at least every grid import value
    def peak_grid_rule(model, t):
        return model.p_peak_new_kw >= model.p_grid_kw[t]

    model.peak_grid = pyo.Constraint(
        model.T,
        rule=peak_grid_rule,
    )

    # New annual peak must be at least previous annual peak
    model.peak_previous = pyo.Constraint(
        expr=model.p_peak_new_kw >= model.p_peak_before_kw
    )

    # Battery state transition
    def battery_dynamics_rule(model, t):
        return (
            model.e_bat_kwh[t + 1]
            == model.e_bat_kwh[t]
            + model.eta_charge * model.p_charge_kw[t] * model.dt_h
            - model.p_discharge_kw[t] * model.dt_h / model.eta_discharge
        )

    model.battery_dynamics = pyo.Constraint(
        model.T,
        rule=battery_dynamics_rule,
    )

    # --- objective ---
    def objective_rule(model):
        energy_cost = (
            model.energy_price * model.dt_h * sum(model.p_grid_kw[t] for t in model.T)
        )

        demand_cost = model.demand_charge * (
            model.p_peak_new_kw - model.p_peak_before_kw
        )

        degradation_cost = (
            model.throughput_cost
            * model.dt_h
            * sum(
                model.eta_charge * model.p_charge_kw[t]
                + model.p_discharge_kw[t] / model.eta_discharge
                for t in model.T
            )
        )
        terminal_value = model.terminal_value * model.e_bat_kwh[n_steps]

        return energy_cost + demand_cost + degradation_cost - terminal_value

    model.objective = pyo.Objective(
        rule=objective_rule,
        sense=pyo.minimize,
    )
    solver = pyo.SolverFactory("highs")
    result = solver.solve(model, tee=False)

    termination = result.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Optimization failed: {termination}")

    # --- extract plan ---
    dispatch_df = df[["timestamp_utc", "pv_kw", "load_kw"]].copy()
    dispatch_df["planned_grid_import_kw"] = [
        pyo.value(model.p_grid_kw[t]) for t in model.T
    ]
    dispatch_df["planned_battery_charge_kw"] = [
        pyo.value(model.p_charge_kw[t]) for t in model.T
    ]
    dispatch_df["planned_battery_discharge_kw"] = [
        pyo.value(model.p_discharge_kw[t]) for t in model.T
    ]
    dispatch_df["planned_battery_action_kw"] = (
        dispatch_df["planned_battery_charge_kw"]
        - dispatch_df["planned_battery_discharge_kw"]
    )
    dispatch_df["planned_pv_curtailment_kw"] = [
        pyo.value(model.p_curt_kw[t]) for t in model.T
    ]
    dispatch_df["planned_soc_kwh"] = [pyo.value(model.e_bat_kwh[t]) for t in model.T]
    dispatch_df["planned_soc_next_kwh"] = [
        pyo.value(model.e_bat_kwh[t + 1]) for t in model.T
    ]
    action_df = dispatch_df[["timestamp_utc", "planned_battery_action_kw"]].rename(
        columns={"planned_battery_action_kw": "action_kw"}
    )

    return {
        "action": action_df,
        "dispatch": dispatch_df,
        "p_peak_new_kw": pyo.value(model.p_peak_new_kw),
        "e_end_kwh": pyo.value(model.e_bat_kwh[n_steps]),
    }
