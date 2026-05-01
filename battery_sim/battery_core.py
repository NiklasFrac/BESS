import math

def validate_spec(spec: dict) -> None:
    if spec["capacity_kwh"] <= 0:
        raise ValueError("capacity_kwh must be positive.")

    if not (0 <= spec["soc_min"] < spec["soc_max"] <= 1):
        raise ValueError("Require 0 <= soc_min < soc_max <= 1.")

    for mode in ("charge", "discharge"):
        cfg = spec[mode]

        if cfg["max_kw"] < 0:
            raise ValueError(f"{mode}.max_kw must be non-negative.")

        if not (0 < cfg["eta_nominal"] <= 1):
            raise ValueError(f"{mode}.eta_nominal must be in (0, 1].")

        if cfg["loss_factor_cold"] < 1.0:
            raise ValueError(f"{mode}.loss_factor_cold must be >= 1.")
        if cfg["loss_factor_hot"] < 1.0:
            raise ValueError(f"{mode}.loss_factor_hot must be >= 1.")

        if not (
            cfg["hard_min"]
            < cfg["optimal_min_temp"]
            <= cfg["optimal_max_temp"]
            < cfg["hard_max"]
        ):
            raise ValueError(
                f"Require {mode}.hard_min < optimal_min_temp <= "
                f"optimal_max_temp < hard_max."
            )
        eta_cold = 1.0 - (1.0 - cfg["eta_nominal"]) * cfg["loss_factor_cold"]
        eta_hot = 1.0 - (1.0 - cfg["eta_nominal"]) * cfg["loss_factor_hot"]

        if eta_cold <= 0.0:
            raise ValueError(f"{mode}.loss_factor_cold makes eta <= 0.")
        if eta_hot <= 0.0:
            raise ValueError(f"{mode}.loss_factor_hot makes eta <= 0.")
        

    
def _derating_factor(T, hard_min, full_min, full_max, hard_max):
    if T <= hard_min or T >= hard_max:
        return 0.0
    if T < full_min:
        return (T - hard_min) / (full_min - hard_min)
    if T > full_max:
        return (hard_max - T) / (hard_max - full_max)
    return 1.0

def _eta_T(T, eta_nominal, hard_min, full_min, full_max, hard_max, f_max_low, f_max_high):
    if T < full_min:
        f = f_max_low - (f_max_low - 1.0) * (T - hard_min) / (full_min - hard_min)
    elif T > full_max:
        f = f_max_high - (f_max_high - 1.0) * (hard_max - T) / (hard_max - full_max)
    else:
        f = 1.0
    return 1.0 - (1.0 - eta_nominal) * f


def step(
        state: dict,
        spec: dict, 
        action_kw: float, 
        dt_h: float,
        battery_temp_degC: float,
        ) -> dict:
    soc_min_kwh = spec["capacity_kwh"] * spec["soc_min"]
    soc_max_kwh = spec["capacity_kwh"] * spec["soc_max"]

    if not math.isfinite(action_kw):
        raise ValueError("action_kw must be finite.")
    if not math.isfinite(dt_h) or dt_h <= 0:
        raise ValueError("dt_h must be positive and finite.")
    if not math.isfinite(battery_temp_degC):
        raise ValueError("battery_temp_degC must be finite.")
    if not (soc_min_kwh <= state["soc_kwh"] <= soc_max_kwh):
        raise ValueError("SoC outside allowed range.")

    soc_before = state["soc_kwh"]
    charge_ac_kwh = 0.0
    discharge_ac_kwh = 0.0

    charge_power_limited_ac_kwh = 0.0
    discharge_power_limited_ac_kwh = 0.0
    
    loss_kwh = 0.0

    eta_charge_effective = None
    eta_discharge_effective = None

    charge_temp_limited_ac_kwh = 0.0
    discharge_temp_limited_ac_kwh = 0.0

    charge_soc_limited_ac_kwh = 0.0
    discharge_soc_limited_ac_kwh = 0.0

    charge_cfg = spec["charge"]
    discharge_cfg = spec["discharge"]

    charge_power_factor = _derating_factor(
        battery_temp_degC,
        charge_cfg["hard_min"],
        charge_cfg["optimal_min_temp"],
        charge_cfg["optimal_max_temp"],
        charge_cfg["hard_max"],
    )

    discharge_power_factor = _derating_factor(
        battery_temp_degC,
        discharge_cfg["hard_min"],
        discharge_cfg["optimal_min_temp"],
        discharge_cfg["optimal_max_temp"],
        discharge_cfg["hard_max"],
    )

    max_charge_kw_effective = charge_cfg["max_kw"] * charge_power_factor
    max_discharge_kw_effective = discharge_cfg["max_kw"] * discharge_power_factor

    charge_allowed = max_charge_kw_effective > 0.0
    discharge_allowed = max_discharge_kw_effective > 0.0

    if action_kw > 0:
        p_requested_kw = action_kw

        p_nominal_kw = min(p_requested_kw, charge_cfg["max_kw"])

        p_kw = min(p_nominal_kw, max_charge_kw_effective)

        charge_power_limited_ac_kwh = max(
            p_requested_kw - charge_cfg["max_kw"],
            0.0,
        ) * dt_h

        charge_temp_limited_ac_kwh = max(
            p_nominal_kw - max_charge_kw_effective,
            0.0,
        ) * dt_h

        if p_kw > 0.0:
            eta_charge_effective = _eta_T(
                battery_temp_degC,
                charge_cfg["eta_nominal"],
                charge_cfg["hard_min"],
                charge_cfg["optimal_min_temp"],
                charge_cfg["optimal_max_temp"],
                charge_cfg["hard_max"],
                charge_cfg["loss_factor_cold"],
                charge_cfg["loss_factor_hot"],
            )

            potential_charge_ac_kwh = p_kw * dt_h
            potential_delta_soc = potential_charge_ac_kwh * eta_charge_effective
            soc_room_kwh = soc_max_kwh - state["soc_kwh"]

            delta_soc = min(potential_delta_soc, soc_room_kwh)

            state["soc_kwh"] += delta_soc

            charge_ac_kwh = delta_soc / eta_charge_effective if delta_soc > 0.0 else 0.0
            charge_soc_limited_ac_kwh = potential_charge_ac_kwh - charge_ac_kwh
            loss_kwh = charge_ac_kwh - delta_soc


    elif action_kw < 0:
        p_requested_kw = -action_kw

        p_nominal_kw = min(p_requested_kw, discharge_cfg["max_kw"])

        p_kw = min(p_nominal_kw, max_discharge_kw_effective)

        discharge_power_limited_ac_kwh = max(
            p_requested_kw - discharge_cfg["max_kw"],
            0.0,
        ) * dt_h

        discharge_temp_limited_ac_kwh = max(
            p_nominal_kw - max_discharge_kw_effective,
            0.0,
        ) * dt_h

        if p_kw > 0.0:
            eta_discharge_effective = _eta_T(
                battery_temp_degC,
                discharge_cfg["eta_nominal"],
                discharge_cfg["hard_min"],
                discharge_cfg["optimal_min_temp"],
                discharge_cfg["optimal_max_temp"],
                discharge_cfg["hard_max"],
                discharge_cfg["loss_factor_cold"],
                discharge_cfg["loss_factor_hot"],
            )

            potential_discharge_ac_kwh = p_kw * dt_h
            potential_delta_soc = potential_discharge_ac_kwh / eta_discharge_effective
            available_soc_kwh = state["soc_kwh"] - soc_min_kwh

            delta_soc = min(potential_delta_soc, available_soc_kwh)

            state["soc_kwh"] -= delta_soc

            discharge_ac_kwh = delta_soc * eta_discharge_effective
            discharge_soc_limited_ac_kwh = potential_discharge_ac_kwh - discharge_ac_kwh
            loss_kwh = delta_soc - discharge_ac_kwh

    soc_after = state["soc_kwh"]

    return {
        "soc_before_kwh": soc_before,
        "soc_after_kwh": soc_after,

        "charge_ac_kwh": charge_ac_kwh,
        "discharge_ac_kwh": discharge_ac_kwh,
        "charge_power_limited_ac_kwh": charge_power_limited_ac_kwh,
        "discharge_power_limited_ac_kwh": discharge_power_limited_ac_kwh,
        "loss_kwh": loss_kwh,
        "charge_allowed": charge_allowed,
        "discharge_allowed": discharge_allowed,
        "charge_temp_limited_ac_kwh": charge_temp_limited_ac_kwh,
        "discharge_temp_limited_ac_kwh": discharge_temp_limited_ac_kwh,
        
        "eta_charge_effective": eta_charge_effective,
        "eta_discharge_effective": eta_discharge_effective,

        "charge_power_factor": charge_power_factor,
        "discharge_power_factor": discharge_power_factor,
        "max_charge_kw_effective": max_charge_kw_effective,
        "max_discharge_kw_effective": max_discharge_kw_effective,

        "charge_soc_limited_ac_kwh": charge_soc_limited_ac_kwh,
        "discharge_soc_limited_ac_kwh": discharge_soc_limited_ac_kwh,
    }