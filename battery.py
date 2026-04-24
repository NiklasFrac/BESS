from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float
    soc_min: float
    soc_max: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_charge: float
    eta_discharge: float

    @classmethod
    def from_yaml(cls, path: Path) -> "BatterySpec":
        raw = yaml.safe_load(path.read_text())["batterie"]
        return cls(**raw)

    @property
    def usable_capacity_kwh(self) -> float:
        return self.capacity_kwh * (self.soc_max - self.soc_min)

    @property
    def soc_min_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min

    @property
    def soc_max_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max


@dataclass
class BatteryState:
    soc_kwh: float

    @classmethod
    def initial(cls, spec: BatterySpec) -> "BatteryState":
        return cls(soc_kwh=spec.soc_min_kwh)


@dataclass(frozen=True)
class StepResult:
    soc_before_kwh: float
    soc_after_kwh: float
    charge_ac_kwh: float     # Energie die vom Netz gezogen wurde
    discharge_ac_kwh: float  # Energie die ans Netz geliefert wurde


def step(state: BatteryState, spec: BatterySpec, action_kw: float, dt_h: float) -> StepResult:
    """
    action_kw: angeforderte AC-seitige Leistung
        > 0  →  Laden   (Netz → Batterie)
        < 0  →  Entladen (Batterie → Netz)
        = 0  →  Idle

    dt_h: Zeitschrittlänge in Stunden
    """
    soc_before = state.soc_kwh

    if action_kw > 0:
        p_kw = min(action_kw, spec.max_charge_kw)
        delta_soc = min(p_kw * dt_h * spec.eta_charge, spec.soc_max_kwh - state.soc_kwh)
        state.soc_kwh += delta_soc
        return StepResult(
            soc_before_kwh=soc_before,
            soc_after_kwh=state.soc_kwh,
            charge_ac_kwh=delta_soc / spec.eta_charge,
            discharge_ac_kwh=0.0,
        )

    if action_kw < 0:
        p_kw = min(-action_kw, spec.max_discharge_kw)
        delta_soc = min(p_kw * dt_h / spec.eta_discharge, state.soc_kwh - spec.soc_min_kwh)
        state.soc_kwh -= delta_soc
        return StepResult(
            soc_before_kwh=soc_before,
            soc_after_kwh=state.soc_kwh,
            charge_ac_kwh=0.0,
            discharge_ac_kwh=delta_soc * spec.eta_discharge,
        )

    return StepResult(
        soc_before_kwh=soc_before,
        soc_after_kwh=state.soc_kwh,
        charge_ac_kwh=0.0,
        discharge_ac_kwh=0.0,
    )