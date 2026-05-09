from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQ = {"timestamp_utc", "load_kw", "grid_import_kw", "energy_cost_eur", "demand_increment_cost_eur"}


def _read(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQ - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _dt_h(df: pd.DataFrame) -> float:
    return df["timestamp_utc"].diff().dt.total_seconds().median() / 3600


def _kpis(df: pd.DataFrame) -> dict[str, float]:
    load = float((df["load_kw"] * _dt_h(df)).sum())
    grid = float(df["grid_import_kwh"].sum())
    energy = float(df["energy_cost_eur"].sum())
    demand = float(df["demand_increment_cost_eur"].sum())
    return {
        "Load [kWh]": load,
        "Grid import [kWh]": grid,
        "Energy cost [€]": energy,
        "Demand cost [€]": demand,
        "Total cost [€]": energy + demand,
        "Peak [kW]": float(df["grid_import_kw"].max()),
        "Autarky [%]": 100 * (1 - grid / load),
    }


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_eval_plots(
    baseline_csv: str | Path,
    system_csv: str | Path,
    plot1_costs_path: str | Path,
    plot2_duration_path: str | Path,
    kpi_table_path: str | Path,
) -> None:
    b, s = _read(baseline_csv), _read(system_csv)

    names = ["Baseline", "System"]
    dfs = [b, s]
    energy = [df["energy_cost_eur"].sum() for df in dfs]
    demand = [df["demand_increment_cost_eur"].sum() for df in dfs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, energy, label="Energy cost")
    ax.bar(names, demand, bottom=energy, label="Demand cost")
    ax.set_ylabel("EUR/year")
    ax.set_title("Annual cost comparison")
    ax.legend()
    _save(fig, plot1_costs_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, df in zip(names, dfs):
        ax.plot(np.sort(df["grid_import_kw"].to_numpy())[::-1], label=name)
    ax.set_ylabel("Grid import [kW]")
    ax.set_xlabel("15-min intervals, sorted descending")
    ax.set_title("Grid import duration curve")
    ax.legend()
    _save(fig, plot2_duration_path)

    k = pd.DataFrame({"Baseline": _kpis(b), "System": _kpis(s)})
    k["Delta"] = k["System"] - k["Baseline"]
    k["Delta [%]"] = 100 * k["Delta"] / k["Baseline"].replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=k.round(2).fillna("").values,
        rowLabels=k.index,
        colLabels=k.columns,
        loc="center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    _save(fig, kpi_table_path)
