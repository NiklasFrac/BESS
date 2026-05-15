import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)


def _daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

    daily = df.resample("D").agg(
        e_net_ac_kwh=("e_net_ac_kwh", "sum"),
    )

    daily["e_net_ac_kwh_14d"] = daily["e_net_ac_kwh"].rolling(14, min_periods=1).mean()

    return daily


def plot_energy_overview(
    energy_path: Path,
    plot_path: Path,
    show: bool = False,
) -> None:
    df = pd.read_csv(
        energy_path,
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc")

    daily = _daily_summary(df)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

    ax.plot(
        daily.index,
        daily["e_net_ac_kwh"],
        linewidth=0.8,
        alpha=0.35,
        label="Daily energy",
    )
    ax.plot(
        daily.index,
        daily["e_net_ac_kwh_14d"],
        linewidth=2.0,
        label="14d rolling mean",
    )

    ax.set_title("PV Energy Overview")
    ax.set_ylabel("Energy [kWh/day]")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    log.info("Plot gespeichert: %s", plot_path)
