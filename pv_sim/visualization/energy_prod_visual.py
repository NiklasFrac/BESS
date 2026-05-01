from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

    daily = df.resample("D").agg(
        e_net_ac_kwh=("e_net_ac_kwh", "sum"),
        poa_global=("poa_global", "mean"),
        p_ac_w=("p_ac_w", "max"),
        TT_10=("TT_10", "mean"),
        t_module_faiman_c=("t_module_faiman_c", "mean"),
    )
    daily["e_net_ac_kwh_30d"] = daily["e_net_ac_kwh"].rolling(30, min_periods=1).mean()
    return daily


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with 'data' folder.")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    energy_path = repo_root / cfg["paths"]["energy"]
    plot_path = repo_root / "results" / "energy_overview.png"

    df = pd.read_csv(
        energy_path,
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc")

    daily = _daily_summary(df)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(daily.index, daily["e_net_ac_kwh"], linewidth=0.8, alpha=0.7)
    axes[0].plot(daily.index, daily["e_net_ac_kwh_30d"], linewidth=1.5)
    axes[0].set_ylabel("kWh/Tag")
    axes[0].set_title("PV-System Overview")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(daily.index, daily["poa_global"], linewidth=0.8)
    axes[1].set_ylabel("POA [W/m²]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(daily.index, daily["p_ac_w"] / 1000.0, linewidth=0.8)
    axes[2].set_ylabel("AC Peak [kW]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(daily.index, daily["TT_10"], linewidth=0.8, label="Air temp")
    axes[3].plot(daily.index, daily["t_module_faiman_c"], linewidth=0.8, label="Module temp")
    axes[3].set_ylabel("Temperature [°C]")
    axes[3].set_xlabel("Time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
