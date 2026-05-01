from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "configs" / "config.yaml").is_file() and (p / "data").is_dir():
            return p
    raise FileNotFoundError("Repo root not found.")


def main() -> None:
    repo = find_repo_root(Path(__file__).resolve().parent)
    config = yaml.safe_load((repo / "configs" / "config.yaml").read_text(encoding="utf-8"))

    data = pd.read_csv(repo / config["paths"]["test_set"], sep=";", skiprows=1).dropna(axis=1, how="all")
    data["timestamp"] = pd.to_datetime(
        data["Time stamp"].astype(str).str.strip().str.replace(r"\s+[ab]$", "", regex=True),
        format="%d.%m.%Y %H:%M:%S",
        errors="raise",
    )

    load_cols = [c for c in data.columns if c.startswith("LG ")]

    daily_energy = (
        data.assign(date=data["timestamp"].dt.date)
        .groupby("date")[load_cols]
        .sum()
        .mul(0.25)
        .reset_index()
    )
    daily_energy["weekday"] = pd.to_datetime(daily_energy["date"]).dt.weekday
    daily_energy["month"] = pd.to_datetime(daily_energy["date"]).dt.month
    daily_energy["season"] = np.where(daily_energy["month"].between(4, 9), "summer", "winter")

    stats = (
        daily_energy
        .groupby(["season", "weekday"])[load_cols]
        .agg([
            ("median_daily_energy_kwh", "median"),
            ("mad_daily_energy_kwh", lambda x: (x - x.median()).abs().median()),
        ])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_2": "company"})
        .sort_values(["company", "season", "weekday"])
    )
    stats["relative_mad"] = np.where(
        stats["median_daily_energy_kwh"] > 0,
        stats["mad_daily_energy_kwh"] / stats["median_daily_energy_kwh"],
        np.nan,
    )

    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    for ax, season in zip(axes, ["winter", "summer"]):
        heat = stats.query("season == @season").pivot(index="company", columns="weekday", values="relative_mad")[range(7)]
        ax.set_title(season)
        ax.set_xticks(range(7), weekday_names)
        ax.set_yticks(range(len(heat.index)), heat.index)
        for i in range(len(heat.index)):
            for j in range(7):
                ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(repo / "data" / "research" / "relative_mad_heatmap_seasonal.png", dpi=200)
    plt.close(fig)

    stats.to_csv(repo / "data" / "research" / "mad_seasonal.csv", index=False, float_format="%.3f")
    daily_energy.to_csv(repo / "data" / "research" / "daily_energy.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()