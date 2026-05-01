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


def plot_heatmap(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(table, aspect="auto")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center", color="white")

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_title("CV der täglichen Lastsumme")
    ax.set_xlabel("Wochentag")
    ax.set_ylabel("Firma")

    fig.colorbar(im, ax=ax, label="CV = std / mean")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pca(features: pd.DataFrame, path: Path) -> tuple[float, float]:
    x = features.to_numpy(dtype=float)

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)

    coords = x @ vt[:2].T
    var_ratio = s**2 / np.sum(s**2)

    pca = pd.DataFrame(coords, index=features.index, columns=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pca["PC1"], pca["PC2"], alpha=0.6, s=100)

    for firm, row in pca.iterrows():
        ax.annotate(firm, (row["PC1"], row["PC2"]), fontsize=9, ha="center")

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} Varianz)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} Varianz)")
    ax.set_title("PCA der CV-Vektoren")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return var_ratio[0], var_ratio[1]


def main() -> None:
    repo = find_repo_root(Path(__file__).resolve().parent)
    config = yaml.safe_load((repo / "configs" / "config.yaml").read_text(encoding="utf-8"))

    data = pd.read_csv(repo / config["paths"]["test_set"], sep=";", skiprows=1)
    data = data.dropna(axis=1, how="all")

    load_cols = [c for c in data.columns if c.startswith("LG")]

    data["timestamp"] = (
        data["Time stamp"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+[ab]$", "", regex=True)
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        format="%d.%m.%Y %H:%M:%S",
        errors="raise",
    )

    data["interval_start"] = data["timestamp"] - pd.Timedelta(minutes=15)
    data["date"] = data["interval_start"].dt.date
    data["weekday_no"] = data["interval_start"].dt.weekday
    data["weekday"] = data["interval_start"].dt.day_name()

    data[load_cols] = data[load_cols].apply(pd.to_numeric, errors="coerce") * 0.25

    daily = (
        data.groupby(["date", "weekday_no", "weekday"], as_index=False)[load_cols]
        .sum()
    )

    daily = daily.melt(
        id_vars=["date", "weekday_no", "weekday"],
        value_vars=load_cols,
        var_name="firm",
        value_name="daily_kwh",
    )

    median = daily.groupby(["firm", "weekday_no"])["daily_kwh"].transform("median")
    filtered = daily[daily["daily_kwh"] >= 0.5 * median].copy()

    result = (
        filtered.groupby(["weekday_no", "weekday", "firm"], as_index=False)["daily_kwh"]
        .agg(mean_daily_kwh="mean", std_daily_kwh="std", n_days="size")
    )

    result["cv_daily_kwh"] = result["std_daily_kwh"] / result["mean_daily_kwh"]

    out_dir = repo / "data" / "firm"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / "weekday_daily_energy_stats.csv"
    heatmap_path = out_dir / "weekday_daily_cv_heatmap.png"
    pca_path = out_dir / "weekday_cv_pca.png"

    result.to_csv(stats_path, index=False)

    heatmap = (
        result.pivot(index="firm", columns="weekday_no", values="cv_daily_kwh")
        .rename(columns={0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    )

    heatmap = heatmap.replace([np.inf, -np.inf], np.nan).dropna()
    heatmap = heatmap.loc[heatmap.mean(axis=1).sort_values().index]

    plot_heatmap(heatmap, heatmap_path)
    pc1, pc2 = plot_pca(heatmap, pca_path)

    print("Rows:", len(data))
    print("Daily rows:", len(daily))
    print("Removed low-load outliers:", len(daily) - len(filtered))
    print("Output:", stats_path)
    print("Heatmap:", heatmap_path)
    print("PCA plot:", pca_path)
    print(f"Explained variance: PC1={pc1:.1%}, PC2={pc2:.1%}")


if __name__ == "__main__":
    main()