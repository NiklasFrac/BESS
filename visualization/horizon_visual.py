from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import matplotlib as mpl
mpl.rcParams["axes3d.mouserotationstyle"] = "azel"

def main() -> None:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    path = Path(cfg["paths"]["pvgis"]).with_suffix(".csv")
    df = pd.read_csv(path)

    az = np.deg2rad(df["azimuth_deg"].to_numpy())
    z = df["horizon_height_deg"].to_numpy()

    az = np.r_[az, az[0]]
    z = np.r_[z, z[0]]

    x = np.sin(az)
    y = np.cos(az)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, lw=2, label="Horizont")
    ax.plot(x, y, np.zeros_like(z), lw=1, alpha=0.4)

    for xi, yi, zi in zip(x, y, z):
        ax.plot([xi, xi], [yi, yi], [0, zi], lw=0.8, alpha=0.5)

    ax.scatter(0, 0, 0, s=30, label="Standort")

    ax.set_xlabel("Ost")
    ax.set_ylabel("Nord")
    ax.set_zlabel("Horizonthöhe [°]")
    ax.set_title("3D-Horizontprofil Augsburg")
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()