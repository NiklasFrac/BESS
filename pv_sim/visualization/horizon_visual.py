from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["axes3d.mouserotationstyle"] = "azel"


def _horizon_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(df["azimuth_deg"].to_numpy(dtype=float))
    z = df["horizon_height_deg"].to_numpy(dtype=float)

    az = np.r_[az, az[0]]
    z = np.r_[z, z[0]]

    x = np.sin(az)
    y = np.cos(az)
    return x, y, z


def plot_horizon_profile(
    horizon_path: Path,
    plot_path: Path,
    station_name: str,
    show: bool = False,
) -> None:
    df = pd.read_csv(horizon_path)

    x, y, z = _horizon_coordinates(df)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, lw=2, label="Horizont")
    ax.plot(x, y, np.zeros_like(z), lw=1, alpha=0.4)

    for xi, yi, zi in zip(x, y, z):
        ax.plot([xi, xi], [yi, yi], [0, zi], lw=0.8, alpha=0.5)

    ax.scatter(0, 0, 0, s=30, label="Standort")

    ax.set_xlabel("Ost")
    ax.set_ylabel("Nord")
    ax.set_zlabel("Horizonthoehe [deg]")
    ax.set_title(f"3D-Horizontprofil {station_name}")
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=25, azim=-60)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)