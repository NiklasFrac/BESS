import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

mpl.rcParams["axes3d.mouserotationstyle"] = "azel"


def _horizon_coordinates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(df["azimuth_deg"].to_numpy(dtype=float))
    z = df["horizon_height_deg"].to_numpy(dtype=float)

    az = np.r_[az, az[0]]
    z = np.r_[z, z[0]]

    x = np.sin(az)   # Osten
    y = np.cos(az)   # Norden
    return x, y, z


def plot_horizon_profile(
    horizon_path: Path,
    plot_path: Path,
    station_name: str,
    show: bool = False,
) -> None:
    df = pd.read_csv(horizon_path)
    x, y, z = _horizon_coordinates(df)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    z_max = max(1.0, float(np.nanmax(z)))

    ax.plot(x, y, np.zeros_like(z), lw=1.0, alpha=0.35, color="black")
    ax.plot(x, y, z, lw=2.5, color="tab:blue", label="Horizont")
    
    step = max(1, len(x) // 36)
    for xi, yi, zi in zip(x[::step], y[::step], z[::step]):
        ax.plot([xi, xi], [yi, yi], [0, zi], lw=0.8, alpha=0.35, color="tab:blue")

    ax.scatter(0, 0, 0, s=55, color="tab:red", label="Standort", depthshade=False)

    label_z = 0.03 * z_max
    ax.text(0, 1.15, label_z, "N", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(1.15, 0, label_z, "O", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0, -1.15, label_z, "S", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(-1.15, 0, label_z, "W", ha="center", va="center", fontsize=12, weight="bold")

    ax.set_xlabel("Ost")
    ax.set_ylabel("Nord")
    ax.set_zlabel("Horizonthöhe [°]")
    ax.set_title(f"3D-Horizontprofil {station_name}")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.25 * z_max)

    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=32, azim=-135)

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    log.info("Plot gespeichert: %s", plot_path)
