import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from pv_sim.visualization.horizon_visual import _horizon_coordinates, plot_horizon_profile


def test_horizon_coordinates_close_loop_and_map_cardinal_directions():
    df = pd.DataFrame(
        {
            "azimuth_deg": [0.0, 90.0, 180.0, 270.0],
            "horizon_height_deg": [1.0, 2.0, 3.0, 4.0],
        }
    )

    x, y, z = _horizon_coordinates(df)

    assert x.tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0, 0.0], abs=1e-12)
    assert y.tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0, 1.0], abs=1e-12)
    assert z.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 1.0])


def test_plot_horizon_profile_writes_plot_file(tmp_path):
    horizon_path = tmp_path / "horizon.csv"
    plot_path = tmp_path / "plots" / "horizon.png"
    pd.DataFrame(
        {
            "azimuth_deg": [0.0, 90.0, 180.0, 270.0],
            "horizon_height_deg": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(horizon_path, index=False)

    plot_horizon_profile(horizon_path, plot_path, "Augsburg")

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
