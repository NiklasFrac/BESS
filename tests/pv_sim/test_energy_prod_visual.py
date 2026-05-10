import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from pv_sim.visualization.energy_prod_visual import _daily_summary, plot_energy_overview


def test_daily_summary_sums_days_and_adds_rolling_mean():
    df = pd.DataFrame(
        {
            "timestamp_utc": [
                "2024-01-02 00:00:00+00:00",
                "2024-01-01 12:00:00+00:00",
                "2024-01-01 00:00:00+00:00",
            ],
            "e_net_ac_kwh": [3.0, 2.0, 1.0],
        }
    )

    daily = _daily_summary(df)

    assert daily["e_net_ac_kwh"].tolist() == pytest.approx([3.0, 3.0])
    assert daily["e_net_ac_kwh_14d"].tolist() == pytest.approx([3.0, 3.0])


def test_plot_energy_overview_writes_plot_file(tmp_path):
    energy_path = tmp_path / "energy.csv"
    plot_path = tmp_path / "plots" / "energy.png"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "e_net_ac_kwh": [1.0, 2.0, 3.0],
        }
    ).to_csv(energy_path, index=False)

    plot_energy_overview(energy_path, plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
