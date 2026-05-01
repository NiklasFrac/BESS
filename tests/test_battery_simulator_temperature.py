import math
from pathlib import Path

import pandas as pd
import pytest

from battery_sim import simulator


def config(freq: str = "1h") -> dict:
    return {
        "time": {"freq": freq},
        "paths": {"energy": "results/energy_curve.csv"},
        "batterie": {
            "capacity_kwh": 100.0,
            "soc_min": 0.05,
            "soc_max": 0.95,
            "max_charge_kw": 100.0,
            "max_discharge_kw": 100.0,
            "eta_charge": 0.9,
            "eta_discharge": 0.8,
        },
        "thermal": {
            "initial_temp_degC": 20.0,
            "thermal_time_constant_h": 6.0,
            "heat_capacity_kwh_per_degC": 50.0,
            "heat_to_battery_fraction": 1.0,
        },
    }


def write_energy_curve(repo_root: Path, rows: list[dict]) -> Path:
    path = repo_root / "results" / "energy_curve.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=simulator.ENERGY_COLUMNS).to_csv(path, index=False)
    return path


def expected_temperature(
    *,
    temp_before: float,
    ambient_temp_degC: float,
    heat_loss_kwh: float,
    thermal_spec: dict,
    dt_h: float,
) -> float:
    heat_to_battery_kw = heat_loss_kwh * thermal_spec["heat_to_battery_fraction"] / dt_h
    thermal_resistance = thermal_spec["thermal_time_constant_h"] / thermal_spec["heat_capacity_kwh_per_degC"]
    equilibrium_temp = ambient_temp_degC + thermal_resistance * heat_to_battery_kw
    decay = math.exp(-dt_h / thermal_spec["thermal_time_constant_h"])
    return equilibrium_temp + (temp_before - equilibrium_temp) * decay


def test_simulate_outputs_temperature_rows_with_expected_schema_and_timestamps(tmp_path: Path):
    cfg = config()
    rows = [
        {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 10.0, "TT_10": 20.0},
        {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": 19.0},
        {"timestamp_utc": "2020-01-01 02:00:00+00:00", "e_net_ac_kwh": -5.0, "TT_10": 18.0},
    ]
    write_energy_curve(tmp_path, rows)

    battery_rows, temperature_rows = simulator.simulate(tmp_path, cfg)

    assert len(battery_rows) == 3
    assert len(temperature_rows) == 3
    assert [row["timestamp_utc"] for row in temperature_rows] == [row["timestamp_utc"] for row in rows]
    assert set(battery_rows[0]) == {
        "timestamp_utc",
        "pv_energy_kwh",
        "action_kw",
        "charge_ac_kwh",
        "discharge_ac_kwh",
        "loss_kwh",
        "soc_kwh",
        "soc_fraction",
    }
    assert set(temperature_rows[0]) == {
        "timestamp_utc",
        "ambient_temp_degC",
        "battery_temp_degC",
        "heat_loss_kwh",
    }


def test_simulate_derives_timestep_and_action_power_from_config_frequency(tmp_path: Path):
    cfg = config(freq="30min")
    write_energy_curve(
        tmp_path,
        [{"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 10.0, "TT_10": 20.0}],
    )

    battery_rows, _ = simulator.simulate(tmp_path, cfg)

    assert battery_rows[0]["action_kw"] == pytest.approx(20.0)
    assert battery_rows[0]["charge_ac_kwh"] == pytest.approx(10.0)
    assert battery_rows[0]["loss_kwh"] == pytest.approx(1.0)


def test_simulate_uses_tt10_as_ambient_temperature_and_passes_battery_losses(tmp_path: Path):
    cfg = config()
    rows = [
        {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 10.0, "TT_10": 12.0},
        {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": -5.0, "TT_10": 13.0},
    ]
    write_energy_curve(tmp_path, rows)

    battery_rows, temperature_rows = simulator.simulate(tmp_path, cfg)

    assert [row["ambient_temp_degC"] for row in temperature_rows] == [12.0, 13.0]
    assert [row["heat_loss_kwh"] for row in temperature_rows] == pytest.approx(
        [row["loss_kwh"] for row in battery_rows]
    )


def test_simulate_temperature_values_match_charge_idle_and_discharge_physics(tmp_path: Path):
    cfg = config()
    rows = [
        {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 10.0, "TT_10": 20.0},
        {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": 20.0},
        {"timestamp_utc": "2020-01-01 02:00:00+00:00", "e_net_ac_kwh": -5.0, "TT_10": 20.0},
    ]
    write_energy_curve(tmp_path, rows)

    battery_rows, temperature_rows = simulator.simulate(tmp_path, cfg)

    assert battery_rows[0]["loss_kwh"] == pytest.approx(1.0)
    assert battery_rows[1]["loss_kwh"] == pytest.approx(0.0)
    assert battery_rows[2]["loss_kwh"] == pytest.approx(1.25)

    expected_first = expected_temperature(
        temp_before=20.0,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.0,
        thermal_spec=cfg["thermal"],
        dt_h=1.0,
    )
    expected_second = expected_temperature(
        temp_before=expected_first,
        ambient_temp_degC=20.0,
        heat_loss_kwh=0.0,
        thermal_spec=cfg["thermal"],
        dt_h=1.0,
    )
    expected_third = expected_temperature(
        temp_before=expected_second,
        ambient_temp_degC=20.0,
        heat_loss_kwh=1.25,
        thermal_spec=cfg["thermal"],
        dt_h=1.0,
    )

    assert temperature_rows[0]["battery_temp_degC"] == pytest.approx(expected_first)
    assert temperature_rows[1]["battery_temp_degC"] == pytest.approx(expected_second)
    assert temperature_rows[2]["battery_temp_degC"] == pytest.approx(expected_third)


def test_simulate_first_temperature_row_starts_from_config_initial_temperature(tmp_path: Path):
    cfg = config()
    cfg["thermal"]["initial_temp_degC"] = 5.0
    write_energy_curve(
        tmp_path,
        [{"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": 20.0}],
    )

    _, temperature_rows = simulator.simulate(tmp_path, cfg)

    expected = expected_temperature(
        temp_before=5.0,
        ambient_temp_degC=20.0,
        heat_loss_kwh=0.0,
        thermal_spec=cfg["thermal"],
        dt_h=1.0,
    )
    assert temperature_rows[0]["battery_temp_degC"] == pytest.approx(expected)
    assert temperature_rows[0]["battery_temp_degC"] > 5.0


def test_simulate_forwards_exact_heat_loss_argument_to_temperature_step(tmp_path: Path, monkeypatch):
    cfg = config()
    write_energy_curve(
        tmp_path,
        [
            {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 10.0, "TT_10": 20.0},
            {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": -5.0, "TT_10": 20.0},
        ],
    )
    captured_heat_losses = []
    original_step_temperature = simulator.step_temperature

    def recording_step_temperature(**kwargs):
        captured_heat_losses.append(kwargs["heat_loss_kwh"])
        return original_step_temperature(**kwargs)

    monkeypatch.setattr(simulator, "step_temperature", recording_step_temperature)

    battery_rows, _ = simulator.simulate(tmp_path, cfg)

    assert captured_heat_losses == pytest.approx([row["loss_kwh"] for row in battery_rows])


def test_simulate_fills_invalid_ambient_temperatures_with_forward_and_backward_fill(tmp_path: Path):
    cfg = config()
    write_energy_curve(
        tmp_path,
        [
            {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "bad"},
            {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "15.0"},
            {"timestamp_utc": "2020-01-01 02:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "inf"},
            {"timestamp_utc": "2020-01-01 03:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "21.0"},
        ],
    )

    _, temperature_rows = simulator.simulate(tmp_path, cfg)

    assert [row["ambient_temp_degC"] for row in temperature_rows] == [15.0, 15.0, 15.0, 21.0]


def test_simulate_rejects_energy_curve_when_all_ambient_temperatures_are_invalid(tmp_path: Path):
    cfg = config()
    write_energy_curve(
        tmp_path,
        [
            {"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "bad"},
            {"timestamp_utc": "2020-01-01 01:00:00+00:00", "e_net_ac_kwh": 0.0, "TT_10": "-inf"},
        ],
    )

    with pytest.raises(ValueError, match="No finite ambient temperature"):
        simulator.simulate(tmp_path, cfg)


def test_simulate_rejects_missing_required_energy_curve_columns(tmp_path: Path):
    cfg = config()
    path = tmp_path / "results" / "energy_curve.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"timestamp_utc": "2020-01-01 00:00:00+00:00", "e_net_ac_kwh": 0.0}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        simulator.simulate(tmp_path, cfg)


def test_simulate_empty_energy_curve_returns_empty_result_lists(tmp_path: Path):
    cfg = config()
    write_energy_curve(tmp_path, [])

    battery_rows, temperature_rows = simulator.simulate(tmp_path, cfg)

    assert battery_rows == []
    assert temperature_rows == []
