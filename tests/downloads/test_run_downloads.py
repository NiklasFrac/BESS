from pathlib import Path

import pytest

import download.run_downloads as run_downloads_module


def write_config(repo_root: Path) -> None:
    config_path = repo_root / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "logging:",
                "  level: INFO",
                "  format: '%(levelname)s:%(message)s'",
                "  datefmt: '%Y-%m-%d'",
                "station:",
                "  id: '00232'",
            ]
        ),
        encoding="utf-8",
    )


def test_main_runs_download_steps_in_order_with_same_config_and_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_config(tmp_path)
    calls: list[tuple[str, dict, Path]] = []

    def record(name: str):
        def step(cfg: dict, repo_root: Path) -> None:
            calls.append((name, cfg, repo_root))

        return step

    monkeypatch.setattr(run_downloads_module, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(
        run_downloads_module,
        "download_station_metadata",
        record("metadata"),
    )
    monkeypatch.setattr(
        run_downloads_module,
        "download_dwd_temp_pressure_wind",
        record("weather"),
    )
    monkeypatch.setattr(
        run_downloads_module,
        "download_dwd_10min_solar",
        record("solar"),
    )
    monkeypatch.setattr(
        run_downloads_module,
        "download_pvgis_horizon",
        record("horizon"),
    )

    run_downloads_module.main()

    assert [name for name, _cfg, _repo_root in calls] == [
        "metadata",
        "weather",
        "solar",
        "horizon",
    ]
    assert {id(cfg) for _name, cfg, _repo_root in calls} == {id(calls[0][1])}
    assert all(repo_root == tmp_path for _name, _cfg, repo_root in calls)


def test_main_stops_when_a_download_step_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_config(tmp_path)
    calls: list[str] = []

    class DownloadStepError(RuntimeError):
        pass

    def metadata(cfg: dict, repo_root: Path) -> None:
        calls.append("metadata")

    def weather(cfg: dict, repo_root: Path) -> None:
        calls.append("weather")
        raise DownloadStepError("weather failed")

    def solar(cfg: dict, repo_root: Path) -> None:
        calls.append("solar")

    def horizon(cfg: dict, repo_root: Path) -> None:
        calls.append("horizon")

    monkeypatch.setattr(run_downloads_module, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(run_downloads_module, "download_station_metadata", metadata)
    monkeypatch.setattr(
        run_downloads_module,
        "download_dwd_temp_pressure_wind",
        weather,
    )
    monkeypatch.setattr(run_downloads_module, "download_dwd_10min_solar", solar)
    monkeypatch.setattr(run_downloads_module, "download_pvgis_horizon", horizon)

    with pytest.raises(DownloadStepError, match="weather failed"):
        run_downloads_module.main()

    assert calls == ["metadata", "weather"]
