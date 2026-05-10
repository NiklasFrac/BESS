from pathlib import Path

import pandas as pd
import pytest

import download.horizon as horizon_module
from download.horizon import (
    _find_repo_root,
    _load_station_coords,
    download_pvgis_horizon,
)


def horizon_cfg(
    metadata_path: str = "metadata/stations.csv",
    output_path: str = "outputs/pvgis/horizon.csv",
    station_id: str = "00232",
) -> dict:
    return {
        "station": {"id": station_id, "name": "Augsburg Bayern"},
        "paths": {"metadata": metadata_path, "pvgis": output_path},
    }


def write_metadata(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_find_repo_root_accepts_start_directory_with_data(tmp_path: Path):
    (tmp_path / "data").mkdir()

    assert _find_repo_root(tmp_path) == tmp_path


def test_find_repo_root_walks_up_to_parent_with_data(tmp_path: Path):
    repo_root = tmp_path / "repo"
    nested = repo_root / "download" / "subdir"
    (repo_root / "data").mkdir(parents=True)
    nested.mkdir(parents=True)

    assert _find_repo_root(nested) == repo_root


def test_find_repo_root_raises_when_data_directory_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Repo-Root"):
        _find_repo_root(Path("missing") / "nested")


def test_load_station_coords_preserves_leading_zeroes_and_strips_whitespace(
    tmp_path: Path,
):
    metadata_path = write_metadata(
        tmp_path / "metadata.csv",
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                " 00232 ,Augsburg,48.3715,10.8985",
                "232,Wrong station,1.0,2.0",
                "00044,Other station,52.9336,8.2370",
            ]
        ),
    )

    lat, lon = _load_station_coords(metadata_path, " 00232 ")

    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert lat == pytest.approx(48.3715)
    assert lon == pytest.approx(10.8985)


def test_load_station_coords_raises_for_unknown_station(tmp_path: Path):
    metadata_path = write_metadata(
        tmp_path / "metadata.csv",
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "00044,Other station,52.9336,8.2370",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Station ID '00232' not found"):
        _load_station_coords(metadata_path, "00232")


def test_load_station_coords_raises_for_missing_required_columns(tmp_path: Path):
    metadata_path = write_metadata(
        tmp_path / "metadata.csv",
        "\n".join(
            [
                "station_id,station_name,latitude",
                "00232,Augsburg,48.3715",
            ]
        ),
    )

    with pytest.raises(KeyError, match="missing columns: longitude"):
        _load_station_coords(metadata_path, "00232")


@pytest.mark.parametrize("bad_value", ["not-a-number", "nan", "inf"])
def test_load_station_coords_raises_for_invalid_coordinates(
    tmp_path: Path,
    bad_value: str,
):
    metadata_path = write_metadata(
        tmp_path / "metadata.csv",
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                f"00232,Augsburg,{bad_value},10.8985",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Invalid coordinates"):
        _load_station_coords(metadata_path, "00232")


def test_download_pvgis_horizon_fetches_transforms_and_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = horizon_cfg()
    write_metadata(
        tmp_path / cfg["paths"]["metadata"],
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "00232,Augsburg,48.3715,10.8985",
            ]
        ),
    )
    calls: list[dict[str, float]] = []

    def fake_get_pvgis_horizon(latitude: float, longitude: float):
        calls.append({"latitude": latitude, "longitude": longitude})
        horizon = pd.Series(
            [1.5, 0.8, 2.0],
            index=pd.Index([0.0, 90.0, 180.0], name="ignored"),
        )
        return horizon, {"metadata": "ignored"}

    monkeypatch.setattr(
        horizon_module.pvlib.iotools,
        "get_pvgis_horizon",
        fake_get_pvgis_horizon,
    )

    df = download_pvgis_horizon(cfg, tmp_path)

    expected = pd.DataFrame(
        {
            "azimuth_deg": [0.0, 90.0, 180.0],
            "horizon_height_deg": [1.5, 0.8, 2.0],
        }
    )
    assert calls == [{"latitude": 48.3715, "longitude": 10.8985}]
    pd.testing.assert_frame_equal(df, expected)

    output_path = tmp_path / cfg["paths"]["pvgis"]
    assert output_path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(output_path), expected)


def test_download_pvgis_horizon_rejects_non_series_response_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = horizon_cfg()
    write_metadata(
        tmp_path / cfg["paths"]["metadata"],
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "00232,Augsburg,48.3715,10.8985",
            ]
        ),
    )

    def fake_get_pvgis_horizon(latitude: float, longitude: float):
        return pd.DataFrame({"height": [1.0]}), {}

    monkeypatch.setattr(
        horizon_module.pvlib.iotools,
        "get_pvgis_horizon",
        fake_get_pvgis_horizon,
    )

    with pytest.raises(TypeError, match="Expected pd.Series"):
        download_pvgis_horizon(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["pvgis"]).exists()


def test_download_pvgis_horizon_propagates_pvgis_errors_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = horizon_cfg()
    write_metadata(
        tmp_path / cfg["paths"]["metadata"],
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "00232,Augsburg,48.3715,10.8985",
            ]
        ),
    )

    class PvgisError(RuntimeError):
        pass

    def fake_get_pvgis_horizon(latitude: float, longitude: float):
        raise PvgisError("PVGIS unavailable")

    monkeypatch.setattr(
        horizon_module.pvlib.iotools,
        "get_pvgis_horizon",
        fake_get_pvgis_horizon,
    )

    with pytest.raises(PvgisError, match="PVGIS unavailable"):
        download_pvgis_horizon(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["pvgis"]).exists()
