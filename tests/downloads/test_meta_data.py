from pathlib import Path

import pandas as pd
import pytest
import requests

import download.meta_data as meta_data_module
from download.meta_data import (
    _decode_text,
    _find_repo_root,
    _parse_station_table,
    download_station_metadata,
)


class FakeResponse:
    def __init__(self, content: bytes, error: Exception | None = None):
        self.content = content
        self.error = error
        self.raise_for_status_calls = 0

    def raise_for_status(self) -> None:
        self.raise_for_status_calls += 1
        if self.error is not None:
            raise self.error


def metadata_cfg(
    url: str = "https://example.test/metadata.txt",
    output_path: str = "outputs/metadata_stations.csv",
) -> dict:
    return {
        "url": {"metadata": url},
        "paths": {"metadata": output_path},
    }


def dwd_text(rows: list[str]) -> str:
    return "\n".join(
        [
            "DWD metadata preamble",
            "",
            (
                "Stations_id von_datum bis_datum Stationshoehe geoBreite "
                "geoLaenge Stationsname Bundesland Abgabe"
            ),
            (
                "----------- --------- --------- ------------- --------- "
                "--------- ----------------------------------------- ---------- ------"
            ),
            *rows,
        ]
    )


def valid_row(
    station_id: str = "00232",
    height: str = "461",
    latitude: str = "48.4252",
    longitude: str = "10.9415",
    name_and_state: str = "Augsburg Bayern",
) -> str:
    return (
        f"{station_id} 20200101 20251231 {height} {latitude} {longitude} "
        f"{name_and_state} Frei"
    )


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


def test_decode_text_decodes_utf8_with_bom():
    assert _decode_text("Augsburg Bayern".encode("utf-8-sig")) == "Augsburg Bayern"


def test_decode_text_decodes_german_characters_from_cp1252():
    assert _decode_text("Großenkneten Niedersachsen".encode("cp1252")) == (
        "Großenkneten Niedersachsen"
    )


def test_decode_text_prefers_cp1252_for_cp1252_specific_bytes():
    assert _decode_text("„Augsburg“".encode("cp1252")) == "„Augsburg“"


def test_parse_station_table_parses_realistic_dwd_rows_and_skips_noise():
    text = dwd_text(
        [
            "too short",
            valid_row(
                station_id="00044",
                height="44",
                latitude="52.9336",
                longitude="8.2370",
                name_and_state="Großenkneten Niedersachsen",
            ),
            valid_row(
                station_id="00071",
                height="759",
                latitude="48.2156",
                longitude="8.9784",
                name_and_state="Albstadt-Badkap Baden-Württemberg",
            ),
        ]
    )

    parsed = _parse_station_table(text)

    expected = pd.DataFrame(
        {
            "station_id": ["00044", "00071"],
            "station_name": [
                "Großenkneten Niedersachsen",
                "Albstadt-Badkap Baden-Württemberg",
            ],
            "latitude": ["52.9336", "48.2156"],
            "longitude": ["8.2370", "8.9784"],
            "height_m_amsl": ["44", "759"],
        }
    )
    pd.testing.assert_frame_equal(parsed, expected)


@pytest.mark.parametrize(
    "text",
    [
        "no matching header\n00044 20200101 20251231 44 52.9336 8.2370 Name Frei",
        dwd_text(["too short"]),
    ],
)
def test_parse_station_table_raises_when_no_station_rows_are_found(text: str):
    with pytest.raises(ValueError, match="Keine Stationszeilen"):
        _parse_station_table(text)


def test_download_station_metadata_fetches_normalizes_and_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()
    response = FakeResponse(
        dwd_text(
            [
                valid_row(
                    station_id="00232",
                    height="461",
                    latitude="48.4252",
                    longitude="10.9415",
                    name_and_state="Augsburg Bayern",
                ),
                valid_row(
                    station_id="00044",
                    height="44",
                    latitude="52.9336",
                    longitude="8.2370",
                    name_and_state="Großenkneten Niedersachsen",
                ),
                valid_row(
                    station_id="44",
                    height="999",
                    latitude="1.0",
                    longitude="2.0",
                    name_and_state="Duplicate Should Drop",
                ),
                valid_row(
                    station_id="00003",
                    height="202",
                    latitude="50.7827",
                    longitude="6.0941",
                    name_and_state="Aachen Nordrhein-Westfalen",
                ),
                valid_row(station_id="bad-id"),
                valid_row(station_id="00055", latitude="not-a-latitude"),
                valid_row(station_id="00056", longitude="not-a-longitude"),
                valid_row(station_id="00057", height="not-a-height"),
            ]
        ).encode("cp1252")
    )
    calls: list[tuple[str, int]] = []

    def fake_get(url: str, timeout: int):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)

    download_station_metadata(cfg, tmp_path)

    assert calls == [(cfg["url"]["metadata"], 120)]
    assert response.raise_for_status_calls == 1

    output_path = tmp_path / cfg["paths"]["metadata"]
    written = pd.read_csv(output_path, dtype={"station_id": str})

    assert list(written.columns) == [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "height_m_amsl",
    ]
    assert written["station_id"].tolist() == ["00003", "00044", "00232"]
    assert written["station_name"].tolist() == [
        "Aachen Nordrhein-Westfalen",
        "Großenkneten Niedersachsen",
        "Augsburg Bayern",
    ]
    assert written["latitude"].tolist() == pytest.approx([50.7827, 52.9336, 48.4252])
    assert written["longitude"].tolist() == pytest.approx([6.0941, 8.2370, 10.9415])
    assert written["height_m_amsl"].tolist() == pytest.approx([202.0, 44.0, 461.0])


def test_download_station_metadata_propagates_http_errors_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()
    response = FakeResponse(
        dwd_text([valid_row()]).encode("utf-8"),
        error=requests.HTTPError("HTTP 404"),
    )

    def fake_get(url: str, timeout: int):
        return response

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError, match="HTTP 404"):
        download_station_metadata(cfg, tmp_path)

    assert response.raise_for_status_calls == 1
    assert not (tmp_path / cfg["paths"]["metadata"]).exists()


def test_download_station_metadata_propagates_request_errors_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()

    def fake_get(url: str, timeout: int):
        raise requests.RequestException("network unavailable")

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)

    with pytest.raises(requests.RequestException, match="network unavailable"):
        download_station_metadata(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["metadata"]).exists()


def test_download_station_metadata_propagates_decode_errors_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()

    def fake_get(url: str, timeout: int):
        return FakeResponse(b"raw")

    def fake_decode_text(content: bytes):
        raise UnicodeDecodeError("fake", content, 0, 1, "cannot decode")

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)
    monkeypatch.setattr(meta_data_module, "_decode_text", fake_decode_text)

    with pytest.raises(UnicodeDecodeError):
        download_station_metadata(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["metadata"]).exists()


def test_download_station_metadata_propagates_parse_errors_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()

    def fake_get(url: str, timeout: int):
        return FakeResponse(b"not a station table")

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)

    with pytest.raises(ValueError, match="Keine Stationszeilen"):
        download_station_metadata(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["metadata"]).exists()


def test_download_station_metadata_rejects_when_all_rows_are_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = metadata_cfg()

    def fake_get(url: str, timeout: int):
        return FakeResponse(
            dwd_text(
                [
                    valid_row(station_id="bad-id"),
                    valid_row(station_id="00055", latitude="not-a-latitude"),
                    valid_row(station_id="00056", longitude="not-a-longitude"),
                    valid_row(station_id="00057", height="not-a-height"),
                ]
            ).encode("utf-8")
        )

    monkeypatch.setattr(meta_data_module.requests, "get", fake_get)

    with pytest.raises(ValueError, match="Keine validen Stationszeilen"):
        download_station_metadata(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["metadata"]).exists()


def test_main_loads_config_and_calls_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "logging:",
                "  level: INFO",
                "  format: '%(levelname)s:%(message)s'",
                "  datefmt: '%Y-%m-%d'",
                "url:",
                "  metadata: https://example.test/metadata.txt",
                "paths:",
                "  metadata: outputs/metadata.csv",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_download_station_metadata(cfg: dict, repo_root: Path):
        calls.append((cfg, repo_root))

    monkeypatch.setattr(meta_data_module, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(
        meta_data_module,
        "download_station_metadata",
        fake_download_station_metadata,
    )

    meta_data_module.main()

    assert len(calls) == 1
    cfg, repo_root = calls[0]
    assert repo_root == tmp_path
    assert cfg["url"]["metadata"] == "https://example.test/metadata.txt"
    assert cfg["paths"]["metadata"] == "outputs/metadata.csv"
