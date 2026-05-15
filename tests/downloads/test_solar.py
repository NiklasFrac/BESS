from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

import download.solar as solar_module
from download.solar import _find_data_member, download_dwd_10min_solar


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.raise_for_status_calls = 0

    def raise_for_status(self) -> None:
        self.raise_for_status_calls += 1


def zip_bytes(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def solar_cfg(output_path: str = "outputs/solar.csv") -> dict:
    return {
        "url": {"solar": "https://example.test/solar.zip"},
        "station": {"id": "00232"},
        "time": {
            "start_utc": "2024-01-01 00:00:00",
            "end_utc": "2024-01-01 00:30:00",
        },
        "paths": {"solar": output_path},
    }


def solar_csv(rows: list[str], header: str | None = None) -> str:
    if header is None:
        header = " MESS_DATUM ; STATIONS_ID ; GS_10 ; DS_10 ; ignored"
    return "\n".join([header, *rows])


def test_find_data_member_selects_only_real_data_txt():
    content = zip_bytes(
        {
            "produkt_zehn_min_sd_00232.txt": "data",
            "Metadaten_sd_00232.txt": "metadata",
            "Beschreibung_sd_00232.txt": "description",
            "notes.csv": "ignored",
        }
    )

    with ZipFile(BytesIO(content)) as zf:
        assert _find_data_member(zf) == "produkt_zehn_min_sd_00232.txt"


@pytest.mark.parametrize(
    "files, error_type",
    [
        ({"Metadaten_sd_00232.txt": "metadata"}, FileNotFoundError),
        (
            {
                "produkt_a.txt": "data-a",
                "produkt_b.txt": "data-b",
            },
            ValueError,
        ),
    ],
)
def test_find_data_member_rejects_missing_or_ambiguous_data_txt(
    files: dict[str, str],
    error_type: type[Exception],
):
    with ZipFile(BytesIO(zip_bytes(files))) as zf:
        with pytest.raises(error_type):
            _find_data_member(zf)


def test_download_dwd_10min_solar_filters_sorts_and_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = solar_cfg()
    response = FakeResponse(
        zip_bytes(
            {
                "produkt_zehn_min_sd_00232.txt": solar_csv(
                    [
                        "202401010020;00232;20;30;x",
                        "202401010000;00232;0;-999;x",
                        "202401010030;00232;999;999;x",
                        "202312312350;00232;111;222;x",
                        "202401010010;00232;10;11;x",
                    ]
                ),
                "Metadaten_sd_00232.txt": "ignored",
            }
        )
    )
    calls: list[tuple[str, int]] = []

    def fake_get(url: str, timeout: int):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(solar_module.requests, "get", fake_get)

    download_dwd_10min_solar(cfg, tmp_path)

    assert calls == [(cfg["url"]["solar"], 120)]
    assert response.raise_for_status_calls == 1

    written = pd.read_csv(tmp_path / cfg["paths"]["solar"])
    assert list(written.columns) == ["timestamp_utc", "GS_10", "DS_10"]
    assert written["timestamp_utc"].tolist() == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:10:00+00:00",
        "2024-01-01 00:20:00+00:00",
    ]
    assert written["GS_10"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert pd.isna(written.loc[0, "DS_10"])
    assert written["DS_10"].iloc[1:].tolist() == pytest.approx([11.0, 30.0])


def test_download_dwd_10min_solar_rejects_wrong_station_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = solar_cfg()
    response = FakeResponse(
        zip_bytes(
            {"produkt_zehn_min_sd_00232.txt": solar_csv(["202401010000;00044;0;0;x"])}
        )
    )

    monkeypatch.setattr(solar_module.requests, "get", lambda url, timeout: response)

    with pytest.raises(ValueError, match="Unerwartete STATIONS_ID"):
        download_dwd_10min_solar(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["solar"]).exists()


@pytest.mark.parametrize(
    "header",
    [
        " STATIONS_ID ; GS_10 ; DS_10 ; ignored",
        " MESS_DATUM ; STATIONS_ID ; DS_10 ; ignored",
    ],
)
def test_download_dwd_10min_solar_rejects_missing_core_columns_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: str,
):
    cfg = solar_cfg()
    response = FakeResponse(
        zip_bytes(
            {
                "produkt_zehn_min_sd_00232.txt": solar_csv(
                    ["202401010000;00232;0;0"], header=header
                )
            }
        )
    )

    monkeypatch.setattr(solar_module.requests, "get", lambda url, timeout: response)

    with pytest.raises(KeyError):
        download_dwd_10min_solar(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["solar"]).exists()


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
                "  solar: https://example.test/solar.zip",
                "station:",
                "  id: '00232'",
                "time:",
                "  start_utc: '2024-01-01 00:00:00'",
                "  end_utc: '2024-01-01 00:30:00'",
                "paths:",
                "  solar: outputs/solar.csv",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_download(cfg: dict, repo_root: Path):
        calls.append((cfg, repo_root))

    monkeypatch.setattr(solar_module, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(solar_module, "download_dwd_10min_solar", fake_download)

    solar_module.main()

    assert len(calls) == 1
    cfg, repo_root = calls[0]
    assert repo_root == tmp_path
    assert cfg["url"]["solar"] == "https://example.test/solar.zip"
    assert cfg["paths"]["solar"] == "outputs/solar.csv"
