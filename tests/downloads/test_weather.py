from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

import download.weather as weather_module
from download.weather import (
    _find_data_member,
    _read_dwd_product,
    download_dwd_temp_pressure_wind,
)


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


def product_csv(rows: list[str], header: str | None = None) -> str:
    if header is None:
        header = " MESS_DATUM ; STATIONS_ID ; TT_10 ; PP_10 ; FF_10"
    return "\n".join([header, *rows])


def weather_cfg(output_path: str = "outputs/meteo.csv") -> dict:
    return {
        "url": {
            "air_temp_url": "https://example.test/temp.zip",
            "wind_url": "https://example.test/wind.zip",
        },
        "station": {"id": "00232"},
        "time": {
            "start_utc": "2024-01-01 00:00:00",
            "end_utc": "2024-01-01 00:30:00",
        },
        "paths": {"meteo": output_path},
    }


def test_find_data_member_selects_only_real_data_txt():
    content = zip_bytes(
        {
            "produkt_zehn_min_tu_00232.txt": "data",
            "Metadaten_tu_00232.txt": "metadata",
            "Beschreibung_tu_00232.txt": "description",
        }
    )

    with ZipFile(BytesIO(content)) as zf:
        assert _find_data_member(zf) == "produkt_zehn_min_tu_00232.txt"


@pytest.mark.parametrize(
    "files, error_type",
    [
        ({"Metadaten_tu_00232.txt": "metadata"}, FileNotFoundError),
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


def test_read_dwd_product_fetches_validates_filters_and_parses_values(
    monkeypatch: pytest.MonkeyPatch,
):
    response = FakeResponse(
        zip_bytes(
            {
                "produkt_zehn_min_tu_00232.txt": product_csv(
                    [
                        "202312312350;00232;99;999;9",
                        "202401010000;00232;4.2;955.1;4.4",
                        "202401010010;00232;-999;-999.0;-999",
                        "202401010030;00232;7.0;958.0;5.0",
                    ]
                )
            }
        )
    )
    calls: list[tuple[str, int]] = []

    def fake_get(url: str, timeout: int):
        calls.append((url, timeout))
        return response

    monkeypatch.setattr(weather_module.requests, "get", fake_get)

    df = _read_dwd_product(
        "https://example.test/temp.zip",
        "00232",
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
    )

    assert calls == [("https://example.test/temp.zip", 120)]
    assert response.raise_for_status_calls == 1
    assert df["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-01-01 00:10:00+00:00"),
    ]
    assert df["TT_10"].iloc[0] == pytest.approx(4.2)
    assert df["PP_10"].iloc[0] == pytest.approx(955.1)
    assert df["FF_10"].iloc[0] == pytest.approx(4.4)
    assert pd.isna(df.loc[df.index[1], "TT_10"])
    assert pd.isna(df.loc[df.index[1], "PP_10"])
    assert pd.isna(df.loc[df.index[1], "FF_10"])


@pytest.mark.parametrize(
    "header",
    [
        " STATIONS_ID ; TT_10 ; PP_10 ; FF_10",
        " MESS_DATUM ; TT_10 ; PP_10 ; FF_10",
    ],
)
def test_read_dwd_product_rejects_missing_required_columns(
    monkeypatch: pytest.MonkeyPatch,
    header: str,
):
    response = FakeResponse(
        zip_bytes(
            {
                "produkt_zehn_min_tu_00232.txt": product_csv(
                    ["202401010000;00232;4.2;955.1"], header=header
                )
            }
        )
    )
    monkeypatch.setattr(weather_module.requests, "get", lambda url, timeout: response)

    with pytest.raises(KeyError):
        _read_dwd_product(
            "https://example.test/temp.zip",
            "00232",
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
        )


def test_read_dwd_product_rejects_wrong_station(monkeypatch: pytest.MonkeyPatch):
    response = FakeResponse(
        zip_bytes(
            {
                "produkt_zehn_min_tu_00232.txt": product_csv(
                    ["202401010000;00044;4.2;955.1;4.4"]
                )
            }
        )
    )
    monkeypatch.setattr(weather_module.requests, "get", lambda url, timeout: response)

    with pytest.raises(ValueError, match="Unerwartete STATIONS_ID"):
        _read_dwd_product(
            "https://example.test/temp.zip",
            "00232",
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
        )


def test_download_dwd_temp_pressure_wind_merges_sorts_and_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = weather_cfg()
    calls: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []

    def fake_read(
        url: str,
        station_id: str,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        calls.append((url, station_id, start_utc, end_utc))
        if url == cfg["url"]["air_temp_url"]:
            return pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(
                        [
                            "2024-01-01 00:20:00+00:00",
                            "2024-01-01 00:00:00+00:00",
                            "2024-01-01 00:10:00+00:00",
                        ],
                        utc=True,
                    ),
                    "TT_10": [6.0, 4.0, 5.0],
                    "PP_10": [957.0, 955.0, 956.0],
                }
            )
        return pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2024-01-01 00:10:00+00:00",
                        "2024-01-01 00:20:00+00:00",
                        "2024-01-01 00:00:00+00:00",
                    ],
                    utc=True,
                ),
                "FF_10": [4.1, 4.2, 4.0],
            }
        )

    monkeypatch.setattr(weather_module, "_read_dwd_product", fake_read)

    download_dwd_temp_pressure_wind(cfg, tmp_path)

    assert calls == [
        (
            cfg["url"]["air_temp_url"],
            "00232",
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
        ),
        (
            cfg["url"]["wind_url"],
            "00232",
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 00:30:00", tz="UTC"),
        ),
    ]

    written = pd.read_csv(tmp_path / cfg["paths"]["meteo"])
    assert list(written.columns) == ["timestamp_utc", "TT_10", "PP_10", "FF_10"]
    assert written["timestamp_utc"].tolist() == [
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:10:00+00:00",
        "2024-01-01 00:20:00+00:00",
    ]
    assert written["TT_10"].tolist() == pytest.approx([4.0, 5.0, 6.0])
    assert written["PP_10"].tolist() == pytest.approx([955.0, 956.0, 957.0])
    assert written["FF_10"].tolist() == pytest.approx([4.0, 4.1, 4.2])


def test_download_dwd_temp_pressure_wind_rejects_duplicate_merge_keys_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = weather_cfg()

    def fake_read(
        url: str,
        station_id: str,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        if url == cfg["url"]["air_temp_url"]:
            return pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(
                        [
                            "2024-01-01 00:00:00+00:00",
                            "2024-01-01 00:00:00+00:00",
                        ],
                        utc=True,
                    ),
                    "TT_10": [4.0, 4.1],
                    "PP_10": [955.0, 955.1],
                }
            )
        return pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    ["2024-01-01 00:00:00+00:00"],
                    utc=True,
                ),
                "FF_10": [4.0],
            }
        )

    monkeypatch.setattr(weather_module, "_read_dwd_product", fake_read)

    with pytest.raises(pd.errors.MergeError):
        download_dwd_temp_pressure_wind(cfg, tmp_path)

    assert not (tmp_path / cfg["paths"]["meteo"]).exists()


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
                "  air_temp_url: https://example.test/temp.zip",
                "  wind_url: https://example.test/wind.zip",
                "station:",
                "  id: '00232'",
                "time:",
                "  start_utc: '2024-01-01 00:00:00'",
                "  end_utc: '2024-01-01 00:30:00'",
                "paths:",
                "  meteo: outputs/meteo.csv",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_download(cfg: dict, repo_root: Path):
        calls.append((cfg, repo_root))

    monkeypatch.setattr(weather_module, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(
        weather_module,
        "download_dwd_temp_pressure_wind",
        fake_download,
    )

    weather_module.main()

    assert len(calls) == 1
    cfg, repo_root = calls[0]
    assert repo_root == tmp_path
    assert cfg["url"]["air_temp_url"] == "https://example.test/temp.zip"
    assert cfg["url"]["wind_url"] == "https://example.test/wind.zip"
    assert cfg["paths"]["meteo"] == "outputs/meteo.csv"
