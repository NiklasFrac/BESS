import logging
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
import yaml


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Repo-Root nicht gefunden. Erwartet ein Verzeichnis mit 'data'-Ordner."
    )


def _find_data_member(zip_file: ZipFile) -> str:
    candidates = [
        name
        for name in zip_file.namelist()
        if name.lower().endswith(".txt")
        and "metadaten" not in name.lower()
        and "beschreibung" not in name.lower()
    ]
    if not candidates:
        raise FileNotFoundError("Keine eigentliche Datendatei im ZIP gefunden.")
    if len(candidates) > 1:
        raise ValueError(f"Mehrdeutige TXT-Dateien im ZIP: {candidates}")
    return candidates[0]


def download_dwd_10min_solar(cfg: dict, repo_root: Path) -> None:
    log = logging.getLogger(__name__)

    url         = cfg["url"]["solar"]
    station_id  = cfg["station"]["id"]
    start_utc   = pd.Timestamp(cfg["time"]["start_utc"], tz="UTC")
    end_utc     = pd.Timestamp(cfg["time"]["end_utc"], tz="UTC")
    output_path = repo_root / cfg["paths"]["solar"]

    log.info("Lade DWD-Solarprodukt: %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as zf:
        member = _find_data_member(zf)
        with zf.open(member) as f:
            df = pd.read_csv(f, sep=";", na_values=[-999, "-999"], low_memory=False)

    df.columns = df.columns.str.strip()

    if "MESS_DATUM" not in df.columns:
        raise KeyError("Spalte 'MESS_DATUM' nicht gefunden.")

    if "STATIONS_ID" not in df.columns:
        raise KeyError("Spalte 'STATIONS_ID' nicht gefunden.")
    ids = pd.to_numeric(df["STATIONS_ID"], errors="coerce").dropna().astype(int).unique()
    if len(ids) != 1 or ids[0] != int(station_id):
        raise ValueError(f"Unerwartete STATIONS_ID. Erwartet: {station_id}, gefunden: {ids.tolist()}")

    df["timestamp_utc"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str).str.strip(),
        format="%Y%m%d%H%M",
        errors="raise",
    ).dt.tz_localize("UTC")

    mask = (df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] < end_utc)
    df = df.loc[mask].copy()
    log.info("Geladene Zeilen nach Zeitfilter: %d", len(df))

    preferred_order = ["timestamp_utc", "STATIONS_ID", "QN", "GS_10", "DS_10", "SD_10", "LS_10"]
    cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[cols].sort_values("timestamp_utc").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Gespeichert nach: %s  (%d Zeilen)", output_path.resolve(), len(df))


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    log_cfg = cfg["logging"]
    logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"], datefmt=log_cfg["datefmt"])

    logging.getLogger(__name__).info("Starte DWD-Solar-Download (Station %s)", cfg["station"]["id"])
    download_dwd_10min_solar(cfg, repo_root)


if __name__ == "__main__":
    main()