import logging
from pathlib import Path

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


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", content, 0, 1, "Konnte Stations-Metadaten nicht dekodieren.")


def _parse_station_table(text: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    in_table = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not in_table:
            if (
                "Stations_id" in line
                and "Stationshoehe" in line
                and "geoBreite" in line
                and "geoLaenge" in line
                and "Stationsname" in line
            ):
                in_table = True
            continue
        if set(line) == {"-"}:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append({
            "station_id":    parts[0],
            "station_name":  " ".join(parts[6:-1]),
            "latitude":      parts[4],
            "longitude":     parts[5],
            "height_m_amsl": parts[3],
        })

    if not rows:
        raise ValueError("Keine Stationszeilen in der DWD-Datei gefunden.")
    return pd.DataFrame(rows)


def download_station_metadata(cfg: dict, repo_root: Path) -> None:
    log = logging.getLogger(__name__)

    url         = cfg["url"]["metadata"]
    output_path = repo_root / cfg["paths"]["metadata"]

    log.info("Lade Stationsmetadaten von %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    text = _decode_text(response.content)
    df = _parse_station_table(text)

    df["station_id"]    = pd.to_numeric(df["station_id"],    errors="coerce")
    df["latitude"]      = pd.to_numeric(df["latitude"],      errors="coerce")
    df["longitude"]     = pd.to_numeric(df["longitude"],     errors="coerce")
    df["height_m_amsl"] = pd.to_numeric(df["height_m_amsl"], errors="coerce")

    df = df.dropna(subset=["station_id", "station_name", "latitude", "longitude", "height_m_amsl"]).copy()

    df["station_id"]   = df["station_id"].astype(int).astype(str).str.zfill(5)
    df["station_name"] = df["station_name"].str.strip()

    df = (
        df[["station_id", "station_name", "latitude", "longitude", "height_m_amsl"]]
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_id")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Gespeichert nach: %s  (%d Stationen)", output_path.resolve(), len(df))


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    log_cfg = cfg["logging"]
    logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"], datefmt=log_cfg["datefmt"])

    logging.getLogger(__name__).info("Starte Metadaten-Download")
    download_station_metadata(cfg, repo_root)


if __name__ == "__main__":
    main()