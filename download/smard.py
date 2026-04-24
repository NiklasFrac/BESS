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


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def _fetch_json(session: requests.Session, url: str) -> dict:
    response = session.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.json()


def _index_url(base_url: str, filter_id: int, region: str, resolution: str) -> str:
    return f"{base_url}/{filter_id}/{region}/index_{resolution}.json"


def _series_url(base_url: str, filter_id: int, region: str, resolution: str, anchor_ms: int) -> str:
    return f"{base_url}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{anchor_ms}.json"


def download_smard_day_ahead_prices(cfg: dict, repo_root: Path) -> None:
    log = logging.getLogger(__name__)

    filter_id   = cfg["smard"]["filter_id"]
    region      = cfg["smard"]["region"]
    resolution  = cfg["smard"]["resolution"]
    base_url    = cfg["smard"]["base_url"]
    start_utc   = pd.Timestamp(cfg["time"]["start_utc"], tz="UTC")
    end_utc     = pd.Timestamp(cfg["time"]["end_utc"], tz="UTC")
    output_path = repo_root / cfg["paths"]["smard"]

    with requests.Session() as session:
        index_payload = _fetch_json(session, _index_url(base_url, filter_id, region, resolution))

        anchor_timestamps = index_payload.get("timestamps")
        if not anchor_timestamps:
            raise ValueError("SMARD index enthält keine 'timestamps'.")

        frames: list[pd.DataFrame] = []

        start_ms = int(start_utc.timestamp() * 1000)
        end_ms   = int(end_utc.timestamp() * 1000)
        week_ms  = 7 * 24 * 3600 * 1000  # SMARD-Chunks sind wöchentlich
        relevant = [a for a in anchor_timestamps if a < end_ms and a + week_ms > start_ms]
        for anchor in relevant:
            payload = _fetch_json(session, _series_url(base_url, filter_id, region, resolution, anchor))
            
            series = payload.get("series")

            if not series:
                continue

            rows = [(item[0], item[1]) for item in series if isinstance(item, list) and len(item) >= 2]

            if not rows:
                continue

            frames.append(pd.DataFrame(rows, columns=["timestamp_ms", "price_eur_per_mwh"]))

    if not frames:
        raise ValueError("Keine Zeitreihendaten von SMARD geladen.")

    df = pd.concat(frames, ignore_index=True)

    df["start_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    df = df.loc[(df["start_utc"] >= start_utc) & (df["start_utc"] < end_utc)].copy()
    df = df.drop_duplicates(subset=["start_utc"]).sort_values("start_utc").reset_index(drop=True)

    
    n_null = df["price_eur_per_mwh"].isna().sum()
    if n_null > 0:
        null_hours = df.loc[df["price_eur_per_mwh"].isna(), "start_utc"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist()
        log.warning("%d fehlende Preiswerte - Stunden: %s", n_null, null_hours)
        

    df["end_utc"]       = df["start_utc"] + pd.Timedelta(hours=1)
    df["start_berlin"]  = df["start_utc"].dt.tz_convert("Europe/Berlin")
    df["end_berlin"]    = df["end_utc"].dt.tz_convert("Europe/Berlin")
    df["market"]        = "Day-ahead auction"
    df["bidding_zone"]  = "Germany/Luxembourg"
    df["source"]        = "SMARD"

    df = df[[
        "start_utc", "end_utc", "start_berlin", "end_berlin",
        "price_eur_per_mwh", "market", "bidding_zone", "source",
    ]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Gespeichert nach: %s  (%d Zeilen)", output_path.resolve(), len(df))


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    log_cfg = cfg["logging"]
    logging.basicConfig(level=log_cfg["level"], format=log_cfg["format"], datefmt=log_cfg["datefmt"])

    logging.getLogger(__name__).info("Starte SMARD-Download")
    download_smard_day_ahead_prices(cfg, repo_root)


if __name__ == "__main__":
    main()