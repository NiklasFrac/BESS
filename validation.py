import json
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
import yaml


HTML_CACHE = {}
METEO_COLS = ["TT_10", "PP_10", "FF_10"]
log = logging.getLogger(__name__)


def _base_url(url: str) -> str:
    return url.rsplit("/", 1)[0] + "/"


def _zip_member(zf: ZipFile) -> str:
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(".txt") and "metadaten" not in low and "beschreibung" not in low:
            return name
    raise FileNotFoundError("No data txt in zip.")


def _dwd_url(base: str, prefix: str, station_id: str, start, end) -> str:
    if base not in HTML_CACHE:
        HTML_CACHE[base] = requests.get(base, timeout=120).text
    station_id = str(station_id).zfill(5)
    pattern = re.compile(
        rf"{re.escape(prefix)}_{station_id}_(\d{{8}})_(\d{{8}})_hist\.zip"
    )
    hits = []
    for href in re.findall(r'href="([^"]+\.zip)"', HTML_CACHE[base]):
        name = href.rsplit("/", 1)[-1]
        match = pattern.fullmatch(name)
        if not match:
            continue
        file_start = pd.Timestamp(match.group(1), tz="UTC")
        file_end = pd.Timestamp(match.group(2), tz="UTC") + pd.Timedelta(days=1)
        if file_start <= start and file_end >= end:
            hits.append((file_end - file_start, name))
    if not hits:
        raise FileNotFoundError(f"No DWD archive for {station_id} in {base}")
    return base + sorted(hits)[0][1]


def _read_dwd_col(url: str, station_id: str, col: str, start, end) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as zf:
        with zf.open(_zip_member(zf)) as f:
            df = pd.read_csv(
                f, sep=";", na_values=[-999, "-999", -999.0, "-999.0"], low_memory=False
            )
    df.columns = df.columns.str.strip()
    ids = pd.to_numeric(df["STATIONS_ID"], errors="coerce").dropna().astype(int).unique()
    if len(ids) != 1 or ids[0] != int(station_id):
        raise ValueError(f"Unexpected station ids: {ids.tolist()}")
    df["timestamp_utc"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str).str.strip(), format="%Y%m%d%H%M", errors="raise"
    ).dt.tz_localize("UTC")
    df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999, pd.NA)
    df = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] < end)]
    return df[["timestamp_utc", col]].set_index("timestamp_utc").sort_index()


def _gaps(series: pd.Series) -> list[pd.DatetimeIndex]:
    gaps, current = [], []
    for ts, missing in series.isna().items():
        if missing:
            current.append(ts)
        elif current:
            gaps.append(pd.DatetimeIndex(current))
            current = []
    if current:
        gaps.append(pd.DatetimeIndex(current))
    return gaps


def _near_stations(cfg: dict, repo_root: Path) -> list[dict]:
    max_distance = float(cfg["validation"]["max_distance"])
    station_id = str(cfg["station"]["id"]).zfill(5)
    metadata = pd.read_csv(repo_root / cfg["paths"]["metadata"], dtype={"station_id": str})
    for col in ["latitude", "longitude", "height_m_amsl"]:
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
    target = metadata.loc[metadata["station_id"].str.zfill(5) == station_id].iloc[0]
    out = []
    for row in metadata.dropna().itertuples(index=False):
        sid = str(row.station_id).zfill(5)
        if sid == station_id:
            continue
        lat_km = (row.latitude - target.latitude) * 111.32
        lon_km = (row.longitude - target.longitude) * 111.32 * math.cos(math.radians(target.latitude))
        height_km = (row.height_m_amsl - target.height_m_amsl) / 1000.0
        distance = math.sqrt(lat_km**2 + lon_km**2 + height_km**2)
        if distance <= max_distance:
            out.append(
                {
                    "station_id": sid,
                    "station_name": row.station_name,
                    "distance": distance,
                }
            )
    return sorted(out, key=lambda x: x["distance"])


def _validate_meteo(cfg: dict, repo_root: Path) -> tuple[dict, list[str]]:
    raw = repo_root / cfg["paths"]["meteo_raw"]
    out = repo_root / cfg["paths"]["meteo"]
    freq = pd.to_timedelta(cfg["time"]["freq"])
    max_gap = pd.to_timedelta(cfg["validation"]["max_gap_length"])
    index = pd.date_range(
        pd.Timestamp(cfg["time"]["start_utc"], tz="UTC"),
        pd.Timestamp(cfg["time"]["end_utc"], tz="UTC") - freq,
        freq=freq,
    )
    df = pd.read_csv(raw, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[["timestamp_utc", *METEO_COLS]].set_index("timestamp_utc").reindex(index)
    df.index.name = "timestamp_utc"
    for col in METEO_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report = {
        "file": cfg["paths"]["meteo_raw"],
        "output": cfg["paths"]["meteo"],
        "columns": {},
    }
    errors, stations = [], _near_stations(cfg, repo_root)
    log.info("Meteo-Validierung: %s | Ersatzstationen: %d", raw, len(stations))
    if not stations:
        return report, [f"No fallback stations within {cfg['validation']['max_distance']} km."]

    for col in METEO_COLS:
        bad_stations = set()
        report["columns"][col] = {"interpolated": [], "filled_from_station": []}
        interpolated = df[col].interpolate(method="time", limit_area="inside")
        gaps = _gaps(df[col])
        log.info("%s: %d Gaps gefunden", col, len(gaps))
        for gap in gaps:
            gap_start, gap_end = gap[0], gap[-1] + freq
            if len(gap) * freq <= max_gap and interpolated.loc[gap].notna().all():
                df.loc[gap, col] = interpolated.loc[gap]
                log.info("%s: interpoliert %s bis %s (%d Werte)", col, gap[0], gap[-1], len(gap))
                report["columns"][col]["interpolated"].append(
                    {"start": str(gap[0]), "end": str(gap[-1]), "count": len(gap)}
                )
                continue

            filled = False
            key, prefix = (
                ("air_temp_url", "10minutenwerte_TU")
                if col in ["TT_10", "PP_10"]
                else ("wind_url", "10minutenwerte_wind")
            )
            for station in stations:
                if station["station_id"] in bad_stations:
                    continue
                try:
                    url = _dwd_url(_base_url(cfg["url"][key]), prefix, station["station_id"], gap_start, gap_end)
                    other = _read_dwd_col(url, station["station_id"], col, gap_start, gap_end).reindex(gap)
                except Exception:
                    bad_stations.add(station["station_id"])
                    continue
                if other[col].notna().all():
                    df.loc[gap, col] = other[col].to_numpy()
                    log.info(
                        "%s: gefuellt aus Station %s (%s bis %s, %d Werte)",
                        col,
                        station["station_id"],
                        gap[0],
                        gap[-1],
                        len(gap),
                    )
                    report["columns"][col]["filled_from_station"].append(
                        {
                            "start": str(gap[0]),
                            "end": str(gap[-1]),
                            "count": len(gap),
                            "station_id": station["station_id"],
                            "station_name": station["station_name"],
                            "distance": round(station["distance"], 3),
                        }
                    )
                    filled = True
                    break
                bad_stations.add(station["station_id"])
            if not filled:
                errors.append(f"{col}: could not fill {gap[0]} to {gap[-1]}")

    report["remaining_nan"] = {col: int(df[col].isna().sum()) for col in METEO_COLS}
    if not errors and not df[METEO_COLS].isna().any().any():
        out.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(out, index=False)
        log.info("Meteo-Validierung gespeichert: %s", out)
    return report, errors


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())
    logging.basicConfig(**cfg["logging"], force=True)
    log.info("Starte Datenvalidierung")
    raw = repo_root / cfg["paths"]["pvgis_raw"]
    out = repo_root / cfg["paths"]["pvgis"]
    report = repo_root / cfg["paths"]["validation_report"]

    df = pd.read_csv(raw)
    replaced = int(df["horizon_height_deg"].isna().sum())
    df["horizon_height_deg"] = df["horizon_height_deg"].fillna(0)
    log.info("Horizontprofil: %d NaNs in horizon_height_deg auf 0 gesetzt", replaced)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "pvgis": {
            "file": cfg["paths"]["pvgis_raw"],
            "horizon_height_deg_nan_replaced": replaced,
        }
    }
    meteo_report, errors = _validate_meteo(cfg, repo_root)
    report_data["meteo"] = meteo_report
    report.write_text(
        json.dumps(report_data, indent=2),
        encoding="utf-8",
    )
    log.info("Validierungsreport gespeichert: %s", report)
    if errors:
        raise ValueError("Meteo validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
