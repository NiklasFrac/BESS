import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

log = logging.getLogger(__name__)


def _read_utc(path: Path, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df

def _infer_interval_seconds(ts: pd.Series) -> float:
    diffs = ts.sort_values().diff().dropna().dt.total_seconds()
    if diffs.empty:
        raise ValueError("Cannot infer interval from fewer than two timestamps.")
    return float(diffs.mode().iloc[0])

def compute_dni(
    solar_path: Path,
    sun_position_path: Path,
    out_path: Path,
    ts_col: str,
    missing: float,
    solar_unit: str,
) -> None:
    solar = _read_utc(solar_path, ts_col)[[ts_col, "GS_10", "DS_10"]]
    solpos = _read_utc(sun_position_path, ts_col)[[ts_col, "solar_zenith_deg"]]

    
    df = (
        solpos.merge(solar, on=ts_col, how="left", validate="one_to_one")
        .sort_values(ts_col)
        .reset_index(drop=True)
    )

    log.info("Merge: %d Zeilen", len(df))

    if solar_unit == "jcm2":
        factor = 10_000.0 / _infer_interval_seconds(solar[ts_col])
    elif solar_unit == "wm2":
        factor = 1.0
    else:
        raise ValueError("solar_unit must be 'jcm2' or 'wm2'")

    df["ghi_wm2"] = (
        pd.to_numeric(df["GS_10"], errors="coerce")
        .replace(missing, np.nan)
        * factor
    )

    df["dhi_wm2"] = (
        pd.to_numeric(df["DS_10"], errors="coerce")
        .replace(missing, np.nan)
        * factor
    )


    zenith = pd.to_numeric(df["solar_zenith_deg"], errors="coerce")

    df["dni_wm2"] = pvlib.irradiance.dni(
        ghi=df["ghi_wm2"],
        dhi=df["dhi_wm2"],
        zenith=zenith,
    )

    valid = df["ghi_wm2"].notna() & df["dhi_wm2"].notna() & zenith.notna()

    log.info(
        "QA: valid=%d  neg_BHI=%d  dni_NaN=%d",
        valid.sum(),
        (df["ghi_wm2"] - df["dhi_wm2"] < 0).sum(),
        df["dni_wm2"].isna().sum(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[[ts_col, "ghi_wm2", "dhi_wm2", "dni_wm2"]].to_csv(out_path, index=False)

    log.info("Gespeichert: %s", out_path)