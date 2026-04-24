import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

JCM2_10MIN_TO_WM2 = 10_000.0 / 600.0
log = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find <repo-root>. Expected a parent directory containing a 'data' folder."
    )


def _read_utc(path: Path, ts_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text(encoding="utf-8"))


    ts_col   = cfg["dni"]["timestamp_col"]
    missing  = float(cfg["dni"]["dwd_missing_value"])
    out_path = repo_root / cfg["paths"]["dni"]

    solar  = _read_utc(repo_root / cfg["paths"]["solar"], ts_col)[[ts_col, "GS_10", "DS_10"]]
    solpos = _read_utc(repo_root / cfg["paths"]["true_sun_position"], ts_col)[[ts_col, "solar_zenith_deg"]]

    df = (
        solpos.merge(solar, on=ts_col, how="left", validate="one_to_one")
            .sort_values(ts_col)
            .reset_index(drop=True)
    )
    log.info("Merge: %d Zeilen", len(df))

    df["ghi_wm2"] = pd.to_numeric(df["GS_10"], errors="coerce").replace(missing, np.nan) * JCM2_10MIN_TO_WM2
    df["dhi_wm2"] = pd.to_numeric(df["DS_10"], errors="coerce").replace(missing, np.nan) * JCM2_10MIN_TO_WM2

    zenith = pd.to_numeric(df["solar_zenith_deg"], errors="coerce")
    df["dni_wm2"] = pvlib.irradiance.dni(ghi=df["ghi_wm2"], dhi=df["dhi_wm2"], zenith=zenith)


    valid = df["ghi_wm2"].notna() & df["dhi_wm2"].notna() & zenith.notna()
    log.info(
        "QA: valid=%d  neg_BHI=%d  dni_NaN=%d",
        valid.sum(),
        (df["ghi_wm2"] - df["dhi_wm2"] < 0).sum(),
        df["dni_wm2"].isna().sum(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[[ts_col, "ghi_wm2", "dhi_wm2", "dni_wm2"]].to_csv(out_path, index=False)
    log.info("Gespeichert: %s", cfg["paths"]["dni"])


if __name__ == "__main__":
    main()