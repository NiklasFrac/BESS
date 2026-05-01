from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib

def get_same_day_last_week(
    df: pd.DataFrame,
    target_day: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    target_day = pd.Timestamp(target_day).date()
    baseline_day = target_day - pd.Timedelta(days=7)

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    return df[df[timestamp_col].dt.date == baseline_day].copy()

def find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "configs" / "config.yaml").is_file() and (p / "data").is_dir():
            return p
    raise FileNotFoundError("Repo root not found.")


def main() -> None:
    repo = find_repo_root(Path(__file__).resolve().parent)
    config = yaml.safe_load((repo / "configs" / "config.yaml").read_text(encoding="utf-8"))

    data = pd.read_csv(
        repo / config["paths"]["test_set"],
        sep=";",
        skiprows=1,
    ).dropna(axis=1, how="all")

    data["timestamp"] = pd.to_datetime(
        data["Time stamp"].astype(str).str.strip().str.replace(r"\s+[ab]$", "", regex=True),
        format="%d.%m.%Y %H:%M:%S",
        errors="raise",
    )

    baseline = get_same_day_last_week(
        df=data,
        target_day="2016-01-08",
        timestamp_col="timestamp",
    )
    print(baseline)



if __name__ == "__main__":
    main()