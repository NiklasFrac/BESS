import math
import pandas as pd


def resample_power_temp(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    temp_col: str,
    source_min: int,
    target_min: int,
    timestamp_is_interval_end: bool = True,
) -> pd.DataFrame:
    base_min = math.gcd(source_min, target_min)
    n = source_min // base_min

    x = df[[timestamp_col, power_col, temp_col]].copy()
    x[timestamp_col] = pd.to_datetime(x[timestamp_col], utc=True)
    x[power_col] = x[power_col].astype(float)
    x[temp_col] = x[temp_col].astype(float)

    if timestamp_is_interval_end:
        x[timestamp_col] -= pd.Timedelta(minutes=source_min)

    parts = []
    for k in range(n):
        y = x.copy()
        y[timestamp_col] += pd.Timedelta(minutes=k * base_min)
        parts.append(y)

    y = (
        pd.concat(parts)
        .sort_values(timestamp_col)
        .set_index(timestamp_col)
        .resample(f"{target_min}min", closed="left", label="left")
        .agg({power_col: "mean", temp_col: "mean"})
        .dropna(subset=[power_col, temp_col])
        .reset_index()
    )

    if timestamp_is_interval_end:
        y[timestamp_col] += pd.Timedelta(minutes=target_min)

    return y
