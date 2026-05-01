from pathlib import Path

import pandas as pd
import yaml


MARKET_TIMEZONE = "Europe/Berlin"


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with 'configs' folder.")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config_firm.yaml").read_text())

    raw_path = repo_root / cfg["paths"]["raw"]
    output_path = repo_root / cfg["paths"]["costs"]

    interval_minutes = cfg["time"]["interval_minutes"]
    interval_hours = interval_minutes / 60.0

    energy_price = cfg["tariff"]["energy_price_eur_per_kwh"]
    demand_charge = cfg["tariff"]["demand_charge_eur_per_kw_year"]

    load = pd.read_csv(raw_path, sep=";", skiprows=1)
    load = load.dropna(axis=1, how="all")

    raw_ts = load["Time stamp"].astype("string").str.strip()
    ts_clean = raw_ts.str.replace(r"\s[ab]$", "", regex=True)

    load["timestamp_local"] = pd.to_datetime(
        ts_clean,
        format="%d.%m.%Y %H:%M:%S",
    )

    lg_cols = [c for c in load.columns if c.startswith("LG ")]

    load[lg_cols] = load[lg_cols].apply(pd.to_numeric, errors="raise")
    load = load.dropna(subset=lg_cols, how="all").copy()

    first_utc = (
        pd.Timestamp(load["timestamp_local"].iloc[0])
        .tz_localize(MARKET_TIMEZONE)
        .tz_convert("UTC")
    )

    load["timestamp_utc"] = pd.date_range(
        start=first_utc,
        periods=len(load),
        freq=f"{interval_minutes}min",
        tz="UTC",
    )

    annual_energy_kwh = load[lg_cols].sum() * interval_hours
    load["month"] = load["timestamp_local"].dt.month
    monthly_peaks = load.groupby("month")[lg_cols].max()
    annual_peak_kw = monthly_peaks.mean()

    result = pd.DataFrame({
        "profile_id": lg_cols,
        "annual_energy_kwh": annual_energy_kwh.values,
        "annual_peak_kw": annual_peak_kw.values,
    })

    result["energy_cost_eur"] = result["annual_energy_kwh"] * energy_price
    result["peak_cost_eur"] = result["annual_peak_kw"] * demand_charge
    result["total_cost_eur"] = result["energy_cost_eur"] + result["peak_cost_eur"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(result)
    print(f"\nSaved costs to: {output_path}")


if __name__ == "__main__":
    main()