from pathlib import Path

import pandas as pd
import pvlib
import yaml


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root with 'data' folder.")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    cfg = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())

    meteo_path = repo_root / cfg["paths"]["meteo"]
    poa_path = repo_root / cfg["paths"]["poa"]
    eff_irr_path = repo_root / cfg["paths"]["effective_irradiance"]

    meteo = pd.read_csv(
        meteo_path,
        usecols=["timestamp_utc", "TT_10", "FF_10"],
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc").reset_index(drop=True)

    poa = pd.read_csv(
        poa_path,
        usecols=["timestamp_utc", "poa_global"],
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc").reset_index(drop=True)

    eff_irr = pd.read_csv(
        eff_irr_path,
        usecols=["timestamp_utc", "effective_irradiance"],
        parse_dates=["timestamp_utc"],
    ).sort_values("timestamp_utc").reset_index(drop=True)

    df = poa.merge(meteo, on="timestamp_utc", how="left", validate="one_to_one")
    df = df.merge(eff_irr, on="timestamp_utc", how="left", validate="one_to_one")

    df["t_module_faiman_c"] = pvlib.temperature.faiman(
        poa_global=df["poa_global"].clip(lower=0),
        temp_air=df["TT_10"],
        wind_speed=df["FF_10"],
        u0=20.0,  # ANNAHME!!! (pvlib-Docs)
        u1=5,     # ANNAHME!!! (pvlib-Docs)
    )

    pdc0_total = cfg["pv"]["module_pdc0"] * cfg["pv"]["module_count"]
    gamma_pdc = cfg["pv"]["gamma_pdc"]

    df["p_dc_gross_w"] = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df["effective_irradiance"].clip(lower=0),
        temp_cell=df["t_module_faiman_c"],
        pdc0=pdc0_total,
        gamma_pdc=gamma_pdc,
    )

    annual_age_loss_pct = cfg["losses"]["annual_age_loss_pct"]
    years_since_start = (
        (df["timestamp_utc"] - df["timestamp_utc"].iloc[0]).dt.total_seconds()
        / (365.25 * 24 * 3600)
    )
    df["age_loss_pct"] = annual_age_loss_pct * years_since_start

    loss_pct = pvlib.pvsystem.pvwatts_losses(
        soiling=2,
        shading=3,
        snow=0,
        mismatch=2,
        wiring=2,
        connections=0.5,
        lid=1.5,
        nameplate_rating=1,
        age=df["age_loss_pct"],
        availability=3,
    )

    loss_factor = 1 - loss_pct / 100.0
    df["p_dc_net_w"] = df["p_dc_gross_w"] * loss_factor

    pac0_total = cfg["inverter"]["pac0_each"] * cfg["inverter"]["inverter_count"]
    eta_inv_nom = cfg["inverter"]["eta_inv_nom"]
    inverter_pdc0 = pac0_total / eta_inv_nom

    df["p_ac_w"] = pvlib.inverter.pvwatts(
        pdc=df["p_dc_net_w"],
        pdc0=inverter_pdc0,
        eta_inv_nom=eta_inv_nom,
    )

    dt_hours = pd.to_timedelta(cfg["time"]["freq"]).total_seconds() / 3600.0
    df["e_net_ac_kwh"] = df["p_ac_w"] / 1000.0 * dt_hours

    output_path = repo_root / cfg["paths"]["energy"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep="NaN")
    print(df[df['poa_global'] > 1200][['timestamp_utc', 'poa_global']].to_string())

if __name__ == "__main__":
    main()