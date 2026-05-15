import logging
from pathlib import Path

import pandas as pd
import pvlib

log = logging.getLogger(__name__)


def compute_energy(
    meteo_path: Path,
    poa_path: Path,
    effective_irradiance_path: Path,
    out_path: Path,
    pv_output_path: Path,
    module_pdc0: float,
    module_count: int,
    gamma_pdc: float,
    annual_age_loss_pct: float,
    pac0_each: float,
    inverter_count: int,
    eta_inv_nom: float,
    freq: str,
) -> None:
    meteo = (
        pd.read_csv(
            meteo_path,
            usecols=["timestamp_utc", "TT_10", "FF_10"],
            parse_dates=["timestamp_utc"],
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    poa = (
        pd.read_csv(
            poa_path,
            usecols=["timestamp_utc", "poa_global"],
            parse_dates=["timestamp_utc"],
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    eff_irr = (
        pd.read_csv(
            effective_irradiance_path,
            usecols=["timestamp_utc", "effective_irradiance"],
            parse_dates=["timestamp_utc"],
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    df = poa.merge(meteo, on="timestamp_utc", how="left", validate="one_to_one")
    df = df.merge(eff_irr, on="timestamp_utc", how="left", validate="one_to_one")

    df["t_module_faiman_c"] = pvlib.temperature.faiman(
        poa_global=df["poa_global"].clip(lower=0),
        temp_air=df["TT_10"],
        wind_speed=df["FF_10"],
        u0=20.0,
        u1=5,
    )

    pdc0_total = module_pdc0 * module_count

    df["p_dc_gross_w"] = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df["effective_irradiance"].clip(lower=0),
        temp_cell=df["t_module_faiman_c"],
        pdc0=pdc0_total,
        gamma_pdc=gamma_pdc,
    )

    years_since_start = (
        df["timestamp_utc"] - df["timestamp_utc"].iloc[0]
    ).dt.total_seconds() / (365.25 * 24 * 3600)
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

    pac0_total = pac0_each * inverter_count
    inverter_pdc0 = pac0_total / eta_inv_nom

    df["p_ac_w"] = pvlib.inverter.pvwatts(
        pdc=df["p_dc_net_w"],
        pdc0=inverter_pdc0,
        eta_inv_nom=eta_inv_nom,
    )

    dt_hours = pd.to_timedelta(freq).total_seconds() / 3600.0
    df["e_net_ac_kwh"] = df["p_ac_w"] / 1000.0 * dt_hours

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, na_rep="NaN")

    pv_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(
        pv_kw=df["p_ac_w"] / 1000.0,
        ambient_temp_degC=df["TT_10"],
    )[["timestamp_utc", "pv_kw", "ambient_temp_degC"]].to_csv(
        pv_output_path,
        index=False,
        float_format="%.3f",
    )
    log.info(
        "Gespeichert: %s, %s | Zeilen: %d | AC_kWh: %.1f",
        out_path,
        pv_output_path,
        len(df),
        df["e_net_ac_kwh"].sum(),
    )
