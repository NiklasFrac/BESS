# EMS - Projektübersicht

Aktuell dokumentiert sind nur die fertigeren Projektteile `download/` und `pv_sim/`.
Der komplette Lauf startet über `python proof_of_concept.py`: zuerst werden die
Eingangsdaten geladen, danach wird daraus die PV-Erzeugung simuliert.

## 1. Downloader (`download/`)

Die Downloader lesen `configs/config.yaml`, laden die externen Daten für die
Station Augsburg Bayern (`00232`) und schreiben sie unter `data/pv/...`.

| Datei | Zweck | Output |
| --- | --- | --- |
| `download/run_downloads.py` | Orchestriert alle Downloads in der richtigen Reihenfolge. | keine eigene Datei |
| `download/meta_data.py` | Lädt DWD-Stationsmetadaten mit ID, Name, Breite, Länge und Höhe; Basis für Standort und PVGIS-Horizont. | `data/pv/general/metadata_stations.csv` |
| `download/weather.py` | Lädt DWD-10-Minuten-Wetterdaten: Lufttemperatur `TT_10`, Luftdruck `PP_10`, Wind `FF_10`; nötig für scheinbare Sonnenposition und Modultemperatur. | `data/pv/actual/dwd_meteo.csv` |
| `download/solar.py` | Lädt DWD-10-Minuten-Solardaten: Globalstrahlung `GS_10` und diffuse Strahlung `DS_10`; Input für DNI und Einstrahlung auf die Modulebene. | `data/pv/actual/dwd_solar_data.csv` |
| `download/horizon.py` | Lädt aus PVGIS das lokale Horizontprofil auf Basis der Stationskoordinaten; später zur Abschattung der Direktstrahlung. | `data/pv/general/pvgis_horizon_augsburg.csv` |

## 2. PV-Simulation (`pv_sim/`)

`pv_sim/runner.py` führt die Pipeline chronologisch aus. Die wichtigsten
Parameter kommen aus `configs/config.yaml`: Zeitraum, 10-Minuten-Frequenz,
Modulausrichtung, Modulanzahl, Wechselrichterdaten und Verlustannahmen.

### Ablauf

1. `true_pos.py` berechnet die geometrische Sonnenposition für den Standort.
   Der Rechenzeitpunkt liegt in der Intervallmitte, der Output-Zeitstempel am
   Intervallende:

   ```text
   t_ref = t_start + Delta t / 2
   t_out = t_start + Delta t
   ```

   Output: `data/pv/actual/true_sp_10min.csv` mit Zenit, Elevation und Azimut.

2. `seen_pos.py` korrigiert die Sonnenposition mit Temperatur und Luftdruck zur
   scheinbaren Sonnenposition. Dazu werden die Meteowerte auf Intervallmitten
   gemittelt:

   ```text
   T_mid = (T_i + T_(i-1)) / 2
   p_Pa = PP_10_mid * 100
   ```

   Output: `data/pv/actual/apparent_sp.csv` mit apparent zenith/elevation/azimuth
   und Refraktionskorrektur.

3. `compute_dni.py` verbindet DWD-Solardaten mit der Sonnenposition. Falls die
   DWD-Werte als `jcm2` konfiguriert sind, werden sie in `W/m2` umgerechnet:

   ```text
   factor = 10000 / Delta t_s
   GHI = GS_10 * factor
   DHI = DS_10 * factor
   DNI approx (GHI - DHI) / cos(theta_z)
   ```

   Output: `data/pv/actual/dni.csv` mit `ghi_wm2`, `dhi_wm2`, `dni_wm2`.

4. `compute_poa.py` interpoliert zuerst die Horizonthöhe auf den aktuellen
   Sonnenazimut. Liegt die Sonne unter dem Horizontprofil, wird die
   Direktstrahlung auf 0 gesetzt:

   ```text
   shaded = apparent_elevation <= horizon_height(azimuth)
   DNI_shaded = 0, wenn shaded, sonst DNI
   ```

   Danach berechnet pvlib mit `perez-driesse` die Einstrahlung auf die geneigte
   Modulebene:

   ```text
   POA_global = POA_direct + POA_diffuse
   ```

   Output: `data/pv/actual/poa.csv` mit globaler, direkter, diffuser, Sky- und
   Boden-Komponente.

   ![Horizon Plot](data/visualisation/horizon_plot.png)

5. `compute_effective_irradiance.py` berechnet den Einfallswinkel auf das Modul
   und reduziert den Direktanteil über den IAM-Faktor:

   ```text
   cos(AOI) = cos(theta_z) * cos(beta)
            + sin(theta_z) * sin(beta) * cos(gamma_s - gamma_p)

   effective_irradiance = POA_direct * IAM(AOI) + POA_diffuse
   ```

   Output: `data/pv/actual/effective_irradiance.csv`.

6. `modul_sim.py` berechnet aus Einstrahlung, Temperatur, Verlusten und
   Wechselrichtermodell die elektrische Energie. Zentrale Gleichungen:

   ```text
   T_module = T_air + POA_global / (u0 + u1 * wind_speed)

   P_dc_gross = P_dc0_total * effective_irradiance / 1000
                * (1 + gamma_pdc * (T_module - 25))

   age_loss_pct = annual_age_loss_pct * years_since_start
   P_dc_net = P_dc_gross * (1 - loss_pct / 100)
   E_net_ac_kwh = P_ac_w / 1000 * Delta t_h
   ```

   Output: `data/pv/actual/energy_curve.csv`.

7. `visualization/energy_prod_visual.py` aggregiert die Energiekurve auf
   Tageswerte und plottet zusätzlich einen 14-Tage-Mittelwert.
   `visualization/horizon_visual.py` visualisiert das PVGIS-Horizontprofil.

### Finaler Output

Die wichtigste Ergebnisdatei ist `data/pv/actual/energy_curve.csv`. Sie enthält
pro 10-Minuten-Zeitpunkt unter anderem:

- `poa_global`: Einstrahlung auf Modulebene
- `effective_irradiance`: für das Modul wirksame Einstrahlung
- `t_module_faiman_c`: berechnete Modultemperatur
- `p_dc_gross_w` / `p_dc_net_w`: DC-Leistung vor und nach Verlusten
- `p_ac_w`: AC-Leistung nach Wechselrichter
- `e_net_ac_kwh`: erzeugte AC-Energie im Intervall

![Energy Plot](data/visualisation/energy_plot.png)
