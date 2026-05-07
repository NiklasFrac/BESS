# PV-Simulation

Diese README beschreibt den fertigen Ablauf in `pv_sim/`. Die Eingangsdaten
kommen aus `download/` und `configs/config.yaml`; orchestriert wird alles in
`pv_sim/runner.py`.

## Chronologischer Ablauf

1. `true_pos.py` berechnet die geometrische Sonnenposition fuer den Standort.
   Der Rechenzeitpunkt liegt in der Intervallmitte, der Output-Zeitstempel am
   Intervallende:

   ```text
   t_ref = t_start + Delta t / 2
   t_out = t_start + Delta t
   ```

   Output: `data/pv/actual/true_sp_10min.csv` mit Zenit, Elevation und Azimut.

2. `seen_pos.py` korrigiert die Sonnenposition mit Temperatur und Luftdruck zur
   scheinbaren Sonnenposition:

   ```text
   T_mid = (T_i + T_(i-1)) / 2
   p_Pa = PP_10_mid * 100
   ```

   Output: `data/pv/actual/apparent_sp.csv` mit apparent zenith/elevation/azimuth
   und Refraktionskorrektur.

3. `compute_dni.py` verbindet DWD-Solardaten mit der Sonnenposition. Bei
   `solar_unit: jcm2` wird auf `W/m2` normiert:

   ```text
   factor = 10000 / Delta t_s
   GHI = GS_10 * factor
   DHI = DS_10 * factor
   DNI approx (GHI - DHI) / cos(theta_z)
   ```

   Output: `data/pv/actual/dni.csv` mit `ghi_wm2`, `dhi_wm2`, `dni_wm2`.

4. `compute_poa.py` interpoliert die Horizonthoehe auf den Sonnenazimut. Wenn
   die Sonne hinter dem Horizont liegt, wird die Direktstrahlung entfernt:

   ```text
   shaded = apparent_elevation <= horizon_height(azimuth)
   DNI_shaded = 0, wenn shaded, sonst DNI
   POA_global = POA_direct + POA_diffuse
   ```

   Output: `data/pv/actual/poa.csv` mit globaler, direkter, diffuser, Sky- und
   Boden-Komponente.

   ![Horizon Plot](../data/visualisation/horizon_plot.png)

5. `compute_effective_irradiance.py` berechnet den Einfallswinkel auf das Modul
   und reduziert den Direktanteil ueber den IAM-Faktor:

   ```text
   cos(AOI) = cos(theta_z) * cos(beta)
            + sin(theta_z) * sin(beta) * cos(gamma_s - gamma_p)

   effective_irradiance = POA_direct * IAM(AOI) + POA_diffuse
   ```

   Output: `data/pv/actual/effective_irradiance.csv`.

6. `modul_sim.py` berechnet die PV-Leistung und Intervallenergie:

   ```text
   T_module = T_air + POA_global / (u0 + u1 * wind_speed)

   P_dc_gross = P_dc0_total * effective_irradiance / 1000
                * (1 + gamma_pdc * (T_module - 25))

   age_loss_pct = annual_age_loss_pct * years_since_start
   P_dc_net = P_dc_gross * (1 - loss_pct / 100)
   E_net_ac_kwh = P_ac_w / 1000 * Delta t_h
   ```

   Output: `data/pv/actual/energy_curve.csv`.

7. `visualization/energy_prod_visual.py` erzeugt den Tagesenergie-Plot mit
   14-Tage-Mittelwert. `visualization/horizon_visual.py` erzeugt den
   Horizon-Plot.

## Finaler Output

Die wichtigste Ergebnisdatei ist `data/pv/actual/energy_curve.csv`. Sie enthaelt
pro 10-Minuten-Zeitpunkt `poa_global`, `effective_irradiance`,
`t_module_faiman_c`, `p_dc_gross_w`, `p_dc_net_w`, `p_ac_w` und
`e_net_ac_kwh`.

![Energy Plot](../data/visualisation/energy_plot.png)
