# Update-Verlauf

Hinweis: Zeiten sind ca.-Angaben, wenn nicht explizit bekannt. Ältere Punkte ggf. mit „Datum & Uhrzeit unbekannt“.

## 2025-10-01 ca. 10:30
- Einführung der NiU-Konvention: `F_NiU_*` (Anzeige/Features, nicht im Training) und `L_NiU_*` (Diagnose-Labels, nicht im Training).
- Training/Evaluierung filtern automatisch alle `F_NiU_`/`L_NiU_` (und legacy `nF_`) aus der Featurematrix.
- Neues Trainingsziel bleibt `LABLE_HalfYear_Target` (halbjährlich konstanter Zielwert; abgeleitet aus `L_NiU_WBZ_BlockMinAbs`).
- README aktualisiert; Helper-Skript `scripts/eval_all_halfyear.py` hinzugefügt.

## 2025-10-01 ca. 09:45
- Halbjahres-Regel implementiert: `LABLE_HalfYear_Target` aus dem maximalen Wert von `_LABLE_WBZ_BlockMinAbs` (später `L_NiU_WBZ_BlockMinAbs`) je 6‑Monats-Fenster.
- Sicherstellung, dass sich das Label bei Bedarf in den Dispo‑Zeitraum fortsetzt (nur für Anzeige/Eval; Training nur historisch).

## 2025-09-30 13:40 (aus Git)
- Commit 2405a7a „New Lable LABLE_HalfYear_Target“: Einführung des halbjährlichen Ziel‑Labels; Anpassungen an `data_pipeline.py`, `train_model.py`, `evaluate_model.py`, Defaults in `scripts/train.py`/`evaluate.py`, sowie ein Evaluations‑Helper.

## 2025-09-30 08:56 (aus Git)
- Commit 48731b0 „Update“: Diverse Ergebnisexports und Plot‑Artefakte für einzelne Teile/Modelle hinzugefügt (Vergleich Full/Dispo‑Zeitfenster, Historie‑Plots für ALL/gb/6–7).

## 2025-09-24 13:35 (aus Git)
- Commit cdfebcc „Update“: Änderungen an `data_pipeline.py`, `train_model.py`, `evaluate_model.py`, sowie den Scripten (`train.py`, `evaluate.py`) und Dokumentation.

## 2025-09-18 (ungefähr; aus Artefaktzeiten/Git abgeleitet)
- ALL‑Modelle trainiert und Evaluierungen generiert (plots/ und später New_Test_Plots/), erste Metriken und Feature‑Importances erzeugt.

## Datum & Uhrzeit unbekannt
- Initiale Pipeline erstellt (CSV‑Parsing, Spaltenmapping via `Spaltenbedeutung.xlsx`, Zeitreihenaggregation, Anzeige‑Spalten), Skripte `build_features.py`, `train.py`, `evaluate.py` angelegt.
- Legacy‑Labels (z. B. `_LABLE_StockOut_MinAdd`) als Diagnose hinzugefügt; Rolling‑Features (`DemandMean_*`, `DemandMax_*`).

## 2025-10-23 10:36
- Doku ergänzt (Backfill mit Alter_SiBe; ALL-Training global sortiert; Tagesgrenzen-Splits; Gewichtungen per Schema/Faktor; Evaluation mit Label-Overlay; Plot-Stabilisierung; Feature-Toggle-GUI).
- Training interaktiv erweitert: --weight_scheme (none|blockmin|flag), --weight_factor.
- Evaluate stabilisiert (Matplotlib Agg, Figure-Close), Label-Overlay hinzugefügt.
- SiBe-Backfill korrigiert: Alter_SiBe rückwärts bis frühestes Lagerbew-Datum, strikt vor erstem Änderungsdatum.

## 2025-10-23 10:48
- Train: Timeseries-Scope interaktiv (global|local). Lokal = pro Teil separat, global = ein Modell über alle Teile. ID-Handling: vorhandene ID wird übernommen, sonst auto pro Zielordner. Gewichtungs-Prompts abgesichert.
