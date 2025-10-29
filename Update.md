# Update‑Verlauf / Werdegang des ML‑Modells

Dieser Verlauf fasst die wichtigsten Schritte von der ursprünglichen Pipeline bis zum aktuellen Stand zusammen. Zeitangaben dienen der Einordnung (soweit bekannt/abschätzbar).

## 2025‑09 (frühe Phase)
- Initiale Pipeline: CSV‑Parsing (inkl. `Spaltenbedeutung.xlsx`), Tagesaggregation, Dispo‑Logik, SiBe‑Verlauf, erste Anzeige‑Spalten.
- Erste Rolling‑Features: `DemandMean_*`, `DemandMax_*`.
- Skripte: `build_features.py`, `train.py`, `evaluate.py`.

## 2025‑09‑30
- Einführung Ziel `L_HalfYear_Target`: halbjährlich konstanter Zielwert aus dem Maximum von `L_NiU_WBZ_BlockMinAbs` pro Halbjahr, leichte Faktoren (WBZ, Aktivität, Volatilität) vorsichtig berücksichtigt.
- Anpassungen in `data_pipeline.py`, `train_model.py`, `evaluate_model.py` und Skripten.

## 2025‑10‑01
- NiU‑Konventionen: `F_NiU_*` (Anzeige/Hilfs‑Features, nicht im Training), `L_NiU_*` (Diagnose‑Labels, nicht im Training). Training/Evaluierung schließen `F_NiU_`/`L_NiU_`/`nF_` automatisch aus.
- README/Docs erweitert, Evaluations‑Helper ergänzt.

## 2025‑10‑23 (Stabilisierung & Bedienung)
- Doku ergänzt (Backfill, globale/regionale Splits, Gewichtungen, Label‑Overlay in Plots).
- Training interaktiv: Gewichtungs‑Schema (`none|blockmin|flag`) und Faktor per Prompt.
- Evaluate: Robustere Plot‑Erzeugung, Label‑Overlay, Encoding‑Toleranz.

## 2025‑10‑25 bis 2025‑10‑28 (aktuelle Arbeiten)

### Pipeline‑Verbesserungen
- Tägliche Reindexierung: Ereignislose Tage werden lückenlos erzeugt.
- Forward‑Fill zentraler Bestände; Demand‑Features werden für alle Tage neu berechnet.
- Essentielle Anzeige/Basis:
  - `F_NiU_EoD_Bestand` und `F_NiU_Hinterlegter SiBe` bleiben erhalten (für Visualisierung),
  - `EoD_Bestand_noSiBe` als feste Basis‑Spalte.
- Test‑Set: identisches Schema und Spaltenreihenfolge wie Features; liegt auf gleicher Ebene wie der Feature‑Ordner; dynamische Trennung je Teil am Dispo‑Start.

### Selektiver Build & GUI
- Neue GUI `scripts/build_features_gui.py`:
  - Checkboxen für Features und Labels; Suche/Scrollen verbessert.
  - Abhängigkeiten werden automatisch berechnet; nur explizit ausgewählte Spalten erscheinen in der Ausgabe (Basisspalten werden – falls nur als Abhängigkeit benötigt – unterdrückt).
  - Immer enthaltene Spalten: `F_NiU_EoD_Bestand`, `F_NiU_Hinterlegter SiBe`, `EoD_Bestand_noSiBe`.
- Interne Tests unter `GPT_Features_Test/` und `GPT_Pipeline_Test/` durchgeführt.

### Lag‑Features (neu)
- Punkt‑Lags: `Lag_EoD_Bestand_noSiBe_{7Tage,28Tage,wbzTage,2xwbzTage}`.
- Mittel‑Lags: `Lag_EoD_Bestand_noSiBe_mean_{7Tage,28Tage,wbzTage,2xwbzTage}`.
- Namensschema vereinheitlicht (`..._7Tage`, `..._28Tage`, `..._wbzTage`, `..._2xwbzTage`).

### Evaluierung – Robustheit
- Fehlende Spalte „Hinterlegter SiBe“ wird robust erkannt (Unterstrich/Leerzeichen); Evaluierung bricht nicht mehr mit KeyError ab.
- Plots/Exports bleiben erhalten; Metriken: MAE, RMSE, R², MAPE (mit Vorsicht bei Ziel=0).

### Training – Bedienung & Monitoring
- Optionaler paralleler Progress‑Balken für `gb` (sklearn) via `--progress` oder interaktiven Prompt.
- Interaktive Prompts für Hyperparameter, CV‑Splits und Gewichtungen überarbeitet.

## Offene To‑Dos / Ideen
- Progress‑Callbacks für `xgb`/`lgbm` analog zu `gb` integrieren.
- Erweiterte Feature‑Registry (z. B. EWM/Quantile‑Fenster) in der GUI auswählbar machen.
- WBZ‑basierte Lags ggf. auf Zeilenebene variabel (derzeit teil‑fixe WBZ je Serie) – Performance beachten.
- Early Stopping, Multi‑Thread‑Konfigurationen und optionale GPU‑Nutzung (modellabhängig) als Konfigurationsoptionen anbieten.

