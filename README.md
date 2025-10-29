# AGENTS_MAKE_ML

Dieses Projekt baut aus Rohdaten tägliche Feature‑Tabellen je Teil und trainiert darauf ML‑Modelle zur Ableitung eines stabilen Sicherheitsbestands (SiBe). Die wichtigsten Ordner sind `Rohdaten/` (Quellen), `Features/` (generierte Merkmale), `Modelle/` (trainierte Modelle) sowie `plots/` (Auswertungen). `Spaltenbedeutung.xlsx` beschreibt die Spalten semantisch.

## Daten & Struktur

```
Rohdaten/
    20250313_M100_Bestand.csv
    20250313_M100_Dispo.csv
    20250313_M100_Lagerbew.csv
    20250313_M100_SiBe.csv
    20250313_M100_TeileWert.csv
    20250313_M100_Teilestamm.csv
    ...
Spaltenbedeutung.xlsx
```

CSV‑Dateien sind typischerweise semikolon‑getrennt und enthalten Bestands‑, Bewegungs‑ und Dispositionsdaten.

## Pipeline & Features (aktuell)

- Parsing der Dateien (Datum im Namen `YYYYMMDD_*`), Normalisierung (Komma‑Dezimal, Trimmen), Filter auf `Lagerort=120`.
- Tägliche Reindexierung: Auch „ereignislose“ Tage werden erzeugt; zentrale Bestandswerte werden per Forward‑Fill fortgetragen.
- Dispo‑Bewegungen werden tagesgenau verrechnet; bedarfsbasierte Features (DemandMean/Max, inkl. `log1p`, `z_`, `robz`) werden für alle Tage neu berechnet.
- SiBe‑Historie wird per asof‑Join zeitlich korrekt angelegt.
- Immer enthaltene Basis‑Spalten (nicht abwählbar):
  - `F_NiU_EoD_Bestand` (Anzeige)
  - `F_NiU_Hinterlegter SiBe` (Anzeige)
  - `EoD_Bestand_noSiBe` (Feature‑Basis)
- Weitere Feature‑Gruppen (Auszug):
  - Nachfrage: `DemandMean_*`, `DemandMax_*` inkl. Varianten `log1p`, `z_`, `robz`
  - Flags: `Flag_StockOut`; WBZ: `WBZ_Days`
  - Labels: `L_NiU_WBZ_BlockMinAbs` (Diagnose), `LABLE_HalfYear_Target` (Training)
- Lag‑Features (neu):
  - Punkt‑Lags: `Lag_EoD_Bestand_noSiBe_{7Tage,28Tage,wbzTage,2xwbzTage}`
  - Mittel‑Lags: `Lag_EoD_Bestand_noSiBe_mean_{7Tage,28Tage,wbzTage,2xwbzTage}`

Abhängigkeiten: Wird ein abgeleitetes Feature (z. B. `DemandMean_25_log1p`) gewählt, werden benötigte Basisspalten intern berechnet, aber nur ausgegeben, wenn sie ebenfalls explizit ausgewählt sind.

Leistung: Große Zeiträume → höhere RAM/IO‑Last. Für Schnelltests nur wenige Teile/Zeiträume wählen.

## Voraussetzungen

- Python 3.11+
- Abhängigkeiten aus `requirements.txt` installieren:

```bash
python -m pip install -r requirements.txt
```

Optional: `xgboost`, `lightgbm` für zusätzliche Modelle.

## Quickstart

1) Features erzeugen (GUI, selektiv):

```bash
python scripts/build_features_gui.py
```

- Checkboxen für Features/Labels; Abhängigkeiten werden automatisch erfüllt.
- Immer enthalten: `F_NiU_EoD_Bestand`, `F_NiU_Hinterlegter SiBe`, `EoD_Bestand_noSiBe`.
- Ausgabe: `Features/<Teil>/features.parquet|xlsx`. Für Tests wurde zusätzlich `GPT_Features_Test/` genutzt.

2) Training:

```bash
python scripts/train.py
```

- Interaktive Abfrage von Feature‑Pfad, Teil (oder `ALL`), Modelltyp (`gb`, optional `xgb`, `lgbm`), Hyperparametern und optionalem Progress‑Balken (`--progress` oder Prompt).
- Modelle und Metriken landen unter `Modelle/<Teil|ALL>/<Modelltyp>/<ID>/`.

3) Evaluierung:

```bash
python scripts/evaluate.py
```

- Interaktive Abfrage analog Training. Plots/Exports unter `plots/<Teil|ALL>/<Modelltyp>/<ID>/`.

## Selektiver Feature‑Build (Details)

- GUI listet alle verfügbaren Features (Demand, `log1p`/`z_`/`robz`, Lags, Flags, …) und Labels.
- Abhängigkeiten: Wird ein abgeleitetes Feature angewählt, werden benötigte Basiswerte gerechnet, aber nur ausgegeben, wenn ebenfalls angewählt.
- „Ereignislose“ Tage werden durch Reindexing erzeugt; Bestandswerte werden vorwärtsgetragen; Demand‑Features werden für alle Tage neu berechnet.

## Training (Details)

- Modelle: `gb` (scikit‑learn), optional `xgb`, `lgbm`.
- Gewichte: `none|blockmin|flag` inkl. Faktor (z. B. 5.0) per Prompt.
- Progress‑Balken (parallel) für `gb` via `--progress` oder Abfrage.
- Splits: Zeitreihen‑konform, optional CV‑Splits per Prompt.

## Evaluierung

- Metriken: MAE, RMSE, R², MAPE (Vorsicht bei Ziel=0).
- Robuster Umgang mit Anzeige‑Spalten (`Hinterlegter SiBe` vs. `Hinterlegter_SiBe`).
- Exporte: `*_predictions.csv|xlsx`, Plots als PNG/HTML unter `plots/...`.

## Hinweise & Grenzen

- Aktuell `Lagerort=120` fix.
- Dateinamen benötigen Datum `YYYYMMDD`.
- Große Zeiträume → hohe RAM‑Last; selektiv bauen.

## Konventionen (NiU)

- `F_NiU_*`: Anzeige/Hilfsspalten, nicht fürs Training.
- `L_NiU_*`: Diagnose‑Labels, nicht fürs Training.
- Trainingsziel: `LABLE_HalfYear_Target` (aus `L_NiU_WBZ_BlockMinAbs`).
- Train/Eval schließen `F_NiU_`/`L_NiU_`/`nF_` automatisch aus.

