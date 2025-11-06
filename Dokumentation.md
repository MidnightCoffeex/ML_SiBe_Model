# AGENTS_MAKE_ML – Technische und fachliche Dokumentation (Stand: Okt 2025)

Diese Dokumentation beschreibt Ziel, Datenbasis, Pipeline (Feature Engineering), Modelle, Auswertung sowie die wichtigsten jüngsten Änderungen. Sie dient auch als Grundlage für die Ausarbeitung (Werdegang siehe `Update.md`).

---

## 1. Zielsetzung

- Ziel: Für jedes Teil einen stabilen, nachvollziehbaren Sicherheitsbestand (SiBe) ableiten.
- Ansatz: Tägliche Zeitreihen je Teil, darauf ML‑Modelle (GB/XGB/LGBM) zur Schätzung eines halbjährlich konstanten Zielwerts.
- Nutzen: Weniger Stockouts, stabilere Bestände, bessere Planbarkeit.

---

## 2. Datenquellen & Struktur

- Rohdaten (`Rohdaten/`): Bestände (Stichtage), Bewegungen (Lagerbew), Disposition (Bedarf/Deckung), Teilestamm (inkl. WBZ), SiBe und SiBe‑Verlauf.
- Spaltenmapping: `Spaltenbedeutung.xlsx` (robuste Zuordnung trotz Encoding/Benennungsvarianten).
- Ergebnisverzeichnisse:
  - `Features/<Teil>/features.parquet|xlsx`
  - `Modelle/<Teil|ALL>/<Modelltyp>/<ID>/`
  - `plots/<Teil|ALL>/<Modelltyp>/<ID>/`
  - Test‑Ordner für Validierung: `GPT_Features_Test/`, `GPT_Pipeline_Test/`

---

## 3. Pipeline – Von Rohdaten zu Features

### 3.1 Einlesen & Normalisierung

- Datum aus Dateinamen (`YYYYMMDD_*`), Encoding‑Toleranz, Dezimal‑Komma → Float.
- Filter `Lagerort=120`.
- SiBe‑Verlauf per asof‑Join (zeitlich korrekt wirksam ab Änderungsdatum).

### 3.2 Tägliche Reindexierung & Forward‑Fill

- Auch „ereignislose“ Tage werden explizit erzeugt (lückenlose Zeitachsen je Teil).
- Zentrale Bestandsgrößen werden vorwärtsgetragen (Forward‑Fill), z. B. `EoD_Bestand_noSiBe` als Basis.
- Nachfrage‑Features (DemandMean/Max) werden für alle Tage konsistent neu berechnet.

### 3.3 EoD‑Logik & Dispo

- Vor Dispo‑Start: Messwerte (Lagerbew/Bestand) auf Tageslevel verdichtet.
- Ab Dispo‑Start: Simulation End‑of‑Day via kumulierter Netto‑Bewegungen (Deckung – Bedarf).

### 3.4 Feature‑Gruppen

- Immer enthalten (nicht abwählbar): `F_NiU_EoD_Bestand`, `F_NiU_Hinterlegter SiBe`, `EoD_Bestand_noSiBe`, `WBZ_Days`, `Price_Material_var`.
- `Price_Material_var` kommt aus den `*_TeileWert.csv`-Exports, wird pro Teil als konstanter Stückpreis in jede Zeile geschrieben und steht dadurch für Label-Faktoren sowie Kapitalbindungs-Analysen bereit.
- Nachfrage: `DemandMean_*`, `DemandMax_*` inkl. Varianten `log1p`, `z_`, `robz`.
- Flags & Stammdaten: `Flag_StockOut`, `WBZ_Days`.
- Labels: `L_WBZ_BlockMinAbs` (Favorit für Training/Evaluierung), `L_NiU_WBZ_BlockMinAbs` (Diagnose), optional `L_HalfYear_Target` (preisfaktorisiertes Derivat).
- Lag‑Features (neu):
  - Punkt‑Lags: `Lag_EoD_Bestand_noSiBe_{7Tage,28Tage,wbzTage,2xwbzTage}` = Wert genau vor X Tagen.
  - Mittel‑Lags: `Lag_EoD_Bestand_noSiBe_mean_{...}` = rückblickendes Fenster (ohne heutigen Tag).

### 3.5 Selektiver Build (GUI)

- `scripts/build_features_gui.py`: Checkboxen für Features/Labels; Abhängigkeiten werden automatisch berechnet. Nicht ausgewählte Basiswerte werden, sofern nur als Abhängigkeit benötigt, im Output unterdrückt.
- Die „Fixe Basis“-Sektion (ausgegraut) umfasst `F_NiU_EoD_Bestand`, `F_NiU_Hinterlegter SiBe`, `EoD_Bestand_noSiBe`, `WBZ_Days`, `Price_Material_var`.
- Eigener Block für Lag-Features; Scrollen und Suchen erleichtern die Auswahl trotz vieler Kennzahlen.
- „Ereignislose“ Tage sowie Recompute der Demand-Features sind integriert.
- Zu jedem Lauf wird die getroffene Auswahl als `build_selection.json` im Zielordner gesichert (Dokumentation für spätere Rebuilds).

### 3.6 Test‑Set & Schema

- Der `Test_Set`-Ordner liegt auf gleicher Ebene wie der gewählte Feature-Ordner und spiegelt dessen Spaltenstruktur (identische Reihenfolge/Benennung) wider.
- Die Trennung in Train/Test erfolgt je Teil über das Exportdatum (`YYYYMMDD` aus dem Dateinamen): bis inkl. Exportdatum → Trainingsordner, ab Folgetag → `..._Test/`.
- Tage ohne Beobachtung werden – falls nötig – bis zum jeweiligen Cut vorwärts aufgefüllt, sodass Evaluationen lauffähig bleiben.

### 3.7 Preisfaktor & Kapitalbindung

- Stückpreise (`Price_Material_var`) stammen aus `*_TeileWert.csv`, werden je Teil als konstanter Wert vorgehalten und laufen in jeder Tabelle mit.
- Das Labeling nutzt einen logarithmischen Preisfaktor (Deckel bei −0,10), damit hochpreisige Teile konservativer vorgeschlagen werden als günstige.
- Auf Basis der Preise werden Kapitalbindungskennzahlen (`SiBe * Price_Material_var`) sowohl pro Teil als auch aggregiert berechnet (siehe Evaluationsausgabe).

---

## 4. Training

- Modelle: `gb` (sklearn Gradient Boosting), optional `xgb` (XGBoost), `lgbm` (LightGBM).
- Standard-Ziel: `L_WBZ_BlockMinAbs` (direkt aus der Zeitreihe, bevorzugte Trainingsgröße); `L_HalfYear_Target` steht optional als preisfaktorisiertes Derivat bereit.
- Gewichte: Schemata `none|blockmin|flag` plus Faktor (z. B. 5.0) per Prompt.
- Splits: Zeitreihen‑konform, optional CV‑Splits.
- Fortschritt: Optionaler Progress‑Balken parallel für `gb` via `--progress` oder Prompt.

---

## 5. Evaluierung

- Metriken: MAE, RMSE, R², MAPE (mit Vorsicht bei Ziel=0). Ergänzende service‑nahe Kennzahlen möglich.
- Robuste Anzeige‑Spalten: Toleranz bei `Hinterlegter SiBe` vs. `Hinterlegter_SiBe`.
- Exporte: `*_predictions.csv|xlsx` und Plots (PNG/HTML) unter `plots/...`.
- Aggregate für ALL-Modelle: `Alle_Teile.html` (Forward-Fill) und `Alle_Teile_no_forward.html` entstehen automatisch und zeigen pro Tag Summen für Hinterlegter/Vorgeschlagener SiBe sowie deren Kapitalbindung.
- Hover-Texte der Kapitalbindungs-Grafen listen die Top-3-Teile je Tag mit Wert und Prozentanteil; Ausreißer lassen sich so unmittelbar erklären.
- Hilfstabellen `Alle_Teile_Tageswerte.xlsx` und `Alle_Teile_Tageswerte_no_forward.xlsx` bündeln alle Tageswerte/Teile und eignen sich für Präsentationen oder zusätzliche Analysen.

---

## 6. Bedienung (Kurz)

1) Feature‑Build (GUI): `python scripts/build_features_gui.py`
2) Train: `python scripts/train.py` (interaktiv; optional `--progress`)
3) Evaluate: `python scripts/evaluate.py` (interaktiv)

---

## 7. Qualitätsregeln

- Keine Zukunfts‑Leakage (rollierende Features nur aus Vergangenheit, SiBe‑asof‑Join).
- NiU‑Spalten sind reine Anzeige/Diagnose und werden nicht als Features genutzt.
- Identisches Schema zwischen Features und Test‑Set.

---

## 8. Grenzen & Hinweise

- Aktuell `Lagerort=120` fix; Dateinamen benötigen Datum `YYYYMMDD`.
- Lange Zeiträume können ressourcenintensiv sein → selektiver Build oder Teilmengen.

