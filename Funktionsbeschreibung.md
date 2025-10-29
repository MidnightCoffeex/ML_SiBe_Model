# Funktionsbeschreibung – Einfach erklärt (Okt 2025)

Diese kurze Erklärung richtet sich an Nutzer ohne Programmier‑/ML‑Vorkenntnisse. Sie beschreibt, was das System macht und wie man es bedient.

---

## Überblick

- Datapipeline: Aus Rohdaten entsteht pro Teil ein lückenloser, täglicher Verlauf.
- Features & Label: Daraus werden sinnvolle Kennzahlen (Features) und ein Zielwert (Label) gebildet.
- Training & Auswertung: Ein Modell lernt aus der Vergangenheit und wird mit verständlichen Kennzahlen bewertet.

---

## Datapipeline – Warum und wie?

Warum? Rohdaten sind uneinheitlich. Wir brauchen eine saubere, tägliche Tabelle je Teil.

Wie?
- Die Dateien werden eingelesen, Spalten vereinheitlicht und auf den relevanten Lagerort gefiltert.
- Es werden alle Tage erzeugt – auch solche ohne Ereignis. Wichtige Bestandswerte werden von gestern nach heute fortgetragen (Forward‑Fill).
- Planbedarfe/‑deckungen (Dispo) und der an jedem Tag gültige Sicherheitsbestand (SiBe) werden passend angelegt.

Ergebnis: Eine verständliche Tages‑Tabelle je Teil als Grundlage für Kennzahlen und das Modell.

---

## Features & Label – Was genau?

Immer enthalten (für Übersicht/Modellbasis):
- `F_NiU_EoD_Bestand` (nur Anzeige)
- `F_NiU_Hinterlegter SiBe` (nur Anzeige)
- `EoD_Bestand_noSiBe` (wichtige Basisgröße)

Weitere Beispiele:
- Nachfrage‑Kennzahlen: `DemandMean_*`, `DemandMax_*`, auch als Varianten `log1p`, `z_`, `robz`.
- Flags: `Flag_StockOut`, Stammdaten wie `WBZ_Days`.

Lag‑Features (neu):
- „Punkt‑Lag“: z. B. `Lag_EoD_Bestand_noSiBe_7Tage` = Wert von genau vor 7 Tagen.
- „Mittel‑Lag“: z. B. `Lag_EoD_Bestand_noSiBe_mean_28Tage` = Durchschnitt der letzten 28 Tage (ohne heute).

Label (Ziel) für das Training:
- `L_HalfYear_Target`: Stabile, halbjährliche Empfehlung, abgeleitet aus dem diagnostischen `L_NiU_WBZ_BlockMinAbs`.

Hinweis zu Abhängigkeiten: Wenn eine abgeleitete Kennzahl (z. B. `..._log1p`) gewählt wird, werden dafür nötige Basiswerte intern berechnet – sie erscheinen aber nur in der Ausgabe, wenn sie ebenfalls angehakt wurden.

---

## Selektiver Feature‑Build (GUI)

Mit `python scripts/build_features_gui.py` öffnet sich eine einfache Oberfläche:
- Häkchen setzen, welche Features/Labels erstellt werden sollen.
- Abhängigkeiten werden automatisch berücksichtigt.
- Die drei Basis‑Spalten (siehe oben) sind immer dabei.
- Ausgabe pro Teil: `Features/<Teil>/features.parquet` und `features.xlsx`.

Praktischer Effekt: Weniger Rechenzeit, weil nur wirklich benötigte Spalten gebaut werden.

---

## Training & Auswertung

Training (`python scripts/train.py`):
- Wähle Feature‑Ordner, Teil oder `ALL`, Modelltyp (`gb`, optional `xgb`, `lgbm`).
- Stelle Hyperparameter ein. Auf Wunsch zeigt ein Progress‑Balken den Trainingsfortschritt (parallel, stört das Training nicht).
- Ergebnisse werden in `Modelle/...` gespeichert (inkl. Metriken und Feature‑Wichtigkeit).

Auswertung (`python scripts/evaluate.py`):
- Berechnet MAE, RMSE, R², MAPE und erzeugt CSV/Excel sowie Plots (PNG/HTML) in `plots/...`.
- Anzeige‑Spalten wie „Hinterlegter SiBe“ werden robust erkannt (Unterstrich/Leerzeichen egal).

---

## Das Wichtigste in einem Satz

Wir verwandeln Rohdaten in einen sauberen, täglichen Verlauf je Teil, bauen sinnvolle Kennzahlen (inkl. Lag‑Features) und ein stabiles, halbjährliches Ziel – und lassen ein bewährtes Lernverfahren daraus eine nachvollziehbare Sicherheitsbestands‑Empfehlung ableiten.

