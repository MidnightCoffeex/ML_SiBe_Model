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
- `Price_Material_var` (Stückpreis aus `*_TeileWert.csv`, pro Teil konstant – Grundlage für Preisfaktor und Kapitalbindung)
- `WBZ_Days` (Lead Time in Tagen)

Weitere Beispiele:
- Nachfrage‑Kennzahlen: `DemandMean_*`, `DemandMax_*`, auch als Varianten `log1p`, `z_`, `robz`.
- Flags: `Flag_StockOut`, Hilfsgrößen wie zusätzliche Lags.

Lag‑Features (neu):
- „Punkt‑Lag“: z. B. `Lag_EoD_Bestand_noSiBe_7Tage` = Wert von genau vor 7 Tagen.
- „Mittel‑Lag“: z. B. `Lag_EoD_Bestand_noSiBe_mean_28Tage` = Durchschnitt der letzten 28 Tage (ohne heute).

Label (Ziel) für das Training:
- `L_HalfYear_Target`: Stabile, halbjährliche Empfehlung, abgeleitet aus dem diagnostischen `L_NiU_WBZ_BlockMinAbs` und mit einem logarithmischen Preisfaktor verfeinert (teure Teile werden etwas konservativer empfohlen als günstige, maximaler Abzug 0,10).

Hinweis zu Abhängigkeiten: Wenn eine abgeleitete Kennzahl (z. B. `..._log1p`) gewählt wird, werden dafür nötige Basiswerte intern berechnet – sie erscheinen aber nur in der Ausgabe, wenn sie ebenfalls angehakt wurden.

---

## Selektiver Feature‑Build (GUI)

Mit `python scripts/build_features_gui.py` öffnet sich eine einfache Oberfläche:
- Häkchen setzen, welche Features/Labels erstellt werden sollen (inkl. separatem Bereich für Lag-Features).
- Abhängigkeiten werden automatisch berücksichtigt; reine Zwischenwerte erscheinen nur, wenn sie ebenfalls ausgewählt werden.
- Die „Fixe Basis“ (immer aktiv) umfasst `F_NiU_EoD_Bestand`, `F_NiU_Hinterlegter SiBe`, `EoD_Bestand_noSiBe`, `WBZ_Days` und `Price_Material_var`.
- Ausgabe pro Teil: `Features/<Teil>/features.parquet` und `features.xlsx` plus `build_selection.json` mit der getroffenen Auswahl.

Praktischer Effekt: Weniger Rechenzeit, weil nur wirklich benötigte Spalten gebaut werden, und gleichzeitig sind Auswahlstände dokumentiert.

---

## Training & Auswertung

Training (`python scripts/train.py`):
- Wähle Feature‑Ordner, Teil oder `ALL`, Modelltyp (`gb`, optional `xgb`, `lgbm`).
- Als Ziel wird standardmäßig `L_WBZ_BlockMinAbs` vorgeschlagen; `L_HalfYear_Target` kann optional gewählt werden, wenn die preisfaktorisierte Variante benötigt wird.
- Stelle Hyperparameter ein. Auf Wunsch zeigt ein Progress‑Balken den Trainingsfortschritt (parallel, stört das Training nicht).
- Ergebnisse werden in `Modelle/...` gespeichert (inkl. Metriken und Feature‑Wichtigkeit).

Auswertung (`python scripts/evaluate.py`):
- Berechnet MAE, RMSE, R², MAPE und erzeugt CSV/Excel sowie Plots (PNG/HTML) in `plots/...`.
- Für ALL-Modelle entsteht zusätzlich `Alle_Teile/` mit zwei HTMLs (Forward-Fill & ohne Forward-Fill) sowie Hilfstabellen (`Alle_Teile_Tageswerte*.xlsx`).
- Die Kapitalbindungs-Grafen zeigen beim Mouseover die Top-3-Teile des jeweiligen Tages mit Wert und Prozentanteil – Ausreißer werden sofort sichtbar.
- Anzeige‑Spalten wie „Hinterlegter SiBe“ werden robust erkannt (Unterstrich/Leerzeichen egal).

---

## Das Wichtigste in einem Satz

Wir verwandeln Rohdaten in einen sauberen, täglichen Verlauf je Teil, bauen sinnvolle Kennzahlen (inkl. Lag‑Features) und ein stabiles, halbjährliches Ziel – und lassen ein bewährtes Lernverfahren daraus eine nachvollziehbare Sicherheitsbestands‑Empfehlung ableiten.

