# AGENTS_MAKE_ML – Technische und fachliche Dokumentation

Diese Dokumentation beschreibt das Ziel, die Datenbasis, die Pipeline zur
Merkmalsgewinnung (Feature Engineering), die Modelle, die Auswertung und die
fachliche Motivation. Sie richtet sich an Leserinnen und Leser ohne tiefe
Vorkenntnisse in KI/ML und erklärt, warum wir welche Schritte durchführen.
Unklare fachliche Entscheidungen sind mit „#Ich frage den User#“ markiert.

---

## 1. Zielsetzung und Problemkontext

- Ziel: Für jedes Teil (Material/Artikel) zu jedem Datum eine datenbasierte
  Empfehlung für den Sicherheitsbestand (SiBe) bzw. den notwendigen
  Zusatzpuffer (kein Stockout) abzuleiten.
- Nutzen: Bessere Verfügbarkeit, weniger Stockouts, planbare Bestände, und
  Unterstützung der Disposition mit nachvollziehbaren Signalen.
- Kernidee: Wir generieren zunächst eine konsistente, tägliche Zeitreihe je
  Teil (Bestände, Bedarfs-/Deckungsplanung, historischer SiBe). Darauf
  trainieren wir Modelle (z. B. Gradient Boosting), die Zielgrößen wie
  „zusätzlicher Bedarf zur Stockout-Vermeidung“ schätzen.

---

## 2. Datenquellen und Verzeichnisstruktur

- Rohdaten („Rohdaten/“): CSV-Exporte aus operativen Systemen (Semikolon
  getrennt, Encoding ISO-8859-1). Wichtigste Tabellen:
  - Bestand: Momentaufnahmen mit Lagerbestand.
  - Lagerbew: Bewegungen mit Zeitstempel (Bestandsverlauf).
  - Dispo: Geplante Bedarfe (Abgänge) und Deckungen (Zugänge) mit Terminen.
  - Teilestamm: Stammdaten inkl. „WBZ“ (Wiederbeschaffungszeit, Lead Time).
  - SiBe: Aktuell hinterlegter Sicherheitsbestand (Stichtagebene).
  - SiBeVerlauf: Historie der SiBe-Änderungen (neu: ein gemeinsamer Export je
    Datum mit Spalten „Teil“, „Datum Änderung“, „aktiver SiBe“).
- Begleitdatei: „Spaltenbedeutung.xlsx“ (Mapping, um relevante Spalten
  robust zu identifizieren).
- Ergebnisverzeichnisse:
  - Features/<Teil>/features.parquet|xlsx
  - Modelle/<Teil oder ALL>/<Modelltyp>/<ID>/
  - plots/<Teil oder ALL>/<Modelltyp>/<ID>/ (Diagnose-Grafiken und Exporte)

Hinweis: Die Pipeline ist robust gegenüber kleineren Spaltenabweichungen und
kodierungsbedingten Varianten (z. B. „Datum Änderung“ vs. „Datum �nderung“).

---

## 3. Pipeline – Von Rohdaten zu Features

### 3.1 Einlesen und Normalisierung

- Erkennung der CSV-Trennzeichen (auto), Encoding ISO-8859-1.
- Spaltenbereinigung (Trimmen, Umbenennen „Teil “ → „Teil“).
- Dezimalzahlen mit Komma in Gleitkommazahlen umwandeln.
- Filtern auf „Lagerort = 120“ (relevanter Lagerort für die Betrachtung).
- Spaltenauswahl über „Spaltenbedeutung.xlsx“ und dataset-spezifische Logik.

Begründung: Einheitliche, saubere und robuste Datenbasis ist Vorbedingung
für sinnvolle ML-Features und vergleichbare Zeitreihen.

### 3.2 Aggregation pro Dataset

- Lagerbew: Zeitstempel → Tageslevel runden; pro (Teil, Tag) den letzten
  Lagerbestand (End-of-Day) verwenden.
- Dispo: Bedarfe („Bedarfsmenge“) und Deckungen („Deckungsmenge“) auf Tages-
  level summieren; Netto = Deckung − Bedarf.
- Bestand/Teilestamm: Stichtagswerte pro (Teil, Tag) übernehmen.

Begründung: Konsistente tägliche Zeitscheiben erleichtern die spätere
Zusammenführung und vermeiden Doppellogik in den Modellen.

### 3.3 SiBeVerlauf – Historie korrekt anwenden

- Alte Struktur: je Teil eine Datei, Spalten wie „AudEreignis-ZeitPkt“ und
  „Im Sytem hinterlgeter SiBe“.
- Neue Struktur: ein gemeinsamer Export je Datum mit „Teil“, „Datum Änderung“,
  „aktiver SiBe“.
- Vereinheitlichung: Spalten werden robust erkannt (auch bei Zeichensatz-
  varianten) und auf ein gemeinsames Schema gemappt.
- Zeitliche Anwendung (Intervall-Logik):
  - Die Änderungszeitpunkte pro Teil werden chronologisch sortiert.
  - Für jedes Feature-Datum wird der zuletzt gültige Änderungswert mittels
    „asof-Join (direction=backward)“ zugewiesen.
  - Vor dem ersten Eintrag gilt 0; nach dem letzten Eintrag gilt der letzte
    Wert fortlaufend.

Begründung: Genau das bildet die Realität ab: Ein geänderter SiBe gilt ab dem
Änderungsdatum, bis ein neuer Wert eingetragen wird.

### 3.4 Bestandsbasis und EoD-Simulation

- „Baseline“: Zum Start der Dispo-Zeit (erstes Dispo-Datum) wird, wenn möglich,
  der zuletzt bekannte Lagerbew-/Bestand-Wert als Ausgangsbestand verwendet.
- EoD_Bestand: Vor Dispo-Start übernehmen wir gemessene Lagerbew-Werte; ab
  Dispo-Start simulieren wir den End-of-Day-Bestand: Baseline + kumulierte
  Netto-Bewegungen („cum_net“) aus Dispo.

Begründung: So verbinden wir tatsächlichen Verlauf und Planungen, um für jeden
Tag einen plausiblen Bestand zu haben.

### 3.5 Abgeleitete Features

- nF_EoD_Bestand (nur Anzeige/Plots, nicht fürs Training): End-of-Day-Bestand
  (siehe oben); als „nF_“ gekennzeichnet, damit es nicht ins Modell fließt.
- nF_Hinterlegter SiBe (nur Anzeige/Plots): Aus SiBeVerlauf abgeleiteter,
  gültiger SiBe je Tag.
- EoD_Bestand_noSiBe: nF_EoD_Bestand − nF_Hinterlegter SiBe.
- Flag_StockOut: 1, wenn EoD_Bestand_noSiBe ≤ 0, sonst 0.
- WBZ_Days: Wiederbeschaffungszeit (Tage) aus Teilestamm (falls vorhanden).
- Rollierende Verbrauchsmerkmale aus EoD_Bestand_noSiBe:
  - DemandMean_*, DemandMax_* mit Fenstern in Relation zur WBZ (100%, 66%,
    50%, 25%) – basierend auf gelaggten Differenzen, um Zukunfts-Leakage zu
    vermeiden.

Bewusst entfernt (fachlich/ML-bedingt):
- DaysToEmpty: nutzte zukünftige Information → Future-Leakage-Risiko.
- BestandDelta_7T: geringes/instabiles Signal.

Begründung: Features sollen robust, kausal plausibel und ohne Zukunftswissen
sein.

### 3.6 Labels (Zielgrößen)

- _LABLE_StockOut_MinAdd (behalten): Schätzt die zusätzliche Menge, die
  benötigt wäre, um innerhalb eines vorausschauenden Fensters (Lookahead)
  Stockouts zu vermeiden.
  - Lookahead ≈ 1.25 × WBZ (#Ich frage den User#: Warum genau 1.25?)
  - Konstruktionsprinzip: Kombination aus „progressiver“ Annäherung an den
    Stockout-Zeitpunkt und „rollierender“ Defizit-Summe über das Fenster.
  - Interpretation: „Wie viel muss ich minimal addieren, um auf Sicht keine
    Unterdeckung zu haben?“
- Entfernte Labels: LABLE_SiBe_STD95, LABLE_SiBe_AvgMax, LABLE_SiBe_Percentile. _LABLE_StockOut_MinAdd bleibt als Hintergrundspalte erhalten (nicht f�rs Training). Gründe: begrenzter Zusatznutzen, teils nicht zur
  Optimierung passend (MSE vs. Quantil), Gefahr der Zielunschärfe.

---

## 4. Training – Modelle und Splits

- Feature-Selektion: Es werden ausschließlich numerische Prädiktoren genutzt;
  „nF_*“ und Target-Spalten werden explizit ausgeschlossen.
- Zeitreihen-Splitting (TimeSeriesSplit):
  - Training/Validierung/Test sind zeitlich sortiert und nicht vermischt.
  - Optionale CV über mehrere Folds möglich.
- Modellfamilien:
  - Gradient Boosting (scikit-learn)
  - XGBoost (optional)
  - LightGBM (optional)
- Gewichte: Zeilen mit _LABLE_StockOut_MinAdd > 0 können höher gewichtet
  werden (#Ich frage den User#: Welche Gewichtung ist fachlich gewünscht?).
- ALL vs. Teil-spezifisch: Ein Modell über alle Teile (ALL) erfasst Muster
  global; Teil-spezifische Modelle können individuelle Charakteristika besser
  treffen (#Ich frage den User#: wann ALL vs. pro Teil einsetzen?).

Begründung: Zeitbasierte Splits verhindern Datenleckage. GBMs sind starke,
interpretierbare Baselines für strukturierte Daten.

---

## 5. Evaluation – Messgrößen und Visualisierung

- Standardmetriken: MAE, RMSE, R². Hinweis: MAPE ist bei Zielwert 0 nicht
  zuverlässig; idealerweise zusätzlich WAPE/SMAPE/MASE verwenden (#Ich frage den User#: Welche Metriken sollen offiziell berichtet werden?).
- Service-Level-bezogene Metriken (empfohlen):
  - Anteil Tage ohne Unterdeckung (EoD_Bestand_noSiBe + Prognose ≥ 0)
  - Summe/Max/Serienlänge von Unterdeckungen (Praxisrelevanz)
- Plots/Exports:
  - Actual vs. Predicted (Scatter, Zeitverlauf)
  - Zeitreihen-Overlays: EoD_Bestand_noSiBe, vorhergesagte Puffer, und Summe
  - Trainingshistorie (sofern Modell verfügbar)

Begründung: Neben klassischen Fehlermaßen zeigen servicelevel-nahe Kennzahlen,
ob das betriebliche Ziel erreicht wird.

---

## 6. Bedienung – Schritt für Schritt

1) Features erzeugen

```bash
python scripts/build_features.py --input Rohdaten --output Features
```
- Fragt Pfade ggf. interaktiv ab. Pro Teil entsteht „features.parquet“ und
  „features.xlsx“.

2) Modelle trainieren (Beispiel: ALL, GBM)

```bash
python scripts/train.py --data Features --part ALL \
  --model-dir Modelle --models gb \
  --targets _LABLE_StockOut_MinAdd \
  --n_estimators 600 --learning_rate 0.05 --max_depth 4 --subsample 0.8
```

3) Evaluieren (Beispiel: Teil 1100831 mit ALL‑Modell)

```bash
python scripts/evaluate.py --features Features --part ALL \
  --model-dir Modelle --model-type gb --model-id 1 \
  --targets _LABLE_StockOut_MinAdd --plots plots
```
- Bei „part=ALL“ fragt das Script intern nach der konkreten Teilnummer für die
  Auswertung.

---

## 7. Qualitätsregeln und Entscheidungen

- Keine Zukunfts-Leakage: Rollierende Features nutzen nur Vergangenheitswerte
  (via shift/lag). SiBe-Verlauf wird zeitlich korrekt per asof-Join angewendet.
- „nF_*“‑Spalten: Für Transparenz/Plots sichtbar, aber kein Input fürs Modell.
- Robustheit: Toleranz gegenüber abweichenden Spaltennamen/Encodings.
- Trennung Ziel/Diagnose: Training auf definierten Labels; Vergleich gegen
  „(nF_)Hinterlegter SiBe“ nur als Diagnose – nicht als Trainingsziel.

---

## 8. Bekannte Grenzen und typische Fehlerquellen

- MAPE bei vielen Nullen im Ziel wenig aussagekräftig → alternative Kennzahlen
  nutzen.
- Baseline (Startbestand) beeinflusst EoD-Simulation: Ist die letzte Messung
  zu weit vom Dispo-Beginn entfernt, kann das Niveau verschoben sein.
- ALL‑Modelle können teil-spezifische Besonderheiten verwässern; umgekehrt
  leiden Teilmodelle unter Datenknappheit.

---

## 9. Weiterentwicklung (Empfehlungen)

- Metriken: WAPE/SMAPE/MASE und Service-Level-Kurven in den Standard-Report.
- Modelle: Für quantilartige Ziele (z. B. p90/p95‑Puffer) Quantil‑Loss
  einsetzen; Early Stopping mit zeitlichen Validierungsfenstern.
- Features: Kalendermerkmale (Wochentag, Monat, Quartal, Feiertage), Trend-
  und Volatilitätsmerkmale für Dispo‑Qualität.
- Backtesting: Mehrere rollierende Testfenster (Mittelwert + Streuung der
  Kennzahlen) zur Stabilitätsaussage.

---

## 10. Offene fachliche Punkte – bitte klären

- Faktor 1.25 × WBZ für Lookahead-Fenster: #Ich frage den User#
- Gewichtung von Zeilen mit Stockout‑Nähe (z. B. 5x): #Ich frage den User#
- Sollen „Hinterlegter SiBe“‑Werte in die Modelle einfließen oder nur als
  Diagnose dienen? (Derzeit nur Diagnose über nF_): #Ich frage den User#
- Bevorzugtes Zielkriterium im Betrieb: Service-Level, Kostenfunktion
  (Holding vs. Stockout‑Penalty) oder gemischte Zielfunktion? #Ich frage den User#
- Wann ALL-Modelle vs. Teilmodelle? Schwelle (Datenmenge/Volatilität)?
  #Ich frage den User#

---

## 11. Glossar

- SiBe (Sicherheitsbestand): Bestandspuffer zur Abdeckung von Unsicherheit.
- Stockout: Bestand fällt (unter Berücksichtigung des SiBe) auf ≤ 0.
- WBZ (Wiederbeschaffungszeit): Zeit vom Auslösen bis zum Eintreffen der Ware.
- Feature: Eingangsgröße eines Modells (Prädiktor).
- Label: Zielgröße, die das Modell lernen soll.
- asof-Join: Verknüpfung, die für jeden Zeitpunkt den zuletzt früheren Wert
  übernimmt (typisch für Zeitreihenhistorien).

---

## 12. Technische Umgebung (Kurz)

- Python 3.11, pandas, NumPy, scikit‑learn, matplotlib, pyarrow, openpyxl,
  plotly; optional: xgboost, lightgbm.
- Encoding: ISO‑8859‑1; Separator: auto; robustes Spalten‑Mapping.
- Scripts: `build_features.py`, `train.py`, `evaluate.py`.

---

Stand dieses Dokuments entspricht der aktuellen Pipeline-Logik mit
zeitkorrekter Anwendung des SiBe‑Verlaufs und der Trennung von
„nF_*“‑Spalten (nur Anzeige) und Modellfeatures.

\n\n### Neues Label: LABLE_WBZ_NegBlockSum\n\n- Ziel: Im WBZ-Fenster [t, t+WBZ) alle zusammenh�ngenden negativen Bl�cke in EoD_Bestand_noSiBe betrachten.\n- F�r jeden Block wird das Minimum (st�rkste Unterdeckung) einmalig herangezogen.\n- Das Label ist die Summe der Absolutbetr�ge dieser Block-Minima innerhalb des Fensters (positiver Wert).\n- Motivation: Mehrere getrennte Engp�sse im Planhorizont sollen additiv abgesichert werden; WBZ steuert die zeitliche Reichweite der Entscheidung.\n
