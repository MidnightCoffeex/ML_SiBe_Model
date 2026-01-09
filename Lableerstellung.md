# Lableerstellung - L_WBZ_BlockMinAbs

Diese Datei beschreibt, wie das Label `L_WBZ_BlockMinAbs` in der Pipeline berechnet wird.
Quelle im Code: `AGENTS_MAKE_ML/src/data_pipeline.py` (Voll-Build und selektiver Build).

---

## Kurzuebersicht

Grundidee: Fuer jeden Tag t wird ein Vorwaertsfenster der Laenge WBZ gebildet und auf
`EoD_Bestand_noSiBe` das Minimum gesucht. Negative Tiefpunkte sind Unterdeckungen.
Das Block-Label ist der positive Betrag dieser Unterdeckung.

Im aktuellen Stand gilt:
- `L_WBZ_BlockMinAbs_noFactors` = Block-Base (reines Blockminimum)
- `L_WBZ_BlockMinAbs` = Block-Base * (1 + Endfaktor)
- `L_WBZ_BlockMinAbs_Factor` = Endfaktor (0.00 bis 0.40)
- `L_NiU_WBZ_BlockMinAbs` wird im Voll-Build als Diagnose geschrieben und ist dort
  identisch zu `L_WBZ_BlockMinAbs` (faktorisiert). Im selektiven Build ist es nicht
  als auswaehlbares Label vorgesehen.

---

## Formel (pro Tag t)

Fenster:
- Voll-Build: H = int(WBZ_Days) mit Fallback 1
- Selektiver Build: H = max(1, round(WBZ_Days)) mit Fallback 14
- Fenster ist [t, t + H) (Start inkl., Ende exkl.)

BlockBase(t):

```
BlockBase(t) = max(0, - min_{u in [t, t+H)} EoD_Bestand_noSiBe(u))
```

Endfaktor:

```
f_total(t) = clip( f_wbz + f_freq + f_vol + f_price , 0.00 , 0.40 )
```

Labels:

```
L_WBZ_BlockMinAbs_noFactors(t) = BlockBase(t)
L_WBZ_BlockMinAbs(t)          = BlockBase(t) * (1 + f_total(t))
L_WBZ_BlockMinAbs_Factor(t)   = f_total(t)
L_NiU_WBZ_BlockMinAbs(t)      = BlockBase(t) * (1 + f_total(t))   # Voll-Build
```

---

## Legende und Detailregeln

EoD_Bestand_noSiBe(u):
- Tagesendbestand ohne hinterlegten SiBe.
- Berechnung: `F_NiU_EoD_Bestand - F_NiU_Hinterlegter SiBe`.

WBZ_Days:
- Wiederbeschaffungszeit in Tagen (pro Teil aus Teilestamm).
- Voll-Build: `lead_time = int(WBZ_Days)`; falls fehlt/ungueltig -> 1 Tag.
- Selektiver Build: `H = max(1, round(WBZ_Days))`; falls fehlt/ungueltig -> 14 Tage.

f_wbz (WBZ-Faktor):
- Abhaengig von WBZ_Days, pro 6-Monats-Fenster konstant.
- <= 28 Tage: 0.00
- 28..84 Tage: linear von 0.00 bis 0.10
- > 84 Tage: 0.20

f_freq (Frequenz-Faktor):
- 0.00..0.20, pro 6-Monats-Fenster konstant.
- Basierend auf Anzahl positiver Entnahmen im Fenster:
  `demand = (EoD_Bestand_noSiBe.shift(1) - EoD_Bestand_noSiBe)`, nur > 0.
- Ereignisse werden auf WBZ skaliert und log-gedaempft:
  `f_freq = 0.20 * log1p(est_events_wbz) / log1p(8.0)`, danach 0..0.20 gecappt.

f_vol (Volatilitaet):
- 0.00..0.20, pro 6-Monats-Fenster konstant.
- Variationskoeffizient der Entnahmen im Fenster:
  `CV = sigma / mu` (falls mu klein -> 0).
  - CV <= 0.3 -> 0.00
  - CV >= 0.8 -> 0.20
  - dazwischen linear.

f_price (Preisfaktor):
- <= 0, daempft (max. -0.10), pro 6-Monats-Fenster konstant.
- Berechnung aus `Price_Material_var`:
  - `p_log = log1p(Preis)`
  - `scale = 95%-Perzentil(p_log > 0)`
  - `f_price = -0.10 * (p_log / scale)` und auf [-0.10, 0] gecappt.

clip(x, a, b):
- Begrenzung auf Intervall [a, b].
- Gesamtfaktor darf nie < 0 werden (Preis kann Zuschlaege nur reduzieren).

---

## Zeitfenster (6-Monats-Logik)

Die Faktoren f_wbz, f_freq, f_vol, f_price werden nicht taeglich, sondern
fensterweise (ca. 6 Monate) berechnet und innerhalb des Fensters konstant gesetzt.
Das Fenster wird anhand der vorhandenen Tagesdaten auf das naechste verfuegbare Datum
aufgerundet, wenn genau 6 Monate nicht im Raster liegen.

---

## Hinweis zur Teiletrennung

Die Berechnung erfolgt strikt pro Teil. Es gibt keine Vermischung zwischen Teilen.
WBZ, Preis und Nachfrageverlauf sind teilindividuell und beeinflussen die Faktoren.
