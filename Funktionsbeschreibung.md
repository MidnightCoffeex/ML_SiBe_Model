# Funktionsbeschreibung (einfach erklärt)

Diese Beschreibung richtet sich an Einsteiger ohne Programmier‑ und Mathe‑Vorkenntnisse. Sie erklärt in einfachen Worten, was dieses Projekt macht, warum es das macht und wie die Ergebnisse zu verstehen sind.

## Ein Überblick in drei Teilen
- Datapipeline: Aus Rohdaten wird eine saubere, tägliche Verlaufstabelle je Teil erstellt.
- Features & Labels: Aus der Verlaufstabelle werden nützliche Kennzahlen (Features) und ein Zielwert (Label) gebildet.
- Training & Evaluierung: Ein Lernverfahren baut aus den Features eine Vorhersage für das Label; die Güte wird gemessen und plausibel visualisiert.

---

## Datapipeline
### Warum?
Rohdaten aus verschiedenen Systemen (Bestände, Bewegungen, Planung) sind uneinheitlich und schwer direkt nutzbar. Die Pipeline macht sie vergleichbar: Ein sauberer, täglicher Zeitverlauf je Teil entsteht. Nur so können wir verlässlich rechnen und lernen.

### Was wird gemacht?
- Dateien lesen: CSV‑Exporte (z. B. Bestände, Bewegungen „Lagerbew“, Planung „Dispo“, Stammdaten „Teilestamm“, Sicherheitsbestand „SiBeVerlauf“).
- Aufbereiten: Spaltennamen begradigen, Dezimal‑Kommas in Zahlen umwandeln, auf den relevanten Lagerort filtern.
- Tägliche Reihe bilden: Für jedes Teil und jeden Tag wird ein Datensatz erzeugt (wie ein Tagebuch).
- Zusammenführen: Reale Bewegungen, geplante Zu‑/Abgänge und der an dem Tag gültige Sicherheitsbestand werden passend zusammengelegt.

### Wie (grob)?
- Datum aus den Dateien lesen, pro Tag zusammenfassen.
- „Ende‑des‑Tages‑Bestand“ (EoD) nachzeichnen: vor dem Planungsstart aus gemessenen Werten, danach mit Hilfe geplanter Zu‑/Abgänge weiterführen.
- Den am Tag gültigen Sicherheitsbestand („Hinterlegter SiBe“) rückwirkend korrekt beilegen.

### Wieso so?
- Ein tagesgenauer Verlauf ist die Basis für verlässliche Kennzahlen.
- Das strikte „Vergangenheit vor Zukunft“ verhindert, dass das Modell heimlich in die Zukunft schaut (keine Verzerrung, kein „Ratespiel“).

---

## Features & Labels
### Was sind Features?
Features sind verständliche Kennzahlen, die das Modell als Eingangsinformation nutzt – z. B.:
- „EoD_Bestand_noSiBe“: Der Bestand ohne Sicherheitsbestand – sagt, wie „eng“ es wirklich wird.
- „Flag_StockOut“: Markiert Tage, an denen es ohne Sicherheitsbestand auf oder unter Null geht.
- „WBZ_Days“: Wiederbeschaffungszeit (wie lange Lieferung/Ersatz dauert).
- „DemandMean_* / DemandMax_*“: Einfache Mittel‑/Spitzenwerte der entnommenen Menge über zurückliegende Zeitfenster (z. B. „in letzter Zeit im Schnitt so viel entnommen“).

Es gibt auch Anzeige‑/Diagnose‑Spalten (beginnen mit „F_NiU_“ bzw. „L_NiU_“). Diese sind „Not in Use“ – sie helfen beim Verständnis, werden aber bewusst nicht als Eingabe fürs Modell verwendet.

### Was ist das Label (Ziel)?
Das Label ist der Wert, den wir schätzen wollen: eine robuste, halbjährlich konstante Sicherheitsbestand‑Empfehlung.

- L_NiU_WBZ_BlockMinAbs (Diagnose): Schaut in ein typisches Zeitfenster (etwa die WBZ) und sucht den „tiefsten Punkt“ unter Null. Er zeigt, wie viel Puffer nötig gewesen wäre, um in diesem Fenster nicht ins Minus zu rutschen.
- LABLE_HalfYear_Target (Trainingsziel): Wir teilen die Zeit in Halbjahres‑Abschnitte. Für jeden Abschnitt nehmen wir den höchsten Diagnose‑Wert innerhalb des Abschnitts als Basis und 
  modulieren ihn nur vorsichtig mit drei leichtgewichtigen Faktoren:
  1) WBZ‑Länge (kurz/mittel/lang): Längere WBZ → etwas mehr Puffer.
  2) Häufigkeit von Entnahmen in einem Zeitraum: Viele Entnahmetage → etwas mehr Puffer.
  3) Schwankung der Entnahmen (Variabilität): Unruhige Entnahmemengen → etwas mehr Puffer.
  
  Diese Faktoren sind bewusst klein, werden gedeckelt (nicht zu groß) und gelten konstant für das Halbjahr. So bleiben Empfehlungen stabil und nachvollziehbar.

Warum halbjährlich?
- Sicherheitsbestände sollen nicht dauernd „springen“. Halbjahres‑Blöcke geben Ruhe, sind gut planbar und unterstützen Prozesse.

---

## Training (Modell lernen)
### Wozu?
Das Modell lernt aus den Beispielen der Vergangenheit, wie es aus den Features (Kennzahlen) auf die Zielgröße (Label) schließen kann. Es soll Muster erkennen: Wann brauchen wir tendenziell mehr Puffer, wann weniger?

### Welches Modell und warum?
Wir nutzen einen „Gradient Boosting“‑Ansatz (auch XGBoost/LightGBM möglich). In einfachen Worten:
- Viele kleine Entscheidungsbäumchen werden nacheinander gebaut.
- Jedes neue Bäumchen versucht, dort besser zu werden, wo die bisherigen noch daneben lagen.
- Am Ende wird die Summe vieler kleiner, zielgerichteter Verbesserungen zu einer guten Gesamtvorhersage.

Warum passt das hier gut?
- Funktioniert sehr gut mit tabellarischen Daten.
- Geht robust mit unterschiedlich skalierten Kennzahlen um.
- Liefert vernünftige Ergebnisse, ohne die Daten „überzuformen“ – wenn man es maßvoll einstellt.

### Wie lernt das Modell (für Laien)?
Stellen Sie sich vor, Sie schätzen anfangs grob. Danach schauen Sie, wo Sie am meisten daneben lagen – und verbessern dort gezielt. Dies wiederholen Sie oft in kleinen Schritten. So nähert man sich an die echten Werte an.

### Wichtige Einstellwerte (nur in Worten)
- Anzahl „Bäumchen“ (n_estimators): Mehr Schritte = potenziell genauer, aber auch längere Laufzeit und Risiko des Überlernens.
- Lernrate (learning_rate): Wie stark korrigiert jeder Schritt? Klein = vorsichtig/stabil, braucht mehr Schritte. Groß = schneller, aber risikoreicher.
- Tiefe (max_depth): Wie komplex darf ein Bäumchen sein? Größer = genauer fürs Training, aber empfindlicher (Überanpassung).
- Teillose ziehen (subsample): Nicht immer alle Daten auf einmal – macht robuster.

### Über‑/Unteranpassung (Overfitting/Underfitting)
- Überanpassung: Im Training top, auf neuen Daten schlecht. Abhilfe: Weniger Tiefe/Schritte, vorsichtigere Lernrate, Querschnittsprüfungen (Validierung).
- Unteranpassung: Zu einfaches Modell verkennt Muster. Abhilfe: Etwas mehr Schritte/Tiefe, bessere Merkmale.

### Wie prüfen wir, ob es gut ist?
- Zeitlich getrennte Prüfungen (Validierung/Test): Wir messen Fehler (z. B. MAE, RMSE) nicht nur im Training, sondern auch auf neueren Abschnitten. Gute Modelle sind dort stabil.
- Permutation Importance (Bedeutung der Kennzahlen): Wir „mischen“ testweise eine Kennzahl – wenn die Vorhersage dann deutlich schlechter wird, war diese Kennzahl wichtig. Das hilft, Entscheidungen zu erklären.

### Kann man „weitertrainieren“ (Feintuning)?
Ja, aber sinnvoll ist oft: bestehendes Modell als Ausgang nehmen, mit maßvollen Parametern weiterlernen oder mit einem kleineren Schrittmaß (Lernrate) nachschärfen. Alternativ: Neu trainieren mit leicht veränderten Einstellungen/Daten – je nach Ziel.

---

## Evaluierung (Ergebnisse prüfen)
### Wie?
- Wir lassen das gelernte Modell auf Daten laufen, die es nicht gesehen hat (neue Zeitabschnitte/Teile).
- Wir speichern Zahlen (CSV/Excel) und anschauliche Grafiken (Zeitverläufe, Ist vs. Prognose) je Teil und Abschnitt.

### Was sagen die Kennzahlen?
- MAE (durchschnittliche Abweichung) und RMSE (Abweichung mit Fokus auf Ausreißer): je kleiner, desto besser.
- R² (Erklärte Varianz): Wie gut werden Unterschiede nachvollzogen? Bei sehr kleinen Datenreihen oft nicht aussagekräftig.
- MAPE (prozentuale Abweichung): Vorsicht bei Ist‑Werten nahe 0 – kann irreführend groß werden.

### Wie interpretieren?
- Zeitverläufe zeigen, ob die Empfehlung plausibel „über der Nulllinie“ genug Puffer gibt.
- Die halbjährlichen Auszüge (Entscheidungen) helfen zu prüfen, ob Empfehlungen stabil und nachvollziehbar sind.
- Die Bedeutung der Kennzahlen (Permutation) macht verständlich, welche Faktoren das Modell wirklich „nutzt“.

---

## Häufige Fragen (FAQ)
**Warum sind manche Spalten als „NiU“ (Not in Use) markiert?**
- Zur Transparenz und Nachvollziehbarkeit. Diese Spalten dienen nur der Anzeige/Diagnose, werden aber bewusst nicht als Eingabe fürs Lernen genutzt.

**Warum 6‑Monats‑Blöcke?**
- Sicherheitsbestände sollen nicht dauernd wechseln. Halbjahres‑Blöcke sind ein guter Kompromiss aus Stabilität und Aktualität.

**Warum kleine Korrekturfaktoren (WBZ/Häufigkeit/Schwankung)?**
- Wir wollen die Basiserkenntnis (tiefster Punkt im Fenster) nicht übersteuern. Die Faktoren sind „moderate Nachjustierungen“, gedeckelt und stabil über das Halbjahr.

**Was mache ich mit auffälligen Ergebnissen?**
- Einzelteile genauer anschauen (Plots/CSV), Faktoren prüfen (z. B. sehr lange WBZ, sehr hohe Entnahme‑Häufigkeit) und ggf. Fachwissen einfließen lassen (MOQ/Packgrößen, kritische Teile).

---

## Das Wichtigste in einem Satz
Wir verwandeln Rohdaten in einen sauberen Tagesverlauf je Teil, bauen einfache, sinnvolle Kennzahlen und ein stabiles, halbjährliches Ziel – und lassen ein bewährtes Lernverfahren aus den Mustern der Vergangenheit eine nachvollziehbare Sicherheitsbestand‑Empfehlung ableiten.


---

## Nachtrag (Okt 2025)

- Vor erster Änderung im SiBe: Wir nehmen den ältesten "Alter_SiBe"-Wert aus der Historie und tragen ihn rückwärts bis zum frühesten Lagerbew-Datum ein (für Tage < erstem Änderungsdatum). Ab der ersten Änderung gilt wieder der aktive SiBe.
- ALL-Training: Alle Teile gemeinsam, global nach Datum (und Teil) sortiert. Zeit-Splits an Tagesgrenzen, damit nicht derselbe Kalendertag gleichzeitig in Training und Test liegt.
- Gewichte: Standard ist `blockmin` (Tage mit `L_NiU_StockOut_MinAdd > 0` zählen stärker, z. B. Faktor 5). Alternativ `flag` (bei `Flag_StockOut == 1`) oder `none` (keine Gewichtung). Einstellung beim Start des Trainings (Fragen im Terminal) oder per Argumenten.
- Auswertung: Das Ziel (z. B. `LABLE_HalfYear_Target`) wird im ersten Plot als graue Linie angezeigt.
- Spalten schnell an-/ausschalten: `scripts/feature_toggle_gui.py` bietet Checkboxen; abgewählte Spalten werden mit Prefix `nF_` versehen und so vom Training ignoriert.
- Timeseries-Scope: Du kannst wählen: „global“ (alle Teile zusammen, ein Modell) oder „lokal“ (jedes Teil bekommt sein eigenes Modell; wir gehen Teil für Teil den Zeitverlauf entlang). Die Abfrage kommt beim Start von scripts/train.py.
