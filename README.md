# AGENTS_MAKE_ML

This repository contains raw CSV files for a machine learning project.
The folder `Rohdaten/` holds the source data while `Spaltenbedeutung.xlsx` explains each column.

## Project overview

The goal is to predict an optimal *Sicherheitsbestand* (safety stock) for each
part on specific dates.  A preprocessing pipeline converts the raw CSV exports
into a single feature table.  On top of these features a Gradient Boosting
model is trained to suggest suitable safety stock levels.

## Dataset structure

```
Rohdaten/
    20250313_M100_Bestand.csv
    20250313_M100_Dispo.csv
    20250313_M100_Lagerbew.csv
    20250313_M100_SiBe.csv
    20250313_M100_TeileWert.csv
    20250313_M100_Teilestamm.csv
    ... (other dated CSV files)
Spaltenbedeutung.xlsx   # Excel sheet describing columns
```

CSV files are semicolon separated and contain inventory and planning data.

## Pipeline and features

Each CSV file name starts with a date in ``YYYYMMDD`` format followed by the
dataset type and in some cases a part number.  The pipeline parses these
filenames, loads the tables and filters for warehouse location ``120`` via the
``Lagerort`` column.  Decimal numbers written with commas are converted to
standard floating point values.

From the movement history and planning tables a daily time series per part is
generated. Planned receipts (``Deckungsmenge``) increase the inventory while
planned demand (``Bedarfsmenge``) reduces it. The safety stock history
(``SiBeVerlauf``) is joined using the last known value up to each date.

Each produced feature file contains exactly these columns in the given order:

- ``Teil`` – part number
- ``Datum`` – date of the record
- ``EoD_Bestand`` – simulated end-of-day stock including planned movements
- ``Hinterlegter SiBe`` – safety stock active on that day
- ``EoD_Bestand_noSiBe`` – stock minus safety stock
- ``Flag_StockOut`` – ``1`` if ``EoD_Bestand_noSiBe`` <= 0
- ``LABLE_StockOut_MinAdd`` – minimal replenishment to avoid stock-out within
  lead time
- ``WBZ_Days`` – lead time from the part master data
- ``LABLE_SiBe_STD95`` – safety stock based on demand variance
- ``LABLE_SiBe_AvgMax`` – safety stock as difference of max and average demand
- ``LABLE_SiBe_Percentile`` – 90th percentile demand minus average
  demand
  
Additional columns ``DemandMean_*`` and ``DemandMax_*`` hold rolling
consumption statistics derived from ``EoD_Bestand_noSiBe``.

The resulting table contains one row per part and date and forms the input for
model training.

Processing all raw files can require substantial memory depending on the
number of dates included. When running the full dataset, ensure enough RAM
is available or process only a subset of files.

## Required software

- Python 3.11
- pandas
- NumPy
- scikit-learn
- Matplotlib
- pyarrow
- openpyxl (for Excel output)
- plotly (for interactive graphs)

Install the dependencies with pip:

```bash
python3.11 -m pip install pandas numpy scikit-learn matplotlib pyarrow openpyxl
```

### Local setup

Clone the repository and install the packages listed in ``requirements.txt``:

```bash
git clone <repository-url>
cd AGENTS_MAKE_ML
python3.11 -m pip install -r requirements.txt
```

All scripts assume Python 3.11 or later.

## Running the pipeline

To generate the feature table from the raw CSV files use the wrapper in
`scripts/`:

```bash
python3.11 scripts/build_features.py
```

Beim Aufruf werden die benötigten Pfade interaktiv abgefragt. Die Pipeline legt
für jedes Teil einen Unterordner unter ``Features/`` an und speichert dort
sowohl eine ``features.parquet`` als auch eine ``features.xlsx`` Datei.

## Training models

After preprocessing, train the model via:

```bash
python3.11 scripts/train.py
```

Beim Start werden Feature-Ordner, Teilnummer (oder ``ALL``) sowie eine
fortlaufende Modellnummer abgefragt. Anschließend können die wichtigsten
Hyperparameter des ``GradientBoostingRegressor`` interaktiv gesetzt werden
(``n_estimators``, ``learning_rate``, ``max_depth`` und ``subsample``). Wird bei
der Eingabe einfach ``Enter`` gedrückt, gelten die Standardwerte. Das
trainierte Modell landet unter ``Modelle/<Teil>/<Nummer>/`` zusammen mit
zusätzlichen Ausgaben wie der berechneten Feature-Importance.

## Evaluating the model

Once a model has been trained, run the evaluation script to compute metrics and
create diagnostic plots:

```bash
python3.11 scripts/evaluate.py
```

Auch hier werden die benötigten Pfade interaktiv abgefragt.

The script reports MAE, RMSE, R² and MAPE on a test split and writes several
plot files to the specified directory.  All graphs are also saved as HTML so
they can be opened in any browser:

- ``actual_vs_pred.png`` / ``.html`` – scatter plot of predicted versus actual values
- ``predictions_over_time.png`` / ``.html`` – comparison of predictions and actual values by date
- ``training_history.png`` / ``.html`` – model deviance over boosting iterations

Die ausgegebene Feature-Importance basiert auf einer Permutation Importance des
Test-Splits. Hohe Werte bedeuten, dass die jeweilige Spalte einen großen
Einfluss auf die Vorhersage hat.

## Known limitations

- Only rows with ``Lagerort`` 120 are processed.
- Date information is extracted from the filename and must follow the
  ``YYYYMMDD`` pattern.
- The raw exports need to be complete; missing tables for a date lead to sparse
  feature rows.
- Building the feature set for many dates can require several gigabytes of
  memory.  If resource limits are reached, process only a subset of the files.
