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

Each produced feature file contains key columns (excerpt):

- ``Teil`` / ``Datum``
- ``F_NiU_EoD_Bestand`` (display only)
- ``F_NiU_Hinterlegter SiBe`` (display only)
- ``EoD_Bestand_noSiBe``, ``Flag_StockOut``, ``WBZ_Days``
- ``L_NiU_StockOut_MinAdd`` (diagnostic)
- ``L_NiU_WBZ_BlockMinAbs`` (diagnostic)
- ``LABLE_HalfYear_Target`` (training target)
- ``DemandMean_*`` / ``DemandMax_*`` (rolling consumption)

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

All scripts assume Python 3.11 or later. For XGBoost and LightGBM support
additional packages are required:

```bash
python3.11 -m pip install xgboost lightgbm
```

## Step-by-step guide

1. **Features erzeugen**

   ```bash
   python3.11 scripts/build_features.py
   ```

   - Pfade zu ``Rohdaten`` und ``Features`` werden abgefragt.
   - Für jedes Teil entsteht ``Features/<Teil>/features.parquet`` und ``features.xlsx``.
   - Die Konsole zeigt den Fortschritt je verarbeiteter Datei.

2. **Modelle trainieren**

   ```bash
   python3.11 scripts/train.py --models gb,xgb,lgbm
   ```

   - Wähle Feature-Ordner, Teilnummer (``ALL`` für alle) und eine Modell-ID.
   - Es werden ``model.joblib``, ``metrics.csv`` und ``feature_importances.csv`` unter ``Modelle/<Teil>/<Modelltyp>/<ID>/`` gespeichert.
   - Die Ausgabe enthält Validierungs- und Testmetriken (MAE, RMSE, R², MAPE).

3. **Modelle auswerten**

   ```bash
   python3.11 scripts/evaluate.py --model-type gb --model-id 1
   ```

   - Gibt Features-Pfad, Teil, Modellverzeichnis und Zielordner für Plots an.
   - Ergebnisse landen unter ``New_Test_Plots/<Teil|ALL>/<Modelltyp>/<ID>/`` und umfassen ``*_predictions.csv``/``.xlsx`` sowie PNG/HTML-Grafiken.
   - Die Konsole meldet erneut MAE, RMSE, R² und MAPE.

## Running the pipeline

To generate the feature table from the raw CSV files use the wrapper in
`scripts/`:

```bash
python3.11 scripts/build_features.py
```

Beim Aufruf werden die benötigten Pfade interaktiv abgefragt. Die Pipeline legt
für jedes Teil einen Unterordner unter ``Features/`` an und speichert dort
sowohl eine ``features.parquet`` als auch eine ``features.xlsx`` Datei. Die
Parquet-Datei dient als direkte Eingabe für die Modelle, die Excel-Datei zur
manuellen Prüfung.

## Training models

After preprocessing, train one or more models via:

```bash
python3.11 scripts/train.py --models gb,xgb,lgbm
```

The script supports the scikit-learn Gradient Boosting regressor (``gb``) as
well as XGBoost (``xgb``) and LightGBM (``lgbm``). Multiple types can be
specified as a comma separated list and will be trained sequentially using the
same train/validation/test split. If no model type is given, ``gb`` is used for
backwards compatibility.

During the interactive prompts the feature directory, part number (or ``ALL``)
and a model identifier are requested. Hyperparameters such as
``n_estimators``, ``learning_rate``, ``max_depth`` and ``subsample`` can be
entered manually or left at their defaults. Results are stored under
``Modelle/<Teil|ALL>/<Modelltyp>/<Modellnummer>/`` containing the trained model,
metrics and feature importances. ``metrics.csv`` listet Kennzahlen wie MAE
(Mean Absolute Error), RMSE, R² und MAPE, während ``feature_importances.csv``
den Einfluss jeder Spalte auf die Vorhersage zeigt.

Training assigns a higher weight to rows with imminent stock-outs
(``LABLE_StockOut_MinAdd`` > 0). An optional time-series cross-validation can
be enabled via ``--cv`` to obtain more robust performance estimates.

## Evaluating the model

Once a model has been trained, run the evaluation script to compute metrics and
create diagnostic plots:

```bash
python3.11 scripts/evaluate.py --model-type gb --model-id 1
```

The evaluator infers the model type from the directory layout if not provided.
Plots and CSV exports are written to ``New_Test_Plots/<Teil|ALL>/<Modelltyp>/<Modellnummer>/``.
It reports MAE, RMSE, R² and MAPE on the test split and saves several graphs
both as PNG and HTML files:

- ``actual_vs_pred.png`` / ``.html`` – scatter plot of predicted versus actual values
- ``predictions_over_time.png`` / ``.html`` – comparison of predictions and actual values by date
- ``training_history.png`` / ``.html`` � model deviance over boosting iterations

``*_predictions.csv`` und ``*_predictions.xlsx`` enthalten die berechneten
Werte je Datum. MAE (Mean Absolute Error) misst die durchschnittliche Abweichung,
RMSE die quadratische Abweichung, R² den Anteil erklärter Varianz und MAPE die
prozentuale Abweichung.

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


## Naming & Exclusions (NiU)
- ``F_NiU_*``: feature/display not in use (never used as model input).
- ``L_NiU_*``: label/diagnostic not in use (helper columns; not inputs).
- Training target: ``LABLE_HalfYear_Target`` (semiannual constant target per window based on the max of ``L_NiU_WBZ_BlockMinAbs``).
- Train/Eval automatically exclude any columns with ``F_NiU_``/``L_NiU_`` (and legacy ``nF_``) from the feature matrix.

