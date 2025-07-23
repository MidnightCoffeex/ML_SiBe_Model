# AGENTS_MAKE_ML

This repository contains raw CSV files for a machine learning project.
The folder `Rohdaten/` holds the source data while `Spaltenbedeutung.xlsx` explains each column.

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

## Required software

- Python 3.11
- pandas
- NumPy
- scikit-learn
- Matplotlib

Install the dependencies with pip:

```bash
python3.11 -m pip install pandas numpy scikit-learn matplotlib
```

## Running the pipeline

To generate the feature table from the raw CSV files use the wrapper in
`scripts/`:

```bash
python3.11 scripts/build_features.py --input Rohdaten --output data/features.parquet
```

The command reads the raw CSV files and saves a merged feature file to
`data/features.parquet`.

## Training models

After preprocessing, train the model via:

```bash
python3.11 scripts/train.py --data data/features.parquet
```

This trains the ML model using the processed dataset and writes the resulting
model to `models/`.

## Evaluating the model

Once a model has been trained, run the evaluation script to compute metrics and
create diagnostic plots:

```bash
python3.11 scripts/evaluate.py --data data/features.parquet --model models/gb_regressor.joblib --plots plots
```
