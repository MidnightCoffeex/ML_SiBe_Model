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

A preprocessing pipeline can be executed with:

```bash
python3.11 pipeline.py --input Rohdaten --output data/processed
```

This reads the raw CSV files and creates processed data in `data/processed/`.

## Training models

After preprocessing, run the training script:

```bash
python3.11 train_model.py --data data/processed
```

This trains the ML model using the processed dataset and saves outputs to `models/`.
