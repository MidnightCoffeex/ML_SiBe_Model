#!/usr/bin/env python3
# Training-Skript für ein einfaches neuronales Regressionsmodell (MLP) auf Basis der bestehenden Feature-Dateien.
# Es sammelt Eingaben interaktiv, lädt die gewünschten Features, trainiert ein Modell und speichert es im NN-spezifischen Modellordner.

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import nn_data, nn_models, train_model


def prompt(text: str, default: str | None = None) -> str:
    """Einfacher Prompt mit Default-Wert."""
    if default is None:
        return input(f"{text}: ").strip()
    value = input(f"{text} [{default}]: ").strip()
    return value or default


def main() -> None:
    print("=== Neuronales Training (MLP) ===")
    features_root = Path(prompt("Pfad zu Features", "CODEX_Features_2"))
    part_selection = prompt("Teil-Nummer oder ALL", "ALL")
    model_root = Path(prompt("Ausgabe-Ordner für NN-Modelle", "NN_Modelle"))
    targets_raw = prompt("Target-Label(s) (Komma)", "L_WBZ_BlockMinAbs")
    target_labels = [t.strip() for t in targets_raw.split(",") if t.strip()]

    feature_input = prompt("Feature-Liste (Komma, leer = automatisch)", "")
    selected_features = (
        [f.strip() for f in feature_input.split(",") if f.strip()]
        if feature_input
        else None
    )

    hidden_layers = nn_models.build_hidden_layers(prompt("Hidden-Layer (z.B. 128,64)", "128,64"))
    activation = prompt("Aktivierung (identity|logistic|tanh|relu)", "relu")
    alpha = float(prompt("L2-Regularisierung alpha", "0.0001"))
    learning_rate = float(prompt("Learning Rate", "0.001"))
    max_iter = int(prompt("Max. Iterationen", "200"))

    df_all, parts = nn_data.collect_dataframe(features_root, part_selection)
    print(f"Eingelesene Teile: {', '.join(parts)} (n={len(df_all)})")

    X_df, y_df = train_model.prepare_data(
        df_all,
        target_labels,
        selected_features=selected_features,
    )
    selected_cols = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    y = y_df.to_numpy(dtype=float)

    if len(X) < 5:
        raise RuntimeError("Zu wenige Datenpunkte für das Training (<5).")

    if len(X) >= 200:
        n_splits = 5
    elif len(X) >= 60:
        n_splits = 4
    elif len(X) >= 20:
        n_splits = 3
    else:
        n_splits = 2

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    base_reg = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=0,
        shuffle=False,
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(base_reg)),
        ]
    )
    pipeline.fit(X[train_idx], y[train_idx])
    y_pred = pipeline.predict(X[test_idx])

    mae = mean_absolute_error(y[test_idx], y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred, multioutput="raw_values"))
    r2 = r2_score(y[test_idx], y_pred, multioutput="raw_values")
    print("Validierungs-MAE:", mae)
    print("Validierungs-RMSE:", rmse)
    print("Validierungs-R2:", r2)

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(base_reg)),
        ]
    )
    final_pipeline.fit(X, y)

    scope_name = nn_models.determine_scope_name(part_selection, parts)
    run_dir = nn_models.choose_run_directory(model_root / scope_name)

    model_path = run_dir / "model.joblib"
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "scope": scope_name,
        "parts": parts,
        "targets": target_labels,
        "selected_features": selected_cols,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "n_samples": len(X),
        "n_features": len(selected_cols),
        "validation_mae": mae.tolist() if hasattr(mae, "tolist") else float(mae),
        "validation_rmse": rmse.tolist() if hasattr(rmse, "tolist") else float(rmse),
        "validation_r2": r2.tolist() if hasattr(r2, "tolist") else float(r2),
        "features_root": str(features_root),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Modell gespeichert in {run_dir}")
    print("Training abgeschlossen.")


if __name__ == "__main__":
    main()
