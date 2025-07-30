import argparse
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

from .train_model import load_features, prepare_data


def _dispo_date_range(raw_dir: str, part: str) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return min and max date for the given part in the Dispo CSV exports."""
    dates: list[pd.Series] = []
    for csv in Path(raw_dir).glob("*_Dispo.csv"):
        try:
            df = pd.read_csv(csv, encoding="ISO-8859-1", sep=None, engine="python", dtype=str)
        except Exception:
            continue
        if "Teil " in df.columns and "Teil" not in df.columns:
            df.rename(columns={"Teil ": "Teil"}, inplace=True)
        if "Teil" not in df.columns:
            continue
        df = df[df["Teil"].astype(str).str.strip() == str(part)]
        if df.empty:
            continue
        for col in ["Termin", "Solltermin"]:
            if col in df.columns:
                d = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce", dayfirst=True)
                dates.append(d)
    if dates:
        all_d = pd.concat(dates).dropna()
        if not all_d.empty:
            return all_d.min(), all_d.max()
    return None, None


def _evaluate_range(
    results: pd.DataFrame, prefix: str, target: str, output_dir: str
) -> None:
    """Save predictions and plots for a given subset."""
    if results.empty:
        return
    results = results.sort_values("Datum")
    pred_col = f"pred_{target}"

    mae = mean_absolute_error(results["Hinterlegter SiBe"], results[pred_col])
    rmse = np.sqrt(mean_squared_error(results["Hinterlegter SiBe"], results[pred_col]))
    r2 = r2_score(results["Hinterlegter SiBe"], results[pred_col])
    mape = mean_absolute_percentage_error(results["Hinterlegter SiBe"], results[pred_col])
    print(
        f"{prefix} MAE vs Hinterlegter SiBe: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | MAPE: {mape:.3f}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
    try:
        results.to_excel(Path(output_dir) / f"{prefix}_predictions.xlsx", index=False)
    except Exception:
        pass

    plt.figure()
    sns.scatterplot(x=results["Hinterlegter SiBe"], y=results[pred_col])
    plt.xlabel("Actual Hinterlegter SiBe")
    plt.ylabel(f"Predicted {target}")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_actual_vs_pred.png")

    fig = px.scatter(
        results,
        x="Hinterlegter SiBe",
        y=pred_col,
        labels={"Hinterlegter SiBe": "Actual Hinterlegter SiBe", pred_col: f"Predicted {target}"},
        title="Actual vs Predicted",
    )
    fig.write_html(Path(output_dir) / f"{prefix}_actual_vs_pred.html")

    plt.figure()
    sns.lineplot(x="Datum", y="Hinterlegter SiBe", data=results, label="Actual")
    sns.lineplot(x="Datum", y=pred_col, data=results, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.title("Predictions Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_predictions_over_time.png")

    fig2 = px.line(
        results,
        x="Datum",
        y=["Hinterlegter SiBe", pred_col],
        labels={"value": target, "variable": "Serie"},
        title="Predictions Over Time",
    )
    fig2.write_html(Path(output_dir) / f"{prefix}_predictions_over_time.html")


def run_evaluation(
    features_path: str,
    model_path: str,
    targets: list[str],
    output_dir: str,
    raw_dir: str = "Rohdaten",
) -> None:
    """Evaluate the trained model and generate plots for multiple time frames."""
    df = load_features(features_path)
    X, y = prepare_data(df, targets)
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-2]
    train_full_idx, test_idx = splits[-1]
    model = joblib.load(model_path)

    y_pred_test = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], y_pred_test, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred_test, multioutput="raw_values"))
    r2 = r2_score(y.iloc[test_idx], y_pred_test, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y.iloc[test_idx], y_pred_test)
    print("Test Metrics -> MAE:", mae, "RMSE:", rmse, "R2:", r2, "MAPE:", mape)

    # predictions for the entire feature set
    X_full = df.drop(columns=targets + ["EoD_Bestand"], errors="ignore")
    X_full = X_full.select_dtypes(include=["number"]).fillna(0)
    full_pred = model.predict(X_full)
    results_full = df.copy()
    for i, col in enumerate(targets):
        results_full[f"pred_{col}"] = full_pred[:, i]

    part = str(df["Teil"].iloc[0]) if "Teil" in df.columns else ""
    dispo_start, dispo_end = _dispo_date_range(raw_dir, part)

    # Evaluation for specified ranges
    full_prefix = "Full_Time"
    _evaluate_range(results_full, full_prefix, targets[0], output_dir)

    if dispo_start is not None and dispo_end is not None:
        mask = (results_full["Datum"] >= dispo_start) & (results_full["Datum"] <= dispo_end)
        dispo_results = results_full.loc[mask].copy()
        _evaluate_range(dispo_results, "Dispo_Time", targets[0], output_dir)

    # Training history from model
    if hasattr(model, "train_score_"):
        plt.figure()
        sns.lineplot(x=range(1, len(model.train_score_) + 1), y=model.train_score_)
        plt.xlabel("Iteration")
        plt.ylabel("Deviance")
        plt.title("Training History")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_history.png")

        fig_hist = px.line(
            x=list(range(1, len(model.train_score_) + 1)),
            y=model.train_score_,
            labels={"x": "Iteration", "y": "Deviance"},
            title="Training History",
        )
        fig_hist.write_html(Path(output_dir) / "training_history.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Features parquet file"
    )
    parser.add_argument(
        "--model", default="models/gb_regressor.joblib", help="Trained model file"
    )
    parser.add_argument(
        "--targets",
        default="LABLE_SiBe_STD95,LABLE_SiBe_AvgMax,LABLE_SiBe_Percentile",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", default="plots", help="Directory to store plots")
    parser.add_argument("--raw", default="Rohdaten", help="Directory with raw CSV files")
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    run_evaluation(args.data, args.model, target_list, args.plots, args.raw)
