import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

from .train_model import load_features, prepare_data


def run_evaluation(
    features_path: str, model_path: str, targets: list[str], output_dir: str
) -> None:
    """Evaluate the trained model on a test split and generate plots."""
    df = load_features(features_path)
    X, y = prepare_data(df, targets)
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-2]
    train_full_idx, test_idx = splits[-1]
    model = joblib.load(model_path)

    y_pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred, multioutput="raw_values"))
    print("Test MAE:", mae)
    print("Test RMSE:", rmse)

    results = pd.DataFrame(y.iloc[test_idx].copy())
    for i, col in enumerate(targets):
        results[f"pred_{col}"] = y_pred[:, i]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(output_dir) / "predictions.csv", index=False)
    try:
        results.to_excel(Path(output_dir) / "predictions.xlsx", index=False)
    except Exception:
        pass

    # Actual vs predicted scatter plot
    plt.figure()
    sns.scatterplot(x=y.iloc[test_idx][targets[0]], y=y_pred[:, 0])
    plt.xlabel(f"Actual {targets[0]}")
    plt.ylabel(f"Predicted {targets[0]}")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "actual_vs_pred.png")

    # Training history from model
    if hasattr(model, "train_score_"):
        plt.figure()
        sns.lineplot(x=range(1, len(model.train_score_) + 1), y=model.train_score_)
        plt.xlabel("Iteration")
        plt.ylabel("Deviance")
        plt.title("Training History")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_history.png")

    # Predicted vs actual over time
    if "Datum" in df.columns:
        test_df = df.loc[y.iloc[test_idx].index].copy()
        test_df["predicted"] = y_pred[:, 0]
        test_df = test_df.sort_values("Datum")
        plt.figure()
        sns.lineplot(x="Datum", y=targets[0], data=test_df, label="Actual")
        sns.lineplot(x="Datum", y="predicted", data=test_df, label="Predicted")
        plt.xlabel("Date")
        plt.ylabel(targets[0])
        plt.title("Predictions Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "predictions_over_time.png")


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
        default="SiBe_STD95,SiBe_AvgMax,SiBe_Percentile",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", default="plots", help="Directory to store plots")
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    run_evaluation(args.data, args.model, target_list, args.plots)
