import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

from .train_model import load_features, prepare_data


def run_evaluation(
    features_path: str, model_path: str, target: str, output_dir: str
) -> None:
    """Evaluate the trained model on a test split and generate plots."""
    df = load_features(features_path)
    X, y = prepare_data(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {mae:.3f}")

    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(output_dir) / "predictions.csv", index=False)
    try:
        results.to_excel(Path(output_dir) / "predictions.xlsx", index=False)
    except Exception:
        pass

    # Actual vs predicted scatter plot
    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Safety Stock")
    plt.ylabel("Predicted Safety Stock")
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
        test_df = df.loc[X_test.index].copy()
        test_df["predicted"] = y_pred
        test_df = test_df.sort_values("Datum")
        plt.figure()
        sns.lineplot(x="Datum", y=target, data=test_df, label="Actual")
        sns.lineplot(x="Datum", y="predicted", data=test_df, label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Safety Stock")
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
        "--target",
        default="Hinterlegter SiBe",
        help="Target column",
    )
    parser.add_argument("--plots", default="plots", help="Directory to store plots")
    args = parser.parse_args()

    run_evaluation(args.data, args.model, args.target, args.plots)
