import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_features(path: str) -> pd.DataFrame:
    """Load pre-computed features from a parquet file."""
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature matrix ``X`` and target ``y`` with rows containing NaN
    in any target removed."""
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) {missing} not found in dataset")
    df = df.dropna(subset=targets)
    y = df[targets]
    X = df.drop(columns=targets + ["EoD_Bestand"])  # exclude EoD_Bestand from features
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y


def train_model(X: pd.DataFrame, y: pd.DataFrame) -> MultiOutputRegressor:
    """Train a multi-output Gradient Boosting regressor."""
    base = GradientBoostingRegressor(random_state=0)
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def run_training_df(df: pd.DataFrame, model_path: str, targets: list[str]) -> tuple[list[float], list[float]]:
    """Train a model from an already loaded DataFrame."""
    X, y = prepare_data(df, targets)
    if len(X) < 50:
        print(
            "Warning: very few training samples; results may be unreliable"
        )

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-2]
    train_full_idx, test_idx = splits[-1]

    model = train_model(X.iloc[train_idx], y.iloc[train_idx])
    val_pred = model.predict(X.iloc[val_idx])
    val_mae = mean_absolute_error(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], val_pred, multioutput="raw_values"))

    # refit on all data except test
    model = train_model(X.iloc[train_full_idx], y.iloc[train_full_idx])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    test_pred = model.predict(X.iloc[test_idx])
    test_mae = mean_absolute_error(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], test_pred, multioutput="raw_values"))
    print("Validation MAE:", val_mae)
    print("Test MAE:", test_mae)
    return test_mae.tolist(), test_rmse.tolist()


def run_training(features_path: str, model_path: str, targets: list[str]) -> tuple[list[float], list[float]]:
    df = load_features(features_path)
    return run_training_df(df, model_path, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety stock model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Path to features parquet"
    )
    parser.add_argument(
        "--output", default="models/gb_regressor.joblib", help="Output model file"
    )
    parser.add_argument(
        "--targets",
        default="SiBe_STD95,SiBe_AvgMax,SiBe_Percentile",
        help="Comma separated target column names",
    )
    args = parser.parse_args()
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    run_training(args.data, args.output, target_list)
