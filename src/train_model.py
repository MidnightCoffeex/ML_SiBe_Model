import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def load_features(path: str) -> pd.DataFrame:
    """Load pre-computed features from a parquet file."""
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix ``X`` and target ``y`` with rows containing NaN
    in ``target`` removed."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    """Train a Gradient Boosting regressor."""
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X, y)
    return model


def run_training(features_path: str, model_path: str, target: str) -> None:
    df = load_features(features_path)
    X, y = prepare_data(df, target)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = train_model(X_train, y_train)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"Validation MAE: {mae:.3f}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety stock model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Path to features parquet"
    )
    parser.add_argument(
        "--output", default="models/gb_regressor.joblib", help="Output model file"
    )
    parser.add_argument(
        "--target", default="SiBe_Sicherheitsbest", help="Target column name"
    )
    args = parser.parse_args()
    run_training(args.data, args.output, args.target)
