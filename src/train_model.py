import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance


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


def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> MultiOutputRegressor:
    """Train a multi-output Gradient Boosting regressor."""
    base = GradientBoostingRegressor(
        random_state=0,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def run_training_df(
    df: pd.DataFrame,
    model_path: str,
    targets: list[str],
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_splits: int | None = None,
) -> tuple[list[float], list[float]]:
    """Train a model from an already loaded DataFrame."""
    X, y = prepare_data(df, targets)
    weights = np.ones(len(y))
    if 'LABLE_StockOut_MinAdd' in y.columns:
        weights[y['LABLE_StockOut_MinAdd'] > 0] = 5.0
    if len(X) < 50:
        print(
            "Warning: very few training samples; results may be unreliable"
        )

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-2]
    train_full_idx, test_idx = splits[-1]

    model = train_model(
        X.iloc[train_idx],
        y.iloc[train_idx],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        sample_weight=weights[train_idx],
    )
    val_pred = model.predict(X.iloc[val_idx])
    val_mae = mean_absolute_error(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_rmse = np.sqrt(
        mean_squared_error(y.iloc[val_idx], val_pred, multioutput="raw_values")
    )
    val_r2 = r2_score(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_mape = mean_absolute_percentage_error(y.iloc[val_idx], val_pred)

    # refit on all data except test
    model = train_model(
        X.iloc[train_full_idx],
        y.iloc[train_full_idx],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        sample_weight=weights[train_full_idx],
    )
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    test_pred = model.predict(X.iloc[test_idx])
    test_mae = mean_absolute_error(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_rmse = np.sqrt(
        mean_squared_error(y.iloc[test_idx], test_pred, multioutput="raw_values")
    )
    test_r2 = r2_score(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_mape = mean_absolute_percentage_error(y.iloc[test_idx], test_pred)
    print(
        "Validation:",
        "MAE",
        val_mae,
        "RMSE",
        val_rmse,
        "R2",
        val_r2,
        "MAPE",
        val_mape,
    )
    print(
        "Test:",
        "MAE",
        test_mae,
        "RMSE",
        test_rmse,
        "R2",
        test_r2,
        "MAPE",
        test_mape,
    )

    try:
        importances = []
        for i in range(len(targets)):
            r = permutation_importance(
                model.estimators_[i],
                X.iloc[test_idx],
                y.iloc[test_idx].iloc[:, i],
                n_repeats=5,
                random_state=0,
                scoring="neg_mean_absolute_error",
            )
            importances.append(r.importances_mean)
        mean_imp = np.mean(importances, axis=0)
        order = np.argsort(mean_imp)[::-1]
        print("Permutation Importance:")
        for idx in order:
            print(f"  {X.columns[idx]}: {mean_imp[idx]:.4f}")
    except Exception as exc:
        print("Permutation Importance konnte nicht berechnet werden:", exc)

    if cv_splits and cv_splits > 1:
        cv = TimeSeriesSplit(n_splits=cv_splits)
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []
        fold_mape: list[float] = []
        for i, (tr, val) in enumerate(cv.split(X), 1):
            m = train_model(
                X.iloc[tr],
                y.iloc[tr],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[tr],
            )
            pred = m.predict(X.iloc[val])
            fold_mae.append(mean_absolute_error(y.iloc[val], pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[val], pred)))
            fold_r2.append(r2_score(y.iloc[val], pred))
            fold_mape.append(mean_absolute_percentage_error(y.iloc[val], pred))
            print(
                f"Fold {i}: MAE {fold_mae[-1]:.3f} RMSE {fold_rmse[-1]:.3f} R2 {fold_r2[-1]:.3f} MAPE {fold_mape[-1]:.3f}"
            )
        print(
            "CV Mean:",
            "MAE",
            np.mean(fold_mae),
            "RMSE",
            np.mean(fold_rmse),
            "R2",
            np.mean(fold_r2),
            "MAPE",
            np.mean(fold_mape),
        )

    return test_mae.tolist(), test_rmse.tolist()


def run_training(
    features_path: str,
    model_path: str,
    targets: list[str],
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_splits: int | None = None,
) -> tuple[list[float], list[float]]:
    df = load_features(features_path)
    return run_training_df(
        df,
        model_path,
        targets,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        cv_splits=cv_splits,
    )


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
        default="LABLE_StockOut_MinAdd",
        help="Comma separated target column names",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--cv_splits", type=int, default=None)
    args = parser.parse_args()
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    run_training(
        args.data,
        args.output,
        target_list,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        cv_splits=args.cv_splits,
    )
