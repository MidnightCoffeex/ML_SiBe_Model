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


def _compute_block_min_abs_label(df: pd.DataFrame, horizon_floor_days: int = 14) -> pd.Series:
    """Compute LABLE_WBZ_BlockMinAbs on-the-fly:
    For each date, horizon H = max(WBZ_Days, horizon_floor_days),
    label = max(0, -min(EoD_Bestand_noSiBe) over [t, t+H)).
    """
    if 'EoD_Bestand_noSiBe' not in df.columns or 'Datum' not in df.columns:
        raise ValueError("Columns 'Datum' and 'EoD_Bestand_noSiBe' are required")
    wbz = pd.to_numeric(df.get('WBZ_Days', 0), errors='coerce').fillna(0).astype(float)
    dates = pd.to_datetime(df['Datum'], errors='coerce')
    vals = pd.to_numeric(df['EoD_Bestand_noSiBe'], errors='coerce').fillna(0).to_numpy()
    H = np.maximum(wbz.to_numpy(), float(horizon_floor_days))
    out = np.zeros(len(df), dtype=float)
    darr = dates.to_numpy()
    for i in range(len(df)):
        h = int(max(1, round(H[i])))
        start = darr[i]
        if pd.isna(start):
            out[i] = 0.0
            continue
        end = start + np.timedelta64(h, 'D')
        mask = (darr >= start) & (darr < end)
        y = vals[mask]
        mmin = float(np.nanmin(y)) if y.size else 0.0
        out[i] = max(0.0, -mmin)
    return pd.Series(out, index=df.index, name='LABLE_WBZ_BlockMinAbs')


def _compute_halfyear_label_from_block(df: pd.DataFrame, block_col: str = 'L_NiU_WBZ_BlockMinAbs') -> pd.Series:
    """Compute the half-year windowed constant label from a per-date block-min-abs series.
    For each start date, pick the next date >= start+6 months as window end; set
    the label for all rows in that window to the maximum value of the block series
    within the window. Continue until the end of the series.
    """
    if 'Datum' not in df.columns:
        raise ValueError("'Datum' column required")
    if block_col not in df.columns:
        raise ValueError(f"'{block_col}' column required")
    sdf = df[['Datum', block_col]].copy()
    sdf['Datum'] = pd.to_datetime(sdf['Datum'], errors='coerce')
    sdf = sdf.sort_values('Datum').reset_index()
    dates = sdf['Datum'].tolist()
    vals = pd.to_numeric(sdf[block_col], errors='coerce').fillna(0).to_numpy()
    out = np.zeros(len(sdf), dtype=float)
    i = 0
    while i < len(sdf):
        start = dates[i]
        target = start + pd.DateOffset(months=6)
        j = i
        while j + 1 < len(sdf) and dates[j + 1] < target:
            j += 1
        if j + 1 < len(sdf) and dates[j] < target <= dates[j + 1]:
            j = j + 1
        window_max = float(np.nanmax(vals[i:j + 1])) if j >= i else float(vals[i])
        out[i:j + 1] = window_max
        i = j + 1
    result = pd.Series(out, index=sdf['index'])
    result = result.reindex(df.index).fillna(method='ffill').fillna(method='bfill')
    result.name = 'LABLE_HalfYear_Target'
    return result

def load_features(path: str) -> pd.DataFrame:
    """Load pre-computed features from a parquet file."""
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature matrix ``X`` and target ``y`` with rows containing NaN
    in any target removed. Excludes non-feature columns such as nF_* and
    legacy display columns from X."""
    # ensure HalfYear label exists if requested; compute from block-min-abs
    if 'LABLE_HalfYear_Target' in targets and 'LABLE_HalfYear_Target' not in df.columns:
        df = df.copy()
        # prefer using precomputed hidden block label; else compute
        block_col = 'L_NiU_WBZ_BlockMinAbs'
        if block_col not in df.columns:
            # try legacy names for compatibility
            if 'LABLE_WBZ_BlockMinAbs' in df.columns:
                block_col = 'LABLE_WBZ_BlockMinAbs'
            elif '_LABLE_WBZ_BlockMinAbs' in df.columns:
                block_col = '_LABLE_WBZ_BlockMinAbs'
            else:
                df['LABLE_WBZ_BlockMinAbs'] = _compute_block_min_abs_label(df)
                block_col = 'LABLE_WBZ_BlockMinAbs'
        df['LABLE_HalfYear_Target'] = _compute_halfyear_label_from_block(df, block_col)
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) {missing} not found in dataset")
    df = df.dropna(subset=targets)
    y = df[targets]
    drop_cols = set(targets)
    # Exclude display-only and legacy columns
    drop_cols.update([
        "EoD_Bestand",
        "Hinterlegter SiBe",
        "nF_EoD_Bestand",
        "nF_Hinterlegter SiBe",
    ])
    # Exclude any Not-in-Use columns
    drop_cols.update([c for c in df.columns if isinstance(c, str) and (c.startswith("F_NiU_") or c.startswith("L_NiU_") or c.startswith("nF_"))])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y


def train_gradient_boosting_model(
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


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> MultiOutputRegressor:
    """Train a multi-output XGBoost regressor."""
    from xgboost import XGBRegressor

    base = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        objective="reg:squarederror",
        random_state=0,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> MultiOutputRegressor:
    """Train a multi-output LightGBM regressor."""
    from lightgbm import LGBMRegressor

    base = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=0,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# Backwards compatibility
def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    **kwargs,
) -> MultiOutputRegressor:
    return train_gradient_boosting_model(X, y, **kwargs)


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
    model_type: str = "gb",
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    weight_scheme: str = "blockmin",
    weight_factor: float = 5.0,
) -> tuple[list[float], list[float]]:
    """Train a model from an already loaded DataFrame."""
    X, y = prepare_data(df, targets)
    # Default: alle Gewichte = 1
    weights = np.ones(len(y), dtype=float)
    # Optionale Gewichtung: Zeilen mit L_NiU_StockOut_MinAdd > 0 hÃ¶her gewichten
    # Wichtig: auf dieselben Zeilen indizieren wie X/y (Index beibehalten)
    if weight_scheme:
        scheme = (weight_scheme or "none").strip().lower()
        if scheme == 'blockmin' and 'L_NiU_StockOut_MinAdd' in df.columns:
            sw = pd.to_numeric(df['L_NiU_StockOut_MinAdd'], errors='coerce').fillna(0)
            try:
                sw = sw.loc[y.index]
                weights = np.where(sw.to_numpy() > 0, float(weight_factor), 1.0)
            except Exception:
                weights = np.ones(len(y), dtype=float)
        elif scheme == 'flag' and 'Flag_StockOut' in df.columns:
            fl = pd.to_numeric(df['Flag_StockOut'], errors='coerce').fillna(0)
            try:
                fl = fl.loc[y.index]
                weights = np.where(fl.to_numpy() == 1, float(weight_factor), 1.0)
            except Exception:
                weights = np.ones(len(y), dtype=float)
    if len(X) < 50:
        print("Warning: very few training samples; results may be unreliable")

    if split_indices is None:
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-2]
        train_full_idx, test_idx = splits[-1]
    else:
        train_idx, val_idx, train_full_idx, test_idx = split_indices

    if model_type == "gb":
        trainer = train_gradient_boosting_model
    elif model_type == "xgb":
        trainer = train_xgboost_model
    elif model_type == "lgbm":
        trainer = train_lightgbm_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = trainer(
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
    val_rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], val_pred, multioutput="raw_values"))
    val_r2 = r2_score(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_mape = mean_absolute_percentage_error(y.iloc[val_idx], val_pred)

    # refit on all data except test
    model = trainer(
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
    test_rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], test_pred, multioutput="raw_values"))
    test_r2 = r2_score(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_mape = mean_absolute_percentage_error(y.iloc[test_idx], test_pred)
    print(
        "Validation:",
        "MAE", val_mae,
        "RMSE", val_rmse,
        "R2", val_r2,
        "MAPE", val_mape,
    )
    print(
        "Test:",
        "MAE", test_mae,
        "RMSE", test_rmse,
        "R2", test_r2,
        "MAPE", test_mape,
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
        fi_df = pd.DataFrame({"feature": X.columns[order], "importance": mean_imp[order]})
        print("Permutation Importance:")
        for _, row in fi_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        fi_df.to_csv(Path(model_path).with_name("feature_importances.csv"), index=False)
    except Exception as exc:
        print("Permutation Importance konnte nicht berechnet werden:", exc)

    metrics_df = pd.DataFrame(
        {
            "target": targets,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "val_mape": val_mape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mape": test_mape,
        }
    )
    metrics_df.to_csv(Path(model_path).with_name("metrics.csv"), index=False)

    if cv_splits and cv_splits > 1:
        cv = TimeSeriesSplit(n_splits=cv_splits)
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []
        fold_mape: list[float] = []
        for i, (tr, val) in enumerate(cv.split(X), 1):
            m = trainer(
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
            "MAE", np.mean(fold_mae),
            "RMSE", np.mean(fold_rmse),
            "R2", np.mean(fold_r2),
            "MAPE", np.mean(fold_mape),
        )

    return test_mae.tolist(), test_rmse.tolist()

def run_training(
    features_path: str,
    model_dir: str,
    targets: list[str],
    model_types: list[str],
    *,
    model_id: str = "1",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_splits: int | None = None,
    weight_scheme: str = "blockmin",
    weight_factor: float = 5.0,
) -> None:
    df = load_features(features_path)
    X, _ = prepare_data(df, targets)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    split_indices = (*splits[-2], *splits[-1])  # train, val, train_full, test

    for mtype in model_types:
        model_path = Path(model_dir) / mtype / model_id / "model.joblib"
        run_training_df(
            df,
            str(model_path),
            targets,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            cv_splits=cv_splits,
            model_type=mtype,
            split_indices=split_indices,
            weight_scheme=weight_scheme,
            weight_factor=weight_factor,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety stock model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Path to features parquet"
    )
    parser.add_argument(
        "--model-dir", default="models", help="Directory to store models"
    )
    parser.add_argument("--model-id", default="1", help="Model identifier")
    parser.add_argument(
        "--models", default="gb", help="Comma separated model types (gb,xgb,lgbm)"
    )
    parser.add_argument(
        "--targets",
        default="LABLE_HalfYear_Target",
        help="Comma separated target column names",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--cv_splits", type=int, default=None)
    parser.add_argument("--weight_scheme", default="blockmin", help="Weighting: none|blockmin|flag")
    parser.add_argument("--weight_factor", type=float, default=5.0, help="Weight factor for selected scheme")
    args = parser.parse_args()
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    run_training(
        args.data,
        args.model_dir,
        target_list,
        model_types,
        model_id=args.model_id,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        cv_splits=args.cv_splits,
        weight_scheme=args.weight_scheme,
        weight_factor=args.weight_factor,
    )

