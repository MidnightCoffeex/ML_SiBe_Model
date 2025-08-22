#!/usr/bin/env python3
"""Evaluate trained models on historical or dispo features."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_pipeline, train_model


def _load_model(model_dir: Path, scope: str, model_type: str, model_id: str):
    base = model_dir / scope / model_type / model_id
    model = joblib.load(base / "model.joblib")
    with open(base / "feature_cols.json", "r", encoding="utf-8") as fh:
        info = json.load(fh)
    return model, info["feature_cols"], info.get("target")


def _evaluate_predictions(y_true: pd.Series, pred: pd.Series) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": mean_squared_error(y_true, pred, squared=False),
        "R2": r2_score(y_true, pred),
        "MAPE": mean_absolute_percentage_error(y_true, pred),
    }


def _save_plot(df: pd.DataFrame, target: str, pred_col: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["Datum"], df[target], label="actual")
    plt.plot(df["Datum"], df[pred_col], label="pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--mode", choices=["full", "dispo"], default="full")
    parser.add_argument("--features-h", help="Historical features directory")
    parser.add_argument("--features-d", help="Dispo features directory")
    parser.add_argument("--model-dir", default="Models")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--part", default="ALL")
    parser.add_argument("--plots", default="plots")
    args = parser.parse_args()

    scope = args.part if args.part != "ALL" else "ALL"
    model_dir = Path(args.model_dir)
    model, feature_cols, target = _load_model(model_dir, scope, args.model_type, args.model_id)

    def _load_feats(base: str | Path, part: str) -> pd.DataFrame:
        base_p = Path(base)
        if part == "ALL":
            frames = []
            for p in base_p.glob("*/features.csv"):
                frames.append(data_pipeline.safe_read_features(p.parent))
            return pd.concat(frames, ignore_index=True)
        return data_pipeline.safe_read_features(base_p / part)

    if args.mode == "full":
        df = _load_feats(args.features_h, args.part)
        if target is None:
            raise RuntimeError("feature_cols.json missing target info")
        df = df.dropna(subset=[target])
        X, y = train_model.split_X_y(df, [target])
        X = X[feature_cols]
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X)//50)))
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        preds = model.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx][target]
        metrics = _evaluate_predictions(y_true, pd.Series(preds, index=y_true.index))
        out_dir = Path(args.plots)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = df.iloc[test_idx].copy()
        res["pred"] = preds
        res.to_csv(out_dir / "predictions.csv", index=False)
        _save_plot(res, target, "pred", out_dir / "pred_vs_actual.png")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
    else:
        df = _load_feats(args.features_d, args.part)
        X = df.reindex(columns=feature_cols, fill_value=0)
        preds = model.predict(X)
        out = df[['Teil', 'Datum']].copy()
        out['prediction'] = preds
        out_dir = Path(args.plots)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "dispo_predictions.csv", index=False)


if __name__ == "__main__":
    main()
