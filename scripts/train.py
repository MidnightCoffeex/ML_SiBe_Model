#!/usr/bin/env python3
"""Train models on historical feature tables."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_model, data_pipeline


def _load_features(features_dir: Path, part: str) -> pd.DataFrame:
    if part == "ALL":
        frames = []
        for p in features_dir.glob("*/features.csv"):
            frames.append(data_pipeline.safe_read_features(p.parent))
        return pd.concat(frames, ignore_index=True)
    else:
        return data_pipeline.safe_read_features(features_dir / part)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--features-dir", required=True, help="Directory with historical features")
    parser.add_argument("--part", default="ALL", help="Part number or ALL")
    parser.add_argument("--targets", required=True, help="Comma separated target columns")
    parser.add_argument("--cv-splits", default="auto", help="auto or explicit integer")
    parser.add_argument("--model-dir", default="Models", help="Output directory for models")
    parser.add_argument("--models", default="gb,xgb,lgbm", help="Comma separated model types")
    parser.add_argument("--model-id", default="1", help="Model identifier")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    df = _load_features(features_dir, args.part)
    scope = args.part if args.part != "ALL" else "ALL"

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]

    for target in target_list:
        df_t = df.dropna(subset=[target]).copy()
        X, y = train_model.split_X_y(df_t, [target])
        feature_cols = X.columns.tolist()

        if args.cv_splits == "auto":
            n_splits = min(5, max(2, len(X) // 50))
        else:
            n_splits = int(args.cv_splits)
        if len(X) <= n_splits:
            n_splits = max(2, len(X) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        if len(splits) >= 2:
            split_indices = (*splits[-2], *splits[-1])
        else:
            tr, te = splits[-1]
            split_indices = (tr, te, tr, te)

        for mtype in model_types:
            out_dir = Path(args.model_dir) / scope / mtype / args.model_id
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / "model.joblib"
            train_model.run_training_df(
                df_t,
                str(model_path),
                [target],
                cv_splits=n_splits,
                model_type=mtype,
                split_indices=split_indices,
            )
            with open(out_dir / "feature_cols.json", "w", encoding="utf-8") as fh:
                json.dump({"feature_cols": feature_cols, "target": target}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
