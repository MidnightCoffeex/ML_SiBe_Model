#!/usr/bin/env python3
"""Wrapper script for model training."""
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the safety stock model")
    parser.add_argument("--data", help="Root directory with feature folders")
    parser.add_argument("--part", help="Specific part number or 'ALL'")
    parser.add_argument("--model-dir", help="Base directory for models")
    parser.add_argument("--model-id", help="Run identifier")
    parser.add_argument(
        "--models",
        help="Comma separated model types (gb,xgb,lgbm) or 'ALL'",
    )
    parser.add_argument(
        "--targets",
        default="LABLE_StockOut_MinAdd",
        help="Comma separated target column names",
    )
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--cv", type=int, help="Number of cross-validation folds")
    args = parser.parse_args()

    if not args.data:
        args.data = input("Pfad zu Features [Features]: ") or "Features"
    if not args.part:
        args.part = input("Teil-Nummer oder ALL [ALL]: ") or "ALL"
    if not args.model_dir:
        args.model_dir = "Modelle"

    # Model type selection (interactive if not provided)
    if not args.models:
        entry = input("Modelltyp(e) [gb|xgb|lgbm|ALL] [gb]: ")
        args.models = entry.strip() if entry else "gb"

    # Hyperparameters with explanatory prompts
    if args.n_estimators is None:
        entry = input("n_estimators [100] – Anzahl Bäume/Iterationen (höher = genauer, aber langsamer/überfitting möglich): ")
        args.n_estimators = int(entry) if entry else 100
    if args.learning_rate is None:
        entry = input("learning_rate [0.1] – Schrittweite je Baum (kleiner = stabiler, aber mehr Bäume nötig): ")
        args.learning_rate = float(entry) if entry else 0.1
    if args.max_depth is None:
        entry = input("max_depth [3] – maximale Baumtiefe (höher = komplexer, Risiko Überanpassung): ")
        args.max_depth = int(entry) if entry else 3
    if args.subsample is None:
        entry = input("subsample [1.0] – Stichprobenanteil pro Baum (kleiner kann generalisieren, 1.0 = alle Daten): ")
        args.subsample = float(entry) if entry else 1.0
    if args.cv is None:
        entry = input("cv_splits [0] – Anzahl Zeitreihen-CV-Folds (0 = aus): ")
        args.cv = int(entry) if entry else 0

    part_name = args.part if args.part else "ALL"

    # Expand 'ALL' to all three
    if args.models.strip().upper() == 'ALL':
        model_types = ['gb', 'xgb', 'lgbm']
    else:
        model_types = [m.strip() for m in args.models.split(',') if m.strip()]

    if not args.model_id:
        part_dir = Path(args.model_dir) / part_name / model_types[0]
        existing = [int(p.name) for p in part_dir.glob('*') if p.is_dir() and p.name.isdigit()]
        next_id = max(existing, default=0) + 1
        args.model_id = str(next_id)
        print(f"Automatisch gewählte Nummer: {args.model_id}")

    if args.part.upper() == "ALL":
        frames = []
        for f in Path(args.data).glob('*/features.parquet'):
            frames.append(pd.read_parquet(f))
        df = pd.concat(frames, ignore_index=True)
        part_name = "ALL"
    else:
        df = pd.read_parquet(Path(args.data) / args.part / 'features.parquet')
        part_name = args.part

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    X, _ = train_model.prepare_data(df, target_list)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    split_indices = (splits[-2][0], splits[-2][1], splits[-1][0], splits[-1][1])

    for mtype in model_types:
        out_dir = Path(args.model_dir) / part_name / mtype / args.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        print(
            "Model-{} wird mit diesen Parametern trainiert: n_estimators={}, "
            "learning_rate={}, max_depth={}, subsample={}.".format(
                mtype,
                args.n_estimators,
                args.learning_rate,
                args.max_depth,
                args.subsample,
            )
        )
        print(f"Speichere Modell in {out_dir}")
        train_model.run_training_df(
            df,
            str(model_path),
            target_list,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            cv_splits=args.cv if args.cv and args.cv > 1 else None,
            model_type=mtype,
            split_indices=split_indices,
        )


if __name__ == "__main__":
    main()
