#!/usr/bin/env python3
"""Train models using pre-computed historical features."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_model, data_pipeline


def _prompt(prompt: str, default: str | None = None) -> str:
    val = input(prompt)
    if not val and default is not None:
        return default
    return val


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the safety stock model")
    parser.add_argument("--features-dir", help="Verzeichnis mit H-Features")
    parser.add_argument("--part", help="Teil-Nummer oder ALL")
    parser.add_argument("--targets", help="Kommagetrennte Target-Spalten")
    parser.add_argument("--cv-splits", help="Anzahl CV-Splits oder auto")
    parser.add_argument("--model-dir", help="Ausgabeordner für Modelle")
    parser.add_argument("--models", help="Kommagetrennte Modelltypen (gb,xgb,lgbm)")
    parser.add_argument("--model-id", help="Laufende Nummer")
    args = parser.parse_args()

    if not args.features_dir:
        args.features_dir = _prompt("Pfad zu Features_H [Features_H]: ", "Features_H")
    if not args.part:
        args.part = _prompt("Teil-Nummer oder ALL [ALL]: ", "ALL")
    if not args.targets:
        args.targets = _prompt("Targets [LABLE_StockOut_MinAdd]: ", "LABLE_StockOut_MinAdd")
    if not args.cv_splits:
        args.cv_splits = _prompt("cv_splits [auto]: ", "auto")
    if not args.model_dir:
        args.model_dir = _prompt("Modelldir [Models]: ", "Models")
    if not args.models:
        args.models = _prompt("Modelle (gb,xgb,lgbm) [gb,xgb,lgbm]: ", "gb,xgb,lgbm")
    if not args.model_id:
        args.model_id = ""

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    model_types = [m.strip() for m in args.models.split(',') if m.strip()]

    part_name = args.part
    if args.part.upper() == "ALL":
        frames = []
        for f in Path(args.features_dir).glob('*/features.csv'):
            frames.append(data_pipeline.safe_read_features(f))
        df = __import__('pandas').concat(frames, ignore_index=True)
        part_name = "ALL"
    else:
        df = data_pipeline.safe_read_features(Path(args.features_dir) / args.part / 'features.csv')
        part_name = args.part

    X, _ = train_model.prepare_data(df, targets)

    if args.cv_splits.lower() == 'auto':
        n_splits = min(5, max(2, len(X) // 50))
    else:
        try:
            n_splits = int(args.cv_splits)
        except ValueError:
            n_splits = 2
    n_splits = max(2, min(n_splits, max(2, len(X) - 1)))
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    split_indices = (splits[-2][0], splits[-2][1], splits[-1][0], splits[-1][1])

    if not args.model_id:
        part_dir = Path(args.model_dir) / part_name / model_types[0]
        existing = [int(p.name) for p in part_dir.glob('*') if p.is_dir() and p.name.isdigit()]
        args.model_id = str(max(existing, default=0) + 1)
        print(f"Automatisch gewählte Nummer: {args.model_id}")

    for mtype in model_types:
        out_dir = Path(args.model_dir) / part_name / mtype / args.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / 'model.joblib'
        print(f"Trainiere {mtype} → {model_path}")
        train_model.run_training_df(
            df,
            str(model_path),
            targets,
            model_type=mtype,
            cv_splits=n_splits,
            split_indices=split_indices,
        )


if __name__ == "__main__":
    main()
