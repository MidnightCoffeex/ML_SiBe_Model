#!/usr/bin/env python3
"""Wrapper script for model training."""
from pathlib import Path
import sys
import pandas as pd

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
        "--targets",
        default="LABLE_SiBe_STD95,LABLE_SiBe_AvgMax,LABLE_SiBe_Percentile",
        help="Comma separated target column names",
    )
    args = parser.parse_args()

    if not args.data:
        args.data = input("Pfad zu Features [Features]: ") or "Features"
    if not args.part:
        args.part = input("Teil-Nummer oder ALL [ALL]: ") or "ALL"
    if not args.model_dir:
        args.model_dir = "Modelle"
    if not args.model_id:
        args.model_id = input("Laufende Nummer [1]: ") or "1"

    if args.part.upper() == "ALL":
        frames = []
        for f in Path(args.data).glob('*/features.parquet'):
            frames.append(pd.read_parquet(f))
        df = pd.concat(frames, ignore_index=True)
        part_name = "ALL"
    else:
        df = pd.read_parquet(Path(args.data) / args.part / 'features.parquet')
        part_name = args.part

    out_dir = Path(args.model_dir) / part_name / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model.joblib'
    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    train_model.run_training_df(df, str(model_path), target_list)


if __name__ == "__main__":
    main()
