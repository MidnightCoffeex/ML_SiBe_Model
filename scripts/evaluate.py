#!/usr/bin/env python3
"""Run model evaluation and generate plots."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import evaluate_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--features", help="Root directory with feature folders")
    parser.add_argument("--part", help="Teil-Nummer oder ALL")
    parser.add_argument("--model-dir", help="Directory containing trained models")
    parser.add_argument("--model-id", help="Numerical model identifier")
    parser.add_argument("--raw", help="Directory with raw CSV files")
    parser.add_argument(
        "--targets",
        default="LABLE_SiBe_STD95,LABLE_SiBe_AvgMax,LABLE_SiBe_Percentile",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", help="Directory to store evaluation results")
    args = parser.parse_args()

    if not args.features:
        args.features = input("Pfad zu Features [Features]: ") or "Features"
    if not args.part:
        args.part = input("Teil-Nummer oder ALL [ALL]: ") or "ALL"
    if not args.model_dir:
        args.model_dir = "Modelle"
    if not args.raw:
        args.raw = "Rohdaten"

    eval_part = args.part
    if args.part.upper() == "ALL":
        eval_part = input("Teilnummer für Auswertung: ")

    if not args.model_id:
        mdir = Path(args.model_dir) / args.part
        existing = [p.name for p in mdir.glob('*') if p.is_dir() and p.name.isdigit()]
        if existing:
            print("Verfügbare Modelle:", ", ".join(sorted(existing)))
        args.model_id = input("Modellnummer: ")

    features_path = Path(args.features) / eval_part / 'features.parquet'
    model_path = Path(args.model_dir) / args.part / args.model_id / 'model.joblib'
    if not args.plots:
        args.plots = input("Ordner für Ergebnisse [plots]: ") or "plots"

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    evaluate_model.run_evaluation(str(features_path), str(model_path), target_list, args.plots, args.raw)


if __name__ == "__main__":
    main()
