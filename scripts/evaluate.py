#!/usr/bin/env python3
"""Run model evaluation and generate plots."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import evaluate_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--data", help="Features parquet file")
    parser.add_argument("--model", help="Trained model file")
    parser.add_argument(
        "--targets",
        default="SiBe_STD95,SiBe_AvgMax,SiBe_Percentile",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", help="Directory to store plots")
    args = parser.parse_args()

    if not args.data:
        args.data = input("Pfad zu Feature-Datei: ")
    if not args.model:
        args.model = input("Pfad zum Modell: ")
    if not args.plots:
        args.plots = input("Ordner für Plots [plots]: ") or "plots"

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    evaluate_model.run_evaluation(args.data, args.model, target_list, args.plots)


if __name__ == "__main__":
    main()
