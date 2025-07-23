#!/usr/bin/env python3
"""Run model evaluation and generate plots."""
from src import evaluate_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Features parquet file"
    )
    parser.add_argument(
        "--model", default="models/gb_regressor.joblib", help="Trained model file"
    )
    parser.add_argument(
        "--target", default="SiBe_Sicherheitsbest", help="Target column"
    )
    parser.add_argument(
        "--plots", default="plots", help="Directory to store plots"
    )
    args = parser.parse_args()
    evaluate_model.run_evaluation(args.data, args.model, args.target, args.plots)


if __name__ == "__main__":
    main()
