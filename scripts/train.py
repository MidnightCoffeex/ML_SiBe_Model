#!/usr/bin/env python3
"""Wrapper script for model training."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the safety stock model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Path to features parquet"
    )
    parser.add_argument(
        "--output", default="models/gb_regressor.joblib", help="Output model file"
    )
    parser.add_argument(
        "--target", default="SiBe_Sicherheitsbest", help="Target column name"
    )
    args = parser.parse_args()
    train_model.run_training(args.data, args.output, args.target)


if __name__ == "__main__":
    main()
