#!/usr/bin/env python3
"""Command-line wrapper for feature engineering."""
from src import data_pipeline


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate feature table from raw CSVs")
    parser.add_argument("--input", default="Rohdaten", help="Directory with raw CSV files")
    parser.add_argument(
        "--output", default="data/features.parquet", help="Output parquet file"
    )
    args = parser.parse_args()
    data_pipeline.create_features(args.input, args.output)


if __name__ == "__main__":
    main()
