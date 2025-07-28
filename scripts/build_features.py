#!/usr/bin/env python3
"""Command-line wrapper for feature engineering."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_pipeline


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate feature tables from raw CSVs")
    parser.add_argument("--input", help="Directory with raw CSV files")
    parser.add_argument("--output", help="Output directory for features")
    args = parser.parse_args()

    if not args.input:
        args.input = input("Pfad zu Rohdaten [Rohdaten]: ") or "Rohdaten"
    if not args.output:
        args.output = input("Ausgabeordner für Features [Features]: ") or "Features"

    data_pipeline.run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
