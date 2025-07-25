import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DECIMAL_REGEX = re.compile(r"^-?\d{1,3}(\.\d{3})*,\d+$")


def _convert_comma_decimal(series: pd.Series) -> pd.Series:
    """Convert comma decimal strings to floats."""
    if series.dtype != object:
        return series
    if not series.str.contains(',', na=False).any():
        return series
    # remove thousand separators '.' and replace decimal comma
    cleaned = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    try:
        return cleaned.astype(float)
    except ValueError:
        return series


def load_csv_file(path: str, part: str | None = None) -> pd.DataFrame:
    """Load a single CSV using project conventions."""
    df = pd.read_csv(path, encoding="ISO-8859-1", delimiter=";", dtype=str)
    df.columns = df.columns.str.strip()
    if 'Teil' not in df.columns and 'Teil ' in df.columns:
        df.rename(columns={'Teil ': 'Teil'}, inplace=True)
    if 'Teil' not in df.columns and part is not None:
        df['Teil'] = part
    if 'Lagerort' in df.columns:
        df = df[df['Lagerort'].astype(str).str.strip() == '120']
    for col in df.columns:
        df[col] = _convert_comma_decimal(df[col])
    if 'Teil' in df.columns and df['Teil'].duplicated().any():
        df = df.drop_duplicates(subset='Teil', keep='first')
    return df


def load_all_csvs(directory: str) -> List[Tuple[str, str, pd.DataFrame]]:
    """Load all CSV files from ``directory``.

    Returns a list of tuples ``(date, dataset, df)``.
    """
    records = []
    csv_files = sorted(Path(directory).glob('*.csv'))
    pattern = re.compile(r"(\d{8})_M100_(.*)\.csv$", re.IGNORECASE)
    for csv in csv_files:
        m = pattern.match(csv.name)
        if not m:
            continue
        date_str, rest = m.groups()
        date = pd.to_datetime(date_str, format='%Y%m%d')
        parts = rest.split('_')
        part = None
        if len(parts) > 1 and parts[0].isdigit():
            part = parts[0]
            dataset = '_'.join(parts[1:])
        else:
            dataset = rest
        df = load_csv_file(str(csv), part)
        df['Datum'] = date
        records.append((date_str, dataset, df))
    return records


def build_features(directory: str) -> pd.DataFrame:
    """Merge different tables by ``Teil`` and date."""
    records = load_all_csvs(directory)
    frames_by_date: Dict[str, Dict[str, pd.DataFrame]] = {}
    for date_str, dataset, df in records:
        frames_by_date.setdefault(date_str, {})[dataset] = df
    feature_frames = []
    for date_str, tables in frames_by_date.items():
        merged: pd.DataFrame | None = None
        for dataset, df in tables.items():
            if 'Teil' not in df.columns:
                continue
            df = df.set_index('Teil')
            df = df.add_prefix(f"{dataset}_")
            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how='outer')
        if merged is not None:
            merged['Datum'] = pd.to_datetime(date_str, format='%Y%m%d')
            feature_frames.append(merged.reset_index())
    if not feature_frames:
        return pd.DataFrame()
    features = pd.concat(feature_frames, ignore_index=True)
    features = features.sort_values(['Teil', 'Datum'])
    return features


def run_pipeline(raw_dir: str, output_path: str) -> None:
    """Run complete preprocessing pipeline and save features.

    The output format is determined by the file extension. ``.parquet`` (default)
    writes a Parquet file, ``.csv`` saves to CSV and ``.xlsx`` writes an Excel
    workbook.
    """
    features = build_features(raw_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = Path(output_path).suffix.lower()
    if ext == ".csv":
        features.to_csv(output_path, index=False)
    elif ext in {".xlsx", ".xls"}:
        features.to_excel(output_path, index=False)
    else:
        features.to_parquet(output_path, index=False)


def create_features(raw_dir: str, output_path: str = "data/features.parquet") -> None:
    """Convenience wrapper for ``run_pipeline``.

    This function mirrors the interface expected by command-line wrappers and
    simply executes :func:`run_pipeline` with the provided arguments.
    """
    run_pipeline(raw_dir, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process raw CSV data")
    parser.add_argument("--input", default="Rohdaten", help="Input directory with raw CSV files")
    parser.add_argument("--output", default="data/features.parquet", help="Output parquet file")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
