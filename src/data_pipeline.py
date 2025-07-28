import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


DECIMAL_REGEX = re.compile(r"^-?\d{1,3}(\.\d{3})*,\d+$")


###############################
# Helpers
###############################

def _convert_comma_decimal(series: pd.Series) -> pd.Series:
    """Convert comma decimals like ``1.234,5`` to floats."""
    if series.dtype != object:
        return series
    if not series.str.contains(',', na=False).any():
        return series
    cleaned = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    try:
        return cleaned.astype(float)
    except ValueError:
        return series


def _load_column_map(xlsx_path: str) -> Dict[str, List[str]]:
    """Return mapping of dataset name to required columns."""
    xl = pd.read_excel(xlsx_path)
    col_map: Dict[str, List[str]] = {}
    for _, row in xl.iterrows():
        base = re.sub(r'^\d{8}_M100_', '', str(row['Ursprungstabelle']))
        base = base.split('.csv')[0]
        if base == '1100831_SiBeVerlauf':
            base = 'SiBeVerlauf'
        col_map.setdefault(base, []).append(str(row['Spaltenname']).strip())
    return col_map


def load_csv_file(path: Path, part: str | None = None) -> pd.DataFrame:
    """Load a single CSV file and apply basic cleaning."""
    df = pd.read_csv(path, encoding='ISO-8859-1', delimiter=';', dtype=str)
    df.columns = df.columns.str.strip()
    if 'Teil' not in df.columns and 'Teil ' in df.columns:
        df.rename(columns={'Teil ': 'Teil'}, inplace=True)
    if 'Teil' not in df.columns and part is not None:
        df['Teil'] = part
    if 'Lagerort' in df.columns:
        df = df[df['Lagerort'].astype(str).str.strip() == '120']
    for col in df.columns:
        df[col] = _convert_comma_decimal(df[col])
    return df


###############################
# Loading raw tables
###############################

def load_all_tables(directory: str, column_map: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Load all CSV files below ``directory`` and return them grouped by dataset."""
    pattern = re.compile(r"(\d{8})_M100_(.*)\.csv$", re.IGNORECASE)
    grouped: Dict[str, List[pd.DataFrame]] = {}
    for csv in sorted(Path(directory).glob('*.csv')):
        m = pattern.match(csv.name)
        if not m:
            continue
        date_str, rest = m.groups()
        parts = rest.split('_')
        part = None
        if len(parts) > 1 and parts[0].isdigit():
            part = parts[0]
            dataset = '_'.join(parts[1:])
        else:
            dataset = rest
        dataset = dataset.split('.csv')[0]
        if dataset.startswith('Teile'):
            dataset = dataset.replace('Teile', '')
        if dataset not in column_map:
            continue  # ignore unrelated exports
        df = load_csv_file(csv, part)
        keep_cols = [c for c in column_map[dataset] if c in df.columns]
        if 'Teil' in df.columns and 'Teil' not in keep_cols:
            keep_cols.append('Teil')
        df = df[keep_cols]
        df['ExportDatum'] = pd.to_datetime(date_str, format='%Y%m%d')
        df['Dataset'] = dataset
        grouped.setdefault(dataset, []).append(df)
    out: Dict[str, pd.DataFrame] = {}
    for dataset, dfs in grouped.items():
        out[dataset] = pd.concat(dfs, ignore_index=True)
    return out


###############################
# Feature engineering per part
###############################

def _parse_date(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    for c in columns:
        if c in df.columns:
            return pd.to_datetime(df[c], errors='coerce', dayfirst=True)
    return pd.NaT


def _aggregate_dataset(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    df['Datum'] = _parse_date(df, date_columns)
    df = df.dropna(subset=['Datum'])
    num_cols = df.select_dtypes(include='number').columns
    agg = {c: 'sum' if c in num_cols else 'first' for c in df.columns if c not in {'Teil', 'Datum', 'Dataset', 'ExportDatum'}}
    grouped = df.groupby(['Teil', 'Datum'], as_index=False).agg(agg)
    return grouped


def build_features_by_part(raw_dir: str, xlsx_path: str = 'Spaltenbedeutung.xlsx') -> Dict[str, pd.DataFrame]:
    """Process all raw files and return a dict of part -> feature DataFrame."""
    column_map = _load_column_map(xlsx_path)
    tables = load_all_tables(raw_dir, column_map)

    processed: Dict[str, pd.DataFrame] = {}
    # Aggregate relevant datasets
    agg_tables: Dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if name == 'Lagerbew':
            agg_tables[name] = _aggregate_dataset(df, ['BuchDat'])
        elif name == 'Dispo':
            agg_tables[name] = _aggregate_dataset(df, ['Termin', 'Solltermin'])
        elif name == 'SiBe':
            agg_tables[name] = _aggregate_dataset(df, ['Laufzeit'])
        elif name == 'SiBeVerlauf':
            agg_tables[name] = _aggregate_dataset(df, ['AudEreignis-ZeitPkt'])
        elif name in {'Bestand', 'Teilestamm'}:
            # use export date as datum
            df = df.copy()
            df['Datum'] = df['ExportDatum']
            agg = {c: 'first' for c in df.columns if c not in {'Teil', 'Datum', 'Dataset', 'ExportDatum'}}
            agg_tables[name] = df.groupby(['Teil', 'Datum'], as_index=False).agg(agg)
        else:
            continue

    # Determine all parts
    parts: set[str] = set()
    for df in agg_tables.values():
        parts.update(df['Teil'].astype(str).unique())

    for part in sorted(parts):
        frames = []
        for name, df in agg_tables.items():
            part_df = df[df['Teil'].astype(str) == str(part)].copy()
            if part_df.empty:
                continue
            part_df = part_df.drop(columns=['Teil'])
            part_df = part_df.add_prefix(f"{name}_")
            part_df.rename(columns={f"{name}_Datum": 'Datum'}, inplace=True)
            frames.append(part_df)
        if not frames:
            continue
        merged = frames[0]
        for f in frames[1:]:
            merged = pd.merge(merged, f, on='Datum', how='outer')
        merged = merged.sort_values('Datum').reset_index(drop=True)
        merged['Teil'] = part
        # derived features
        merged['EoD_Bestand'] = merged.get('Lagerbew_Lagerbestand').fillna(merged.get('Bestand_Bestand'))
        merged['WBZ_Days'] = merged.get('Teilestamm_WBZ')
        merged['SiBe'] = merged.get('SiBe_Sicherheitsbest')
        merged['EoD_Bestand_noSiBe'] = merged['EoD_Bestand'] - merged['SiBe'].fillna(0)
        merged['Flag_StockOut'] = (merged['EoD_Bestand_noSiBe'] <= 0).astype(int)
        # normalize numeric columns
        num_cols = merged.select_dtypes(include='number').columns
        merged[num_cols] = merged[num_cols].fillna(0)
        processed[part] = merged
    return processed


def save_feature_folders(features: Dict[str, pd.DataFrame], output_dir: str = 'Features') -> None:
    """Write each part's features to ``output_dir/<part>/features.parquet``."""
    out_base = Path(output_dir)
    for part, df in features.items():
        part_dir = out_base / str(part)
        part_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(part_dir / 'features.parquet', index=False)


def run_pipeline(raw_dir: str, output_dir: str = 'Features') -> None:
    """Complete preprocessing pipeline producing one file per part."""
    features = build_features_by_part(raw_dir)
    save_feature_folders(features, output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process raw CSV data')
    parser.add_argument('--input', default='Rohdaten', help='Input directory with raw CSVs')
    parser.add_argument('--output', default='Features', help='Output directory for feature folders')
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
