import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


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
    """Load a single CSV file and apply basic cleaning.

    Some exports use a comma instead of a semicolon as delimiter. ``pandas`` can
    automatically detect the separator when ``sep=None`` and ``engine='python'``
    is used.  This ensures all files are parsed correctly.
    """
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=None, engine='python', dtype=str)
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
            cleaned = dataset.replace('Teile', '')
            if cleaned in column_map:
                dataset = cleaned
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
    """Parse the first available column in ``columns`` as datetime.

    Some source tables mix date-only and date-time strings. ``pandas`` will
    sometimes fail to infer the correct format when such mixtures occur.
    Using ``format='mixed'`` ensures each entry is parsed individually.
    """
    for c in columns:
        if c in df.columns:
            col = df[c].astype(str).str.strip()
            return pd.to_datetime(col, errors='coerce', dayfirst=True, format='mixed')
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
        elif name == 'SiBeVerlauf':
            agg_tables[name] = _aggregate_dataset(df, ['AudEreignis-ZeitPkt'])
        elif name in {'Bestand', 'Teilestamm'}:
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
        data: Dict[str, pd.DataFrame] = {}
        for name, df in agg_tables.items():
            part_df = df[df['Teil'].astype(str) == str(part)].copy()
            if not part_df.empty:
                data[name] = part_df
        if not data:
            continue

        # collect all relevant dates from Lagerbew and Dispo
        date_sets = []
        for name in ['Lagerbew', 'Dispo']:
            if name in data:
                date_sets.append(pd.to_datetime(data[name]['Datum']))
        if not date_sets:
            continue
        all_dates = pd.concat(date_sets).dropna().sort_values().unique()
        feat = pd.DataFrame({'Datum': all_dates})
        feat['Teil'] = part

        # baseline inventory
        if 'Lagerbew' in data:
            lb = data['Lagerbew'][['Datum', 'Lagerbestand']].copy()
            lb['Datum'] = pd.to_datetime(lb['Datum'])
            feat = pd.merge(feat, lb, on='Datum', how='left')
        else:
            feat['Lagerbestand'] = pd.NA

        if 'Bestand' in data:
            best = data['Bestand'][['Datum', 'Bestand']].copy()
            best['Datum'] = pd.to_datetime(best['Datum'])
            feat = pd.merge(feat, best, on='Datum', how='left')
            feat['Lagerbestand'] = feat['Lagerbestand'].combine_first(feat['Bestand'])
            feat.drop(columns=['Bestand'], inplace=True)

        feat.sort_values('Datum', inplace=True)
        feat['Lagerbestand'] = feat['Lagerbestand'].ffill().fillna(0)

        # planned movements from Dispo
        if 'Dispo' in data:
            dispo_raw = data['Dispo'][['Datum', 'Bedarfsmenge', 'Deckungsmenge']].copy()
            dispo_raw['Bedarfsmenge'] = pd.to_numeric(dispo_raw['Bedarfsmenge'], errors='coerce').fillna(0)
            dispo_raw['Deckungsmenge'] = pd.to_numeric(dispo_raw['Deckungsmenge'], errors='coerce').fillna(0)
            # keep daily demand for later feature engineering
            demand_daily = dispo_raw.groupby('Datum', as_index=False)['Bedarfsmenge'].sum()
            feat = pd.merge(feat, demand_daily, on='Datum', how='left')
            feat['Bedarfsmenge'] = feat['Bedarfsmenge'].fillna(0)

            dispo_raw['net'] = dispo_raw['Deckungsmenge'] - dispo_raw['Bedarfsmenge']
            dispo = dispo_raw.groupby('Datum', as_index=False)['net'].sum()
            feat = pd.merge(feat, dispo, on='Datum', how='left')
            feat['net'] = feat['net'].fillna(0)
            feat['cum_dispo'] = feat['net'].cumsum()
        else:
            feat['cum_dispo'] = 0
            feat['Bedarfsmenge'] = 0

        feat['EoD_Bestand'] = feat['Lagerbestand'] + feat['cum_dispo']
        feat['EoD_Bestand'] = pd.to_numeric(feat['EoD_Bestand'], errors='coerce').fillna(0)

        # safety stock history
        if 'SiBeVerlauf' in data:
            sibe = data['SiBeVerlauf'][['Datum', 'Im Sytem hinterlgeter SiBe']].copy()
            sibe['Datum'] = pd.to_datetime(sibe['Datum'])
            sibe = sibe.sort_values('Datum')
            sibe.rename(columns={'Im Sytem hinterlgeter SiBe': 'Hinterlegter SiBe'}, inplace=True)
            feat = pd.merge_asof(feat.sort_values('Datum'), sibe, on='Datum', direction='backward')
            feat['Hinterlegter SiBe'] = pd.to_numeric(feat['Hinterlegter SiBe'], errors='coerce').fillna(0)
        else:
            feat['Hinterlegter SiBe'] = 0

        feat['EoD_Bestand_noSiBe'] = feat['EoD_Bestand'] - feat['Hinterlegter SiBe']
        feat['Flag_StockOut'] = (feat['EoD_Bestand_noSiBe'] <= 0).astype(int)

        # Days until stock depletes when no additional supply arrives
        # Iterate backwards while keeping track of the next stock-out date
        next_so = pd.NaT
        horizon = (feat['Datum'].max() - feat['Datum'].min()).days + 1
        days_to_empty = np.empty(len(feat), dtype=float)
        for i in range(len(feat) - 1, -1, -1):
            current_date = feat.loc[i, 'Datum']
            if feat.loc[i, 'Flag_StockOut'] == 1:
                next_so = current_date
                days_to_empty[i] = 0
            else:
                if pd.isna(next_so):
                    days_to_empty[i] = horizon
                else:
                    days_to_empty[i] = (next_so - current_date).days
        feat['DaysToEmpty'] = days_to_empty

        # Inventory change over the last 7 days
        prev = feat[['Datum', 'EoD_Bestand']].copy()
        prev['Datum'] = prev['Datum'] + pd.Timedelta(days=7)
        prev.rename(columns={'EoD_Bestand': 'EoD_Bestand_prev7'}, inplace=True)
        feat = pd.merge(feat, prev, on='Datum', how='left')
        feat['BestandDelta_7T'] = feat['EoD_Bestand'] - feat['EoD_Bestand_prev7']
        feat['BestandDelta_7T'] = feat['BestandDelta_7T'].fillna(0)
        feat.drop(columns=['EoD_Bestand_prev7'], inplace=True)

        # WBZ from Teilestamm
        wbz = None
        if 'Teilestamm' in data:
            w = data['Teilestamm']['WBZ'].dropna()
            if not w.empty:
                wbz = float(pd.to_numeric(w.iloc[0], errors='coerce'))
        feat['WBZ_Days'] = wbz

        lead_time = int(wbz) if wbz and wbz > 0 else 1
        window = max(1, int(np.ceil(lead_time * 1.25)))

        # cumulative replenishment needed to avoid repeated stock-outs
        deficit = (-feat['EoD_Bestand_noSiBe']).clip(lower=0)
        future_deficit = deficit[::-1].rolling(window, min_periods=1).sum()[::-1]
        feat['LABLE_StockOut_MinAdd'] = future_deficit

        # ----- pseudo label calculation -----
        demand_series = feat['EoD_Bestand_noSiBe'].shift(1) - feat['EoD_Bestand_noSiBe']
        demand_series = demand_series.clip(lower=0).fillna(0)
        roll_mean = demand_series.rolling(lead_time, min_periods=1).mean()
        roll_var = demand_series.rolling(lead_time, min_periods=1).var().fillna(0)
        roll_max = demand_series.rolling(lead_time, min_periods=1).max()
        lt_demand = demand_series.rolling(lead_time, min_periods=1).sum()
        lt_p90 = lt_demand.expanding().quantile(0.9)

        feat['LABLE_SiBe_STD95'] = 1.65 * np.sqrt(lead_time * roll_var)
        feat['LABLE_SiBe_AvgMax'] = roll_max - roll_mean
        feat['LABLE_SiBe_Percentile'] = lt_p90 - roll_mean

        # ----- rolling time features -----
        for fac in [1.0, 2/3, 1/2, 1/4]:
            w = max(1, int(round(lead_time * fac)))
            key = str(int(fac*100)).rjust(2, '0')
            feat[f'DemandMean_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).mean()
            feat[f'DemandMax_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).max()

        feat = feat[
            [
                'Teil',
                'Datum',
                'EoD_Bestand',
                'Hinterlegter SiBe',
                'EoD_Bestand_noSiBe',
                'Flag_StockOut',
                'DaysToEmpty',
                'BestandDelta_7T',
                'WBZ_Days',
                'LABLE_StockOut_MinAdd',
                'LABLE_SiBe_STD95',
                'LABLE_SiBe_AvgMax',
                'LABLE_SiBe_Percentile',
            ]
            + [
                c
                for c in feat.columns
                if c.startswith('DemandMean_') or c.startswith('DemandMax_')
            ]
        ]

        processed[part] = feat

    return processed


def save_feature_folders(features: Dict[str, pd.DataFrame], output_dir: str = 'Features') -> None:
    """Write each part's features to ``output_dir/<part>``.

    Both ``features.parquet`` and ``features.xlsx`` are created so the
    tables can be used with tools that do not support Parquet.
    """
    out_base = Path(output_dir)
    for part, df in features.items():
        part_dir = out_base / str(part)
        part_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(part_dir / 'features.parquet', index=False)
        df.to_excel(part_dir / 'features.xlsx', index=False)


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
