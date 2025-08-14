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
        loc = pd.to_numeric(df['Lagerort'].astype(str).str.strip(), errors='coerce')
        df = df[loc == 120]
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
    """Parse date columns with priority.

    Columns are parsed in the given order (e.g. ``Termin`` before
    ``Solltermin``).  Per row the first valid date is chosen.  This implements
    the rule *"Termin bevorzugen, sonst der frühere von beiden"*.
    """
    parsed: List[pd.Series] = []
    for c in columns:
        if c in df.columns:
            col = df[c].astype(str).str.strip()
            parsed.append(pd.to_datetime(col, errors='coerce', dayfirst=True, format='mixed'))
    if not parsed:
        return pd.Series(pd.NaT, index=df.index)
    result = parsed[0]
    for add in parsed[1:]:
        result = result.combine_first(add)
    return result


def _aggregate_dataset(
    df: pd.DataFrame,
    date_columns: List[str],
    last_cols: set[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df['Datum_raw'] = _parse_date(df, date_columns)
    df = df.dropna(subset=['Datum_raw'])
    df.sort_values('Datum_raw', inplace=True)
    df['Datum'] = df['Datum_raw'].dt.floor('D')
    num_cols = df.select_dtypes(include='number').columns
    agg: Dict[str, str] = {}
    for c in df.columns:
        if c in {'Teil', 'Datum', 'Datum_raw', 'Dataset', 'ExportDatum'}:
            continue
        if last_cols and c in last_cols:
            agg[c] = 'last'
        elif c in num_cols:
            agg[c] = 'sum'
        else:
            agg[c] = 'first'
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
            agg_tables[name] = _aggregate_dataset(df, ['BuchDat'], last_cols={'Lagerbestand'})
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

        # extract per-table subsets
        lb = data.get('Lagerbew', pd.DataFrame(columns=['Datum', 'Lagerbestand']))[
            ['Datum', 'Lagerbestand']
        ]
        best = data.get('Bestand', pd.DataFrame(columns=['Datum', 'Bestand']))[
            ['Datum', 'Bestand']
        ]
        dispo = data.get(
            'Dispo', pd.DataFrame(columns=['Datum', 'Bedarfsmenge', 'Deckungsmenge'])
        )[['Datum', 'Bedarfsmenge', 'Deckungsmenge']]

        for df_part in [lb, best, dispo]:
            if not df_part.empty:
                df_part['Datum'] = pd.to_datetime(df_part['Datum'])

        if not dispo.empty:
            dispo['Bedarfsmenge'] = pd.to_numeric(
                dispo['Bedarfsmenge'], errors='coerce'
            ).fillna(0)
            dispo['Deckungsmenge'] = pd.to_numeric(
                dispo['Deckungsmenge'], errors='coerce'
            ).fillna(0)
            # net = Deckungsmenge - Bedarfsmenge (positive = supply, negative = demand)
            dispo['net'] = dispo['Deckungsmenge'] - dispo['Bedarfsmenge']

        first_dispo_date = dispo['Datum'].min() if not dispo.empty else None
        baseline = 0.0
        baseline_date = None
        if first_dispo_date is not None:
            if not lb.empty:
                lb_before = lb[lb['Datum'] <= first_dispo_date]
                if not lb_before.empty:
                    row = lb_before.sort_values('Datum').iloc[-1]
                    baseline = float(pd.to_numeric(row['Lagerbestand'], errors='coerce'))
                    baseline_date = row['Datum']
            if baseline_date is None and not best.empty:
                best_before = best[best['Datum'] <= first_dispo_date]
                if not best_before.empty:
                    row = best_before.sort_values('Datum').iloc[-1]
                    baseline = float(pd.to_numeric(row['Bestand'], errors='coerce'))
                    baseline_date = row['Datum']
        else:
            if not lb.empty:
                row = lb.sort_values('Datum').iloc[-1]
                baseline = float(pd.to_numeric(row['Lagerbestand'], errors='coerce'))
                baseline_date = row['Datum']
            elif not best.empty:
                row = best.sort_values('Datum').iloc[-1]
                baseline = float(pd.to_numeric(row['Bestand'], errors='coerce'))
                baseline_date = row['Datum']

        # date skeleton
        date_set: set[pd.Timestamp] = set()
        if not lb.empty:
            date_set.update(lb['Datum'])
        if not dispo.empty:
            date_set.update(dispo['Datum'])
        if baseline_date is not None:
            date_set.add(baseline_date)
        if not date_set:
            continue
        feat = pd.DataFrame({'Datum': sorted(date_set)})
        feat['Teil'] = part

        # merge demand and net values
        if not dispo.empty:
            dispo_agg = dispo[['Datum', 'Bedarfsmenge', 'net']].rename(
                columns={'Bedarfsmenge': 'Bedarfsmenge_daily'}
            )
            feat = pd.merge(feat, dispo_agg, on='Datum', how='left')
            feat['Bedarfsmenge_daily'] = feat['Bedarfsmenge_daily'].fillna(0)
            feat['net'] = feat['net'].fillna(0)
            dispo_sorted = dispo.sort_values('Datum').copy()
            dispo_sorted['cum_net'] = dispo_sorted['net'].cumsum()
            feat = pd.merge(feat, dispo_sorted[['Datum', 'cum_net']], on='Datum', how='left')
            feat['cum_net'] = feat['cum_net'].fillna(method='ffill').fillna(0)
        else:
            feat['Bedarfsmenge_daily'] = 0
            feat['net'] = 0
            feat['cum_net'] = 0

        # merge Lagerbestand and apply baseline if needed
        if not lb.empty:
            feat = pd.merge(feat, lb, on='Datum', how='left')
        else:
            feat['Lagerbestand'] = np.nan
        if baseline_date is not None:
            mask = feat['Datum'] == baseline_date
            feat.loc[mask, 'Lagerbestand'] = feat.loc[mask, 'Lagerbestand'].fillna(baseline)

        if first_dispo_date is not None:
            feat['EoD_Bestand'] = np.where(
                feat['Datum'] < first_dispo_date,
                feat['Lagerbestand'],
                baseline + feat['cum_net'],
            )
        else:
            feat['EoD_Bestand'] = feat['Lagerbestand'].fillna(baseline)
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

        # Progressive & Rolling Differential Prozesse for label calculation
        deficit = (-feat['EoD_Bestand_noSiBe']).clip(lower=0)
        deficit_arr = deficit.to_numpy()
        dte = feat['DaysToEmpty'].to_numpy()
        idx = np.arange(len(deficit_arr))

        # Progressive Differential Prozess: gradual ramp-up towards stock-out
        future_idx = np.clip(idx + dte.astype(int), 0, len(deficit_arr) - 1)
        deficit_at_so = deficit_arr[future_idx]
        progressive = np.where(
            dte < window,
            deficit_at_so * (1 - dte / window),
            0,
        )

        # Rolling Differential Prozess: sum of deficits over the look-ahead window
        rolling = (
            pd.Series(deficit_arr[::-1])
            .rolling(window, min_periods=1)
            .sum()
            .to_numpy()[::-1]
        )

        feat['LABLE_StockOut_MinAdd'] = np.maximum(progressive, rolling)

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
                'Bedarfsmenge_daily',
                'net',
                'cum_net',
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
