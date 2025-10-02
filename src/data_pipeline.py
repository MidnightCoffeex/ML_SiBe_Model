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
        loc = df['Lagerort'].astype(str).str.replace(',', '.').str.strip()
        df = df[pd.to_numeric(loc, errors='coerce') == 120]
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
        # For SiBeVerlauf we now allow new unified export schema
        if dataset == 'SiBeVerlauf':
            # Prefer explicit selection to ensure required columns survive
            sibe_candidates = ['Teil', 'AudEreignis-ZeitPkt', 'Datum Änderung', 'Im Sytem hinterlgeter SiBe', 'aktiver SiBe']
            # Be robust to encoding variations like 'Datum �nderung'
            colmap = {c: c for c in df.columns}
            # find a column that looks like 'Datum Änderung'
            for c in df.columns:
                cs = str(c)
                lcs = cs.lower()
                if 'datum' in lcs and 'nderung' in lcs and 'änder' in lcs or 'nd' in lcs:
                    colmap['Datum Änderung'] = c
                if 'aktiver' in lcs and 'sibe' in lcs and 'im sytem' not in lcs:
                    colmap['aktiver SiBe'] = c
            keep_cols = []
            for k in sibe_candidates:
                src = colmap.get(k)
                if src in df.columns and src not in keep_cols:
                    keep_cols.append(src)
            if 'Teil' not in keep_cols and 'Teil' in df.columns:
                keep_cols.append('Teil')
        else:
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
    """Parse multiple possible date columns with fallback.

    Columns are checked in order and missing values are filled by the next
    available column.  Parsing is done with ``dayfirst`` and ``format='mixed'`` to
    handle heterogeneous date formats robustly.
    """
    date = pd.Series(pd.NaT, index=df.index)
    for c in columns:
        if c in df.columns:
            col = df[c].astype(str).str.strip()
            parsed = pd.to_datetime(col, errors='coerce', dayfirst=True, format='mixed')
            date = date.fillna(parsed)
    return date


def _aggregate_dataset(
    df: pd.DataFrame, date_columns: List[str], last_cols: List[str] | None = None
) -> pd.DataFrame:
    df = df.copy()
    full_dt = _parse_date(df, date_columns)
    df['Datum'] = full_dt.dt.floor('D')
    df['_sort'] = full_dt
    df = df.dropna(subset=['Datum'])
    num_cols = df.select_dtypes(include='number').columns
    agg: Dict[str, str] = {}
    for c in df.columns:
        if c in {'Teil', 'Datum', 'Dataset', 'ExportDatum', '_sort'}:
            continue
        if last_cols and c in last_cols:
            agg[c] = 'last'
        elif c in num_cols:
            agg[c] = 'sum'
        else:
            agg[c] = 'first'
    grouped = df.sort_values('_sort').groupby(['Teil', 'Datum'], as_index=False).agg(agg)
    return grouped


def _prepare_lagerbew(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``Lagerbew`` to daily end-of-day stock per part."""
    df = df.copy()
    full_dt = _parse_date(df, ['BuchDat'])
    df['Datum'] = full_dt.dt.floor('D')
    df['_sort'] = full_dt
    df['Lagerbestand'] = pd.to_numeric(df.get('Lagerbestand'), errors='coerce')
    df = df.dropna(subset=['Datum', 'Lagerbestand'])
    df = df.sort_values('_sort')
    return df.groupby(['Teil', 'Datum'], as_index=False)['Lagerbestand'].last()


def _prepare_dispo(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``Dispo`` to daily net requirements per part."""
    df = df.copy()
    df['Datum'] = _parse_date(df, ['Termin', 'Solltermin'])
    df = df.dropna(subset=['Datum'])
    df['Bedarfsmenge'] = pd.to_numeric(df.get('Bedarfsmenge'), errors='coerce').fillna(0)
    df['Deckungsmenge'] = pd.to_numeric(df.get('Deckungsmenge'), errors='coerce').fillna(0)
    df['net'] = df['Deckungsmenge'] - df['Bedarfsmenge']
    agg = df.groupby(['Teil', 'Datum'], as_index=False).agg({
        'net': 'sum',
        'Bedarfsmenge': 'sum',
    })
    return agg


def build_features_by_part(raw_dir: str, xlsx_path: str = 'Spaltenbedeutung.xlsx') -> Dict[str, Dict[str, pd.DataFrame | pd.Timestamp | None]]:
    """Process all raw files and return a dict of
    part -> { 'df': feature DataFrame, 'first_dispo_date': Timestamp | None }.

    The first_dispo_date per part is used downstream to split into
    historical (pre-Dispo) and Dispo-period subsets.
    """
    column_map = _load_column_map(xlsx_path)
    tables = load_all_tables(raw_dir, column_map)

    processed: Dict[str, pd.DataFrame] = {}
    # Aggregate relevant datasets
    agg_tables: Dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if name == 'Lagerbew':
            agg_tables[name] = _prepare_lagerbew(df)
        elif name == 'Dispo':
            agg_tables[name] = _prepare_dispo(df)
        elif name == 'SiBeVerlauf':
            df_s = df.copy()
            # Unify new schema to legacy names so downstream stays stable
            # tolerate columns like 'Datum �nderung'
            dchange = None
            for c in df_s.columns:
                lcs = str(c).lower()
                if 'datum' in lcs and 'nderung' in lcs:
                    dchange = c
                    break
            if dchange and 'AudEreignis-ZeitPkt' not in df_s.columns:
                df_s.rename(columns={dchange: 'AudEreignis-ZeitPkt'}, inplace=True)
            if 'aktiver SiBe' in df_s.columns and 'Im Sytem hinterlgeter SiBe' not in df_s.columns:
                df_s.rename(columns={'aktiver SiBe': 'Im Sytem hinterlgeter SiBe'}, inplace=True)
            # ensure Teil exists; if not, try to derive from Primärschlüssel text
            if 'Teil' not in df_s.columns and 'Primärschlüssel' in df_s.columns:
                part = df_s['Primärschlüssel'].astype(str).str.extract(r'Teil\s+(\d+)')[0]
                df_s['Teil'] = part
            agg_tables[name] = _aggregate_dataset(df_s, ['AudEreignis-ZeitPkt'], last_cols=['Im Sytem hinterlgeter SiBe'])
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

        # collect relevant dates from Lagerbewegung and Dispo
        date_set: set[pd.Timestamp] = set()
        first_dispo_date = None
        if 'Lagerbew' in data:
            data['Lagerbew']['Datum'] = pd.to_datetime(data['Lagerbew']['Datum'])
            date_set.update(data['Lagerbew']['Datum'].unique())
        if 'Dispo' in data:
            data['Dispo']['Datum'] = pd.to_datetime(data['Dispo']['Datum'])
            date_set.update(data['Dispo']['Datum'].unique())
            first_dispo_date = data['Dispo']['Datum'].min()

        baseline = 0.0
        baseline_date = None
        if first_dispo_date is not None and 'Lagerbew' in data:
            lb_before = data['Lagerbew'][data['Lagerbew']['Datum'] <= first_dispo_date]
            if not lb_before.empty:
                last_lb = lb_before.sort_values('Datum').iloc[-1]
                baseline = float(last_lb['Lagerbestand'])
                baseline_date = last_lb['Datum']
        if baseline_date is None and 'Bestand' in data:
            best = data['Bestand'][['Datum', 'Bestand']].copy()
            best['Datum'] = pd.to_datetime(best['Datum'])
            if first_dispo_date is not None:
                best_before = best[best['Datum'] <= first_dispo_date]
                if not best_before.empty:
                    last_best = best_before.sort_values('Datum').iloc[-1]
                    baseline = float(last_best['Bestand'])
                    baseline_date = last_best['Datum']
            elif not best.empty:
                last_best = best.sort_values('Datum').iloc[-1]
                baseline = float(last_best['Bestand'])
                baseline_date = last_best['Datum']
        if baseline_date is not None:
            date_set.add(baseline_date)
        if not date_set:
            continue

        feat = pd.DataFrame({'Datum': sorted(date_set)})
        feat['Teil'] = part

        # merge Lagerbewegung inventory
        if 'Lagerbew' in data:
            lb = data['Lagerbew'][['Datum', 'Lagerbestand']]
            feat = feat.merge(lb, on='Datum', how='left')
        else:
            feat['Lagerbestand'] = np.nan
        if baseline_date is not None and feat.loc[feat['Datum'] == baseline_date, 'Lagerbestand'].isna().all():
            feat.loc[feat['Datum'] == baseline_date, 'Lagerbestand'] = baseline

        # merge Dispo movements
        if 'Dispo' in data:
            dispo = data['Dispo'][['Datum', 'net', 'Bedarfsmenge']]
            feat = feat.merge(dispo, on='Datum', how='left')
            feat['net'] = feat['net'].fillna(0)
            feat['Bedarfsmenge'] = feat['Bedarfsmenge'].fillna(0)
            feat['cum_net'] = 0.0
            if first_dispo_date is not None:
                mask = feat['Datum'] >= first_dispo_date
                feat.loc[mask, 'cum_net'] = feat.loc[mask, 'net'].cumsum()
        else:
            feat['net'] = 0
            feat['Bedarfsmenge'] = 0
            feat['cum_net'] = 0

        # compute end-of-day stock
        if first_dispo_date is not None:
            pre_mask = feat['Datum'] < first_dispo_date
            feat.loc[pre_mask, 'EoD_Bestand'] = feat.loc[pre_mask, 'Lagerbestand']
            post_mask = feat['Datum'] >= first_dispo_date
            feat.loc[post_mask, 'EoD_Bestand'] = baseline + feat.loc[post_mask, 'cum_net']
        else:
            feat['EoD_Bestand'] = feat['Lagerbestand'].fillna(baseline)
        feat['EoD_Bestand'] = pd.to_numeric(feat['EoD_Bestand'], errors='coerce').fillna(0)

        # safety stock history: apply latest change as of each feature date (backward asof),
        # before the first change -> 0, after last change -> last value
        if 'SiBeVerlauf' in data:
            sibe = data['SiBeVerlauf'][['Datum', 'Im Sytem hinterlgeter SiBe']].copy()
            sibe['Datum'] = pd.to_datetime(sibe['Datum'], errors='coerce').dt.floor('D')
            sibe = sibe.dropna(subset=['Datum'])
            sibe = sibe.sort_values('Datum')
            merged = pd.merge_asof(
                feat.sort_values('Datum'),
                sibe,
                on='Datum',
                direction='backward',
            )
            merged['Im Sytem hinterlgeter SiBe'] = pd.to_numeric(
                merged['Im Sytem hinterlgeter SiBe'], errors='coerce'
            ).fillna(0)
            feat = merged
            feat.rename(columns={'Im Sytem hinterlgeter SiBe': 'Hinterlegter SiBe'}, inplace=True)
        else:
            feat['Hinterlegter SiBe'] = 0

        # rename display-only (no-feature) columns and compute training series
        feat.rename(columns={'EoD_Bestand': 'F_NiU_EoD_Bestand'}, inplace=True)
        feat.rename(columns={'Hinterlegter SiBe': 'F_NiU_Hinterlegter SiBe'}, inplace=True)
        feat['EoD_Bestand_noSiBe'] = feat['F_NiU_EoD_Bestand'] - feat['F_NiU_Hinterlegter SiBe']
        feat['Flag_StockOut'] = (feat['EoD_Bestand_noSiBe'] <= 0).astype(int)

        # (DaysToEmpty/BestandDelta_7T entfernt – nicht mehr Teil der Ausgabe)

        # WBZ from Teilestamm
        wbz = None
        if 'Teilestamm' in data:
            w = data['Teilestamm']['WBZ'].dropna()
            if not w.empty:
                wbz = float(pd.to_numeric(w.iloc[0], errors='coerce'))
        feat['WBZ_Days'] = wbz

        lead_time = int(wbz) if wbz and wbz > 0 else 1
        window = max(1, int(np.ceil(lead_time * 1.25)))

        # Legacy label (kept as background only, renamed)
        deficit = (-feat['EoD_Bestand_noSiBe']).clip(lower=0)
        deficit_arr = deficit.to_numpy()
        # simple rolling sum over window as proxy (keine DaysToEmpty mehr)
        rolling = (
            pd.Series(deficit_arr[::-1])
            .rolling(window, min_periods=1)
            .sum()
            .to_numpy()[::-1]
        )
        feat['L_NiU_StockOut_MinAdd'] = rolling

        # Neues Label: größter negativer Ausschlag innerhalb WBZ-Fenster (als positiver Wert)
        # LABLE_WBZ_BlockMinAbs = max(0, -min(EoD_Bestand_noSiBe im Fenster))
        feat = feat.sort_values('Datum').reset_index(drop=True)
        date_list = pd.to_datetime(feat['Datum']).tolist()
        vals = feat['EoD_Bestand_noSiBe'].to_numpy()
        lbl_block = np.zeros(len(feat), dtype=float)
        for i in range(len(feat)):
            start = date_list[i]
            end = start + pd.Timedelta(days=int(lead_time))
            # Fenstermaske über Index, da date_list eine Liste ist
            y_vals = []
            for k in range(i, len(feat)):
                if date_list[k] < end:
                    y_vals.append(vals[k])
                else:
                    break
            y = np.array(y_vals, dtype=float)
            mmin = float(np.nanmin(y)) if y.size else 0.0
            lbl_block[i] = max(0.0, -mmin)
        # ausblenden als Diagnose (Label Not-in-Use)
        feat['L_NiU_WBZ_BlockMinAbs'] = lbl_block

        # Halbjahres-Regel mit Faktoren: Fensterweise (ca. 6 Monate) denselben Wert setzen,
        # und zwar den Höchstwert von L_NiU_WBZ_BlockMinAbs innerhalb des Fensters,
        # moduliert durch marginale Faktoren (WBZ, Frequenz, Volatilität), jeweils gecappt.
        lbl_halfyear_base = np.zeros(len(feat), dtype=float)
        lbl_halfyear = np.zeros(len(feat), dtype=float)
        f_wbz_arr = np.zeros(len(feat), dtype=float)
        f_freq_arr = np.zeros(len(feat), dtype=float)
        f_vol_arr = np.zeros(len(feat), dtype=float)
        # Nachfrage-Ereignisse aus demand_series: Ereignis, wenn positive Nachfrage
        demand_series = feat['EoD_Bestand_noSiBe'].shift(1) - feat['EoD_Bestand_noSiBe']
        demand_series = pd.to_numeric(demand_series, errors='coerce').clip(lower=0).fillna(0).to_numpy()
        i = 0
        while i < len(feat):
            start = date_list[i]
            # 6 Monate nach vorn; nächstes verfügbares Datum >= diesem Ziel
            target = start + pd.DateOffset(months=6)
            # finde j: erstes Index mit Datum >= target
            j = i
            while j + 1 < len(feat) and date_list[j + 1] < target:
                j += 1
            # falls nächstes Datum hinter target existiert, auf dieses "aufrunden"
            if j + 1 < len(feat) and date_list[j] < target <= date_list[j + 1]:
                j = j + 1
            # Fenster [i..j]
            window_max = float(np.nanmax(lbl_block[i:j + 1])) if j >= i else float(lbl_block[i])
            # Faktoren berechnen (fensterweise konstant)
            wbz_eff = float(pd.to_numeric(feat.loc[i, 'WBZ_Days'], errors='coerce')) if 'WBZ_Days' in feat.columns else 0.0
            if not np.isfinite(wbz_eff) or wbz_eff <= 0:
                wbz_eff = 14.0
            wbz_eff = max(14.0, wbz_eff)
            # f_wbz: 0..0.20 je nach WBZ-Kategorie (sanft)
            if wbz_eff <= 28:
                f_wbz = 0.0
            elif wbz_eff <= 84:
                f_wbz = 0.10 * (wbz_eff - 28.0) / (84.0 - 28.0)
            else:
                f_wbz = 0.20
            # Frequenzfaktor: Ereignisse pro Fenster -> auf WBZ skaliert
            window_days = max(1, int((date_list[j] - date_list[i]).days) + 1)
            events_window = int(np.sum(demand_series[i:j + 1] > 0))
            est_events_wbz = events_window * (wbz_eff / window_days)
            f_freq = 0.20 * (np.log1p(est_events_wbz) / np.log1p(8.0))
            f_freq = float(min(0.20, max(0.0, f_freq)))
            # Volatilität (CV) im Fenster
            dwin = demand_series[i:j + 1]
            mu = float(np.nanmean(dwin)) if dwin.size else 0.0
            sigma = float(np.nanstd(dwin)) if dwin.size else 0.0
            cv = (sigma / mu) if mu > 1e-12 else 0.0
            if cv <= 0.3:
                f_vol = 0.0
            elif cv >= 0.8:
                f_vol = 0.20
            else:
                f_vol = 0.20 * (cv - 0.3) / (0.8 - 0.3)
            total_factor = min(0.40, f_wbz + f_freq + f_vol)

            base_val = window_max
            final_val = base_val * (1.0 + total_factor)
            lbl_halfyear_base[i:j + 1] = base_val
            lbl_halfyear[i:j + 1] = final_val
            f_wbz_arr[i:j + 1] = f_wbz
            f_freq_arr[i:j + 1] = f_freq
            f_vol_arr[i:j + 1] = f_vol
            i = j + 1
        feat['L_NiU_HalfYear_Base'] = lbl_halfyear_base
        feat['F_NiU_Factor_WBZ'] = f_wbz_arr
        feat['F_NiU_Factor_Freq'] = f_freq_arr
        feat['F_NiU_Factor_Vol'] = f_vol_arr
        feat['LABLE_HalfYear_Target'] = lbl_halfyear

        # ----- rolling time features -----
        demand_series = feat['EoD_Bestand_noSiBe'].shift(1) - feat['EoD_Bestand_noSiBe']
        demand_series = demand_series.clip(lower=0).fillna(0)
        for fac in [1.0, 2/3, 1/2, 1/4]:
            w = max(1, int(round(lead_time * fac)))
            key = str(int(fac*100)).rjust(2, '0')
            feat[f'DemandMean_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).mean()
            feat[f'DemandMax_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).max()

        feat = feat[
            [
                'Teil',
                'Datum',
                'F_NiU_EoD_Bestand',
                'F_NiU_Hinterlegter SiBe',
                'EoD_Bestand_noSiBe',
                'Flag_StockOut',
                'WBZ_Days',
                'L_NiU_StockOut_MinAdd',
                'L_NiU_WBZ_BlockMinAbs',
                'L_NiU_HalfYear_Base',
                'F_NiU_Factor_WBZ',
                'F_NiU_Factor_Freq',
                'F_NiU_Factor_Vol',
                'LABLE_HalfYear_Target',
            ]
            + [
                c
                for c in feat.columns
                if c.startswith('DemandMean_') or c.startswith('DemandMax_')
            ]
        ]

        processed[part] = {'df': feat, 'first_dispo_date': first_dispo_date}

    return processed


def save_feature_folders_split(
    features: Dict[str, Dict[str, pd.DataFrame | pd.Timestamp | None]],
    features_dir: str = 'Features',
    test_dir: str = 'Test_Set',
) -> None:
    """Write each part's features into two folders:
    - ``features_dir/<part>``: only historical Lagerbewegung period (pre-Dispo)
    - ``test_dir/<part>``: Dispo period (including and after first Dispo date)

    Both Parquet and Excel are written for convenience.
    """
    out_feat = Path(features_dir)
    out_test = Path(test_dir)
    for part, bundle in features.items():
        df = bundle['df']  # type: ignore[assignment]
        first_dispo_date = bundle.get('first_dispo_date')  # type: ignore[assignment]
        if isinstance(first_dispo_date, pd.Timestamp):
            hist_df = df[df['Datum'] < first_dispo_date].copy()
            dispo_df = df[df['Datum'] >= first_dispo_date].copy()
        else:
            hist_df = df.copy()
            dispo_df = df.iloc[0:0].copy()

        # write historical subset
        part_dir = out_feat / str(part)
        part_dir.mkdir(parents=True, exist_ok=True)
        if not hist_df.empty:
            hist_df.to_parquet(part_dir / 'features.parquet', index=False)
            try:
                hist_df.to_excel(part_dir / 'features.xlsx', index=False)
            except Exception:
                pass
        else:
            # create empty placeholder parquet to signal existence
            hist_df.to_parquet(part_dir / 'features.parquet', index=False)

        # write dispo subset
        part_dir_t = out_test / str(part)
        part_dir_t.mkdir(parents=True, exist_ok=True)
        if not dispo_df.empty:
            dispo_df.to_parquet(part_dir_t / 'features.parquet', index=False)
            try:
                dispo_df.to_excel(part_dir_t / 'features.xlsx', index=False)
            except Exception:
                pass
        else:
            dispo_df.to_parquet(part_dir_t / 'features.parquet', index=False)


def run_pipeline(raw_dir: str, output_dir: str = 'Features') -> None:
    """Complete preprocessing pipeline producing split outputs per part.

    - ``output_dir`` contains historical (pre-Dispo) subsets per part.
    - ``Test_Set`` sibling folder contains Dispo-period subsets per part.
    """
    features = build_features_by_part(raw_dir)
    # Write historical into output_dir and Dispo-period into Test_Set
    save_feature_folders_split(features, output_dir, 'Test_Set')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process raw CSV data')
    parser.add_argument('--input', default='Rohdaten', help='Input directory with raw CSVs')
    parser.add_argument('--output', default='Features', help='Output directory for feature folders')
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
