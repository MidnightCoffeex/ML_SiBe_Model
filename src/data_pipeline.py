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
            sibe_candidates = ['Teil', 'AudEreignis-ZeitPkt', 'Datum Ã„nderung', 'Im Sytem hinterlgeter SiBe', 'aktiver SiBe']
            # Be robust to encoding variations like 'Datum ï¿½nderung'
            colmap = {c: c for c in df.columns}
            # find a column that looks like 'Datum Ã„nderung'
            for c in df.columns:
                cs = str(c)
                lcs = cs.lower()
                if 'datum' in lcs and 'nderung' in lcs and 'Ã¤nder' in lcs or 'nd' in lcs:
                    colmap['Datum Ã„nderung'] = c
                if 'aktiver' in lcs and 'sibe' in lcs and 'im sytem' not in lcs:
                    colmap['aktiver SiBe'] = c
            keep_cols = []
            for k in sibe_candidates:
                src = colmap.get(k)
                if src in df.columns and src not in keep_cols:
                    keep_cols.append(src)
            # If an 'alter sibe' like column exists but was not mapped, include it
            for c in df.columns:
                lcs = str(c).lower()
                if 'alter' in lcs and 'sibe' in lcs and c not in keep_cols:
                    keep_cols.append(c)
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
            # tolerate columns like 'Datum ï¿½nderung'
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
            # Normalize any variant of 'alter SiBe' to a canonical 'Alter_SiBe'
            if 'Alter_SiBe' not in df_s.columns:
                for c in list(df_s.columns):
                    lcs = str(c).lower()
                    if 'alter' in lcs and 'sibe' in lcs:
                        df_s.rename(columns={c: 'Alter_SiBe'}, inplace=True)
                        break
            # ensure Teil exists; if not, try to derive from PrimÃ¤rschlÃ¼ssel text
            if 'Teil' not in df_s.columns and 'PrimÃ¤rschlÃ¼ssel' in df_s.columns:
                part = df_s['PrimÃ¤rschlÃ¼ssel'].astype(str).str.extract(r'Teil\s+(\d+)')[0]
                df_s['Teil'] = part
            agg_tables[name] = _aggregate_dataset(
                df_s,
                ['AudEreignis-ZeitPkt'],
                last_cols=['Im Sytem hinterlgeter SiBe', 'Alter_SiBe'],
            )
        elif name in {'Bestand', 'Teilestamm'}:
            df = df.copy()
            df['Datum'] = df['ExportDatum']
            agg = {c: 'first' for c in df.columns if c not in {'Teil', 'Datum', 'Dataset', 'ExportDatum'}}
            agg_tables[name] = df.groupby(['Teil', 'Datum'], as_index=False).agg(agg)
        elif name == 'TeileWert':
            df_tw = df.copy()
            # Versuche die Spalte fÃ¼r Materialpreis zu erkennen
            price_col = None
            for c in df_tw.columns:
                lcs = str(c).lower()
                if 'dpr' in lcs and 'material' in lcs and 'var' in lcs:
                    price_col = c
                    break
            if price_col is None:
                for c in df_tw.columns:
                    if 'material' in str(c).lower():
                        price_col = c
                        break
            if price_col is not None:
                df_tw = df_tw[['Teil', 'ExportDatum', price_col]].copy()
                df_tw.rename(columns={price_col: 'Price_Material_var', 'ExportDatum': 'Datum'}, inplace=True)
                df_tw['Datum'] = pd.to_datetime(df_tw['Datum'], errors='coerce').dt.floor('D')
                df_tw = df_tw.dropna(subset=['Datum'])
                agg_tables[name] = df_tw.groupby(['Teil', 'Datum'], as_index=False)['Price_Material_var'].last()
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

        # Build a continuous daily timeline between earliest and latest relevant dates
        # This ensures eventless days are represented for robust rolling windows and lags
        min_date = pd.to_datetime(min(date_set))
        max_date = pd.to_datetime(max(date_set))
        full_days = pd.date_range(min_date, max_date, freq='D')
        feat = pd.DataFrame({'Datum': full_days})
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

        # compute end-of-day stock on a daily grid
        # - before first_dispo_date: carry forward last known Lagerbestand (no backfill to avoid leakage)
        # - on/after first_dispo_date: baseline + cumulative net movements
        lb_num = pd.to_numeric(feat['Lagerbestand'], errors='coerce')
        if first_dispo_date is not None:
            pre_mask = feat['Datum'] < first_dispo_date
            # forward-fill Lagerbestand over pre-Dispo window only
            feat.loc[pre_mask, 'EoD_Bestand'] = lb_num.loc[pre_mask].ffill()
            post_mask = feat['Datum'] >= first_dispo_date
            feat.loc[post_mask, 'EoD_Bestand'] = baseline + feat.loc[post_mask, 'cum_net']
        else:
            # no Dispo present: forward-fill across entire range
            feat['EoD_Bestand'] = lb_num.ffill()
        # final numeric coercion; keep unknowns as 0 to stay backward-compatible
        feat['EoD_Bestand'] = pd.to_numeric(feat['EoD_Bestand'], errors='coerce').fillna(0)

        # safety stock history: apply latest change as of each feature date (backward asof),
        # before the first change -> 0, after last change -> last value
        if 'SiBeVerlauf' in data:
            # include optional previous SiBe if provided (e.g. 'alter SiBE')
            use_cols = ['Datum', 'Im Sytem hinterlgeter SiBe']
            if 'Alter_SiBe' in data['SiBeVerlauf'].columns:
                use_cols.append('Alter_SiBe')
            sibe = data['SiBeVerlauf'][use_cols].copy()
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
            # Backfill: For all dates strictly before the first change, use the earliest Alter_SiBe.
            # Der 'alte SiBe' gilt ab Serienstart bis zum ersten Ã„nderungsdatum; danach gilt der aktive SiBe.
            if 'Alter_SiBe' in sibe.columns:
                try:
                    first_change = sibe['Datum'].min()
                    alter_val = pd.to_numeric(
                        sibe.loc[sibe['Datum'] == first_change, 'Alter_SiBe'], errors='coerce'
                    ).iloc[0]
                except Exception:
                    first_change = None
                    alter_val = None
                if first_change is not None and pd.notna(alter_val):
                    mask_range = (merged['Datum'] < first_change)
                    merged.loc[mask_range, 'Im Sytem hinterlgeter SiBe'] = float(alter_val)
            feat = merged
            feat.rename(columns={'Im Sytem hinterlgeter SiBe': 'Hinterlegter SiBe'}, inplace=True)
        else:
            feat['Hinterlegter SiBe'] = 0

        # Materialpreis asof-join (falls vorhanden)
        if 'TeileWert' in data:
            tw = data['TeileWert'][['Datum', 'Price_Material_var']].copy()
            if not tw.empty:
                tw = tw.dropna(subset=['Datum']).sort_values('Datum')
                feat = pd.merge_asof(feat.sort_values('Datum'), tw, on='Datum', direction='backward')

        # rename display-only (no-feature) columns and compute training series
        feat.rename(columns={'EoD_Bestand': 'F_NiU_EoD_Bestand'}, inplace=True)
        feat.rename(columns={'Hinterlegter SiBe': 'F_NiU_Hinterlegter SiBe'}, inplace=True)
        # FÃ¼r Auswertung/Plots zusÃ¤tzlich eine nicht-NiU-Kopie behalten
        if 'F_NiU_Hinterlegter SiBe' in feat.columns and 'Hinterlegter SiBe' not in feat.columns:
            feat['Hinterlegter SiBe'] = feat['F_NiU_Hinterlegter SiBe']
        feat['EoD_Bestand_noSiBe'] = feat['F_NiU_EoD_Bestand'] - feat['F_NiU_Hinterlegter SiBe']
        # Neutral always-present columns for downstream usage
        feat['EoD_Bestand'] = feat['F_NiU_EoD_Bestand']
        feat['Hinterlegter_SiBe'] = feat['F_NiU_Hinterlegter SiBe']
        feat['Flag_StockOut'] = (feat['EoD_Bestand_noSiBe'] <= 0).astype(int)

        # (DaysToEmpty/BestandDelta_7T entfernt â€“ nicht mehr Teil der Ausgabe)

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

        # Neues Label: grÃ¶ÃŸter negativer Ausschlag innerhalb WBZ-Fenster (als positiver Wert)
        # LABLE_WBZ_BlockMinAbs = max(0, -min(EoD_Bestand_noSiBe im Fenster))
        feat = feat.sort_values('Datum').reset_index(drop=True)
        date_list = pd.to_datetime(feat['Datum']).tolist()
        vals = feat['EoD_Bestand_noSiBe'].to_numpy()
        lbl_block = np.zeros(len(feat), dtype=float)
        for i in range(len(feat)):
            start = date_list[i]
            end = start + pd.Timedelta(days=int(lead_time))
            # Fenstermaske Ã¼ber Index, da date_list eine Liste ist
            y_vals = []
            for k in range(i, len(feat)):
                if date_list[k] < end:
                    y_vals.append(vals[k])
                else:
                    break
            y = np.array(y_vals, dtype=float)
            mmin = float(np.nanmin(y)) if y.size else 0.0
            lbl_block[i] = max(0.0, -mmin)
        # Basiswert des Block-Min-Abs (wird nach Faktorberechnung angepasst)
        block_base = lbl_block.copy()

        # Halbjahres-Regel mit Faktoren: Fensterweise (ca. 6 Monate) denselben Wert setzen,
        # und zwar den HÃ¶chstwert von L_NiU_WBZ_BlockMinAbs innerhalb des Fensters,
        # moduliert durch marginale Faktoren (WBZ, Frequenz, VolatilitÃ¤t), jeweils gecappt.
        lbl_halfyear_base = np.zeros(len(feat), dtype=float)
        lbl_halfyear = np.zeros(len(feat), dtype=float)
        f_wbz_arr = np.zeros(len(feat), dtype=float)
        f_freq_arr = np.zeros(len(feat), dtype=float)
        f_vol_arr = np.zeros(len(feat), dtype=float)
        f_price_arr = np.zeros(len(feat), dtype=float)
        # Nachfrage-Ereignisse aus demand_series: Ereignis, wenn positive Nachfrage
        demand_series = feat['EoD_Bestand_noSiBe'].shift(1) - feat['EoD_Bestand_noSiBe']
        demand_series = pd.to_numeric(demand_series, errors='coerce').clip(lower=0).fillna(0).to_numpy()
        i = 0
        while i < len(feat):
            start = date_list[i]
            # 6 Monate nach vorn; nÃ¤chstes verfÃ¼gbares Datum >= diesem Ziel
            target = start + pd.DateOffset(months=6)
            # finde j: erstes Index mit Datum >= target
            j = i
            while j + 1 < len(feat) and date_list[j + 1] < target:
                j += 1
            # falls nÃ¤chstes Datum hinter target existiert, auf dieses "aufrunden"
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
            # VolatilitÃ¤t (CV) im Fenster
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
            # Preisfaktor (logarithmisch, nur reduzierend, Cap 0.10)
            if 'Price_Material_var' in feat.columns:
                p = pd.to_numeric(feat['Price_Material_var'], errors='coerce').fillna(0).astype(float).to_numpy()
                p_log = np.log1p(np.maximum(0.0, p))
                scale = np.nanpercentile(p_log[p_log > 0], 95) if np.any(p_log > 0) else 1.0
                f_price = -0.10 * (p_log / scale) if scale > 0 else np.zeros_like(p_log)
                f_price = float(np.clip(f_price[i], -0.10, 0.0)) if i < len(f_price) else 0.0
            else:
                f_price = 0.0
            total_factor = min(0.40, max(0.0, f_wbz + f_freq + f_vol + f_price))

            base_val = window_max
            final_val = base_val * (1.0 + total_factor)
            lbl_halfyear_base[i:j + 1] = base_val
            lbl_halfyear[i:j + 1] = final_val
            f_wbz_arr[i:j + 1] = f_wbz
            f_freq_arr[i:j + 1] = f_freq
            f_vol_arr[i:j + 1] = f_vol
            f_price_arr[i:j + 1] = f_price
            i = j + 1
        feat['L_NiU_HalfYear_Base'] = lbl_halfyear_base
        feat['F_NiU_Factor_WBZ'] = f_wbz_arr
        feat['F_NiU_Factor_Freq'] = f_freq_arr
        feat['F_NiU_Factor_Vol'] = f_vol_arr
        feat['F_NiU_Factor_Price'] = f_price_arr
        feat['L_HalfYear_Target'] = lbl_halfyear
        # Faktoren auch auf den per-Date BlockMinAbs anwenden (Clip 0..0.40)
        total_factor_arr = np.clip(f_wbz_arr + f_freq_arr + f_vol_arr + f_price_arr, 0.0, 0.40)
        feat['L_NiU_WBZ_BlockMinAbs'] = block_base * (1.0 + total_factor_arr)

        # ----- rolling time features -----
        demand_series = feat['EoD_Bestand_noSiBe'].shift(1) - feat['EoD_Bestand_noSiBe']
        demand_series = demand_series.clip(lower=0).fillna(0)
        for fac in [1.0, 2/3, 1/2, 1/4]:
            w = max(1, int(round(lead_time * fac)))
            key = str(int(fac*100)).rjust(2, '0')
            feat[f'DemandMean_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).mean()
            feat[f'DemandMax_{key}'] = demand_series.shift(1).rolling(w, min_periods=1).max()

        # ----- transformations: log1p and per-SKU (Teil) normalization for key quantities -----
        # 1) EoD_Bestand_noSiBe transformations
        try:
            # log1p on non-negative part; separate positive deficit magnitude as log1p too
            eod = pd.to_numeric(feat['EoD_Bestand_noSiBe'], errors='coerce')
            feat['EoD_Bestand_noSiBe_log1p'] = np.log1p(eod.clip(lower=0))
            deficit_pos = (-eod).clip(lower=0)
            feat['DeficitPos_log1p'] = np.log1p(deficit_pos)
            # per-SKU z-score and robust z-score
            if 'Teil' in feat.columns:
                grp = feat.groupby('Teil')
                mean = grp['EoD_Bestand_noSiBe'].transform('mean')
                std = grp['EoD_Bestand_noSiBe'].transform('std')
                std = std.replace(0, np.nan)
                z = (eod - mean) / std
                feat['EoD_Bestand_noSiBe_z_Teil'] = z.fillna(0)
                med = grp['EoD_Bestand_noSiBe'].transform('median')
                q75 = grp['EoD_Bestand_noSiBe'].transform(lambda s: s.quantile(0.75))
                q25 = grp['EoD_Bestand_noSiBe'].transform(lambda s: s.quantile(0.25))
                iqr = (q75 - q25).replace(0, np.nan)
                robz = (eod - med) / iqr
                feat['EoD_Bestand_noSiBe_robz_Teil'] = robz.fillna(0)
        except Exception:
            pass

        # 2) DemandMean_* and DemandMax_* transformations per column
        demand_cols = [c for c in feat.columns if c.startswith('DemandMean_') or c.startswith('DemandMax_')]
        for c in demand_cols:
            try:
                vals = pd.to_numeric(feat[c], errors='coerce').fillna(0)
                # log1p
                feat[f'{c}_log1p'] = np.log1p(vals.clip(lower=0))
                if 'Teil' in feat.columns:
                    grp = feat.groupby('Teil')
                    mean = grp[c].transform('mean')
                    std = grp[c].transform('std').replace(0, np.nan)
                    feat[f'{c}_z_Teil'] = ((vals - mean) / std).fillna(0)
                    med = grp[c].transform('median')
                    q75 = grp[c].transform(lambda s: s.quantile(0.75))
                    q25 = grp[c].transform(lambda s: s.quantile(0.25))
                    iqr = (q75 - q25).replace(0, np.nan)
                    feat[f'{c}_robz_Teil'] = ((vals - med) / iqr).fillna(0)
            except Exception:
                continue

        feat = feat[
            [
                'Teil',
                'Datum',
                'F_NiU_EoD_Bestand',
                'F_NiU_Hinterlegter SiBe',
                'EoD_Bestand_noSiBe',
                'EoD_Bestand_noSiBe_log1p',
                'EoD_Bestand_noSiBe_z_Teil',
                'EoD_Bestand_noSiBe_robz_Teil',
                'DeficitPos_log1p',
                'Flag_StockOut',
                'WBZ_Days',
                'L_NiU_StockOut_MinAdd',
                'L_NiU_WBZ_BlockMinAbs',
                'L_NiU_HalfYear_Base',
                'F_NiU_Factor_WBZ',
                'F_NiU_Factor_Freq',
                'F_NiU_Factor_Vol',
                'F_NiU_Factor_Price',
                'L_HalfYear_Target',
                'Price_Material_var',
            ]
            + [
                c
                for c in feat.columns
                if c.startswith('DemandMean_') or c.startswith('DemandMax_')
            ]
        ]

        # Drop Not-in-Use diagnostics before persisting, but KEEP essential NiU display columns
        essential_niu = {"F_NiU_EoD_Bestand", "F_NiU_Hinterlegter SiBe"}
        keep_cols = []
        for c in feat.columns:
            cs = str(c)
            if cs.startswith("L_NiU_"):
                continue
            if cs.startswith("F_NiU_") and cs not in essential_niu:
                continue
            keep_cols.append(c)
        feat = feat[keep_cols]

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

    def _save_parquet_or_csv(df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path, index=False)
        except Exception:
            # Fallback when parquet engine is unavailable: write CSV next to intended parquet
            csv_path = path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
    for part, bundle in features.items():
        df = bundle['df']  # type: ignore[assignment]
        first_dispo_date = bundle.get('first_dispo_date')  # type: ignore[assignment]
        if isinstance(first_dispo_date, pd.Timestamp):
            hist_df = df[df['Datum'] < first_dispo_date].copy()
            dispo_df = df[df['Datum'] >= first_dispo_date].copy()
        else:
            hist_df = df.copy()
            dispo_df = df.iloc[0:0].copy()

        # Recompute forward-looking labels inside historical subset only,
        # so that no Test_Set future bleeds into training labels.
        if not hist_df.empty:
            hist_df = hist_df.sort_values('Datum').reset_index(drop=True)
            date_list = pd.to_datetime(hist_df['Datum']).tolist()
            vals = pd.to_numeric(hist_df['EoD_Bestand_noSiBe'], errors='coerce').fillna(0).to_numpy()
            # BlockMinAbs per row (WBZ-days window; restrict to hist end)
            lbl_block = np.zeros(len(hist_df), dtype=float)
            for ii in range(len(hist_df)):
                start = date_list[ii]
                # WBZ pro Zeile (fallback 14â†’1 Tag fÃ¼r BlockMinAbs wie zuvor minimal 1)
                wbz_row = pd.to_numeric(hist_df.loc[ii, 'WBZ_Days'], errors='coerce') if 'WBZ_Days' in hist_df.columns else np.nan
                lead_days = int(wbz_row) if pd.notna(wbz_row) and wbz_row > 0 else 1
                end = start + pd.Timedelta(days=lead_days)
                # Collect values until end (or hist end)
                y_vals = []
                for kk in range(ii, len(hist_df)):
                    if date_list[kk] < end:
                        y_vals.append(vals[kk])
                    else:
                        break
                y = np.array(y_vals, dtype=float)
                mmin = float(np.nanmin(y)) if y.size else 0.0
                lbl_block[ii] = max(0.0, -mmin)
            block_base = lbl_block.copy()

            # Half-year window with factors (within hist only)
            lbl_halfyear_base = np.zeros(len(hist_df), dtype=float)
            lbl_halfyear = np.zeros(len(hist_df), dtype=float)
            f_wbz_arr = np.zeros(len(hist_df), dtype=float)
            f_freq_arr = np.zeros(len(hist_df), dtype=float)
            f_vol_arr = np.zeros(len(hist_df), dtype=float)
            demand_series = (
                hist_df['EoD_Bestand_noSiBe'].shift(1) - hist_df['EoD_Bestand_noSiBe']
            )
            demand_series = pd.to_numeric(demand_series, errors='coerce').clip(lower=0).fillna(0).to_numpy()
            ii = 0
            while ii < len(hist_df):
                start = date_list[ii]
                target = start + pd.DateOffset(months=6)
                jj = ii
                while jj + 1 < len(hist_df) and date_list[jj + 1] < target:
                    jj += 1
                if jj + 1 < len(hist_df) and date_list[jj] < target <= date_list[jj + 1]:
                    jj = jj + 1
                window_max = float(np.nanmax(lbl_block[ii:jj + 1])) if jj >= ii else float(lbl_block[ii])
                wbz_eff = float(pd.to_numeric(hist_df.loc[ii, 'WBZ_Days'], errors='coerce')) if 'WBZ_Days' in hist_df.columns else 0.0
                if not np.isfinite(wbz_eff) or wbz_eff <= 0:
                    wbz_eff = 14.0
                wbz_eff = max(14.0, wbz_eff)
                # f_wbz
                if wbz_eff <= 28:
                    f_wbz = 0.0
                elif wbz_eff <= 84:
                    f_wbz = 0.10 * (wbz_eff - 28.0) / (84.0 - 28.0)
                else:
                    f_wbz = 0.20
                # f_freq (events per window scaled by WBZ)
                window_days = max(1, int((date_list[jj] - date_list[ii]).days) + 1)
                events_window = int(np.sum(demand_series[ii:jj + 1] > 0))
                est_events_wbz = events_window * (wbz_eff / window_days)
                f_freq = 0.20 * (np.log1p(est_events_wbz) / np.log1p(8.0))
                f_freq = float(min(0.20, max(0.0, f_freq)))
                # f_vol (CV in window)
                dwin = demand_series[ii:jj + 1]
                mu = float(np.nanmean(dwin)) if dwin.size else 0.0
                sigma = float(np.nanstd(dwin)) if dwin.size else 0.0
                cv = (sigma / mu) if mu > 1e-12 else 0.0
                if cv <= 0.3:
                    f_vol = 0.0
                elif cv >= 0.8:
                    f_vol = 0.20
                else:
                    f_vol = 0.20 * (cv - 0.3) / (0.8 - 0.3)
                # Preisfaktor (logarithmisch, reduzierend, Cap 0.10)
                if 'Price_Material_var' in hist_df.columns:
                    p = pd.to_numeric(hist_df['Price_Material_var'], errors='coerce').fillna(0).astype(float).to_numpy()
                    p_log = np.log1p(np.maximum(0.0, p))
                    scale = np.nanpercentile(p_log[p_log > 0], 95) if np.any(p_log > 0) else 1.0
                    f_price = -0.10 * (p_log / scale) if scale > 0 else np.zeros_like(p_log)
                    f_price = float(np.clip(f_price[ii], -0.10, 0.0)) if ii < len(f_price) else 0.0
                else:
                    f_price = 0.0
                total_factor = min(0.40, max(0.0, f_wbz + f_freq + f_vol + f_price))
                base_val = window_max
                final_val = base_val * (1.0 + total_factor)
                lbl_halfyear_base[ii:jj + 1] = base_val
                lbl_halfyear[ii:jj + 1] = final_val
                f_wbz_arr[ii:jj + 1] = f_wbz
                f_freq_arr[ii:jj + 1] = f_freq
                f_vol_arr[ii:jj + 1] = f_vol
                # Preisfaktor im Fenster als konstanten Wert Ã¼bernehmen
                if 'Price_Material_var' in hist_df.columns:
                    f_price_win = np.full(jj - ii + 1, f_price, dtype=float)
                else:
                    f_price_win = np.zeros(jj - ii + 1, dtype=float)
                # HÃ¤nge als temporÃ¤re Liste an; final wird volle LÃ¤nge vergeben
                ii = jj + 1
            hist_df['L_NiU_HalfYear_Base'] = lbl_halfyear_base
            hist_df['F_NiU_Factor_WBZ'] = f_wbz_arr
            hist_df['F_NiU_Factor_Freq'] = f_freq_arr
            hist_df['F_NiU_Factor_Vol'] = f_vol_arr
            # FÃ¼r Konsistenz: Preisfaktor pro Zeile berechnen (erneut, robust gegen Fenstergrenzen)
            if 'Price_Material_var' in hist_df.columns:
                p = pd.to_numeric(hist_df['Price_Material_var'], errors='coerce').fillna(0).astype(float).to_numpy()
                p_log = np.log1p(np.maximum(0.0, p))
                scale = np.nanpercentile(p_log[p_log > 0], 95) if np.any(p_log > 0) else 1.0
                f_price_arr = -0.10 * (p_log / scale) if scale > 0 else np.zeros_like(p_log)
                f_price_arr = np.clip(f_price_arr, -0.10, 0.0)
            else:
                f_price_arr = np.zeros(len(hist_df), dtype=float)
            hist_df['F_NiU_Factor_Price'] = f_price_arr
            # Finales Target mit Gesamtfaktor (Clip 0..0.40)
            total_arr = np.clip(f_wbz_arr + f_freq_arr + f_vol_arr + f_price_arr, 0.0, 0.40)
            hist_df['L_HalfYear_Target'] = lbl_halfyear_base * (1.0 + total_arr)
        # Ensure identical column structure between hist_df and dispo_df
        all_cols = list(hist_df.columns)
        for c in all_cols:
            if c not in dispo_df.columns:
                dispo_df[c] = np.nan
        dispo_df = dispo_df[all_cols]

        # write historical subset
        part_dir = out_feat / str(part)
        part_dir.mkdir(parents=True, exist_ok=True)
        if not hist_df.empty:
            _save_parquet_or_csv(hist_df, part_dir / 'features.parquet')
            try:
                hist_df.to_excel(part_dir / 'features.xlsx', index=False)
            except Exception:
                pass
        else:
            # create empty placeholder (CSV) to signal existence if parquet not available
            _save_parquet_or_csv(hist_df, part_dir / 'features.parquet')

        # write dispo subset
        part_dir_t = out_test / str(part)
        part_dir_t.mkdir(parents=True, exist_ok=True)
        if not dispo_df.empty:
            _save_parquet_or_csv(dispo_df, part_dir_t / 'features.parquet')
            try:
                dispo_df.to_excel(part_dir_t / 'features.xlsx', index=False)
            except Exception:
                pass
        else:
            _save_parquet_or_csv(dispo_df, part_dir_t / 'features.parquet')


###############################
# Selective build (feature registry)
###############################

def _list_registry_features() -> list[str]:
    """Names of selectable features exposed to the GUI."""
    return [
        # Materialpreis (direkt aus TeileWert)
        'Price_Material_var',
        # Transforms on EoD_noSiBe
        'EoD_Bestand_noSiBe_log1p',
        'EoD_Bestand_noSiBe_z_Teil',
        'EoD_Bestand_noSiBe_robz_Teil',
        'DeficitPos_log1p',
        # Demand rollings (WBZ-relative facs like in legacy pipeline)
        'DemandMean_100', 'DemandMax_100',
        'DemandMean_66', 'DemandMax_66',
        'DemandMean_50', 'DemandMax_50',
        'DemandMean_25', 'DemandMax_25',
        # Demand transforms
        'DemandMean_100_log1p', 'DemandMean_100_z_Teil', 'DemandMean_100_robz_Teil',
        'DemandMax_100_log1p', 'DemandMax_100_z_Teil', 'DemandMax_100_robz_Teil',
        'DemandMean_66_log1p', 'DemandMean_66_z_Teil', 'DemandMean_66_robz_Teil',
        'DemandMax_66_log1p', 'DemandMax_66_z_Teil', 'DemandMax_66_robz_Teil',
        'DemandMean_50_log1p', 'DemandMean_50_z_Teil', 'DemandMean_50_robz_Teil',
        'DemandMax_50_log1p', 'DemandMax_50_z_Teil', 'DemandMax_50_robz_Teil',
        'DemandMean_25_log1p', 'DemandMean_25_z_Teil', 'DemandMean_25_robz_Teil',
        'DemandMax_25_log1p', 'DemandMax_25_z_Teil', 'DemandMax_25_robz_Teil',
        # Lag features (Punkt-Lags und Mittelwert-Lags)
        # Punkt-Lags: Wert von EoD_noSiBe vor n Tagen
        'Lag_EoD_Bestand_noSiBe_7Tage',
        'Lag_EoD_Bestand_noSiBe_28Tage',
        'Lag_EoD_Bestand_noSiBe_wbzTage',
        'Lag_EoD_Bestand_noSiBe_2xwbzTage',
        # Mittelwert-Lags: gleitender Mittelwert ueber n Tage (closed=left)
        'Lag_EoD_Bestand_noSiBe_mean_7Tage',
        'Lag_EoD_Bestand_noSiBe_mean_28Tage',
        'Lag_EoD_Bestand_noSiBe_mean_wbzTage',
        'Lag_EoD_Bestand_noSiBe_mean_2xwbzTage',
    ]


def _list_registry_labels() -> list[str]:
    return [
        'L_WBZ_BlockMinAbs',
        'L_HalfYear_Target',
    ]


def list_available_feature_names() -> list[str]:
    return _list_registry_features()


def list_available_label_names() -> list[str]:
    return _list_registry_labels()


def _compute_selected_features(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Compute selected features on top of a core dataframe.

    Core df must contain: Teil, Datum, EoD_Bestand_noSiBe, WBZ_Days.
    """
    out = df.copy()
    # base series
    eod = pd.to_numeric(out.get('EoD_Bestand_noSiBe'), errors='coerce')
    wbz = pd.to_numeric(out.get('WBZ_Days'), errors='coerce').fillna(14)

    # demand events from inventory deltas (legacy convention)
    demand_series = (out['EoD_Bestand_noSiBe'].shift(1) - out['EoD_Bestand_noSiBe']).clip(lower=0).fillna(0)

    # helpers
    def _z_by_part(s: pd.Series) -> pd.Series:
        if 'Teil' not in out.columns:
            return s * 0
        grp = out.groupby('Teil')[s.name]
        mean = grp.transform('mean')
        std = grp.transform('std').replace(0, np.nan)
        return ((s - mean) / std).fillna(0)

    def _robz_by_part(s: pd.Series) -> pd.Series:
        if 'Teil' not in out.columns:
            return s * 0
        grp = out.groupby('Teil')[s.name]
        med = grp.transform('median')
        q75 = grp.transform(lambda x: x.quantile(0.75))
        q25 = grp.transform(lambda x: x.quantile(0.25))
        iqr = (q75 - q25).replace(0, np.nan)
        return ((s - med) / iqr).fillna(0)

    # --- EoD transforms ---
    if 'EoD_Bestand_noSiBe_log1p' in selected:
        out['EoD_Bestand_noSiBe_log1p'] = np.log1p(eod.clip(lower=0))
    if 'EoD_Bestand_noSiBe_z_Teil' in selected:
        s = pd.to_numeric(out['EoD_Bestand_noSiBe'], errors='coerce').fillna(0)
        s.name = 'EoD_Bestand_noSiBe'
        out['EoD_Bestand_noSiBe_z_Teil'] = _z_by_part(s)
    if 'EoD_Bestand_noSiBe_robz_Teil' in selected:
        s = pd.to_numeric(out['EoD_Bestand_noSiBe'], errors='coerce').fillna(0)
        s.name = 'EoD_Bestand_noSiBe'
        out['EoD_Bestand_noSiBe_robz_Teil'] = _robz_by_part(s)
    if 'DeficitPos_log1p' in selected:
        out['DeficitPos_log1p'] = np.log1p((-eod).clip(lower=0))

    # --- Demand windows (WBZ-relative factors like legacy: 100,66,50,25) ---
    fac_map = {
        '100': 1.0,
        '66': 2/3,
        '50': 1/2,
        '25': 1/4,
    }
    for key, fac in fac_map.items():
        w = max(1, int(round((wbz if np.isscalar(wbz) else wbz.iloc[0]) * fac)))
        mean_name = f'DemandMean_{key}'
        max_name = f'DemandMax_{key}'
        # prÃ¼fen, ob Base benÃ¶tigt wird (Base oder ein Transform ist ausgewÃ¤hlt)
        mean_trans = [f'{mean_name}_log1p', f'{mean_name}_z_Teil', f'{mean_name}_robz_Teil']
        max_trans = [f'{max_name}_log1p', f'{max_name}_z_Teil', f'{max_name}_robz_Teil']
        need_mean_base = (mean_name in selected) or any(t in selected for t in mean_trans)
        need_max_base = (max_name in selected) or any(t in selected for t in max_trans)
        # Base berechnen, wenn benÃ¶tigt (auch wenn Base selbst nicht ausgewÃ¤hlt ist)
        if need_mean_base and mean_name not in out.columns:
            out[mean_name] = demand_series.shift(1).rolling(w, min_periods=1).mean()
        if need_max_base and max_name not in out.columns:
            out[max_name] = demand_series.shift(1).rolling(w, min_periods=1).max()
        # transforms
        if f'{mean_name}_log1p' in selected and mean_name in out.columns:
            out[f'{mean_name}_log1p'] = np.log1p(pd.to_numeric(out[mean_name], errors='coerce').clip(lower=0))
        if f'{max_name}_log1p' in selected and max_name in out.columns:
            out[f'{max_name}_log1p'] = np.log1p(pd.to_numeric(out[max_name], errors='coerce').clip(lower=0))
        if f'{mean_name}_z_Teil' in selected and mean_name in out.columns:
            s = pd.to_numeric(out[mean_name], errors='coerce').fillna(0); s.name = mean_name
            out[f'{mean_name}_z_Teil'] = _z_by_part(s)
        if f'{max_name}_z_Teil' in selected and max_name in out.columns:
            s = pd.to_numeric(out[max_name], errors='coerce').fillna(0); s.name = max_name
            out[f'{max_name}_z_Teil'] = _z_by_part(s)
        if f'{mean_name}_robz_Teil' in selected and mean_name in out.columns:
            s = pd.to_numeric(out[mean_name], errors='coerce').fillna(0); s.name = mean_name
            out[f'{mean_name}_robz_Teil'] = _robz_by_part(s)
        if f'{max_name}_robz_Teil' in selected and max_name in out.columns:
            s = pd.to_numeric(out[max_name], errors='coerce').fillna(0); s.name = max_name
            out[f'{max_name}_robz_Teil'] = _robz_by_part(s)
        # Wenn Base nicht explizit ausgewÃ¤hlt wurde, am Ende wieder entfernen
        if need_mean_base and (mean_name not in selected) and mean_name in out.columns:
            out.drop(columns=[mean_name], inplace=True)
        if need_max_base and (max_name not in selected) and max_name in out.columns:
            out.drop(columns=[max_name], inplace=True)

    # --- Lags ---
    # Punkt-Lags (reiner Rueckblick auf den Wert vor N Tagen)
    if 'Lag_EoD_Bestand_noSiBe_7Tage' in selected:
        out['Lag_EoD_Bestand_noSiBe_7Tage'] = eod.shift(7)
    if 'Lag_EoD_Bestand_noSiBe_28Tage' in selected:
        out['Lag_EoD_Bestand_noSiBe_28Tage'] = eod.shift(28)
    if 'Lag_EoD_Bestand_noSiBe_wbzTage' in selected:
        w = int(max(1, (wbz.iloc[0] if len(wbz) else 14)))
        out['Lag_EoD_Bestand_noSiBe_wbzTage'] = eod.shift(w)
    if 'Lag_EoD_Bestand_noSiBe_2xwbzTage' in selected:
        w = int(max(1, (wbz.iloc[0] if len(wbz) else 14) * 2))
        out['Lag_EoD_Bestand_noSiBe_2xwbzTage'] = eod.shift(w)
    # Mittelwert-Lags (closed=left)
    if 'Lag_EoD_Bestand_noSiBe_mean_7Tage' in selected:
        out['Lag_EoD_Bestand_noSiBe_mean_7Tage'] = eod.shift(1).rolling(7, min_periods=1).mean()
    if 'Lag_EoD_Bestand_noSiBe_mean_28Tage' in selected:
        out['Lag_EoD_Bestand_noSiBe_mean_28Tage'] = eod.shift(1).rolling(28, min_periods=1).mean()
    if 'Lag_EoD_Bestand_noSiBe_mean_wbzTage' in selected:
        w = int(max(1, (wbz.iloc[0] if len(wbz) else 14)))
        out['Lag_EoD_Bestand_noSiBe_mean_wbzTage'] = eod.shift(1).rolling(w, min_periods=1).mean()
    if 'Lag_EoD_Bestand_noSiBe_mean_2xwbzTage' in selected:
        w = int(max(1, (wbz.iloc[0] if len(wbz) else 14) * 2))
        out['Lag_EoD_Bestand_noSiBe_mean_2xwbzTage'] = eod.shift(1).rolling(w, min_periods=1).mean()

    return out


def _compute_selected_labels(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    out = df.copy()
    if 'L_WBZ_BlockMinAbs' in selected:
        # WBZ-window min of EoD_noSiBe (positive)
        dates = pd.to_datetime(out['Datum'])
        eod = pd.to_numeric(out['EoD_Bestand_noSiBe'], errors='coerce').fillna(0).to_numpy()
        wbz = pd.to_numeric(out.get('WBZ_Days'), errors='coerce').fillna(14)
        darr = dates.to_numpy()
        block = np.zeros(len(out), dtype=float)
        for i in range(len(out)):
            h = int(max(1, round(wbz.iloc[i] if len(wbz) else 14)))
            end = darr[i] + np.timedelta64(h, 'D')
            mask = (darr >= darr[i]) & (darr < end)
            y = eod[mask]
            mmin = float(np.nanmin(y)) if y.size else 0.0
            block[i] = max(0.0, -mmin)
        # optional Faktoren (wie im Hauptpfad), grob angenÃ¤hert per Frequenz/VolatilitÃ¤t
        # (bei Bedarf weiter verfeinern)
        out['L_WBZ_BlockMinAbs'] = block
    if 'L_HalfYear_Target' in selected:
        # Ableitung aus vorhandenem Label, falls vorhanden
        if 'LABLE_HalfYear_Target' in out.columns:
            out['L_HalfYear_Target'] = pd.to_numeric(out['LABLE_HalfYear_Target'], errors='coerce')
    return out


def _build_core_by_part(raw_dir: str, xlsx_path: str) -> dict[str, dict]:
    """Lightweight core build: only keys, core stocks, WBZ, and SiBe history.

    Returns: part -> core dataframe with at least
      Teil, Datum, F_NiU_EoD_Bestand, F_NiU_Hinterlegter SiBe, EoD_Bestand_noSiBe, WBZ_Days
    """
    column_map = _load_column_map(xlsx_path)
    tables = load_all_tables(raw_dir, column_map)
    # aggregate
    agg_tables: Dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if name == 'Lagerbew':
            agg_tables[name] = _prepare_lagerbew(df)
        elif name == 'Dispo':
            agg_tables[name] = _prepare_dispo(df)
        elif name == 'SiBeVerlauf':
            df_s = df.copy()
            dchange = None
            for c in df_s.columns:
                if 'datum' in str(c).lower() and 'nderung' in str(c).lower():
                    dchange = c; break
            if dchange and 'AudEreignis-ZeitPkt' not in df_s.columns:
                df_s.rename(columns={dchange: 'AudEreignis-ZeitPkt'}, inplace=True)
            if 'aktiver SiBe' in df_s.columns and 'Im Sytem hinterlgeter SiBe' not in df_s.columns:
                df_s.rename(columns={'aktiver SiBe': 'Im Sytem hinterlgeter SiBe'}, inplace=True)
            if 'Alter_SiBe' not in df_s.columns:
                for c in list(df_s.columns):
                    if 'alter' in str(c).lower() and 'sibe' in str(c).lower():
                        df_s.rename(columns={c: 'Alter_SiBe'}, inplace=True); break
            agg_tables[name] = _aggregate_dataset(df_s, ['AudEreignis-ZeitPkt'], last_cols=['Im Sytem hinterlgeter SiBe', 'Alter_SiBe'])
        elif name in {'Bestand', 'Teilestamm'}:
            df2 = df.copy(); df2['Datum'] = df2['ExportDatum']
            agg = {c: 'first' for c in df2.columns if c not in {'Teil','Datum','Dataset','ExportDatum'}}
            agg_tables[name] = df2.groupby(['Teil','Datum'], as_index=False).agg(agg)
        elif name == 'TeileWert':
            df_tw = df.copy()
            price_col = None
            for c in df_tw.columns:
                if 'material' in str(c).lower() and ('dpr' in str(c).lower() or 'preis' in str(c).lower() or 'wert' in str(c).lower()):
                    price_col = c; break
            if price_col is not None:
                df_tw = df_tw[['Teil','ExportDatum',price_col]].copy()
                df_tw.rename(columns={price_col:'Price_Material_var','ExportDatum':'Datum'}, inplace=True)
                df_tw['Datum'] = pd.to_datetime(df_tw['Datum'], errors='coerce').dt.floor('D')
                df_tw = df_tw.dropna(subset=['Datum'])
                agg_tables[name] = df_tw.groupby(['Teil','Datum'], as_index=False)['Price_Material_var'].last()

    parts: set[str] = set()
    for df in agg_tables.values():
        parts.update(df['Teil'].astype(str).unique())

    processed: dict[str, dict] = {}
    for part in sorted(parts):
        data: Dict[str, pd.DataFrame] = {}
        for name, df in agg_tables.items():
            part_df = df[df['Teil'].astype(str) == str(part)].copy()
            if not part_df.empty:
                data[name] = part_df
        if not data:
            continue
        date_set: set[pd.Timestamp] = set()
        first_dispo_date = None
        if 'Lagerbew' in data:
            data['Lagerbew']['Datum'] = pd.to_datetime(data['Lagerbew']['Datum'])
            date_set.update(data['Lagerbew']['Datum'].unique())
        if 'Dispo' in data:
            data['Dispo']['Datum'] = pd.to_datetime(data['Dispo']['Datum'])
            date_set.update(data['Dispo']['Datum'].unique())
            first_dispo_date = data['Dispo']['Datum'].min()
        baseline = 0.0; baseline_date = None
        if first_dispo_date is not None and 'Lagerbew' in data:
            lb_before = data['Lagerbew'][data['Lagerbew']['Datum'] <= first_dispo_date]
            if not lb_before.empty:
                last_lb = lb_before.sort_values('Datum').iloc[-1]
                baseline = float(last_lb['Lagerbestand'])
                baseline_date = last_lb['Datum']
        if baseline_date is None and 'Bestand' in data:
            best = data['Bestand'][['Datum','Bestand']].copy(); best['Datum'] = pd.to_datetime(best['Datum'])
            if first_dispo_date is not None:
                best_before = best[best['Datum'] <= first_dispo_date]
                if not best_before.empty:
                    last_best = best_before.sort_values('Datum').iloc[-1]
                    baseline = float(last_best['Bestand']); baseline_date = last_best['Datum']
            elif not best.empty:
                last_best = best.sort_values('Datum').iloc[-1]
                baseline = float(last_best['Bestand']); baseline_date = last_best['Datum']
        if baseline_date is not None:
            date_set.add(baseline_date)
        if not date_set:
            continue
        # daily grid
        full_days = pd.date_range(min(date_set), max(date_set), freq='D')
        feat = pd.DataFrame({'Datum': full_days}); feat['Teil'] = part
        # merge Lagerbew
        if 'Lagerbew' in data:
            lb = data['Lagerbew'][['Datum','Lagerbestand']]
            feat = feat.merge(lb, on='Datum', how='left')
        else:
            feat['Lagerbestand'] = np.nan
        if baseline_date is not None and feat.loc[feat['Datum']==baseline_date, 'Lagerbestand'].isna().all():
            feat.loc[feat['Datum']==baseline_date, 'Lagerbestand'] = baseline
        # merge dispo
        if 'Dispo' in data:
            dispo = data['Dispo'][['Datum','net','Bedarfsmenge']]
            feat = feat.merge(dispo, on='Datum', how='left')
            feat['net'] = feat['net'].fillna(0); feat['Bedarfsmenge'] = feat['Bedarfsmenge'].fillna(0)
            feat['cum_net'] = 0.0
            if first_dispo_date is not None:
                mask = feat['Datum'] >= first_dispo_date
                feat.loc[mask,'cum_net'] = feat.loc[mask,'net'].cumsum()
        else:
            feat['net'] = 0; feat['Bedarfsmenge'] = 0; feat['cum_net'] = 0
        # EoD
        if first_dispo_date is not None:
            pre = feat['Datum'] < first_dispo_date
            feat.loc[pre,'EoD_Bestand'] = pd.to_numeric(feat['Lagerbestand'], errors='coerce').ffill()
            post = feat['Datum'] >= first_dispo_date
            feat.loc[post,'EoD_Bestand'] = baseline + feat.loc[post,'cum_net']
        else:
            feat['EoD_Bestand'] = pd.to_numeric(feat['Lagerbestand'], errors='coerce').ffill()
        feat['EoD_Bestand'] = pd.to_numeric(feat['EoD_Bestand'], errors='coerce').fillna(0)
        # Preis (TeileWert) asof-Join
        if 'TeileWert' in data:
            tw = data['TeileWert'][['Datum','Price_Material_var']].copy()
            if not tw.empty:
                tw = tw.dropna(subset=['Datum']).sort_values('Datum')
                feat = pd.merge_asof(feat.sort_values('Datum'), tw, on='Datum', direction='backward')

        # SiBe Verlauf (asof)
        if 'SiBeVerlauf' in data:
            sibe = data['SiBeVerlauf'][['Datum','Im Sytem hinterlgeter SiBe','Alter_SiBe']].copy()
            sibe['Datum'] = pd.to_datetime(sibe['Datum'], errors='coerce').dt.floor('D'); sibe = sibe.dropna(subset=['Datum']).sort_values('Datum')
            merged = pd.merge_asof(feat.sort_values('Datum'), sibe, on='Datum', direction='backward')
            merged['Im Sytem hinterlgeter SiBe'] = pd.to_numeric(merged['Im Sytem hinterlgeter SiBe'], errors='coerce').fillna(0)
            # Backfill vor erstem Ã„nderungsdatum mit Alter_SiBe (falls vorhanden)
            try:
                first_change = sibe['Datum'].min()
                alter_val = pd.to_numeric(
                    sibe.loc[sibe['Datum'] == first_change, 'Alter_SiBe'], errors='coerce'
                ).iloc[0]
            except Exception:
                first_change = None
                alter_val = None
            if first_change is not None and pd.notna(alter_val):
                pre_mask = (merged['Datum'] < first_change)
                merged.loc[pre_mask, 'Im Sytem hinterlgeter SiBe'] = float(alter_val)
            feat = merged.rename(columns={'Im Sytem hinterlgeter SiBe': 'F_NiU_Hinterlegter SiBe'})
        else:
            feat['F_NiU_Hinterlegter SiBe'] = 0
        feat['F_NiU_EoD_Bestand'] = feat['EoD_Bestand']
        feat['Hinterlegter_SiBe'] = feat['F_NiU_Hinterlegter SiBe']
        feat['EoD_Bestand_noSiBe'] = feat['EoD_Bestand'] - feat['Hinterlegter_SiBe']
        feat['Flag_StockOut'] = (feat['EoD_Bestand_noSiBe'] <= 0).astype(int)
        # WBZ from Teilestamm
        wbz = None
        if 'Teilestamm' in data:
            w = data['Teilestamm']['WBZ'].dropna()
            if not w.empty:
                wbz = float(pd.to_numeric(w.iloc[0], errors='coerce'))
        feat['WBZ_Days'] = wbz
        processed[part] = {"df": feat, "first_dispo_date": first_dispo_date}
    return processed


def run_pipeline_selective(
    raw_dir: str,
    output_dir: str,
    features_to_build: list[str],
    labels_to_build: list[str],
) -> None:
    try:
        default_xlsx = Path(__file__).resolve().parents[1] / 'Spaltenbedeutung.xlsx'
    except Exception:
        default_xlsx = Path('Spaltenbedeutung.xlsx')
    core = _build_core_by_part(raw_dir, str(default_xlsx))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    test_root = out_root.parent / f"{out_root.name}_Test"
    test_root.mkdir(parents=True, exist_ok=True)
    for part, bundle in core.items():
        df = bundle["df"]
        first_dispo_date = bundle.get("first_dispo_date")
        if df.empty:
            continue
        df = df.sort_values('Datum').reset_index(drop=True)
        # compute selected features/labels
        df_feat = _compute_selected_features(df, features_to_build)
        df_lab = _compute_selected_labels(df_feat, labels_to_build)
        # final selection: locked + selected
        locked = ['Teil', 'Datum', 'F_NiU_EoD_Bestand', 'F_NiU_Hinterlegter SiBe', 'EoD_Bestand_noSiBe', 'WBZ_Days']
        final_cols = [c for c in locked + features_to_build + labels_to_build if c in df_lab.columns]
        out_full = df_lab[final_cols].copy()
        # split into historical and dispo period per part
        if isinstance(first_dispo_date, pd.Timestamp):
            hist_df = out_full[out_full['Datum'] < first_dispo_date].copy()
            dispo_df = out_full[out_full['Datum'] >= first_dispo_date].copy()
        else:
            hist_df = out_full.copy()
            dispo_df = out_full.iloc[0:0].copy()

        # write hist (features)
        part_dir = out_root / str(part)
        part_dir.mkdir(parents=True, exist_ok=True)
        try:
            hist_df.to_parquet(part_dir / 'features.parquet', index=False)
        except Exception:
            hist_df.to_csv(part_dir / 'features.csv', index=False)
        try:
            hist_df.to_excel(part_dir / 'features.xlsx', index=False)
        except Exception:
            pass

        # write dispo (test set) with identical schema
        # ensure identical columns
        for c in final_cols:
            if c not in dispo_df.columns:
                dispo_df[c] = np.nan
        dispo_df = dispo_df[final_cols]
        part_dir_t = test_root / str(part)
        part_dir_t.mkdir(parents=True, exist_ok=True)
        try:
            dispo_df.to_parquet(part_dir_t / 'features.parquet', index=False)
        except Exception:
            dispo_df.to_csv(part_dir_t / 'features.csv', index=False)
        try:
            dispo_df.to_excel(part_dir_t / 'features.xlsx', index=False)
        except Exception:
            pass


def run_pipeline(raw_dir: str, output_dir: str = 'Features') -> None:
    """Complete preprocessing pipeline producing split outputs per part.

    - ``output_dir`` contains historical (pre-Dispo) subsets per part.
    - ``Test_Set`` sibling folder contains Dispo-period subsets per part.
    """
    # Resolve column spec sheet next to the AGENTS_MAKE_ML package
    try:
        default_xlsx = Path(__file__).resolve().parents[1] / 'Spaltenbedeutung.xlsx'
    except Exception:
        default_xlsx = Path('Spaltenbedeutung.xlsx')
    features = build_features_by_part(raw_dir, xlsx_path=str(default_xlsx))
    # Write historical into output_dir and Dispo-period into a sibling folder at the same level
    out_base = Path(output_dir)
    test_subdir = str(out_base.parent / 'Test_Set')
    save_feature_folders_split(features, output_dir, test_subdir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process raw CSV data')
    parser.add_argument('--input', default='Rohdaten', help='Input directory with raw CSVs')
    parser.add_argument('--output', default='Features', help='Output directory for feature folders')
    args = parser.parse_args()
    run_pipeline(args.input, args.output)

