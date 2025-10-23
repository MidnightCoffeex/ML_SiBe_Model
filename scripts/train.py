#!/usr/bin/env python3
"""Wrapper script for model training.

Erweiterungen:
- Bei ALL-Training werden alle Teile global zusammengeführt und nach Datum (und Teil) sortiert.
- Splitting erfolgt datumsbasiert an Tagesgrenzen (TimeSeriesSplit über eindeutige Datumstage),
  damit keine Zeilen desselben Tages train/test mischen.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import train_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the safety stock model")
    parser.add_argument("--data", help="Root directory with feature folders")
    parser.add_argument("--part", help="Specific part number or 'ALL'")
    parser.add_argument("--model-dir", help="Base directory for models")
    parser.add_argument("--model-id", help="Run identifier")
    parser.add_argument(
        "--models",
        help="Comma separated model types (gb,xgb,lgbm) or 'ALL'",
    )
    parser.add_argument(
        "--targets",
        default="LABLE_HalfYear_Target",
        help="Comma separated target column names",
    )
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--cv", type=int, help="Number of cross-validation folds")
    parser.add_argument("--ts_scope", help="Timeseries scope: global|local")
    parser.add_argument("--weight_scheme", help="Weighting: none|blockmin|flag")
    parser.add_argument("--weight_factor", type=float, help="Weight factor for selected scheme")
    args = parser.parse_args()

    if not args.data:
        args.data = input("Pfad zu Features [Features]: ") or "Features"
    if not args.part:
        args.part = input("Teil-Nummer oder ALL [ALL]: ") or "ALL"
    if not args.model_dir:
        args.model_dir = "Modelle"

    # Model type selection (interactive if not provided)
    if not args.models:
        entry = input("Modelltyp(e) [gb|xgb|lgbm|ALL] [gb]: ")
        args.models = entry.strip() if entry else "gb"

    # Hyperparameters with explanatory prompts
    if args.n_estimators is None:
        entry = input("n_estimators [100] – Anzahl Bäume/Iterationen (höher = genauer, aber langsamer/überfitting möglich): ")
        args.n_estimators = int(entry) if entry else 100
    if args.learning_rate is None:
        entry = input("learning_rate [0.1] – Schrittweite je Baum (kleiner = stabiler, aber mehr Bäume nötig): ")
        args.learning_rate = float(entry) if entry else 0.1
    if args.max_depth is None:
        entry = input("max_depth [3] – maximale Baumtiefe (höher = komplexer, Risiko Überanpassung): ")
        args.max_depth = int(entry) if entry else 3
    if args.subsample is None:
        entry = input("subsample [1.0] – Stichprobenanteil pro Baum (kleiner kann generalisieren, 1.0 = alle Daten): ")
        args.subsample = float(entry) if entry else 1.0
    if args.cv is None:
        entry = input("cv_splits [0] – Anzahl Zeitreihen-CV-Folds (0 = aus): ")
        args.cv = int(entry) if entry else 0

    # Timeseries-Scope abfragen, falls nicht gesetzt
    if not getattr(args, 'ts_scope', None):
        entry = input("Timeseries-Scope [global|local] [global]: ")
        args.ts_scope = entry.strip() if entry else "global"

    part_name = args.part if args.part else "ALL"

    # Expand 'ALL' to all three
    if args.models.strip().upper() == 'ALL':
        model_types = ['gb', 'xgb', 'lgbm']
    else:
        model_types = [m.strip() for m in args.models.split(',') if m.strip()]

    if not args.model_id:
        part_dir = Path(args.model_dir) / part_name / model_types[0]
        existing = [int(p.name) for p in part_dir.glob('*') if p.is_dir() and p.name.isdigit()]
        next_id = max(existing, default=0) + 1
        args.model_id = str(next_id)
        print(f"Automatisch gewählte Nummer: {args.model_id}")

    if args.part.upper() == "ALL":
        frames = []
        for f in Path(args.data).glob('*/features.parquet'):
            frames.append(pd.read_parquet(f))
        df = pd.concat(frames, ignore_index=True)
        # global nach Datum (und Teil) sortieren
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datum']).sort_values(['Datum', 'Teil']).reset_index(drop=True)
        part_name = "ALL"
    else:
        df = pd.read_parquet(Path(args.data) / args.part / 'features.parquet')
        # sicherheitshalber nach Datum sortieren
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datum']).sort_values(['Datum']).reset_index(drop=True)
        part_name = args.part

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    # Datumsbasierte Splits an Tagesgrenzen: wir führen TimeSeriesSplit über die eindeutigen Tage aus
    # und mappen anschließend zurück auf Zeilenindizes des gefilterten DataFrames.
    def compute_date_based_split_indices(df_full: pd.DataFrame, targets: list[str], n_splits: int = 5):
        df_f = df_full.dropna(subset=targets).copy()
        df_f['__d'] = pd.to_datetime(df_f['Datum'], errors='coerce').dt.floor('D')
        uniq_days = pd.Index(df_f['__d'].unique()).sort_values()
        # Falls zu wenig Tage vorhanden sind, reduzieren
        ns = min(n_splits, max(2, len(uniq_days) - 1))
        tscv_local = TimeSeriesSplit(n_splits=ns)
        splits_local = list(tscv_local.split(range(len(uniq_days))))
        # letzte zwei Folds: (-2) für Val, (-1) für Test
        tr_d_ix, val_d_ix = splits_local[-2]
        trfull_d_ix, test_d_ix = splits_local[-1]
        tr_days = set(uniq_days[tr_d_ix])
        val_days = set(uniq_days[val_d_ix])
        trfull_days = set(uniq_days[trfull_d_ix])
        test_days = set(uniq_days[test_d_ix])
        arr = df_f['__d'].to_numpy()
        train_idx = np.nonzero(pd.Series(arr).isin(tr_days).to_numpy())[0]
        val_idx = np.nonzero(pd.Series(arr).isin(val_days).to_numpy())[0]
        train_full_idx = np.nonzero(pd.Series(arr).isin(trfull_days).to_numpy())[0]
        test_idx = np.nonzero(pd.Series(arr).isin(test_days).to_numpy())[0]
        return (train_idx, val_idx, train_full_idx, test_idx)

    # Sonderfall: ALL + lokaler Zeitreihen-Pool (pro Teil separat)
    if args.part.upper() == 'ALL' and (args.ts_scope or '').strip().lower() == 'local':
        parts = sorted(p.parent.name for p in Path(args.data).glob('*/features.parquet'))
        # Gewichtungs-Parameter absichern (falls nicht via Parser gesetzt)
        if not getattr(args, 'weight_scheme', None):
            entry = input("Gewichtungs-Schema [none|blockmin|flag] [blockmin]: ")
            args.weight_scheme = entry.strip() if entry else "blockmin"
        if getattr(args, 'weight_factor', None) is None:
            entry = input("Gewichtungs-Faktor [5.0]: ")
            try:
                args.weight_factor = float(entry) if entry else 5.0
            except Exception:
                args.weight_factor = 5.0
        for p in parts:
            df_p = pd.read_parquet(Path(args.data) / p / 'features.parquet')
            df_p['Datum'] = pd.to_datetime(df_p['Datum'], errors='coerce')
            df_p = df_p.dropna(subset=['Datum']).sort_values(['Datum']).reset_index(drop=True)
            # Tagesgrenzen-Splits pro Teil
            def _split(df_full: pd.DataFrame):
                df_f = df_full.dropna(subset=target_list).copy()
                df_f['__d'] = pd.to_datetime(df_f['Datum'], errors='coerce').dt.floor('D')
                uniq_days = pd.Index(df_f['__d'].unique()).sort_values()
                ns = min(5, max(2, len(uniq_days) - 1))
                tscv_local = TimeSeriesSplit(n_splits=ns)
                sl = list(tscv_local.split(range(len(uniq_days))))
                tr_d_ix, val_d_ix = sl[-2]
                trfull_d_ix, test_d_ix = sl[-1]
                tr_days = set(uniq_days[tr_d_ix]); val_days = set(uniq_days[val_d_ix])
                trfull_days = set(uniq_days[trfull_d_ix]); test_days = set(uniq_days[test_d_ix])
                arr = df_f['__d'].to_numpy()
                import numpy as _np
                train_idx = _np.nonzero(pd.Series(arr).isin(tr_days).to_numpy())[0]
                val_idx = _np.nonzero(pd.Series(arr).isin(val_days).to_numpy())[0]
                train_full_idx = _np.nonzero(pd.Series(arr).isin(trfull_days).to_numpy())[0]
                test_idx = _np.nonzero(pd.Series(arr).isin(test_days).to_numpy())[0]
                return (train_idx, val_idx, train_full_idx, test_idx)
            split_indices = _split(df_p)
            for mtype in model_types:
                # Zielverzeichnis ermitteln (ID ggf. auto)
                out_dir = Path(args.model_dir) / p / mtype / (args.model_id if args.model_id else '1')
                if not args.model_id:
                    base = out_dir.parent; base.mkdir(parents=True, exist_ok=True)
                    existing = [int(d.name) for d in base.glob('*') if d.is_dir() and d.name.isdigit()]
                    out_dir = base / str(max(existing, default=0) + 1)
                out_dir.mkdir(parents=True, exist_ok=True)
                model_path = out_dir / 'model.joblib'
                print(f"[Teil {p}] Model-{mtype}: n_estimators={args.n_estimators}, learning_rate={args.learning_rate}, max_depth={args.max_depth}, subsample={args.subsample}")
                print(f"Speichere Modell in {out_dir}")
                train_model.run_training_df(
                    df_p,
                    str(model_path),
                    target_list,
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=args.subsample,
                    cv_splits=args.cv if args.cv and args.cv > 1 else None,
                    model_type=mtype,
                    split_indices=split_indices,
                    weight_scheme=args.weight_scheme,
                    weight_factor=args.weight_factor,
                )
        return

    split_indices = compute_date_based_split_indices(df, target_list, n_splits=5)

    # Falls die Gewichtungsparameter fehlen, hier interaktiv abfragen
    try:
        _ = args.weight_scheme
    except AttributeError:
        args.weight_scheme = None
    try:
        _ = args.weight_factor
    except AttributeError:
        args.weight_factor = None
    if not args.weight_scheme:
        entry = input("Gewichtungs-Schema [none|blockmin|flag] [blockmin]: ")
        args.weight_scheme = entry.strip() if entry else "blockmin"
    if args.weight_factor is None:
        entry = input("Gewichtungs-Faktor [5.0]: ")
        try:
            args.weight_factor = float(entry) if entry else 5.0
        except Exception:
            args.weight_factor = 5.0

    for mtype in model_types:
        out_dir = Path(args.model_dir) / part_name / mtype / args.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        print(
            "Model-{} wird mit diesen Parametern trainiert: n_estimators={}, "
            "learning_rate={}, max_depth={}, subsample={}.".format(
                mtype,
                args.n_estimators,
                args.learning_rate,
                args.max_depth,
                args.subsample,
            )
        )
        print(f"Speichere Modell in {out_dir}")
        train_model.run_training_df(
            df,
            str(model_path),
            target_list,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            cv_splits=args.cv if args.cv and args.cv > 1 else None,
            model_type=mtype,
            split_indices=split_indices,
            weight_scheme=args.weight_scheme,
            weight_factor=args.weight_factor,
        )


if __name__ == "__main__":
    main()

