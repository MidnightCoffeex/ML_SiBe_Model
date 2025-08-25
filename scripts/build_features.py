#!/usr/bin/env python3
"""Command-line wrapper for feature engineering.

This version can generate separate historical (H) and dispo (D) feature
tables.  Arguments that are not supplied are requested from the user one after
another, keeping the behaviour similar to the previous interactive scripts.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_pipeline


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature tables")
    parser.add_argument("--mode", choices=["H", "D", "both"], help="Welche Tabellen sollen erzeugt werden")
    parser.add_argument("--cutoff-date", help="Stichtag YYYY-MM-DD für Trennung H/D")
    parser.add_argument("--raw-dir", help="Verzeichnis mit Rohdaten")
    parser.add_argument("--out-h", help="Ausgabeverzeichnis Historie")
    parser.add_argument("--out-d", help="Ausgabeverzeichnis Dispo")
    args = parser.parse_args()

    if not args.mode:
        args.mode = input("Modus {H,D,both} [both]: ") or "both"
    if not args.cutoff_date:
        args.cutoff_date = input("Cutoff-Date YYYY-MM-DD [leer = max Datum]: ") or None
    if not args.raw_dir:
        args.raw_dir = input("Pfad zu Rohdaten [Rohdaten]: ") or "Rohdaten"
    if not args.out_h:
        args.out_h = input("Ausgabeordner Historie [Features_H]: ") or "Features_H"
    if not args.out_d:
        args.out_d = input("Ausgabeordner Dispo [Features_D]: ") or "Features_D"

    cutoff_dt = _parse_date(args.cutoff_date)

    hist = {}
    dispo = {}

    if args.mode in {"H", "both"}:
        hist = data_pipeline.build_historical_features(args.raw_dir, cutoff_dt)
    if args.mode in {"D", "both"}:
        if not hist:
            hist = data_pipeline.build_historical_features(args.raw_dir, cutoff_dt)
        # determine start/end for dispo simulation
        seed_eod = {k: df.tail(1) for k, df in hist.items()}
        start = max(df["Datum"].max() for df in hist.values()) + timedelta(days=1)
        end = start + timedelta(days=30)
        dispo = data_pipeline.build_dispo_features(args.raw_dir, seed_eod, start.date(), end.date())

    if args.mode == "H":
        aligned_h = hist
        aligned_d = {}
    elif args.mode == "D":
        aligned_h = {}
        aligned_d = dispo
    else:
        aligned_h = {}
        aligned_d = {}
        for part in hist.keys() | dispo.keys():
            h = hist.get(part, data_pipeline.pd.DataFrame())
            d = dispo.get(part, data_pipeline.pd.DataFrame())
            h_aligned, d_aligned = data_pipeline.align_schema(h, d)
            aligned_h[part] = h_aligned
            aligned_d[part] = d_aligned

    # write output
    for part, df in aligned_h.items():
        out_dir = Path(args.out_h) / str(part)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "features.csv", index=False)
        try:
            df.to_parquet(out_dir / "features.parquet", index=False)
        except Exception:
            pass
        try:
            df.to_excel(out_dir / "features.xlsx", index=False)
        except Exception:
            pass

    for part, df in aligned_d.items():
        out_dir = Path(args.out_d) / str(part)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "features.csv", index=False)
        try:
            df.to_parquet(out_dir / "features.parquet", index=False)
        except Exception:
            pass
        try:
            df.to_excel(out_dir / "features.xlsx", index=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
