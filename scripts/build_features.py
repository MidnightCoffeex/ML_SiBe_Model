#!/usr/bin/env python3
"""Command-line wrapper for the feature pipeline."""
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_pipeline


def _write_part_tables(features: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, df in features.items():
        pdir = out_dir / str(part)
        pdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(pdir / "features.csv", index=False)
        try:
            df.to_parquet(pdir / "features.parquet", index=False)
        except Exception:
            pass
        try:
            df.to_excel(pdir / "features.xlsx", index=False)
        except Exception:
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature tables")
    parser.add_argument("--mode", choices=["H", "D", "both"], default="both")
    parser.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD for splitting history and dispo")
    parser.add_argument("--raw-dir", default="Rohdaten", help="Input raw data directory")
    parser.add_argument("--out-h", default="Features_H", help="Output directory for historical features")
    parser.add_argument("--out-d", default="Features_D", help="Output directory for dispo features")
    args = parser.parse_args()

    hist = {}
    dispo = {}
    cutoff = None

    if args.mode in {"H", "both"}:
        hist, cutoff = data_pipeline.build_historical_features(args.raw_dir, args.cutoff_date)

    if args.mode in {"D", "both"}:
        if cutoff is None:
            # Need historical features to determine seed
            hist, cutoff = data_pipeline.build_historical_features(args.raw_dir, args.cutoff_date)
        seeds = []
        for part, df in hist.items():
            last = df[df["Datum"] == cutoff]
            if not last.empty:
                seeds.append({"Teil": part, "EoD_Bestand": float(last["EoD_Bestand"].iloc[0])})
        seed_df = pd.DataFrame(seeds)
        start = cutoff + pd.Timedelta(days=1)
        dispo = data_pipeline.build_dispo_features(args.raw_dir, seed_df, start)

    if args.mode == "both":
        for part in set(hist.keys()) | set(dispo.keys()):
            h = hist.get(part, pd.DataFrame(columns=["Teil", "Datum"]))
            d = dispo.get(part, pd.DataFrame(columns=["Teil", "Datum"]))
            h_aligned, d_aligned = data_pipeline.align_schema(h, d)
            if not h.empty:
                hist[part] = h_aligned
            if not d.empty:
                dispo[part] = d_aligned

    if args.mode in {"H", "both"}:
        _write_part_tables(hist, Path(args.out_h))
    if args.mode in {"D", "both"}:
        _write_part_tables(dispo, Path(args.out_d))


if __name__ == "__main__":
    main()
