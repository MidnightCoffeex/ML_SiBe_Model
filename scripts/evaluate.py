#!/usr/bin/env python3
"""Evaluate trained models on historical data or run inference on dispo features."""
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import evaluate_model, data_pipeline


def _prompt(prompt: str, default: str | None = None) -> str:
    val = input(prompt)
    if not val and default is not None:
        return default
    return val


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--mode", choices=["full", "dispo"], help="Evaluationsmodus")
    parser.add_argument("--features-h", help="Pfad zu H-Features")
    parser.add_argument("--features-d", help="Pfad zu D-Features")
    parser.add_argument("--model-dir", help="Modelldir")
    parser.add_argument("--model-type", help="Modelltyp")
    parser.add_argument("--model-id", help="Modell-ID")
    parser.add_argument("--part", help="Teil-Nummer oder ALL")
    parser.add_argument("--plots", help="Ausgabepfad für Plots")
    parser.add_argument("--targets", help="Kommagetrennte Targets", default="LABLE_StockOut_MinAdd")
    args = parser.parse_args()

    if not args.mode:
        args.mode = _prompt("Modus {full,dispo} [full]: ", "full")
    if not args.model_dir:
        args.model_dir = _prompt("Modelldir [Models]: ", "Models")
    if not args.model_type:
        args.model_type = _prompt("Modelltyp (gb,xgb,lgbm): ")
    if not args.model_id:
        args.model_id = _prompt("Modell-ID: ")
    if not args.part:
        args.part = _prompt("Teil-Nummer oder ALL [ALL]: ", "ALL")
    if not args.plots:
        args.plots = _prompt("Plot-Verzeichnis [plots]: ", "plots")
    if not args.targets:
        args.targets = _prompt("Targets [LABLE_StockOut_MinAdd]: ", "LABLE_StockOut_MinAdd")

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]

    if args.mode == 'full':
        if not args.features_h:
            args.features_h = _prompt("Pfad zu Features_H [Features_H]: ", "Features_H")
        features_path = Path(args.features_h) / args.part / 'features.csv'
        model_path = Path(args.model_dir) / args.part / args.model_type / args.model_id / 'model.joblib'
        plot_dir = Path(args.plots) / args.part / args.model_type / args.model_id
        evaluate_model.run_evaluation(str(features_path), str(model_path), targets, str(plot_dir), 'Rohdaten', model_type=args.model_type)
    else:
        if not args.features_d:
            args.features_d = _prompt("Pfad zu Features_D [Features_D]: ", "Features_D")
        features_path = Path(args.features_d) / args.part / 'features.csv'
        model_dir = Path(args.model_dir) / args.part / args.model_type / args.model_id
        model_path = model_dir / 'model.joblib'
        cols_path = model_dir / 'feature_cols.json'
        feat = data_pipeline.safe_read_features(features_path)
        with open(cols_path, 'r', encoding='utf-8') as fh:
            feature_cols = json.load(fh)
        X = feat.reindex(columns=feature_cols, fill_value=0)
        model = __import__('joblib').load(model_path)
        preds = model.predict(X)
        res = feat[['Teil', 'Datum']].copy()
        for i, col in enumerate(targets):
            res[f'pred_{col}'] = preds[:, i]
        out_dir = Path(args.plots) / args.part / args.model_type / args.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_dir / 'predictions.csv', index=False)
        try:
            res.to_excel(out_dir / 'predictions.xlsx', index=False)
        except Exception:
            pass


if __name__ == '__main__':
    main()
