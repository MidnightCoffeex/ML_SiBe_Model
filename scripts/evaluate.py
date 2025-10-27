#!/usr/bin/env python3
"""Run model evaluation and generate plots."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import evaluate_model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--features", help="Root directory with feature folders")
    parser.add_argument("--part", help="Teil-Nummer oder ALL")
    parser.add_argument("--model-dir", help="Directory containing trained models")
    parser.add_argument("--model-type", help="Model type (gb,xgb,lgbm)")
    parser.add_argument("--model-id", help="Numerical model identifier")
    parser.add_argument("--raw", help="Directory with raw CSV files")
    parser.add_argument(
        "--targets",
        default="LABLE_HalfYear_Target",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", help="Directory to store evaluation results")
    parser.add_argument("--feature-list", help="Path to selected_features.json")
    args = parser.parse_args()

    if not args.features:
        args.features = input("Pfad zu Features [Features]: ") or "Features"
    if not args.part:
        args.part = input("Teil-Nummer oder ALL [ALL]: ") or "ALL"
    if not args.model_dir:
        args.model_dir = "Modelle"
    if not args.raw:
        args.raw = "Rohdaten"
    # Target-Label interaktiv abfragen (optional mehrere, komma-getrennt)
    try:
        current_targets = args.targets if isinstance(args.targets, str) else ""
    except AttributeError:
        current_targets = "LABLE_HalfYear_Target"
    entry = input(f"Target-Label(s) [{current_targets}]: ")
    if entry.strip():
        args.targets = entry.strip()
    # Load optional feature list once for reuse
    selected_features = None
    sel_path = Path(args.feature_list) if getattr(args, 'feature_list', None) else (Path(args.features) / 'selected_features.json')
    if sel_path and sel_path.exists():
        try:
            import json
            data = json.loads(sel_path.read_text(encoding='utf-8'))
            selected_features = [str(x) for x in data.get('selected', [])]
            print(f"Feature-Liste geladen: {sel_path} (n={len(selected_features)})")
        except Exception as exc:
            print(f"Warnung: Konnte Feature-Liste nicht laden: {exc}")

    eval_part = args.part
    if args.part.upper() == "ALL":
        eval_part = input("Teilnummer fÃ¼r Auswertung: ")

    if not args.model_type:
        mdir = Path(args.model_dir) / args.part
        types = [p.name for p in mdir.glob('*') if p.is_dir()]
        if types:
            print("VerfÃ¼gbare Modelltypen:", ", ".join(sorted(types)))
        args.model_type = input("Modelltyp: ")

    if not args.model_id:
        mdir = Path(args.model_dir) / args.part / args.model_type
        existing = [p.name for p in mdir.glob('*') if p.is_dir() and p.name.isdigit()]
        if existing:
            print("VerfÃ¼gbare Modelle:", ", ".join(sorted(existing)))
        args.model_id = input("Modellnummer: ")

    # Optional: evaluate ALL parts when using an ALL model
    if args.part and args.part.upper() == 'ALL':
        sel = eval_part.strip().upper() if isinstance(eval_part, str) else ''
        if sel == 'ALL':
            parts = sorted(p.parent.name for p in Path(args.features).glob('*/features.parquet'))
            model_path = Path(args.model_dir) / args.part / args.model_type / args.model_id / 'model.joblib'
            if not args.plots:
                args.plots = input("Ordner fÇ¬r Ergebnisse [plots]: ") or "plots"
            base_plot_dir = Path(args.plots) / args.part / args.model_type / args.model_id
            target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
            for p in parts:
                features_path = Path(args.features) / p / 'features.parquet'
                plot_dir = base_plot_dir / p
                plot_dir.mkdir(parents=True, exist_ok=True)
                evaluate_model.run_evaluation(str(features_path), str(model_path), target_list, str(plot_dir), args.raw, model_type=args.model_type, selected_features=selected_features)
            return

    features_path = Path(args.features) / eval_part / 'features.parquet'
    model_path = Path(args.model_dir) / args.part / args.model_type / args.model_id / 'model.joblib'
    if not args.plots:
        args.plots = input("Ordner fÃ¼r Ergebnisse [plots]: ") or "plots"
    plot_dir = Path(args.plots) / args.part / args.model_type / args.model_id

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    evaluate_model.run_evaluation(str(features_path), str(model_path), target_list, str(plot_dir), args.raw, model_type=args.model_type, selected_features=selected_features)


if __name__ == "__main__":
    main()

