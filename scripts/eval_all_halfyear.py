#!/usr/bin/env python3
"""Evaluate ALL parts with the ALL/gb model using L_HalfYear_Target.
Outputs to New_Test_Plots/ALL/gb/<model-id>/<part>/

Usage:
  python scripts/eval_all_halfyear.py --model-id 12
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import evaluate_model  # type: ignore


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    features_root = Path('AGENTS_MAKE_ML/Features')
    parts = sorted(p.parent.name for p in features_root.glob('*/features.parquet'))
    model_path = f'AGENTS_MAKE_ML/Modelle/ALL/gb/{args.model_id}/model.joblib'
    plots_root = Path('New_Test_Plots') / 'ALL' / 'gb' / args.model_id
    raw_dir = 'AGENTS_MAKE_ML/Rohdaten'
    targets = ['L_HalfYear_Target']

    for part in parts:
        features_path = features_root / part / 'features.parquet'
        out_dir = plots_root / part
        out_dir.mkdir(parents=True, exist_ok=True)
        evaluate_model.run_evaluation(str(features_path), model_path, targets, str(out_dir), raw_dir, model_type='gb')


if __name__ == '__main__':
    main()

