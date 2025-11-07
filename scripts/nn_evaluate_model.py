#!/usr/bin/env python3
# Evaluation-Skript für das separat trainierte neuronale Modell (MLP).
# Lädt gespeicherte Modelle, berechnet Vorhersagen je Teil und erzeugt dieselben Auswertungen wie die bestehende Pipeline.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import evaluate_model, nn_data


def prompt(text: str, default: str | None = None) -> str:
    """Einfacher Prompt mit optionalem Default-Wert."""
    if default is None:
        return input(f"{text}: ").strip()
    value = input(f"{text} [{default}]: ").strip()
    return value or default


def main() -> None:
    print("=== Neuronale Auswertung (MLP) ===")
    features_root = Path(prompt("Pfad zu Test-Features", "CODEX_Features_2_Test"))
    part_selection = prompt("Teil-Nummer oder ALL", "ALL")
    model_run_path = Path(
        prompt("Pfad zum Modell-Lauf (z.B. NN_Modelle/ALL/1)", "NN_Modelle/ALL/1")
    ).resolve()
    model_file = model_run_path / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Modell-Datei {model_file} nicht gefunden.")

    metadata = nn_data.load_metadata(model_run_path)
    default_targets = ",".join(metadata.get("targets", [])) or "L_WBZ_BlockMinAbs"
    targets_raw = prompt("Target-Label(s) (Komma)", default_targets)
    targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
    if not targets:
        raise RuntimeError("Es muss mindestens ein Target angegeben werden.")

    selected_features = metadata.get("selected_features")
    raw_dir = prompt("Rohdaten-Ordner", "Rohdaten_Aktuell")
    output_root = Path(prompt("Ausgabeordner für Plots", "NN_Plots")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    part_dirs = nn_data.resolve_part_dirs(features_root, part_selection)
    if not part_dirs:
        raise RuntimeError("Keine passenden Teil-Ordner gefunden.")

    results_for_aggregate: Dict[str, object] = {}
    for part_dir in part_dirs:
        try:
            feature_file = nn_data.find_feature_file(part_dir)
        except FileNotFoundError as exc:
            print(exc)
            continue
        out_dir = output_root / part_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Auswertung Teil {part_dir.name} ...")
        try:
            res = evaluate_model.run_evaluation(
                str(feature_file),
                str(model_file),
                targets,
                str(out_dir),
                raw_dir=raw_dir,
                model_type="nn",
                selected_features=selected_features,
            )
        except Exception as exc:
            print(f"Fehler bei Teil {part_dir.name}: {exc}")
            continue
        if res and res.get("full") is not None:
            results_for_aggregate[part_dir.name] = res["full"]

    if results_for_aggregate:
        aggregate_dir = output_root / "Alle_Teile"
        try:
            evaluate_model.aggregate_all_parts(results_for_aggregate, str(aggregate_dir))
            print(f"Aggregation nach {aggregate_dir} geschrieben.")
        except Exception as exc:
            print(f"Aggregation fehlgeschlagen: {exc}")
    else:
        print("Keine Ergebnisse für eine Aggregation vorhanden.")

    print("Auswertung abgeschlossen.")


if __name__ == "__main__":
    main()
