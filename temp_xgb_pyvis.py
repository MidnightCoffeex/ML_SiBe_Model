"""SuperTree basierte Visualisierung fuer gespeicherte XGBoost-Modelle."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd
from supertree import SuperTree

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "Modelle" / "ALL" / "xgb" / "3" / "model.joblib"
DEFAULT_OUTPUT_DIR = HERE / "Modell-Visual"
DEFAULT_TREES_PER_TARGET = 3
DEFAULT_START_DEPTH = 5
DEFAULT_MAX_SAMPLES = 7500
DEFAULT_PART = "ALL"
DEFAULT_TARGET_COLUMN = "L_WBZ_BlockMinAbs"
if (HERE / "Final_Features_Test").exists():
    DEFAULT_FEATURES_ROOT = HERE / "Final_Features_Test"
elif (HERE / "Final_Features").exists():
    DEFAULT_FEATURES_ROOT = HERE / "Final_Features"
else:
    DEFAULT_FEATURES_ROOT = HERE / "Features"
DEFAULT_SAMPLE_ROWS = 10_000
DEFAULT_TREE_OFFSET = 0


@dataclass
class VisualizationSample:
    """Hae lt ein gefiltertes DataFrame fuer SuperTree-Scatterplots."""

    frame: pd.DataFrame
    target_columns: list[str]


def ask_path(prompt: str, default: pathlib.Path) -> pathlib.Path:
    """Fragt interaktiv nach einem Pfad und erlaubt die Enter-Bestaetigung."""
    raw = input(f"{prompt} [{default}]: ").strip()
    value = pathlib.Path(raw) if raw else default
    return value.expanduser()


def ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    """Fragt so lange nach einer Ganzzahl, bis die Mindestgrenze erreicht ist."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        current = default if not raw else raw
        try:
            number = int(current)
        except ValueError:
            print("Bitte eine ganze Zahl eingeben.")
            continue
        if number < minimum:
            print(f"Bitte mindestens {minimum} waehlen.")
            continue
        return number


def flatten_estimators(value) -> Iterable[object]:
    """Reduziert verschachtelte Estimator-Sammlungen zu einem Iterator."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened: list[object] = []
        for item in value:
            flattened.extend(list(flatten_estimators(item)))
        return flattened
    return [value]


def collect_estimators(
    model: object, target_labels: Sequence[str] | None = None
) -> list[tuple[int, str, object]]:
    """Liefert (index, label, estimator) fuer alle Zielmodelle."""
    estimators_attr = getattr(model, "estimators_", None)
    resolved_labels = list(target_labels or [])
    if estimators_attr is None:
        label = resolved_labels[0] if resolved_labels else "target1"
        return [(1, label, model)]
    flattened = [
        est for est in flatten_estimators(estimators_attr) if est is not None
    ]
    if not flattened:
        label = resolved_labels[0] if resolved_labels else "target1"
        return [(1, label, model)]
    while len(resolved_labels) < len(flattened):
        resolved_labels.append(f"target{len(resolved_labels) + 1}")
    return [
        (idx, resolved_labels[idx - 1], est)
        for idx, est in enumerate(flattened, start=1)
    ]


def feature_names_for(estimator: object) -> list[str]:
    """Bestimmt Feature-Namen oder erzeugt generische Platzhalter."""
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    names = getattr(estimator, "feature_names", None)
    if names is not None:
        return [str(name) for name in names]
    n_features = getattr(estimator, "n_features_in_", None)
    if isinstance(n_features, int) and n_features > 0:
        return [f"feature{i+1}" for i in range(n_features)]
    return []


def target_names_for(estimator: object, fallback_label: str) -> list[str]:
    """Verwendet vorhandene Klassen oder erzeugt einen Zielnamen."""
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        try:
            return [str(name) for name in classes]
        except TypeError:
            pass
    return [fallback_label]


def resolve_booster(estimator: object):
    """Versucht, ein XGBoost-Booster-Objekt aus dem Estimator zu extrahieren."""
    if hasattr(estimator, "get_booster"):
        return estimator.get_booster()
    booster = getattr(estimator, "booster_", None)
    if booster is not None:
        return booster
    return None


def count_available_trees(estimator: object) -> int:
    """Bestimmt, wie viele Baeume im Estimator vorhanden sind."""
    booster = resolve_booster(estimator)
    if booster is None or not hasattr(booster, "get_dump"):
        raise ValueError("Kein Booster mit get_dump() gefunden.")
    dumps = booster.get_dump(with_stats=True, dump_format="json")
    return len(dumps)


def read_table(path: pathlib.Path) -> pd.DataFrame:
    """Laedt eine einzelne Feature-Datei."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path}")


def find_feature_files(
    feature_file: pathlib.Path | None,
    features_root: pathlib.Path,
    part: str,
) -> list[pathlib.Path]:
    """Ermittelt alle relevanten Feature-Dateien fuer die Visualisierungsdaten."""
    if feature_file:
        return [feature_file]
    base = features_root
    if not base.exists():
        return []
    files: list[pathlib.Path] = []
    part_upper = part.upper()
    if part_upper == "ALL":
        for suffix in (".parquet", ".csv"):
            pattern = sorted(base.glob(f"*/features{suffix}"))
            if pattern:
                files.extend(pattern)
                break
    else:
        for suffix in (".parquet", ".csv"):
            candidate = base / part / f"features{suffix}"
            if candidate.exists():
                files.append(candidate)
                break
    return files


def build_sample_data(
    feature_names: list[str],
    target_columns: list[str],
    *,
    feature_file: pathlib.Path | None,
    features_root: pathlib.Path,
    part: str,
    sample_rows: int,
) -> VisualizationSample | None:
    """Laedt die Features/Targets als Stichprobe fuer SuperTree."""
    if not feature_names or not target_columns:
        return None
    files = find_feature_files(feature_file, features_root, part)
    if not files:
        print(
            "Hinweis: Keine Feature-Dateien fuer Visualisierungsdaten gefunden "
            f"(Root: {features_root}, Teil: {part})."
        )
        return None
    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frames.append(read_table(file_path))
        except Exception as error:  # noqa: BLE001
            print(f"  Konnte {file_path} nicht laden: {error}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        print(
            "Hinweis: Einige benoetigte Feature-Spalten fehlen in den Daten "
            f"und die Punktwolken werden deaktiviert: {missing_features}"
        )
        return None
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        print(
            "Hinweis: Zielspalten fehlen in den Daten und die Punktwolken "
            f"werden deaktiviert: {missing_targets}"
        )
        return None
    df = df.dropna(subset=target_columns)
    if df.empty:
        print("Hinweis: Keine Zeilen mit vollstaendigen Zielwerten gefunden.")
        return None
    columns_needed = list(dict.fromkeys([*feature_names, *target_columns]))
    df = df[columns_needed]
    if sample_rows and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
    df = df.reset_index(drop=True)
    file_note = ", ".join(str(p.name) for p in files[:3])
    if len(files) > 3:
        file_note += ", ..."
    print(
        f"SuperTree-Probendaten: {len(df)} Zeilen aus {len(files)} Datei(en) "
        f"({file_note})."
    )
    return VisualizationSample(frame=df, target_columns=target_columns)


def extract_feature_array(
    sample: VisualizationSample | None, feature_names: list[str]
) -> Optional[pd.DataFrame]:
    """Gibt die numerische Feature-Matrix fuer SuperTree zurueck."""
    if not sample:
        return None
    missing = [name for name in feature_names if name not in sample.frame.columns]
    if missing:
        print(f"  Hinweis: Feature-Spalten fuer Punktwolken fehlen: {missing}")
        return None
    numeric = sample.frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric


def extract_target_array(
    sample: VisualizationSample | None, label: str
) -> Optional[pd.Series]:
    """Gibt die Zielwerte fuer SuperTree zurueck."""
    if not sample:
        return None
    if label not in sample.frame.columns:
        if sample.target_columns:
            fallback = sample.target_columns[0]
            print(
                f"  Hinweis: Ziel '{label}' fehlt in den Punktwolken-Daten, "
                f"verwende '{fallback}' stattdessen."
            )
            label = fallback
        else:
            return None
    series = pd.to_numeric(sample.frame[label], errors="coerce").fillna(0.0)
    return series


def visualize_model(
    estimators: list[tuple[int, str, object]],
    output_dir: pathlib.Path,
    max_trees: int,
    start_depth: int,
    max_samples: int,
    dataset: VisualizationSample | None,
    *,
    tree_offset: int = 0,
    use_tail: bool = False,
) -> int:
    """Exportiert die gewuenschten Baeume als HTML-Dateien."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    for target_idx, target_label, estimator in estimators:
        try:
            tree_count = count_available_trees(estimator)
        except Exception as error:  # noqa: BLE001
            print(
                f"Ziel {target_idx}: Booster nicht nutzbar ({error}).",
            )
            continue

        if tree_count == 0:
            print(f"Ziel {target_idx}: Keine Baeume gefunden.")
            continue

        if use_tail:
            start_index = max(tree_count - max_trees, 0)
        else:
            start_index = min(tree_offset, max(tree_count - 1, 0))

        if start_index >= tree_count:
            print(
                f"Ziel {target_idx}: Startindex {tree_offset} liegt ausserhalb "
                f"der verfuegbaren Baeume (0..{tree_count - 1})."
            )
            continue

        limit = min(max_trees, tree_count - start_index)
        if limit <= 0:
            print(f"Ziel {target_idx}: Keine Baeume im gewaehlten Bereich.")
            continue
        print(
            f"Ziel {target_idx}: Exportiere {limit} Baeume (Start {start_index}, "
            f"insgesamt {tree_count}) nach {output_dir}.",
        )

        feature_names = feature_names_for(estimator)
        feature_array = extract_feature_array(dataset, feature_names)
        target_array = extract_target_array(dataset, target_label)
        if feature_array is None or target_array is None:
            feature_arg = None
            target_arg = None
        else:
            feature_arg = feature_array.to_numpy()
            target_arg = target_array.to_numpy()

        try:
            visualizer = SuperTree(
                estimator,
                feature_data=feature_arg,
                target_data=target_arg,
                feature_names=feature_names,
                target_names=target_names_for(estimator, target_label),
            )
        except Exception as error:  # noqa: BLE001
            print(f"Ziel {target_idx}: SuperTree konnte nicht erstellt werden ({error}).")
            continue

        for tree_index in range(start_index, start_index + limit):
            out_file = output_dir / f"{target_label}_tree{tree_index + 1}.html"
            try:
                visualizer.save_html(
                    str(out_file),
                    which_tree=tree_index,
                    which_iteration=0,
                    start_depth=start_depth,
                    max_samples=max_samples,
                )
            except Exception as error:  # noqa: BLE001
                print(
                    f"  Baum {tree_index}: Fehler beim Speichern ({error}).",
                )
                continue
            print(f"  -> Visualisierung gespeichert: {out_file}")
            exported += 1

    return exported


def parse_targets(text: str | None) -> list[str]:
    """Zerlegt Komma-Listen in eine Target-Liste."""
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_targets(model_path: pathlib.Path, override: str | None) -> list[str]:
    """Bestimmt Ziellabels (Option, metrics.csv oder Default)."""
    if override:
        return parse_targets(override)
    metrics_file = model_path.with_name("metrics.csv")
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            targets = [str(t).strip() for t in df.get("target", []) if str(t).strip()]
            if targets:
                return targets
        except Exception:  # noqa: BLE001
            pass
    return [DEFAULT_TARGET_COLUMN]


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Argument-Parser."""
    parser = argparse.ArgumentParser(
        description="Visualisiert XGBoost-Modelle mit SuperTree.",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help="Pfad zur model.joblib (Standard: Modelle/ALL/xgb/3/model.joblib).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Zielordner fuer die HTML-Dateien (Standard: Modell-Visual).",
    )
    parser.add_argument(
        "--trees-per-target",
        type=int,
        default=None,
        help="Wieviele Baeume pro Ziel ausgegeben werden sollen.",
    )
    parser.add_argument(
        "--start-depth",
        type=int,
        default=None,
        help="Starttiefe fuer SuperTree (Standard: 5).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Sample-Limit fuer SuperTree (Standard: 7500).",
    )
    parser.add_argument(
        "--feature-file",
        type=pathlib.Path,
        default=None,
        help="Optional direkte features.parquet-Datei (ueberschreibt Root/Teil).",
    )
    parser.add_argument(
        "--features-root",
        type=pathlib.Path,
        default=None,
        help="Basisordner mit Feature-Unterordnern (Standard: Final_Features_Test oder Features).",
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Teil-Nummer oder 'ALL' fuer aggregierte Daten (Standard: ALL).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Kommagetrennte Zielspalten (Standard: metrics.csv oder L_WBZ_BlockMinAbs).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Maximale Zeilenzahl fuer Punktwolken (0 = alle, Standard: 10000).",
    )
    parser.add_argument(
        "--tree-offset",
        type=int,
        default=None,
        help="Index des ersten auszugebenden Baums (0-basiert, Standard: 0).",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Gibt die letzten N Baeume statt der ersten aus.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Interaktive Rueckfragen unterdruecken und nur CLI-Werte nutzen.",
    )
    return parser


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[
    pathlib.Path,
    pathlib.Path,
    int,
    int,
    int,
    pathlib.Path | None,
    pathlib.Path,
    str,
    list[str],
    int,
    int,
    bool,
]:
    """Leitet finale Pfade und Parameter aus Argumenten bzw. Prompts ab."""
    model_default = args.model or DEFAULT_MODEL_PATH
    output_default = args.output_dir or DEFAULT_OUTPUT_DIR
    trees_default = args.trees_per_target or DEFAULT_TREES_PER_TARGET
    start_depth = args.start_depth if args.start_depth is not None else DEFAULT_START_DEPTH
    max_samples = args.max_samples if args.max_samples is not None else DEFAULT_MAX_SAMPLES
    feature_file = args.feature_file
    features_root = args.features_root or DEFAULT_FEATURES_ROOT
    part = args.part or DEFAULT_PART
    sample_rows = args.sample_rows if args.sample_rows is not None else DEFAULT_SAMPLE_ROWS
    tree_offset = args.tree_offset if args.tree_offset is not None else DEFAULT_TREE_OFFSET
    use_tail = bool(args.tail)

    if args.no_prompt:
        model_path = model_default
        output_dir = output_default
        trees_per_target = trees_default
    else:
        model_path = ask_path("Pfad zur Modelldatei", model_default)
        output_dir = ask_path("Zielordner fuer HTML-Dateien", output_default)
        trees_per_target = ask_int(
            "Wie viele Baeume pro Ziel exportieren?", trees_default, minimum=1
        )

    if trees_per_target < 1:
        raise SystemExit("Mindestens ein Baum muss ausgegeben werden.")
    if start_depth < 0:
        raise SystemExit("Die Starttiefe muss >= 0 sein.")
    if max_samples < 1:
        raise SystemExit("max_samples muss >= 1 sein.")
    if sample_rows < 0:
        raise SystemExit("sample_rows muss >= 0 sein.")
    if tree_offset < 0:
        raise SystemExit("tree-offset muss >= 0 sein.")

    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Modelldatei wurde nicht gefunden: {model_path}")

    feature_file = feature_file.expanduser().resolve() if feature_file else None
    features_root = features_root.expanduser().resolve()
    if feature_file is None and not features_root.exists():
        raise SystemExit(f"Features-Wurzel wurde nicht gefunden: {features_root}")

    targets = resolve_targets(model_path, args.targets)
    return (
        model_path,
        output_dir,
        trees_per_target,
        start_depth,
        max_samples,
        feature_file,
        features_root,
        part,
        targets,
        sample_rows,
        tree_offset,
        use_tail,
    )


def main() -> None:
    """CLI-Einstiegspunkt."""
    parser = build_parser()
    args = parser.parse_args()
    (
        model_path,
        output_dir,
        trees_per_target,
        start_depth,
        max_samples,
        feature_file,
        features_root,
        part,
        target_labels,
        sample_rows,
        tree_offset,
        use_tail,
    ) = resolve_inputs(args)

    print(f"Lade Modell: {model_path}")
    model = joblib.load(model_path)
    estimators = collect_estimators(model, target_labels)
    if not estimators:
        raise SystemExit("Keine Estimatoren gefunden.")

    reference_features = feature_names_for(estimators[0][2])
    dataset = build_sample_data(
        reference_features,
        target_labels,
        feature_file=feature_file,
        features_root=features_root,
        part=part,
        sample_rows=sample_rows,
    )
    if dataset is None:
        print("Fahre ohne Punktwolken fort (SuperTree erhaelt keine Stichprobe).")

    total_files = visualize_model(
        estimators=estimators,
        output_dir=output_dir,
        max_trees=trees_per_target,
        start_depth=start_depth,
        max_samples=max_samples,
        dataset=dataset,
        tree_offset=tree_offset,
        use_tail=use_tail,
    )
    if total_files == 0:
        print("Es wurden keine Visualisierungen erstellt.")
    else:
        print(f"Insgesamt {total_files} Visualisierungen gespeichert.")


if __name__ == "__main__":
    main()
