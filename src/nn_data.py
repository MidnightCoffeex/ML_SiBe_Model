"""Hilfsfunktionen rund um das Laden und Kombinieren der Feature-Dateien für neuronale Modelle."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import json
import pandas as pd

from . import train_model


def resolve_part_dirs(root: Path, selection: str) -> List[Path]:
    """Ermittelt alle Teil-Ordner, die ausgewertet werden sollen."""
    if selection.upper() == "ALL":
        return sorted([p for p in root.iterdir() if p.is_dir()])
    dirs: List[Path] = []
    for token in {s.strip() for s in selection.split(",") if s.strip()}:
        candidate = root / token
        if candidate.is_dir():
            dirs.append(candidate)
        else:
            print(f"Warnung: Ordner {candidate} existiert nicht und wird übersprungen.")
    return dirs


def load_feature_frame(part_dir: Path) -> pd.DataFrame:
    """Lädt die Feature-Datei eines Teils (Parquet bevorzugt, sonst CSV/XLSX)."""
    parquet_path = part_dir / "features.parquet"
    if parquet_path.exists():
        return train_model.load_features(str(parquet_path))
    csv_path = part_dir / "features.csv"
    if csv_path.exists():
        return train_model.load_features(str(csv_path))
    xlsx_path = part_dir / "features.xlsx"
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError(f"Keine Feature-Datei in {part_dir} gefunden.")


def collect_dataframe(features_root: Path, part_selection: str) -> Tuple[pd.DataFrame, List[str]]:
    """Lädt alle ausgewählten Teile und liefert einen kombinierten DataFrame plus Liste der Teilnamen."""
    part_dirs = resolve_part_dirs(features_root, part_selection)
    if not part_dirs:
        raise RuntimeError("Keine passenden Teil-Ordner gefunden.")
    frames = []
    parts = []
    for part_dir in part_dirs:
        try:
            df = load_feature_frame(part_dir)
        except Exception as exc:
            print(f"Überspringe {part_dir.name}: {exc}")
            continue
        if df.empty:
            continue
        frames.append(df)
        parts.append(part_dir.name)
    if not frames:
        raise RuntimeError("Alle ausgewählten Teile waren leer oder fehlerhaft.")
    return pd.concat(frames, ignore_index=True), parts


def load_metadata(run_dir: Path) -> Dict:
    """Liest die metadata.json eines NN-Laufs, falls vorhanden."""
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warnung: Konnte metadata.json nicht lesen ({exc}).")
        return {}


def find_feature_file(part_dir: Path) -> Path:
    """Findet die Feature-Datei eines Teils (Präferenz Parquet)."""
    for name in ("features.parquet", "features.csv", "features.xlsx"):
        candidate = part_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Keine Feature-Datei in {part_dir} gefunden.")
