"""Gemeinsame Helfer für neuronale Modellläufe (Namensgebung, Verzeichnisse, Layer-Konfigurationen)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def determine_scope_name(selection: str, parts: Sequence[str]) -> str:
    """Erzeugt den Scope-Namen für den Modellordner."""
    if selection.upper() == "ALL":
        return "ALL"
    if len(parts) == 1:
        return parts[0]
    return "MULTI"


def build_hidden_layers(text: str) -> tuple[int, ...]:
    """Wandelt eine Kommasequenz in ein Hidden-Layer-Tuple um."""
    layers: list[int] = []
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            layers.append(int(entry))
        except ValueError:
            print(f"Ungültige Layer-Größe '{entry}' – wird ignoriert.")
    return tuple(layers) if layers else (128, 64)


def choose_run_directory(base_dir: Path) -> Path:
    """Erzeugt einen neuen nummerierten Unterordner für den Modell-Lauf."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name)
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    next_id = max(existing, default=0) + 1
    run_dir = base_dir / str(next_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

