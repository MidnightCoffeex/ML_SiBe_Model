#!/usr/bin/env python3
"""
GUI: Feature-Auswahl für Training/Evaluierung – ohne Spalten umzubenennen.

Neues Verhalten:
- Ordner mit Feature-Unterordnern wählen (enthält */features.parquet).
- Checkboxen für auswählbare Spalten (Union über alle Dateien).
- Beim Anwenden wird eine Datei "selected_features.json" im gewählten Root gespeichert.

Hinweise:
- Die Pipeline entfernt F_NiU_/L_NiU_-Spalten. nF_-Spalten sind Hilfswerte und
  werden nicht in die Featureliste aufgenommen.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Set
import traceback
import json
from datetime import datetime

import pandas as pd


PREFIX_EXCLUDE = ("F_NiU_", "L_NiU_", "nF_")
NF_PREFIX = "nF_"
SELECTION_FILE = "selected_features.json"


def list_feature_files(root: Path) -> List[Path]:
    return sorted(root.glob('*/features.parquet'))


def _read_any(p: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(p)
        except Exception:
            csv = p.with_suffix('.csv')
            if csv.exists():
                return pd.read_csv(csv)
            raise


def union_columns(parquet_files: List[Path]) -> List[str]:
    cols: Set[str] = set()
    for p in parquet_files:
        df = _read_any(p)
        for c in df.columns:
            cols.add(str(c))
    filtered = [
        c for c in cols
        if c not in {"Teil", "Datum"}
        and not c.startswith("LABLE_")
        and not c.startswith("L_NiU_")
        and not c.startswith("F_NiU_")
        and not c.startswith("nF_")
    ]
    return sorted(filtered)


def load_existing_selection(root: Path) -> Set[str]:
    sel_path = root / SELECTION_FILE
    if not sel_path.exists():
        return set()
    try:
        data = json.loads(sel_path.read_text(encoding="utf-8"))
        items = data.get("selected", [])
        return set(str(x) for x in items)
    except Exception:
        return set()


def apply_selection(root: Path, selected: Dict[str, tk.BooleanVar]) -> None:
    include_map = {name: var.get() for name, var in selected.items()}
    selected_cols = sorted([name for name, use in include_map.items() if use])
    payload = {
        "selected": selected_cols,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        (root / SELECTION_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        messagebox.showinfo("Gespeichert", f"Auswahl gespeichert in {root/SELECTION_FILE}")
    except Exception:
        messagebox.showerror("Fehler", traceback.format_exc())


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Features auswählen (Training)")
        self.geometry("700x600")

        self.root_dir: Path | None = None
        self.check_vars: Dict[str, tk.BooleanVar] = {}

        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(top, text="Features-Ordner wählen", command=self.pick_dir).pack(side=tk.LEFT)
        self.dir_label = tk.Label(top, text="Kein Ordner gewählt", anchor="w")
        self.dir_label.pack(side=tk.LEFT, padx=10)

        actions = tk.Frame(self)
        actions.pack(fill=tk.X, padx=10)
        tk.Button(actions, text="Alle auswählen", command=self.select_all).pack(side=tk.LEFT)
        tk.Button(actions, text="Alle abwählen", command=self.deselect_all).pack(side=tk.LEFT, padx=10)
        tk.Button(actions, text="Anwenden", command=self.apply).pack(side=tk.RIGHT)

        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.check_frame = tk.Frame(canvas)
        self.check_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.check_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def pick_dir(self) -> None:
        sel = filedialog.askdirectory(title="Features-Ordner (enthält */features.parquet)")
        if not sel:
            return
        self.root_dir = Path(sel)
        self.dir_label.config(text=str(self.root_dir))
        self.load_columns()

    def load_columns(self) -> None:
        if self.root_dir is None:
            return
        pq_files = list_feature_files(self.root_dir)
        if not pq_files:
            messagebox.showerror("Fehler", "Keine Dateien gefunden: */features.parquet")
            return
        cols = union_columns(pq_files)
        prev = load_existing_selection(self.root_dir)
        for w in list(self.check_frame.children.values()):
            w.destroy()
        self.check_vars.clear()
        try:
            df0 = _read_any(pq_files[0])
            existing = set(df0.columns)
        except Exception:
            existing = set()
        for c in cols:
            var = tk.BooleanVar(value=True)
            if prev:
                var.set(c in prev)
            if c.startswith(PREFIX_EXCLUDE) or (NF_PREFIX + c) in existing:
                pass
            self.check_vars[c] = var
        for i, (name, var) in enumerate(sorted(self.check_vars.items())):
            cb = tk.Checkbutton(self.check_frame, text=name, variable=var, anchor="w", justify="left")
            cb.grid(row=i, column=0, sticky="w")

    def select_all(self) -> None:
        for var in self.check_vars.values():
            var.set(True)

    def deselect_all(self) -> None:
        for var in self.check_vars.values():
            var.set(False)

    def apply(self) -> None:
        if self.root_dir is None:
            messagebox.showerror("Fehler", "Bitte zuerst einen Ordner wählen")
            return
        apply_selection(self.root_dir, self.check_vars)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

