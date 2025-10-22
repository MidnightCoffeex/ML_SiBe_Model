#!/usr/bin/env python3
"""
GUI-Hilfstool zum Markieren von Feature-Spalten als "Not in Use" via Prefix.

Funktion:
- Ordner mit Feature-Unterordnern auswählen (jeder enthält features.parquet und ggf. features.xlsx)
- Checkboxen für Spalten anzeigen (Union über alle Parquet-Dateien)
- Beim Anwenden werden Spalten, die abgewählt sind, mit dem Prefix 'nF_' versehen
  (oder zurückbenannt, wenn ausgewählt) – in allen Unterordnern (Parquet + Excel).

Hinweise:
- Training ignoriert Spalten mit Prefix 'F_NiU_', 'L_NiU_' und 'nF_' automatisch.
- Dieses Tool nutzt 'nF_' als Umschalt-Prefix.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Set
import traceback

import pandas as pd


PREFIX_EXCLUDE = ("F_NiU_", "L_NiU_", "nF_")
NF_PREFIX = "nF_"  # muss mit dem Trainingsfilter übereinstimmen


def list_feature_files(root: Path) -> List[Path]:
    return sorted(root.glob('*/features.parquet'))


def union_columns(parquet_files: List[Path]) -> List[str]:
    cols: Set[str] = set()
    for p in parquet_files:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(p)  # fallback
        for c in df.columns:
            cols.add(str(c))
    # sinnvolle Defaults: Schlüsselspalten/Targets/Diagnostics nicht zur Auswahl
    filtered = [
        c for c in cols
        if c not in {"Teil", "Datum"}
        and not c.startswith("LABLE_")
        and not c.startswith("L_NiU_")
    ]
    return sorted(filtered)


def apply_selection(root: Path, selected: Dict[str, tk.BooleanVar]) -> None:
    parquet_files = list_feature_files(root)
    if not parquet_files:
        messagebox.showerror("Fehler", "Keine Parquet-Dateien gefunden (*/features.parquet)")
        return

    # Mappe Soll-Zustände: True=verwenden (Prefix entfernen), False=nicht verwenden (Prefix setzen)
    include_map = {name: var.get() for name, var in selected.items()}

    errors: List[str] = []
    updated = 0
    for pq in parquet_files:
        try:
            # Parquet laden
            try:
                df = pd.read_parquet(pq, engine="pyarrow")
            except Exception:
                df = pd.read_parquet(pq)

            orig_cols = list(df.columns)
            new_cols = orig_cols.copy()
            rename_map: Dict[str, str] = {}

            for col in list(orig_cols):
                base = str(col)
                # Kandidat ist steuerbar, wenn in Auswahl enthalten oder bereits nF_-Variante
                if base in include_map:
                    want_include = include_map[base]
                    if not want_include and not base.startswith(NF_PREFIX):
                        # ausschließen -> Prefix hinzufügen
                        target = NF_PREFIX + base
                        if target in df.columns:
                            # exists already -> drop base
                            df.drop(columns=[base], inplace=True)
                        else:
                            rename_map[base] = target
                    elif want_include and base.startswith(NF_PREFIX):
                        # einschließen -> Prefix entfernen
                        target = base[len(NF_PREFIX):]
                        if target in df.columns:
                            # Ziel existiert: drop die nF_-Spalte
                            df.drop(columns=[base], inplace=True)
                        else:
                            rename_map[base] = target
                elif str(col).startswith(NF_PREFIX):
                    # Spalten, die bereits nF_ sind, aber nicht explizit in der Auswahl erscheinen
                    # (z. B. weil herausgefiltert) lassen wir unverändert.
                    pass

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            # Spaltenreihenfolge: Schlüssel voran, Rest danach
            order = []
            for k in ["Teil", "Datum"]:
                if k in df.columns:
                    order.append(k)
            order += [c for c in df.columns if c not in order]
            df = df[order]

            # Parquet speichern
            df.to_parquet(pq, index=False)

            # Excel ggf. aktualisieren
            xlsx = pq.with_name("features.xlsx")
            if xlsx.exists():
                try:
                    df.to_excel(xlsx, index=False)
                except Exception:
                    # nicht kritisch, nur melden
                    errors.append(f"Excel konnte nicht gespeichert werden: {xlsx}")

            updated += 1
        except Exception:
            errors.append(f"Fehler bei: {pq}\n{traceback.format_exc()}")

    msg = f"Aktualisiert: {updated} Parquet-Datei(en)."
    if errors:
        msg += f"\nHinweise/Fehler: {len(errors)}\n- " + "\n- ".join(errors[:5])
    messagebox.showinfo("Fertig", msg)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Features auswählen (Training)")
        self.geometry("700x600")

        self.root_dir: Path | None = None
        self.check_vars: Dict[str, tk.BooleanVar] = {}

        # UI
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

        # Scrollbarer Bereich für Checkboxen
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
        # Bestehende Widgets entfernen
        for w in list(self.check_frame.children.values()):
            w.destroy()
        self.check_vars.clear()
        # Default-Status: eingeschaltet, außer bereits als nF_ vorhanden
        # dafür schauen wir in die erste Datei, ob eine nF_-Version existiert
        try:
            df0 = pd.read_parquet(pq_files[0])
            existing = set(df0.columns)
        except Exception:
            existing = set()
        for c in cols:
            var = tk.BooleanVar(value=True)
            # wenn in erster Datei nF_-Variante existiert, dann abwählen
            if (NF_PREFIX + c) in existing:
                var.set(False)
            # bereits als F_NiU_ deklariert -> ausblenden (keine Checkbox)
            if c.startswith(PREFIX_EXCLUDE):
                continue
            self.check_vars[c] = var
        # Anzeigen
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

