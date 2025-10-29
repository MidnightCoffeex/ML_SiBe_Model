#!/usr/bin/env python3
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List
import sys

# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import data_pipeline


LOCKED_BASE = ['F_NiU_EoD_Bestand', 'F_NiU_Hinterlegter SiBe', 'EoD_Bestand_noSiBe']


class ScrollableFrame(tk.Frame):
    def __init__(self, master: tk.Misc, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Mouse wheel bindings (Windows/Mac/Linux)
        self.inner.bind_all("<MouseWheel>", self._on_mousewheel, add=False)
        self.inner.bind_all("<Button-4>", self._on_mousewheel_linux, add=False)
        self.inner.bind_all("<Button-5>", self._on_mousewheel_linux, add=False)

    def _on_configure(self, event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        # stretch inner frame to canvas width
        self.canvas.itemconfig(self._win, width=event.width)

    def _on_mousewheel(self, event) -> None:
        # Windows/Mac
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event) -> None:
        # Linux: Button-4 up, Button-5 down
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Build Features (Selective)')
        self.geometry('800x600')

        self.input_dir: Path | None = None
        self.output_dir: Path | None = None

        # Top: paths
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(top, text='Rohdaten wählen', command=self.pick_input).pack(side=tk.LEFT)
        self.in_label = tk.Label(top, text='Rohdaten: (leer)', anchor='w')
        self.in_label.pack(side=tk.LEFT, padx=10)

        bot = tk.Frame(self)
        bot.pack(fill=tk.X, padx=10)
        tk.Button(bot, text='Ausgabe wählen', command=self.pick_output).pack(side=tk.LEFT)
        self.out_label = tk.Label(bot, text='Ausgabe: GPT_Features_Test (Default)', anchor='w')
        self.out_label.pack(side=tk.LEFT, padx=10)

        # Scrollable content (features + labels)
        scroll = ScrollableFrame(self)
        scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Feature section (ohne Lags)
        sec_feat = tk.LabelFrame(scroll.inner, text='Features')
        sec_feat.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.feature_vars: dict[str, tk.BooleanVar] = {}
        try:
            all_feats = data_pipeline.list_available_feature_names()
        except Exception:
            all_feats = []
        non_lag_feats = [f for f in all_feats if not f.startswith('Lag_')]
        for name in non_lag_feats:
            var = tk.BooleanVar(value=False)
            self.feature_vars[name] = var
            tk.Checkbutton(sec_feat, text=name, variable=var, anchor='w', justify='left').pack(fill=tk.X, padx=10)

        # Lag section (eigener Bereich)
        sec_lag = tk.LabelFrame(scroll.inner, text='Lag Features')
        sec_lag.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.lag_vars: dict[str, tk.BooleanVar] = {}
        lag_feats = [f for f in all_feats if f.startswith('Lag_')]
        for name in lag_feats:
            var = tk.BooleanVar(value=False)
            self.lag_vars[name] = var
            tk.Checkbutton(sec_lag, text=name, variable=var, anchor='w', justify='left').pack(fill=tk.X, padx=10)

        # Label section
        sec_lab = tk.LabelFrame(scroll.inner, text='Labels')
        sec_lab.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.label_vars: dict[str, tk.BooleanVar] = {}
        try:
            all_labs = data_pipeline.list_available_label_names()
        except Exception:
            all_labs = []
        for name in all_labs:
            var = tk.BooleanVar(value=False)
            self.label_vars[name] = var
            tk.Checkbutton(sec_lab, text=name, variable=var, anchor='w', justify='left').pack(fill=tk.X, padx=10)

        # Run button
        # Locked base Anzeige
        frame_locked = tk.Frame(self)
        frame_locked.pack(fill=tk.X, padx=10, pady=(0, 0))
        tk.Label(frame_locked, text='Fixe Basis (immer enthalten):').pack(anchor='w')
        for k in LOCKED_BASE:
            cb = tk.Checkbutton(frame_locked, text=k, state=tk.DISABLED)
            cb.select(); cb.pack(anchor='w', padx=10)

        tk.Button(self, text='Build', command=self.run_build).pack(side=tk.BOTTOM, pady=15)

    def pick_input(self) -> None:
        sel = filedialog.askdirectory(title='Rohdaten-Ordner wählen')
        if sel:
            self.input_dir = Path(sel)
            self.in_label.config(text=f'Rohdaten: {self.input_dir}')

    def pick_output(self) -> None:
        sel = filedialog.askdirectory(title='Ausgabe-Ordner wählen')
        if sel:
            self.output_dir = Path(sel)
            self.out_label.config(text=f'Ausgabe: {self.output_dir}')

    def run_build(self) -> None:
        inp = str(self.input_dir or (Path(__file__).resolve().parents[1] / 'Rohdaten'))
        out = str(self.output_dir or (Path(__file__).resolve().parents[1] / 'GPT_Features_Test'))
        features = [k for k, v in self.feature_vars.items() if v.get()]
        features += [k for k, v in getattr(self, 'lag_vars', {}).items() if v.get()]
        labels = [k for k, v in self.label_vars.items() if v.get()]
        try:
            data_pipeline.run_pipeline_selective(inp, out, features, labels)
            messagebox.showinfo('Fertig', f'Features geschrieben nach {out}')
        except Exception as exc:
            messagebox.showerror('Fehler', str(exc))


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
