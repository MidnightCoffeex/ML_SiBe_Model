#!/usr/bin/env python3
"""Simple GUI to run build_features with form inputs.

Opens a window to select input (raw CSV) and output (features) directories,
then calls scripts/build_features.py with those arguments and streams output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


ROOT = Path(__file__).resolve().parents[1]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Build Features – GUI")
        self.geometry("700x420")

        self.var_input = tk.StringVar(value=str(ROOT / "Rohdaten"))
        self.var_output = tk.StringVar(value=str(ROOT / "Features"))

        frm = tk.Frame(self)
        frm.pack(fill=tk.X, padx=10, pady=10)

        # Input
        tk.Label(frm, text="Rohdaten-Ordner:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.var_input, width=70).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(frm, text="Wählen…", command=self.pick_input).grid(row=0, column=2)

        # Output
        tk.Label(frm, text="Features-Ordner:").grid(row=1, column=0, sticky="w", pady=(8,0))
        tk.Entry(frm, textvariable=self.var_output, width=70).grid(row=1, column=1, sticky="we", padx=6, pady=(8,0))
        tk.Button(frm, text="Wählen…", command=self.pick_output).grid(row=1, column=2, pady=(8,0))

        # Actions
        act = tk.Frame(self)
        act.pack(fill=tk.X, padx=10)
        tk.Button(act, text="Start", command=self.run).pack(side=tk.RIGHT)

        # Log
        self.txt = tk.Text(self, height=16)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt.configure(state=tk.DISABLED)

    def pick_input(self) -> None:
        sel = filedialog.askdirectory(title="Rohdaten-Ordner wählen")
        if sel:
            self.var_input.set(sel)

    def pick_output(self) -> None:
        sel = filedialog.askdirectory(title="Features-Ordner wählen")
        if sel:
            self.var_output.set(sel)

    def log(self, s: str) -> None:
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, s)
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def run(self) -> None:
        inp = self.var_input.get().strip()
        out = self.var_output.get().strip()
        if not inp or not out:
            messagebox.showerror("Fehler", "Bitte Eingabe- und Ausgabeordner wählen.")
            return
        cmd = [sys.executable, str(ROOT / "scripts" / "build_features.py"), "--input", inp, "--output", out]
        try:
            subprocess.Popen(cmd)
            # Fenster schließen und beenden
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Fehler", f"Konnte Skript nicht ausführen: {exc}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
