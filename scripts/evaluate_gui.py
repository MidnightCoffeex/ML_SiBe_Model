#!/usr/bin/env python3
# Diese Oberfläche ermöglicht es, Modelle ohne Kommandozeile zu evaluieren.
# Sie nimmt Pfadangaben entgegen und startet anschließend das Auswertungsskript.
# Einfache Tkinter-Oberfläche, um Modelle ohne Kommandozeile zu evaluieren.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


ROOT = Path(__file__).resolve().parents[1]


# Hauptfenster für die Auswertung: sammelt Eingaben und startet den Prozess.
class App(tk.Tk):
    # Richtet alle Eingabefelder ein und hinterlegt Standardwerte.
    def __init__(self) -> None:
        self.title("Evaluate Models – GUI")
        self.title("Evaluate Models – GUI")
        self.geometry("820x520")

        self.v_features = tk.StringVar(value=str(ROOT / "Features"))
        self.v_part = tk.StringVar(value="ALL")
        self.v_model_dir = tk.StringVar(value=str(ROOT / "Modelle"))
        self.v_model_type = tk.StringVar(value="gb")
        self.v_model_id = tk.StringVar(value="1")
        self.v_raw = tk.StringVar(value=str(ROOT / "Rohdaten"))
        self.v_targets = tk.StringVar(value="L_HalfYear_Target")
        self.v_plots = tk.StringVar(value=str(ROOT / "New_Test_Plots"))

        frm = tk.Frame(self)
        frm.pack(fill=tk.X, padx=10, pady=10)

        # Hilfsfunktion, um beschriftete Eingabereihen samt optionalem Dateidialog zu erzeugen.
        def add_row(label: str, var: tk.StringVar, browse: bool = False, is_dir: bool = True):
            r = add_row.row
            tk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            tk.Entry(frm, textvariable=var, width=70).grid(row=r, column=1, sticky="we", padx=6)
            if browse:
                # Öffnet je nach Einstellung einen Datei- oder Ordner-Dialog und übernimmt die Auswahl.
                def pick():
                    sel = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
                    if sel:
                        var.set(sel)
                tk.Button(frm, text="Wählen…", command=pick).grid(row=r, column=2)
            add_row.row += 1
        add_row.row = 0

        add_row("Features-Root:", self.v_features, browse=True, is_dir=True)
        add_row("Teil (oder ALL):", self.v_part)
        add_row("Modelle-Root:", self.v_model_dir, browse=True, is_dir=True)
        add_row("Model-Typ (gb/xgb/lgbm):", self.v_model_type)
        add_row("Model-ID:", self.v_model_id)
        add_row("Rohdaten-Root:", self.v_raw, browse=True, is_dir=True)
        add_row("Targets (Komma):", self.v_targets)
        add_row("Plots-Root:", self.v_plots, browse=True, is_dir=True)

        act = tk.Frame(self)
        act.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(act, text="Start", command=self.run).pack(side=tk.RIGHT)

        self.txt = tk.Text(self, height=16)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt.configure(state=tk.DISABLED)

    # Schreibt Text in das Ausgabefeld, damit der Benutzer Rückmeldungen sieht.
    def log(self, s: str) -> None:
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, s)
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    # Stellt den Befehl zusammen und startet das Evaluierungsskript im Hintergrund.
    def run(self) -> None:
        try:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                "--features", self.v_features.get().strip(),
                "--part", self.v_part.get().strip(),
                "--model-dir", self.v_model_dir.get().strip(),
                "--model-type", self.v_model_type.get().strip(),
                "--model-id", self.v_model_id.get().strip(),
                "--raw", self.v_raw.get().strip(),
                "--targets", self.v_targets.get().strip(),
                "--plots", self.v_plots.get().strip(),
            ]
            subprocess.Popen(cmd)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Fehler", f"Konnte Evaluation nicht starten: {exc}")


# Startet die grafische Oberfläche, wenn das Skript direkt ausgeführt wird.
def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
