#!/usr/bin/env python3
# Diese Tkinter-Oberfläche sammelt Trainingsparameter per Klick statt über die Konsole.
# Sie startet anschließend das Trainingsskript mit den ausgewählten Einstellungen.
# Tkinter-Oberfläche, die Trainingsparameter abfragt und das CLI-Skript startet.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


ROOT = Path(__file__).resolve().parents[1]


# Hauptfenster für das Training: hier werden alle Parameter gepflegt.
class App(tk.Tk):
    # Initialisiert das Fenster und legt alle Eingabefelder für das Training an.
    def __init__(self) -> None:
        self.title("Train Models – GUI")
        self.title("Train Models – GUI")
        self.geometry("820x680")

        # Variablen
        self.v_features = tk.StringVar(value=str(ROOT / "Features"))
        self.v_part = tk.StringVar(value="ALL")
        self.v_model_dir = tk.StringVar(value=str(ROOT / "Modelle"))
        self.v_model_id = tk.StringVar(value="")
        self.v_models = tk.StringVar(value="gb")
        self.v_targets = tk.StringVar(value="L_WBZ_BlockMinAbs")
        self.v_estimators = tk.StringVar(value="700")
        self.v_lr = tk.StringVar(value="0.05")
        self.v_max_depth = tk.StringVar(value="3")
        self.v_subsample = tk.StringVar(value="0.8")
        self.v_cv = tk.StringVar(value="3")
        self.v_weight_scheme = tk.StringVar(value="blockmin")
        self.v_weight_factor = tk.StringVar(value="3.0")
        self.v_ts_scope = tk.StringVar(value="global")  # global | local

        row = 0
        frm = tk.Frame(self)
        frm.pack(fill=tk.X, padx=10, pady=10)

        # Hilfsfunktion, um eine beschriftete Eingabezeile samt optionalem Dateidialog zu erzeugen.
        def add_row(label: str, var: tk.StringVar, browse: bool = False, is_dir: bool = True):
            nonlocal row
            tk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
            tk.Entry(frm, textvariable=var, width=70).grid(row=row, column=1, sticky="we", padx=6)
            if browse:
                # Öffnet je nach Bedarf einen Datei- oder Ordner-Dialog und überträgt die Auswahl in das Eingabefeld.
                def pick():
                    sel = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
                    if sel:
                        var.set(sel)
                tk.Button(frm, text="Wählen…", command=pick).grid(row=row, column=2)
            row += 1

        add_row("Features-Root:", self.v_features, browse=True, is_dir=True)
        add_row("Teil (oder ALL):", self.v_part)
        add_row("Modelle-Root:", self.v_model_dir, browse=True, is_dir=True)
        add_row("Model-ID (optional):", self.v_model_id)
        add_row("Model-Typ(en) (gb,xgb,lgbm|ALL):", self.v_models)
        add_row("Targets (Komma):", self.v_targets)
        add_row("n_estimators:", self.v_estimators)
        add_row("learning_rate:", self.v_lr)
        add_row("max_depth:", self.v_max_depth)
        add_row("subsample:", self.v_subsample)
        add_row("cv_splits:", self.v_cv)

        # Zeitreihen-Scope und Gewichtung
        scf = tk.Frame(self)
        scf.pack(fill=tk.X, padx=10)
        tk.Label(scf, text="Timeseries-Scope:").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(scf, text="global", variable=self.v_ts_scope, value="global").grid(row=0, column=1, sticky="w")
        tk.Radiobutton(scf, text="local", variable=self.v_ts_scope, value="local").grid(row=0, column=2, sticky="w")
        tk.Label(scf, text="Gewichtung:").grid(row=1, column=0, sticky="w", pady=(6,0))
        tk.Entry(scf, textvariable=self.v_weight_scheme, width=18).grid(row=1, column=1, sticky="w", pady=(6,0))
        tk.Entry(scf, textvariable=self.v_weight_factor, width=10).grid(row=1, column=2, sticky="w", pady=(6,0))

        # Aktionen
        act = tk.Frame(self)
        act.pack(fill=tk.X, padx=10, pady=6)
        tk.Button(act, text="Start", command=self.run).pack(side=tk.RIGHT)

        # Protokoll
        self.txt = tk.Text(self, height=18)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.txt.configure(state=tk.DISABLED)

    # Zeigt Protokolleinträge im Fenster an, damit der Benutzer Rückmeldungen erhält.
    def log(self, s: str) -> None:
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, s)
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    # Übersetzt die Eingaben in einen Aufruf des Trainingsskripts und startet ihn.
    def run(self) -> None:
        try:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "train.py"),
                "--data", self.v_features.get().strip(),
                "--part", self.v_part.get().strip(),
                "--model-dir", self.v_model_dir.get().strip(),
                "--models", self.v_models.get().strip(),
                "--targets", self.v_targets.get().strip(),
                "--n_estimators", self.v_estimators.get().strip(),
                "--learning_rate", self.v_lr.get().strip(),
                "--max_depth", self.v_max_depth.get().strip(),
                "--subsample", self.v_subsample.get().strip(),
                "--cv", self.v_cv.get().strip(),
                "--ts_scope", self.v_ts_scope.get().strip(),
                "--weight_scheme", self.v_weight_scheme.get().strip(),
                "--weight_factor", self.v_weight_factor.get().strip(),
            ]
            model_id = self.v_model_id.get().strip()
            if model_id:
                cmd += ["--model-id", model_id]
            subprocess.Popen(cmd)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Fehler", f"Konnte Training nicht starten: {exc}")


# Startet die grafische Oberfläche, wenn das Skript direkt ausgeführt wird.
def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
