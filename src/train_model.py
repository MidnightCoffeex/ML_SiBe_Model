# Dieses Modul formt aus den Feature-Tabellen Trainingsdaten und führt das Modelltraining durch.
# Es berechnet nötige Labels, kümmert sich um Gewichte, optionales Cross-Validation und die Fortschrittsanzeige.

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
import sys
import threading
import time
import os

class _MultiOutputXGB:
    """Einfache Multi-Output-Huelle fuer XGB-Regressoren (picklebar)."""

    def __init__(self, estimators: list):
        self.estimators_ = estimators
        self.n_outputs_ = len(estimators)
        first = estimators[0]
        self.n_features_in_ = getattr(first, "n_features_in_", None)
        self.feature_names_in_ = getattr(first, "feature_names_in_", None)

    def predict(self, Xin):
        preds = [est.predict(Xin) for est in self.estimators_]
        return np.column_stack(preds)


# Ermittelt pro Tag, wie stark der Bestand innerhalb eines WBZ-Zeitraums unter null fallen würde.
def _compute_block_min_abs_label(df: pd.DataFrame, horizon_floor_days: int = 14) -> pd.Series:
    # Berechnet L_WBZ_BlockMinAbs direkt aus den Zeitreihen (negativer Bestand innerhalb der WBZ).
    if 'EoD_Bestand_noSiBe' not in df.columns or 'Datum' not in df.columns:
        raise ValueError("Columns 'Datum' and 'EoD_Bestand_noSiBe' are required")
    wbz = pd.to_numeric(df.get('WBZ_Days', 0), errors='coerce').fillna(0).astype(float)
    dates = pd.to_datetime(df['Datum'], errors='coerce')
    vals = pd.to_numeric(df['EoD_Bestand_noSiBe'], errors='coerce').fillna(0).to_numpy()
    H = np.maximum(wbz.to_numpy(), float(horizon_floor_days))
    out = np.zeros(len(df), dtype=float)
    darr = dates.to_numpy()
    for i in range(len(df)):
        h = int(max(1, round(H[i])))
        start = darr[i]
        if pd.isna(start):
            out[i] = 0.0
            continue
        end = start + np.timedelta64(h, 'D')
        mask = (darr >= start) & (darr < end)
        y = vals[mask]
        mmin = float(np.nanmin(y)) if y.size else 0.0
        out[i] = max(0.0, -mmin)
    return pd.Series(out, index=df.index, name='L_WBZ_BlockMinAbs')


# Lädt Feature-Dateien von der Festplatte und versucht zuerst das schnelle Parquet-Format zu verwenden.
def load_features(path: str) -> pd.DataFrame:
    # Lädt vorberechnete Features bevorzugt aus Parquet und fällt bei Bedarf auf CSV zurück.
    p = Path(path)
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


# Entfernt Anzeige-Spalten, stellt sicher, dass Ziele vorhanden sind und liefert X und y für das Training.
def prepare_data(df: pd.DataFrame, targets: list[str], *, selected_features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Gibt Feature-Matrix X und Ziel y zurück, bereinigt Anzeige-Spalten und entfernt Zeilen mit fehlenden Targets.
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) {missing} not found in dataset")
    df = df.dropna(subset=targets)
    y = df[targets]
    drop_cols = set(targets)
    # Entfernt Anzeige- sowie Altspalten aus dem Trainingssatz.
    drop_cols.update([
        "EoD_Bestand",
        "Hinterlegter SiBe",
        "nF_EoD_Bestand",
        "nF_Hinterlegter SiBe",
    ])
    # Entfernt alle Not-in-Use-Spalten.
    drop_cols.update([c for c in df.columns if isinstance(c, str) and (c.startswith("F_NiU_") or c.startswith("L_NiU_") or c.startswith("nF_"))])
    drop_cols.update([c for c in df.columns if isinstance(c, str) and ("LABLE" in c or "LABEL" in c)])
    # Verhindert Leckage: alle übrigen Label-Spalten entfernen, außer den aktiven Targets.
    drop_cols.update([c for c in df.columns if isinstance(c, str) and c.startswith("L_") and c not in targets])
    if selected_features:
        # Schneidet auf die erlaubte Featureliste zu.
        keep = [c for c in selected_features if c in df.columns and c not in drop_cols]
        X = df[keep]
    else:
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y


# Trainiert das klassische scikit-learn Gradient Boosting Modell mit den gewünschten Parametern.
def train_gradient_boosting_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> MultiOutputRegressor:
    # Trainiert einen Multi-Output-Gradient-Boosting-Regressor aus scikit-learn.
    base = GradientBoostingRegressor(
        random_state=0,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# Trainiert ein XGBoost-Modell, sofern die Bibliothek installiert ist.
def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
    eval_set: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    early_stopping_rounds: int | None = None,
) -> MultiOutputRegressor:
    # Trainiert einen Multi-Output-XGBoost-Regressor.
    from xgboost import XGBRegressor
    import numpy as np

    if eval_set is not None and early_stopping_rounds:
        X_val, y_val = eval_set
        estimators = []
        for col in y.columns:
            est = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                objective="reg:squarederror",
                random_state=0,
            )
            try:
                est.fit(
                    X,
                    y[col],
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val[col])],
                    eval_metric="rmse",
                    verbose=False,
                    early_stopping_rounds=early_stopping_rounds,
                )
            except TypeError:
                # Fallback falls early_stopping_rounds/ eval_metric nicht unterstuetzt sind
                est.fit(
                    X,
                    y[col],
                    sample_weight=sample_weight,
                )
            estimators.append(est)

        return _MultiOutputXGB(estimators)

    base = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        objective="reg:squarederror",
        random_state=0,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# Trainiert ein LightGBM-Modell, sofern die Bibliothek vorhanden ist.
def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    subsample: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> MultiOutputRegressor:
    # Trainiert einen Multi-Output-LightGBM-Regressor.
    from lightgbm import LGBMRegressor

    base = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=0,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# Rückwärtskompatibilität

def run_training_df(
    df: pd.DataFrame,
    model_path: str,
    targets: list[str],
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_splits: int | None = None,
    model_type: str = "gb",
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    weight_scheme: str = "blockmin",
    weight_factor: float = 5.0,
    selected_features: list[str] | None = None,
    progress: bool = False,
    early_stop: bool = False,
    early_stopping_rounds: int = 20,
) -> tuple[list[float], list[float]]:
    # Trainiert ein Modell direkt auf einem bereits geladenen DataFrame.
    X, y = prepare_data(df, targets, selected_features=selected_features)
    # Ab hier arbeiten wir nur noch mit numerischen Features und Zielwerten.
    # Default: alle Gewichte = 1
    weights = np.ones(len(y), dtype=float)
    # Optionale Gewichtung: Zeilen mit L_NiU_StockOut_MinAdd > 0 höher gewichten
    # Wichtig: auf dieselben Zeilen indizieren wie X/y (Index beibehalten)
    if weight_scheme:
        scheme = (weight_scheme or "none").strip().lower()
        if scheme == 'blockmin' and 'L_NiU_StockOut_MinAdd' in df.columns:
            sw = pd.to_numeric(df['L_NiU_StockOut_MinAdd'], errors='coerce').fillna(0)
            try:
                sw = sw.loc[y.index]
                weights = np.where(sw.to_numpy() > 0, float(weight_factor), 1.0)
            except Exception:
                weights = np.ones(len(y), dtype=float)
        elif scheme == 'flag' and 'Flag_StockOut' in df.columns:
            fl = pd.to_numeric(df['Flag_StockOut'], errors='coerce').fillna(0)
            try:
                fl = fl.loc[y.index]
                weights = np.where(fl.to_numpy() == 1, float(weight_factor), 1.0)
            except Exception:
                weights = np.ones(len(y), dtype=float)
    if len(X) < 50:
        print("Warning: very few training samples; results may be unreliable")

    if split_indices is None:
    # Falls keine Splits vorgegeben sind, erstellen wir zeitbasierte Trainings- und Testfenster.
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-2]
        train_full_idx, test_idx = splits[-1]
    else:
        train_idx, val_idx, train_full_idx, test_idx = split_indices

    if model_type == "gb":
        trainer = train_gradient_boosting_model
    elif model_type == "xgb":
        trainer = train_xgboost_model
    elif model_type == "lgbm":
        trainer = train_lightgbm_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Hilfsfunktionen für den Fortschrittsbalken
    # Kleine Fortschrittsanzeige für die Konsole, damit lange Trainingsläufe sichtbar bleiben.
    class _ProgressBar:
        # Legt Gesamtumfang und Status der Fortschrittsanzeige fest.
        def __init__(self, total: int, label: str = "Training") -> None:
            self.total = max(1, int(total))
            self.current = 0
            self.label = label
            self._stop = threading.Event()
            self._lock = threading.Lock()
            self._thread = threading.Thread(target=self._run, daemon=True)
        # Startet den Hintergrund-Thread, der den Balken regelmäßig aktualisiert.
        def start(self) -> None:
            self._thread.start()
        # Stoppt die Anzeige und sorgt dafür, dass zuletzt 100 % gezeigt werden.
        def stop(self) -> None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            with self._lock:
                self.current = self.total
            self._render(final=True)
            sys.stdout.write("\n"); sys.stdout.flush()
        # Erhöht den Fortschritt um n Schritte, typischerweise nach jedem Fit.
        def tick(self, n: int = 1) -> None:
            with self._lock:
                self.current = min(self.total, self.current + n)
        # Arbeitet im Hintergrund und zeichnet den Balken in kurzen Abständen neu.
        def _run(self) -> None:
            while not self._stop.is_set():
                self._render()
                time.sleep(0.1)
        # Baut die Textdarstellung des Balkens zusammen und schreibt sie in die Konsole.
        def _render(self, final: bool = False) -> None:
            with self._lock:
                cur = self.current; total = self.total
            pct = int(100 * cur / total)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#"*filled + "-"*(bar_len - filled)
            sys.stdout.write(f"\r{self.label} [{bar}] {pct:3d}% ({cur}/{total})")
            sys.stdout.flush()

    # Spezialvariante für das scikit-learn Modell: nach jedem Estimator wird der Fortschritt aktualisiert.
    def _train_gb_with_progress(Xtr, Ytr, sw, label: str):
        pbar = _ProgressBar(total=n_estimators * Ytr.shape[1], label=label)
        pbar.start()
        estimators = []
        for i in range(Ytr.shape[1]):
            est = GradientBoostingRegressor(
                random_state=0,
                n_estimators=1,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                warm_start=True,
            )
            for t in range(1, n_estimators + 1):
                est.n_estimators = t
                est.fit(Xtr, Ytr.iloc[:, i], sample_weight=sw)
                pbar.tick(1)
            estimators.append(est)
        pbar.stop()
        m = MultiOutputRegressor(GradientBoostingRegressor())
        m.estimators_ = estimators
        return m

    def _log_early_stop_info(model_obj, phase: str) -> None:
        """Gibt Hinweise aus, ob Early Stopping aktiv war (nur XGB)."""
        if model_type != "xgb":
            return
        ests = getattr(model_obj, "estimators_", None)
        best_iter = None
        best_score = None
        if ests:
            for est in ests:
                best_iter = getattr(est, "best_iteration", None)
                if best_iter is None:
                    best_iter = getattr(est, "best_ntree_limit", None)
                best_score = getattr(est, "best_score", None)
                if best_iter is not None:
                    break
        if best_iter is not None:
            note = ""
            if isinstance(best_score, (int, float)):
                note = f", best_score={best_score:.4f}"
            elif best_score is not None:
                note = f", best_score={best_score}"
            print(
                f"{phase}: Early Stop aktiv -> beste Iteration {best_iter + 1}/{n_estimators}{note}"
            )
        else:
            print(f"{phase}: Early Stop nicht aktiv (trainierte ca. {n_estimators} Iterationen)")

    # Erster Fit
    if model_type == "gb" and (progress or os.environ.get('TRAIN_PROGRESS') == '1'):
        model = _train_gb_with_progress(X.iloc[train_idx], y.iloc[train_idx], weights[train_idx], label="Training gb")
    else:
        if model_type == "xgb" and early_stop:
            model = trainer(
                X.iloc[train_idx],
                y.iloc[train_idx],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[train_idx],
                eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            model = trainer(
                X.iloc[train_idx],
                y.iloc[train_idx],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[train_idx],
            )
    _log_early_stop_info(model, "Training")
    val_pred = model.predict(X.iloc[val_idx])
    val_mae = mean_absolute_error(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], val_pred, multioutput="raw_values"))
    val_r2 = r2_score(y.iloc[val_idx], val_pred, multioutput="raw_values")
    val_mape = mean_absolute_percentage_error(y.iloc[val_idx], val_pred)

    # Anschließend erneuter Fit auf allen Daten außer dem Testbereich
    if model_type == "gb" and (progress or os.environ.get('TRAIN_PROGRESS') == '1'):
        model = _train_gb_with_progress(X.iloc[train_full_idx], y.iloc[train_full_idx], weights[train_full_idx], label="Refit gb")
    else:
        if model_type == "xgb" and early_stop:
            model = trainer(
                X.iloc[train_full_idx],
                y.iloc[train_full_idx],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[train_full_idx],
                eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            model = trainer(
                X.iloc[train_full_idx],
                y.iloc[train_full_idx],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[train_full_idx],
            )
    _log_early_stop_info(model, "Refit")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    test_pred = model.predict(X.iloc[test_idx])
    test_mae = mean_absolute_error(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], test_pred, multioutput="raw_values"))
    test_r2 = r2_score(y.iloc[test_idx], test_pred, multioutput="raw_values")
    test_mape = mean_absolute_percentage_error(y.iloc[test_idx], test_pred)
    print(
        "Validation:",
        "MAE", val_mae,
        "RMSE", val_rmse,
        "R2", val_r2,
        "MAPE", val_mape,
    )
    print(
        "Test:",
        "MAE", test_mae,
        "RMSE", test_rmse,
        "R2", test_r2,
        "MAPE", test_mape,
    )

    try:
        importances = []
        for i in range(len(targets)):
            r = permutation_importance(
                model.estimators_[i],
                X.iloc[test_idx],
                y.iloc[test_idx].iloc[:, i],
                n_repeats=5,
                random_state=0,
                scoring="neg_mean_absolute_error",
            )
            importances.append(r.importances_mean)
        mean_imp = np.mean(importances, axis=0)
        order = np.argsort(mean_imp)[::-1]
        fi_df = pd.DataFrame({"feature": X.columns[order], "importance": mean_imp[order]})
        print("Permutation Importance:")
        for _, row in fi_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        fi_df.to_csv(Path(model_path).with_name("feature_importances.csv"), index=False)
    except Exception as exc:
        print("Permutation Importance konnte nicht berechnet werden:", exc)

    metrics_df = pd.DataFrame(
        {
            "target": targets,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "val_mape": val_mape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mape": test_mape,
        }
    )
    metrics_df.to_csv(Path(model_path).with_name("metrics.csv"), index=False)

    if cv_splits and cv_splits > 1:
        cv = TimeSeriesSplit(n_splits=cv_splits)
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_r2: list[float] = []
        fold_mape: list[float] = []
        for i, (tr, val) in enumerate(cv.split(X), 1):
            m = trainer(
                X.iloc[tr],
                y.iloc[tr],
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                sample_weight=weights[tr],
            )
            pred = m.predict(X.iloc[val])
            fold_mae.append(mean_absolute_error(y.iloc[val], pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[val], pred)))
            fold_r2.append(r2_score(y.iloc[val], pred))
            fold_mape.append(mean_absolute_percentage_error(y.iloc[val], pred))
            print(
                f"Fold {i}: MAE {fold_mae[-1]:.3f} RMSE {fold_rmse[-1]:.3f} R2 {fold_r2[-1]:.3f} MAPE {fold_mape[-1]:.3f}"
            )
        print(
            "CV Mean:",
            "MAE", np.mean(fold_mae),
            "RMSE", np.mean(fold_rmse),
            "R2", np.mean(fold_r2),
            "MAPE", np.mean(fold_mape),
        )

    return test_mae.tolist(), test_rmse.tolist()

# Komfortfunktion für die CLI: lädt Features von der Platte und trainiert alle gewünschten Modellvarianten hintereinander.
def run_training(
    features_path: str,
    model_dir: str,
    targets: list[str],
    model_types: list[str],
    *,
    model_id: str = "1",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    cv_splits: int | None = None,
    weight_scheme: str = "blockmin",
    weight_factor: float = 5.0,
    early_stop: bool = False,
    early_stopping_rounds: int = 20,
) -> None:
    df = load_features(features_path)
    X, _ = prepare_data(df, targets)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    split_indices = (*splits[-2], *splits[-1])  # train, val, train_full, test

    for mtype in model_types:
        model_path = Path(model_dir) / mtype / model_id / "model.joblib"
        run_training_df(
            df,
            str(model_path),
            targets,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            cv_splits=cv_splits,
            model_type=mtype,
            split_indices=split_indices,
            weight_scheme=weight_scheme,
            weight_factor=weight_factor,
            early_stop=early_stop,
            early_stopping_rounds=early_stopping_rounds,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety stock model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Path to features parquet"
    )
    parser.add_argument(
        "--model-dir", default="models", help="Directory to store models"
    )
    parser.add_argument("--model-id", default="1", help="Model identifier")
    parser.add_argument(
        "--models", default="gb", help="Comma separated model types (gb,xgb,lgbm)"
    )
    parser.add_argument(
        "--targets",
        default="L_WBZ_BlockMinAbs",
        help="Comma separated target column names",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--cv_splits", type=int, default=None)
    parser.add_argument("--weight_scheme", default="blockmin", help="Weighting: none|blockmin|flag")
    parser.add_argument("--weight_factor", type=float, default=5.0, help="Weight factor for selected scheme")
    args = parser.parse_args()
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    run_training(
        args.data,
        args.model_dir,
        target_list,
        model_types,
        model_id=args.model_id,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        cv_splits=args.cv_splits,
        weight_scheme=args.weight_scheme,
        weight_factor=args.weight_factor,
    )

