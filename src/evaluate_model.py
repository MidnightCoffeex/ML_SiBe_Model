from pathlib import Path
from typing import Tuple, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

from .train_model import load_features, prepare_data


def _dispo_date_range(raw_dir: str, part: str) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return min and max date for the given part in the Dispo CSV exports."""
    dates: list[pd.Series] = []
    for csv in Path(raw_dir).glob("*_Dispo.csv"):
        try:
            df = pd.read_csv(csv, encoding="ISO-8859-1", sep=None, engine="python", dtype=str)
        except Exception:
            continue
        if "Teil " in df.columns and "Teil" not in df.columns:
            df.rename(columns={"Teil ": "Teil"}, inplace=True)
        if "Teil" not in df.columns:
            continue
        df = df[df["Teil"].astype(str).str.strip() == str(part)]
        if df.empty:
            continue
        for col in ["Termin", "Solltermin"]:
            if col in df.columns:
                d = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce", dayfirst=True)
                dates.append(d)
    if dates:
        all_d = pd.concat(dates).dropna()
        if not all_d.empty:
            return all_d.min(), all_d.max()
    return None, None


def _resolve_col(df: pd.DataFrame, base: str) -> Optional[str]:
    for cand in (base, f"F_NiU_{base}", f"nF_{base}"):
        if cand in df.columns:
            return cand
    return None


def _evaluate_range(
    results: pd.DataFrame, prefix: str, target: str, output_dir: str
) -> None:
    """Save predictions and plots for a given subset."""
    if results.empty:
        return
    results = results.sort_values("Datum")
    pred_col = f"pred_{target}"

    actual_col = (
        "F_NiU_Hinterlegter SiBe"
        if "F_NiU_Hinterlegter SiBe" in results.columns
        else ("nF_Hinterlegter SiBe" if "nF_Hinterlegter SiBe" in results.columns else "Hinterlegter SiBe")
    )
    # Friendly naming and heading (Teil/WBZ)
    name_pred = "KI: Vorgeschlagener SiBe"
    name_eod = "Gesamtbestand ohne SiBe"
    name_actual_sibe = "Aktueller hinterlegter SiBe"
    name_combined = "Gesamtbestand ohne SiBe + Vorgeschlagener SiBe"
    name_peak = "Spitzenlinie Vorschlag"
    name_peak_combined = "Spitzenlinie + Gesamtbestand ohne SiBe"
    part_txt = str(results.get("Teil").iloc[0]) if "Teil" in results.columns and not results.empty else ""
    # WBZ-Days (robust gegenÃ¼ber NiU-Prefix)
    wbz_col = _resolve_col(results, "WBZ_Days")
    wbz_val = results.get(wbz_col).iloc[0] if wbz_col and not results.empty else None
    wbz_txt = f" | WBZ: {int(wbz_val)} Tage" if isinstance(wbz_val, (int, float)) and pd.notna(wbz_val) else ""
    heading = f"Teil {part_txt}{wbz_txt}" if part_txt else (f"WBZ{wbz_txt}" if wbz_txt else "")
    mae = mean_absolute_error(results[actual_col], results[pred_col])
    rmse = np.sqrt(mean_squared_error(results[actual_col], results[pred_col]))
    r2 = r2_score(results[actual_col], results[pred_col])
    mape = mean_absolute_percentage_error(results[actual_col], results[pred_col])
    print(
        f"{prefix} MAE vs Hinterlegter SiBe: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | MAPE: {mape:.3f}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
    try:
        results.to_excel(Path(output_dir) / f"{prefix}_predictions.xlsx", index=False)
    except Exception:
        pass

    plt.figure()
    sns.scatterplot(x=results[actual_col], y=results[pred_col])
    plt.xlabel(name_actual_sibe)
    plt.ylabel(name_pred)
    plt.title(f"{heading}\nAktueller SiBe vs. Vorschlag" if heading else "Aktueller SiBe vs. Vorschlag")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_actual_vs_pred.png")
    plt.close()

    fig = px.scatter(
        results,
        x=actual_col,
        y=pred_col,
        labels={actual_col: name_actual_sibe, pred_col: name_pred},
        title=f"{heading} | Aktueller SiBe vs. Vorschlag" if heading else "Aktueller SiBe vs. Vorschlag",
    )
    fig.write_html(Path(output_dir) / f"{prefix}_actual_vs_pred.html")

    plt.figure()
    sns.lineplot(x="Datum", y=actual_col, data=results, label="Actual")
    sns.lineplot(x="Datum", y=pred_col, data=results, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.title("Predictions Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{prefix}_predictions_over_time.png")
    plt.close()

    # Prepare an upper-envelope (peak) curve of the clamped predictions
    pred_col = f"pred_{target}"
    results["pred_peak_curve"] = (
        results[pred_col]
        .rolling(window=14, min_periods=1)
        .max()
    )

    fig2 = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Vorschlag Ã¼ber die Zeit",
            name_eod,
            name_combined,
            name_peak,
            name_peak_combined,
        ),
    )
    fig2.add_trace(
        go.Scatter(
            x=results["Datum"],
            y=results[actual_col],
            mode="lines",
            name=name_actual_sibe,
        ),
        row=1,
        col=1,
    )
    # Overlay the actual training label as a medium-light gray line (if present)
    if target in results.columns:
        try:
            lab_series = pd.to_numeric(results[target], errors="coerce")
        except Exception:
            lab_series = results[target]
        fig2.add_trace(
            go.Scatter(
                x=results["Datum"],
                y=lab_series,
                mode="lines",
                name=f"Label: {target}",
                line=dict(color="#B0B0B0"),
            ),
            row=1,
            col=1,
        )
    fig2.add_trace(
        go.Scatter(
            x=results["Datum"],
            y=results[pred_col],
            mode="lines",
            name=name_pred,
        ),
        row=1,
        col=1,
    )
    # EoD ohne SiBe (robust gegenÃ¼ber NiU-Prefix)
    eod_col = _resolve_col(results, "EoD_Bestand_noSiBe")
    if eod_col:
        fig2.add_trace(
            go.Scatter(
                x=results["Datum"],
                y=results[eod_col],
                mode="lines",
                name=name_eod,
            ),
            row=2,
            col=1,
        )

    if eod_col:
        combined = results[eod_col] + results[pred_col]
        combined_pos = combined.where(combined > 0)
        combined_neg = combined.where(combined <= 0)

        fig2.add_trace(
            go.Scatter(
                x=results["Datum"],
                y=combined_pos,
                mode="lines",
                name=name_combined,
                line=dict(color="blue"),
            ),
            row=3,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=results["Datum"],
                y=combined_neg,
                mode="lines",
                showlegend=False,
                line=dict(color="red"),
            ),
            row=3,
            col=1,
        )

    # Row 4: peak curve of predictions
    fig2.add_trace(
        go.Scatter(
            x=results["Datum"],
            y=results["pred_peak_curve"],
            mode="lines",
            name=name_peak,
            line=dict(color="purple"),
        ),
        row=4,
        col=1,
    )

    # Row 5: peak curve + EoD_Bestand_noSiBe
    if eod_col:
        combined2 = results[eod_col] + results["pred_peak_curve"]
        fig2.add_trace(
            go.Scatter(
                x=results["Datum"],
                y=combined2,
                mode="lines",
                name="Peak Curve + EoD_Bestand_noSiBe",
                line=dict(color="darkgreen"),
            ),
            row=5,
            col=1,
        )

    fig2.update_layout(
        hovermode="x unified",
        height=1800,
        title_text=(f"{heading} â€“ Zeitverlauf" if heading else "Zeitverlauf"),
    )
    fig2.update_yaxes(title_text=name_pred, row=1, col=1)
    fig2.update_yaxes(title_text=name_eod, row=2, col=1)
    fig2.update_yaxes(title_text=name_combined, row=3, col=1)
    fig2.update_yaxes(title_text=name_peak, row=4, col=1)
    fig2.update_yaxes(title_text=name_peak_combined, row=5, col=1)
    fig2.update_xaxes(title_text="Datum", row=5, col=1)
    fig2.write_html(Path(output_dir) / f"{prefix}_predictions_over_time.html")
    
    # help GC
    del fig2


def run_evaluation(
    features_path: str,
    model_path: str,
    targets: list[str],
    output_dir: str,
    raw_dir: str = "Rohdaten",
    model_type: str | None = None,
    selected_features: list[str] | None = None,
) -> None:
    """Evaluate the trained model and generate plots for multiple time frames."""
    if model_type is None:
        parts = set(Path(model_path).parts)
        for cand in ["gb", "xgb", "lgbm"]:
            if cand in parts:
                model_type = cand
                break

    df = load_features(features_path)
    X, y = prepare_data(df, targets, selected_features=selected_features)
    n = len(X)
    # Not enough samples to evaluate robustly -> skip gracefully
    if n < 3:
        print(f"Skip evaluation ({prefix if 'prefix' in locals() else 'ALL'}): not enough samples (n={n}).")
        return
    if n >= 7:
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-2]
        train_full_idx, test_idx = splits[-1]
    elif n >= 4:
        tscv = TimeSeriesSplit(n_splits=max(2, n - 2))
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-2]
        train_full_idx, test_idx = splits[-1]
    else:
        train_idx = np.arange(max(1, n - 2))
        val_idx = train_idx
        train_full_idx = train_idx
        test_idx = np.arange(max(1, n - 1), n)

    # If no test samples available, skip to avoid metric errors
    if test_idx.size == 0 or train_full_idx.size == 0:
        print(f"Skip evaluation ({prefix if 'prefix' in locals() else 'ALL'}): insufficient split sizes (n={n}).")
        return
    model = joblib.load(model_path)

    # Align features to model's expected columns if available
    def _align_features(Xin: pd.DataFrame) -> pd.DataFrame:
        try:
            ests = getattr(model, 'estimators_', None)
            if ests and len(ests) > 0:
                feat = getattr(ests[0], 'feature_names_in_', None)
                if feat is None:
                    return Xin
                need = list(feat)
                cur = set(Xin.columns)
                for c in need:
                    if c not in cur:
                        Xin[c] = 0.0
                Xin = Xin[need]
        except Exception:
            return Xin
        return Xin

    X_test_aligned = _align_features(X.iloc[test_idx].copy())
    y_pred_test = model.predict(X_test_aligned)
    # Clamp negative predictions to 0
    y_pred_test = np.clip(y_pred_test, 0, None)
    mae = mean_absolute_error(y.iloc[test_idx], y_pred_test, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred_test, multioutput="raw_values"))
    r2 = r2_score(y.iloc[test_idx], y_pred_test, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y.iloc[test_idx], y_pred_test)
    print("Test Metrics -> MAE:", mae, "RMSE:", rmse, "R2:", r2, "MAPE:", mape)

    # predictions for the entire feature set
    drop_cols = set(targets)
    drop_cols.update(["EoD_Bestand", "Hinterlegter SiBe"]) 
    drop_cols.update([c for c in df.columns if isinstance(c, str) and (c.startswith("F_NiU_") or c.startswith("L_NiU_") or c.startswith("nF_"))])
    drop_cols.update([c for c in df.columns if isinstance(c, str) and ("LABLE" in c or "LABEL" in c)])
    X_full = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X_full = X_full.select_dtypes(include=["number"]).fillna(0)
    X_full = _align_features(X_full)
    full_pred = model.predict(X_full)
    # Clamp negative predictions to 0
    full_pred = np.clip(full_pred, 0, None)
    results_full = df.copy()
    for i, col in enumerate(targets):
        results_full[f"pred_{col}"] = full_pred[:, i]

    part = str(df["Teil"].iloc[0]) if "Teil" in df.columns else ""
    # Ensure an 'Hinterlegter SiBe' column exists for plotting/metrics
    if "Hinterlegter SiBe" not in results_full.columns:
        for cand in ("Hinterlegter_SiBe", "F_NiU_Hinterlegter SiBe", "nF_Hinterlegter SiBe"):
            if cand in results_full.columns:
                results_full["Hinterlegter SiBe"] = results_full[cand]
                break
        if "Hinterlegter SiBe" not in results_full.columns:
            # fallback to zeros
            results_full["Hinterlegter SiBe"] = 0.0
    dispo_start, dispo_end = _dispo_date_range(raw_dir, part)

    # Evaluation for specified ranges
    full_prefix = "Full_Time"
    _evaluate_range(results_full, full_prefix, targets[0], output_dir)

    if dispo_start is not None and dispo_end is not None:
        mask = (results_full["Datum"] >= dispo_start) & (results_full["Datum"] <= dispo_end)
        dispo_results = results_full.loc[mask].copy()
        _evaluate_range(dispo_results, "Dispo_Time", targets[0], output_dir)

    # Training history from model
    if hasattr(model, "train_score_"):
        plt.figure()
        sns.lineplot(x=range(1, len(model.train_score_) + 1), y=model.train_score_)
        plt.xlabel("Iteration")
        plt.ylabel("Deviance")
        plt.title("Training History")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_history.png")
        plt.close()

        fig_hist = px.line(
            x=list(range(1, len(model.train_score_) + 1)),
            y=model.train_score_,
            labels={"x": "Iteration", "y": "Deviance"},
            title="Training History",
        )
        fig_hist.write_html(Path(output_dir) / "training_history.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--data", default="data/features.parquet", help="Features parquet file"
    )
    parser.add_argument(
        "--model", default="models/gb_regressor.joblib", help="Trained model file"
    )
    parser.add_argument("--model-type", help="Model type (gb,xgb,lgbm)")
    parser.add_argument(
        "--targets",
        default="LABLE_WBZ_NegBlockSum",
        help="Comma separated target column names",
    )
    parser.add_argument("--plots", default="plots", help="Directory to store plots")
    parser.add_argument("--raw", default="Rohdaten", help="Directory with raw CSV files")
    args = parser.parse_args()

    target_list = [t.strip() for t in args.targets.split(',') if t.strip()]
    run_evaluation(args.data, args.model, target_list, args.plots, args.raw, model_type=args.model_type)



