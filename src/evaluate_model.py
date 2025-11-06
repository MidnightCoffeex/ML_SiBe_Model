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
    name_peak = "Quartal - Spitzenwert"
    name_peak_combined = "Quartal - Spitzenwert + Gesamtbestand ohne SiBe"
    part_txt = str(results.get("Teil").iloc[0]) if "Teil" in results.columns and not results.empty else ""
    # WBZ-Days (robust gegenueber NiU-Prefix)
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
    dates_series = pd.to_datetime(results["Datum"], errors="coerce")
    if not dates_series.empty:
        start_date = dates_series.min().normalize()
        boundaries = [start_date]
        current = start_date
        while current <= dates_series.max():
            current = current + pd.DateOffset(months=3)
            boundaries.append(current)
        boundaries = sorted(set(boundaries))
        block_idx = pd.cut(
            dates_series,
            bins=boundaries,
            right=False,
            labels=False,
            include_lowest=True,
        )
        results["_quarter_block"] = block_idx.astype("Int64")
        results["pred_peak_curve"] = results.groupby("_quarter_block")[pred_col].transform("max")
        results.drop(columns=["_quarter_block"], inplace=True)
    else:
        results["pred_peak_curve"] = results[pred_col]

    fig2 = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Vorschlag ueber die Zeit",
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
    # EoD ohne SiBe (robust gegenueber NiU-Prefix)
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
        title_text=(f"{heading} - Zeitverlauf" if heading else "Zeitverlauf"),
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



def aggregate_all_parts(parts_root: str, out_dir: str) -> None:
    """Aggregate all part-level `Full_Time_predictions` into a combined HTML report."""
    root = Path(parts_root)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    raw_frames: list[pd.DataFrame] = []
    pred_name: Optional[str] = None
    global_min: Optional[pd.Timestamp] = None
    global_max: Optional[pd.Timestamp] = None

    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        if sub.name.lower() == 'alle_teile':
            continue

        csv_path = sub / 'Full_Time_predictions.csv'
        xlsx_path = sub / 'Full_Time_predictions.xlsx'
        df: Optional[pd.DataFrame] = None
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = None
        if df is None and xlsx_path.exists():
            try:
                df = pd.read_excel(xlsx_path)
            except Exception:
                df = None
        if df is None or df.empty:
            continue

        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datum'])
        if df.empty:
            continue

        if 'Hinterlegter SiBe' not in df.columns:
            for cand in ('Hinterlegter_SiBe', 'F_NiU_Hinterlegter SiBe', 'nF_Hinterlegter SiBe'):
                if cand in df.columns:
                    df['Hinterlegter SiBe'] = df[cand]
                    break
        if 'Hinterlegter SiBe' not in df.columns:
            continue

        pred_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('pred_')]
        if not pred_cols:
            continue
        if 'pred_L_WBZ_BlockMinAbs' in pred_cols:
            pred_col = 'pred_L_WBZ_BlockMinAbs'
        else:
            pred_col = pred_cols[0]
        if pred_name is None:
            pred_name = pred_col

        price_col = 'Price_Material_var'
        if price_col not in df.columns:
            df[price_col] = 0.0

        base_cols = ['Datum', 'Hinterlegter SiBe', pred_col, price_col]
        if 'Teil' in df.columns:
            base_cols.insert(0, 'Teil')
        df = df[base_cols].copy()
        if 'Teil' not in df.columns:
            df['Teil'] = sub.name
        df['Teil'] = df['Teil'].astype(str).replace({"": sub.name}).fillna(sub.name)

        df['Hinterlegter SiBe'] = pd.to_numeric(df['Hinterlegter SiBe'], errors='coerce').fillna(0.0)
        df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce').fillna(0.0)
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0.0)

        part_min = df['Datum'].min()
        part_max = df['Datum'].max()
        global_min = part_min if global_min is None else min(global_min, part_min)
        global_max = part_max if global_max is None else max(global_max, part_max)

        df = df.sort_values('Datum')
        raw = df[['Teil', 'Datum', 'Hinterlegter SiBe', pred_col, price_col]].copy()
        raw.rename(columns={pred_col: pred_name or pred_col}, inplace=True)
        raw['Kapitalbindung_Hinterlegter'] = raw['Hinterlegter SiBe'] * raw[price_col]
        raw['Kapitalbindung_Vorgeschlagen'] = raw[pred_name or pred_col] * raw[price_col]
        if pred_name is None:
            pred_name = pred_col
        raw_frames.append(raw[['Teil', 'Datum', 'Hinterlegter SiBe', pred_name,
                               price_col, 'Kapitalbindung_Hinterlegter',
                               'Kapitalbindung_Vorgeschlagen']])

    if not raw_frames or pred_name is None or global_min is None or global_max is None:
        return

    full_index = pd.date_range(global_min, global_max, freq='D')

    def _build_top_hover(source_df: pd.DataFrame, value_col: str, dates: list[pd.Timestamp]) -> list[str]:
        grouped = (
            source_df.groupby(['Datum', 'Teil'])[value_col]
            .sum()
            .reset_index()
        )
        if grouped.empty:
            return ["Keine Kapitalbindung" for _ in dates]
        grouped['Datum'] = pd.to_datetime(grouped['Datum'])
        total_by_date = grouped.groupby('Datum')[value_col].sum()
        grouped_dict = {dt: g for dt, g in grouped.groupby('Datum')}
        hover_texts: list[str] = []
        for date in dates:
            dt = pd.Timestamp(date)
            grp = grouped_dict.get(dt)
            total_val = float(total_by_date.get(dt, 0.0))
            if grp is None or total_val <= 0:
                hover_texts.append("Keine Kapitalbindung")
                continue
            top = grp.sort_values(value_col, ascending=False).head(3)
            lines = []
            for rank, row in enumerate(top.itertuples(), 1):
                val = float(getattr(row, value_col))
                pct = (val / total_val * 100.0) if total_val else 0.0
                lines.append(f"{rank}) Teil {row.Teil}: {val:,.0f} ({pct:.1f}%)")
            hover_texts.append("<br>".join(lines))
        return hover_texts
    forward_frames: list[pd.DataFrame] = []
    for raw in raw_frames:
        part_id = raw['Teil'].iloc[0] if not raw.empty else ''
        df_fwd = raw[['Teil', 'Datum', 'Hinterlegter SiBe', pred_name, 'Price_Material_var']].copy()
        df_fwd = df_fwd.set_index('Datum').reindex(full_index)
        df_fwd['Teil'] = df_fwd['Teil'].ffill().bfill().fillna(part_id)
        df_fwd[['Hinterlegter SiBe', pred_name, 'Price_Material_var']] = (
            df_fwd[['Hinterlegter SiBe', pred_name, 'Price_Material_var']]
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        df_fwd = df_fwd.rename_axis('Datum').reset_index()
        df_fwd['Kapitalbindung_Hinterlegter'] = df_fwd['Hinterlegter SiBe'] * df_fwd['Price_Material_var']
        df_fwd['Kapitalbindung_Vorgeschlagen'] = df_fwd[pred_name] * df_fwd['Price_Material_var']
        forward_frames.append(df_fwd[['Teil', 'Datum', 'Hinterlegter SiBe', pred_name,
                                      'Price_Material_var', 'Kapitalbindung_Hinterlegter',
                                      'Kapitalbindung_Vorgeschlagen']])

    long_df_forward = pd.concat(forward_frames, ignore_index=True)

    helper_path = outp / 'Alle_Teile_Tageswerte.xlsx'
    try:
        long_df_forward.to_excel(helper_path, index=False)
    except Exception:
        long_df_forward.to_csv(helper_path.with_suffix('.csv'), index=False)

    daily = (
        long_df_forward.groupby('Datum', as_index=False)
        .agg({
            'Hinterlegter SiBe': 'sum',
            pred_name: 'sum',
            'Kapitalbindung_Hinterlegter': 'sum',
            'Kapitalbindung_Vorgeschlagen': 'sum',
        })
        .sort_values('Datum')
    )

    daily_dates = list(daily['Datum'])
    hover_forward_actual = _build_top_hover(long_df_forward, 'Kapitalbindung_Hinterlegter', daily_dates)
    hover_forward_pred = _build_top_hover(long_df_forward, 'Kapitalbindung_Vorgeschlagen', daily_dates)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            'Hinterlegter vs Vorgeschlagener SiBe (Summe)',
            'Kapitalbindung (Summe)',
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=daily['Datum'],
            y=daily['Hinterlegter SiBe'],
            mode='lines',
            name='Hinterlegter SiBe',
            line=dict(color='blue'),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily['Datum'],
            y=daily[pred_name],
            mode='lines',
            name='Vorgeschlagener SiBe',
            line=dict(color='green'),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily['Datum'],
            y=daily['Kapitalbindung_Hinterlegter'],
            mode='lines',
            name='Kapitalbindung (Hinterlegter SiBe)',
            line=dict(color='blue'),
            customdata=hover_forward_actual,
            hovertemplate="Kapitalbindung (Hinterlegter SiBe): %{y:,.0f}<br>%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily['Datum'],
            y=daily['Kapitalbindung_Vorgeschlagen'],
            mode='lines',
            name='Kapitalbindung (Vorgeschlagener SiBe)',
            line=dict(color='orange'),
            customdata=hover_forward_pred,
            hovertemplate="Kapitalbindung (Vorgeschlagener SiBe): %{y:,.0f}<br>%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title='Alle Teile - Aggregierte Entwicklung',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(title_text='Datum', row=1, col=1)
    fig.update_xaxes(title_text='Datum', row=2, col=1)
    fig.update_yaxes(title_text='Summe SiBe', row=1, col=1)
    fig.update_yaxes(title_text='EUR (aggregiert)', row=2, col=1)
    fig.write_html(outp / 'Alle_Teile.html')

    # No-forward variant (no datum carry-forward)
    long_df_no_forward = pd.concat(raw_frames, ignore_index=True)
    helper_path_no_forward = outp / 'Alle_Teile_Tageswerte_no_forward.xlsx'
    try:
        long_df_no_forward.to_excel(helper_path_no_forward, index=False)
    except Exception:
        long_df_no_forward.to_csv(helper_path_no_forward.with_suffix('.csv'), index=False)

    daily_no_forward = (
        long_df_no_forward.groupby('Datum', as_index=False)
        .agg({
            'Hinterlegter SiBe': 'sum',
            pred_name: 'sum',
            'Kapitalbindung_Hinterlegter': 'sum',
            'Kapitalbindung_Vorgeschlagen': 'sum',
        })
        .sort_values('Datum')
    )
    daily_no_forward = (
        daily_no_forward.set_index('Datum')
        .reindex(full_index, fill_value=0.0)
        .rename_axis('Datum')
        .reset_index()
    )

    daily_dates_no_forward = list(daily_no_forward['Datum'])
    hover_no_forward_actual = _build_top_hover(long_df_no_forward, 'Kapitalbindung_Hinterlegter', daily_dates_no_forward)
    hover_no_forward_pred = _build_top_hover(long_df_no_forward, 'Kapitalbindung_Vorgeschlagen', daily_dates_no_forward)

    fig_no_forward = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            'Hinterlegter vs Vorgeschlagener SiBe (Summe, ohne Forward-Fill)',
            'Kapitalbindung (Summe, ohne Forward-Fill)',
        ),
    )
    fig_no_forward.add_trace(
        go.Scatter(
            x=daily_no_forward['Datum'],
            y=daily_no_forward['Hinterlegter SiBe'],
            mode='lines',
            name='Hinterlegter SiBe',
            line=dict(color='blue'),
        ),
        row=1,
        col=1,
    )
    fig_no_forward.add_trace(
        go.Scatter(
            x=daily_no_forward['Datum'],
            y=daily_no_forward[pred_name],
            mode='lines',
            name='Vorgeschlagener SiBe',
            line=dict(color='green'),
        ),
        row=1,
        col=1,
    )
    fig_no_forward.add_trace(
        go.Scatter(
            x=daily_no_forward['Datum'],
            y=daily_no_forward['Kapitalbindung_Hinterlegter'],
            mode='lines',
            name='Kapitalbindung (Hinterlegter SiBe)',
            line=dict(color='blue'),
            customdata=hover_no_forward_actual,
            hovertemplate="Kapitalbindung (Hinterlegter SiBe): %{y:,.0f}<br>%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_no_forward.add_trace(
        go.Scatter(
            x=daily_no_forward['Datum'],
            y=daily_no_forward['Kapitalbindung_Vorgeschlagen'],
            mode='lines',
            name='Kapitalbindung (Vorgeschlagener SiBe)',
            line=dict(color='orange'),
            customdata=hover_no_forward_pred,
            hovertemplate="Kapitalbindung (Vorgeschlagener SiBe): %{y:,.0f}<br>%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig_no_forward.update_layout(
        title='Alle Teile - Aggregierte Entwicklung (ohne Forward-Fill)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig_no_forward.update_xaxes(title_text='Datum', row=1, col=1)
    fig_no_forward.update_xaxes(title_text='Datum', row=2, col=1)
    fig_no_forward.update_yaxes(title_text='Summe SiBe', row=1, col=1)
    fig_no_forward.update_yaxes(title_text='EUR (aggregiert)', row=2, col=1)
    fig_no_forward.write_html(outp / 'Alle_Teile_no_forward.html')


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


