#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-analysis for trained centralized model.

- Loads experiment artifacts from --exp_dir
- Reads saved predictions & metrics (two-class or standard)
- Computes extra ML metrics (F1/precision/recall, ROC-AUC, PR-AUC)
- Computes trading KPIs (coverage, win rate, avg/net profit, cumulative PnL)
- Produces plots and a concise Markdown report

Usage:
  python analyze_centralized_model.py --exp_dir <path_to_experiment_dir> [--out <dir>]

Author: you :)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ---------- Utils ----------

def _fmt(bps: float) -> str:
    return f"{bps:.2f} bps"

def _nan_safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size else float("nan")

def _nan_safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x, ddof=1)) if x.size > 1 else float("nan")


from pandas.api import types as pdt

def _ensure_dt(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = df[col]
    # Если не datetime — конвертируем в UTC
    if not (pdt.is_datetime64_any_dtype(s) or pdt.is_datetime64tz_dtype(s)):
        df[col] = pd.to_datetime(s, utc=True, errors="coerce")
        return df
    # Уже datetime: убедимся, что таймзона есть и это UTC
    if pdt.is_datetime64tz_dtype(df[col].dtype):
        # tz-aware: привести к UTC
        try:
            df[col] = df[col].dt.tz_convert("UTC")
        except Exception:
            # если была naive внутри, подстрахуемся
            df[col] = df[col].dt.tz_localize("UTC")
    else:
        # naive -> локализуем в UTC
        df[col] = df[col].dt.tz_localize("UTC")
    return df


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

# ---------- Loaders ----------

def load_experiment(exp_dir: Path) -> Dict[str, Path]:
    exp_dir = exp_dir.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    files = {
        "config": exp_dir / "config.yaml",
        "metrics_two_class": exp_dir / "metrics_two_class.json",
        "metrics_global": exp_dir / "metrics_global.json",
        "metrics_per_symbol": exp_dir / "metrics_per_symbol.csv",
        "metrics_per_symbol_two_class": exp_dir / "metrics_per_symbol_two_class.csv",
        "preds_two_class": exp_dir / "test_predictions_two_class.csv",
        "preds_standard": exp_dir / "test_predictions.csv",
    }
    return files

# ---------- Metric computation ----------

def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def roc_pr_metrics(y_true, prob_pos) -> Dict[str, float]:
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob_pos)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = average_precision_score(y_true, prob_pos)
    except Exception:
        out["pr_auc"] = float("nan")
    return out

def trading_kpis_from_two_class(df: pd.DataFrame, cost_bps: Optional[float]) -> Dict[str, float]:
    # executed trades only
    if "execute" in df.columns:
        exec_mask = df["execute"].astype(bool).values
    else:
        # if missing, assume everything executed
        exec_mask = np.ones(len(df), dtype=bool)

    sub = df.loc[exec_mask].copy()
    n_total = len(df)
    n_exec = len(sub)
    coverage = n_exec / max(n_total, 1)

    # Direction accuracy on executed trades (y: -1/1 or 0/1 ?)
    # In two-class CSV we have y_direction in {-1, 1} and pred_direction in {0,1} or {-1,1}
    # Normalize to {0,1} with 1 == UP
    def _to01(x):
        x = np.asarray(x)
        if set(pd.unique(x)) <= {0, 1}:
            return x
        return (x > 0).astype(int)

    if "y_direction" in sub.columns and "pred_direction" in sub.columns:
        y01 = _to01(sub["y_direction"].values)
        p01 = _to01(sub["pred_direction"].values)
        acc = accuracy_score(y01, p01) if len(sub) else float("nan")
        f1m = f1_score(y01, p01, average="macro", zero_division=0) if len(sub) else float("nan")
    else:
        acc, f1m = float("nan"), float("nan")

    # Profit columns should already be present, but recompute net in case cost changed
    if "ret_bps" in sub.columns and "pred_direction" in sub.columns:
        # gross: up-> +ret, down-> -ret
        gross = np.where(_to01(sub["pred_direction"]) == 1, sub["ret_bps"].values, -sub["ret_bps"].values)
    else:
        gross = sub.get("gross_profit_bps", pd.Series(np.zeros(len(sub)))).values

    if cost_bps is None:
        # try read from CSV if already saved
        if "net_profit_bps" in sub.columns:
            net = sub["net_profit_bps"].values
        else:
            net = gross
    else:
        net = gross - float(cost_bps)

    win_rate = float(np.mean(net > 0)) if len(net) else float("nan")
    avg_gross = _nan_safe_mean(gross)
    avg_net = _nan_safe_mean(net)
    std_net = _nan_safe_std(net)
    # naive sharpe per trade
    sharpe = (avg_net / std_net) * math.sqrt(len(net)) if (len(net) > 1 and std_net > 0) else float("nan")

    return {
        "coverage": coverage,
        "n_total": n_total,
        "n_executed": n_exec,
        "direction_accuracy_on_executed": acc,
        "f1_macro_on_executed": f1m,
        "avg_gross_profit_bps": float(avg_gross),
        "avg_net_profit_bps": float(avg_net),
        "win_rate": win_rate,
        "sharpe_like": sharpe,
    }

# ---------- Plotting ----------

def plot_confusion(y_true, y_pred, out_path: Path, title: str = "Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(4.2, 3.8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Down","Up"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Down","Up"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v)}", ha="center", va="center")
    _savefig(out_path)

def plot_roc(y_true, prob_pos, out_path: Path, title: str = "ROC curve"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    _savefig(out_path)

def plot_pr(y_true, prob_pos, out_path: Path, title: str = "Precision-Recall curve"):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, prob_pos)
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    _savefig(out_path)


def plot_cum_pnl(df: pd.DataFrame, out_path: Path, title: str = "Cumulative Net Profit (bps)"):
    df = df.copy()
    df = _ensure_dt(df, "timestamp")
    # executed only
    if "execute" in df.columns:
        df = df[df["execute"].astype(bool)]
    # choose net column if exists else gross
    col = "net_profit_bps" if "net_profit_bps" in df.columns else ("gross_profit_bps" if "gross_profit_bps" in df.columns else None)
    if not col:
        return
    # сортировка по времени, если есть валидная дата
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp")
        x = df["timestamp"].values
    else:
        x = np.arange(len(df))
    cum = np.cumsum(df[col].values)
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(x, cum)
    plt.title(title)
    plt.xlabel("Time" if "timestamp" in df.columns and df["timestamp"].notna().any() else "Trade index")
    plt.ylabel("bps")
    _savefig(out_path)


def plot_profit_hist(df: pd.DataFrame, out_path: Path, title: str = "Per-trade Net Profit (bps)"):
    if "execute" in df.columns:
        df = df[df["execute"] == True]
    col = "net_profit_bps" if "net_profit_bps" in df.columns else "gross_profit_bps"
    if col not in df.columns:
        return
    plt.figure(figsize=(5.8, 3.8))
    plt.hist(df[col].values, bins=50)
    plt.title(title)
    plt.xlabel("bps")
    plt.ylabel("Count")
    _savefig(out_path)

def plot_symbol_bars(df: pd.DataFrame, out_dir: Path):
    if "symbol" not in df.columns:
        return
    # executed only
    if "execute" in df.columns:
        df = df[df["execute"] == True].copy()
    # normalize labels
    y_true = (df["y_direction"].values > 0).astype(int) if "y_direction" in df.columns else None
    y_pred = (df["pred_direction"].values > 0).astype(int) if "pred_direction" in df.columns else None
    per = []
    for sym, g in df.groupby("symbol"):
        row = {"symbol": sym}
        if y_true is not None and y_pred is not None:
            yt = (g["y_direction"] > 0).astype(int)
            yp = (g["pred_direction"] > 0).astype(int)
            row["accuracy"] = accuracy_score(yt, yp)
        col = "net_profit_bps" if "net_profit_bps" in g.columns else ("gross_profit_bps" if "gross_profit_bps" in g.columns else None)
        if col:
            row["avg_net_profit_bps"] = _nan_safe_mean(g[col].values)
        per.append(row)
    if not per:
        return
    per_df = pd.DataFrame(per).sort_values("avg_net_profit_bps", ascending=False)
    # accuracy
    if "accuracy" in per_df.columns:
        plt.figure(figsize=(6.5, 3.8))
        plt.bar(per_df["symbol"], per_df["accuracy"].values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Direction accuracy by symbol (executed trades)")
        plt.ylabel("Accuracy")
        _savefig(out_dir / "by_symbol_accuracy.png")
    # avg net
    plt.figure(figsize=(6.5, 3.8))
    plt.bar(per_df["symbol"], per_df["avg_net_profit_bps"].values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Avg net profit (bps) by symbol (executed trades)")
    plt.ylabel("bps")
    _savefig(out_dir / "by_symbol_avg_net_profit.png")

# ---------- Analysis flows ----------

def analyze_two_class(exp_dir: Path, files: Dict[str, Path], out_dir: Path) -> None:
    preds_path = files["preds_two_class"]
    if not preds_path.exists():
        print(f"[warn] {preds_path.name} is missing. Nothing to analyze in two-class mode.", file=sys.stderr)
        return

    df = pd.read_csv(preds_path)
    df = _ensure_dt(df, "timestamp")

    # Compute classification set on executed trades
    if "execute" in df.columns:
        exec_df = df[df["execute"] == True].copy()
    else:
        exec_df = df.copy()

    if {"y_direction", "pred_direction"} <= set(exec_df.columns):
        y = (exec_df["y_direction"].values > 0).astype(int)
        p = (exec_df["pred_direction"].values > 0).astype(int)
        cls = classification_metrics(y, p)
    else:
        cls = {"accuracy": float("nan"), "f1_macro": float("nan"), "precision_macro": float("nan"), "recall_macro": float("nan")}

    # ROC/PR if probabilities present
    rocpr = {}
    if {"proba_up"} <= set(df.columns):
        y_all = (df["y_direction"].values > 0).astype(int)
        rocpr = roc_pr_metrics(y_all, df["proba_up"].values)

    # Trading KPIs (need cost; try to read from metrics_two_class.json or config.yaml)
    cost_bps = None
    if files["metrics_two_class"].exists():
        try:
            with open(files["metrics_two_class"], "r", encoding="utf-8") as f:
                m2 = json.load(f)
                cost_bps = m2.get("profit_cost_bps", None)
        except Exception:
            pass

    kpis = trading_kpis_from_two_class(df, cost_bps)

    # --------- Plots ----------
    out_plots = out_dir / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    # Confusion on executed trades
    if {"y_direction", "pred_direction"} <= set(exec_df.columns):
        plot_confusion(
            (exec_df["y_direction"].values > 0).astype(int),
            (exec_df["pred_direction"].values > 0).astype(int),
            out_plots / "confusion_executed.png",
            title="Confusion (executed trades)",
        )

    # ROC / PR
    if {"proba_up"} <= set(df.columns):
        y_all = (df["y_direction"].values > 0).astype(int)
        plot_roc(y_all, df["proba_up"].values, out_plots / "roc_curve.png")
        plot_pr(y_all, df["proba_up"].values, out_plots / "pr_curve.png")

    # Cumulative PnL & histogram
    plot_cum_pnl(df, out_plots / "cumulative_pnl.png")
    plot_profit_hist(df, out_plots / "profit_hist.png")

    # Per symbol
    plot_symbol_bars(df, out_plots)

    # --------- Markdown report ----------
    report = out_dir / "report_two_class.md"
    lines = []
    lines.append(f"# Two-class analysis\n")
    lines.append(f"**Experiment:** `{exp_dir}`\n")
    lines.append(f"**Samples:** total={len(df)}, executed={int(kpis['n_executed'])} (coverage={kpis['coverage']*100:.2f}%)\n")
    lines.append("## Classification (executed trades)\n")
    lines.append(f"- Accuracy: **{cls['accuracy']:.4f}**")
    lines.append(f"- F1 macro: **{cls['f1_macro']:.4f}**")
    lines.append(f"- Precision macro: {cls['precision_macro']:.4f}")
    lines.append(f"- Recall macro: {cls['recall_macro']:.4f}\n")
    if rocpr:
        lines.append("## Probabilistic quality\n")
        lines.append(f"- ROC-AUC: {rocpr.get('roc_auc', float('nan')):.4f}")
        lines.append(f"- PR-AUC: {rocpr.get('pr_auc', float('nan')):.4f}\n")
    lines.append("## Trading KPIs\n")
    lines.append(f"- Win rate: **{kpis['win_rate']*100:.2f}%**")
    lines.append(f"- Avg gross profit: **{_fmt(kpis['avg_gross_profit_bps'])}**")
    lines.append(f"- Avg net profit: **{_fmt(kpis['avg_net_profit_bps'])}**")
    lines.append(f"- Sharpe-like (per trade): {kpis['sharpe_like']:.3f}\n")
    lines.append("## Plots\n")
    for img in ["confusion_executed.png","roc_curve.png","pr_curve.png","cumulative_pnl.png","profit_hist.png","by_symbol_accuracy.png","by_symbol_avg_net_profit.png"]:
        p = out_plots / img
        if p.exists():
            lines.append(f"![{img}](plots/{img})")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] Two-class analysis saved to {report}")

def analyze_standard(exp_dir: Path, files: Dict[str, Path], out_dir: Path) -> None:
    preds_path = files["preds_standard"]
    if not preds_path.exists():
        print(f"[warn] {preds_path.name} is missing. Nothing to analyze in standard mode.", file=sys.stderr)
        return
    df = pd.read_csv(preds_path)
    # Expected columns: symbol, timestamp, y_true, y_pred, optional prob_class_1
    y_true = df["y_true"].values if "y_true" in df.columns else None
    y_pred = df["y_pred"].values if "y_pred" in df.columns else None
    prob_pos = df["prob_class_1"].values if "prob_class_1" in df.columns else None

    out_plots = out_dir / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    cls = {}
    rocpr = {}
    if y_true is not None and y_pred is not None:
        cls = classification_metrics(y_true, y_pred)
        plot_confusion(y_true, y_pred, out_plots / "confusion.png", title="Confusion (standard)")
    if y_true is not None and prob_pos is not None:
        rocpr = roc_pr_metrics(y_true, prob_pos)
        plot_roc(y_true, prob_pos, out_plots / "roc_curve.png", title="ROC (standard)")
        plot_pr(y_true, prob_pos, out_plots / "pr_curve.png", title="PR (standard)")

    report = out_dir / "report_standard.md"
    lines = [f"# Standard classification analysis\n", f"**Experiment:** `{exp_dir}`\n"]
    if cls:
        lines.append("## Classification\n")
        lines.append(f"- Accuracy: **{cls['accuracy']:.4f}**")
        lines.append(f"- F1 macro: **{cls['f1_macro']:.4f}**")
        lines.append(f"- Precision macro: {cls['precision_macro']:.4f}")
        lines.append(f"- Recall macro: {cls['recall_macro']:.4f}\n")
    if rocpr:
        lines.append("## Probabilistic quality\n")
        lines.append(f"- ROC-AUC: {rocpr.get('roc_auc', float('nan')):.4f}")
        lines.append(f"- PR-AUC: {rocpr.get('pr_auc', float('nan')):.4f}\n")
    lines.append("## Plots\n")
    for img in ["confusion.png","roc_curve.png","pr_curve.png"]:
        p = out_plots / img
        if p.exists():
            lines.append(f"![{img}](plots/{img})")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] Standard analysis saved to {report}")

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Analyze trained centralized model artifacts.")
    p.add_argument("--exp_dir", required=True, type=Path, help="Path to experiment directory")
    p.add_argument("--out", type=Path, default=None, help="Output directory for analysis (default: <exp_dir>/analysis)")
    args = p.parse_args()

    files = load_experiment(args.exp_dir)
    out_dir = args.out or (args.exp_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer two-class if present
    if files["preds_two_class"].exists():
        analyze_two_class(args.exp_dir, files, out_dir)
    else:
        analyze_standard(args.exp_dir, files, out_dir)

    print(f"[done] Artifacts in: {out_dir}")

if __name__ == "__main__":
    main()
