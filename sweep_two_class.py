# sweep_two_class.py
import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[SWEEP] failed (exit {e.returncode}): {' '.join(str(c) for c in cmd)}", flush=True)
        # продолжаем остальные прогоны

def parse_grid(s: str):
    # формат "start:end:step" -> (float, float, float)
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError("Grid must be like '0.50:0.98:0.01'")
    a, b, c = map(float, parts)
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and c > 0):
        raise ValueError("Grid values must be in [0,1], step>0")
    return a, b, c

def main():
    p = argparse.ArgumentParser(description="Sweeper for centralized_training_two_classes.py")
    p.add_argument("--py", default=sys.executable, help="Python executable")
    p.add_argument("--trainer", default="centralized_training_two_classes.py")
    p.add_argument("--unified", default="data/processed/unified_dataset.parquet")
    p.add_argument("--out", default="experiments/centralized")
    p.add_argument("--model", default="mlp", choices=["mlp","lstm"])
    p.add_argument("--horizons", nargs="+", type=int, default=[100, 200, 400])
    p.add_argument("--deadbands", nargs="+", type=float, default=[2, 5, 10])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_size", type=int, default=1_000_000)
    p.add_argument("--top_k_features", type=int, default=64)
    p.add_argument("--profit_cost_bps", type=float, default=1.0)
    p.add_argument("--grid", default="0.50:0.98:0.01", help="tau grid as start:end:step")
    p.add_argument("--optimize_tau_by", nargs="*", default=["profit","ev"],
                   help="one or more of: profit, ev, profit_with_min_coverage")
    p.add_argument("--min_coverage", type=float, default=0.005,
                   help="min coverage for profit_with_min_coverage (e.g., 0.005 = 0.5%)")
    p.add_argument("--fixed_tau", nargs="*", type=float,
                   help="if set, runs fixed taus (no optimization)")
    p.add_argument("--calibrate", action="store_true", default=True)
    p.add_argument("--no_calibrate", action="store_true")
    p.add_argument("--use_robust_scaler", action="store_true", default=True)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = p.parse_args()

    # калибровка
    calibrate = (not args.no_calibrate) and args.calibrate
    # проверка грида
    _ = parse_grid(args.grid)

    base = [
        args.py, args.trainer,
        "--unified", args.unified,
        "--out", args.out,
        "--task", "classification",
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--max_train_size", str(args.max_train_size),
        "--top_k_features", str(args.top_k_features),
        "--two_class_mode",
        "--profit_cost_bps", str(args.profit_cost_bps),
        "--device", args.device,
    ]
    if args.use_robust_scaler:
        base += ["--use_robust_scaler"]
    if calibrate:
        base += ["--calibrate_probabilities"]

    # Режим 1: фиксированный τ (если задан --fixed_tau)
    if args.fixed_tau:
        for h in args.horizons:
            for db in args.deadbands:
                for tau in args.fixed_tau:
                    cmd = base + [
                        "--horizon_min", str(h),
                        "--deadband_bps", str(db),
                        "--confidence_tau", str(tau),
                    ]
                    run(cmd)
        return

    # Режим 2: автоподбор τ по критериям
    for h in args.horizons:
        for db in args.deadbands:
            for crit in args.optimize_tau_by:
                cmd = base + [
                    "--horizon_min", str(h),
                    "--deadband_bps", str(db),
                    "--optimize_threshold_for_profit",
                    "--confidence_grid", args.grid,
                    "--optimize_tau_by", crit,
                ]
                if crit == "profit_with_min_coverage":
                    cmd += ["--min_coverage", str(args.min_coverage)]
                run(cmd)

if __name__ == "__main__":
    main()
