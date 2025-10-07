#!/usr/bin/env python3
"""
Plot profit/coverage curves for different confidence thresholds.
No retraining needed - just varies tau on existing model predictions.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_predictions(pred_path: Path):
    """Load model predictions with probabilities."""
    df = pd.read_csv(pred_path)
    
    # Ensure we have needed columns
    required = {'y_direction', 'proba_up', 'ret_bps'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Need: {required}")
    
    return df

def compute_metrics_at_tau(df: pd.DataFrame, tau: float, cost_bps: float):
    """Compute all metrics for a given tau threshold."""
    proba_up = df['proba_up'].values
    confidence = np.maximum(proba_up, 1.0 - proba_up)
    
    # Execution mask
    execute = confidence >= tau
    n_total = len(df)
    n_exec = int(execute.sum())
    coverage = n_exec / n_total if n_total > 0 else 0.0
    
    if n_exec == 0:
        return {
            'tau': tau,
            'coverage': 0.0,
            'n_executed': 0,
            'direction_accuracy': 0.0,
            'avg_profit_bps': -cost_bps,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'ev_per_sample': 0.0,
        }
    
    # Executed subset
    y_true = df['y_direction'].values[execute]
    pred_dir = (proba_up[execute] >= 0.5).astype(int)
    ret_bps = df['ret_bps'].values[execute]
    
    # Direction accuracy
    dir_acc = (pred_dir == y_true).mean()
    
    # Profit
    gross = np.where(pred_dir == 1, ret_bps, -ret_bps)
    net = gross - cost_bps
    
    win_rate = (net > 0).mean()
    avg_profit = net.mean()
    sharpe = avg_profit / (net.std() + 1e-8)
    ev_per_sample = avg_profit * coverage  # Expected value per original sample
    
    return {
        'tau': tau,
        'coverage': coverage,
        'n_executed': n_exec,
        'direction_accuracy': dir_acc,
        'avg_profit_bps': avg_profit,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'ev_per_sample': ev_per_sample,
    }

def plot_curves(results_df: pd.DataFrame, out_dir: Path):
    """Create comprehensive visualization."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Profit vs Coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df['coverage'] * 100, results_df['avg_profit_bps'], 
            'b-', linewidth=2, label='Avg Profit')
    ax.set_xlabel('Coverage (%)', fontsize=12)
    ax.set_ylabel('Avg Profit (bps)', fontsize=12)
    ax.set_title('Profit vs Coverage Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'profit_vs_coverage.png', dpi=150)
    plt.close()
    
    # 2. Multiple metrics on one plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Profit & Win Rate
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(results_df['coverage'] * 100, results_df['avg_profit_bps'], 
             'b-', linewidth=2, label='Avg Profit')
    ax1_twin.plot(results_df['coverage'] * 100, results_df['win_rate'] * 100, 
                  'r--', linewidth=2, label='Win Rate')
    ax1.set_xlabel('Coverage (%)')
    ax1.set_ylabel('Avg Profit (bps)', color='b')
    ax1_twin.set_ylabel('Win Rate (%)', color='r')
    ax1.set_title('Profit & Win Rate vs Coverage')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Top-right: Direction Accuracy
    ax2 = axes[0, 1]
    ax2.plot(results_df['coverage'] * 100, results_df['direction_accuracy'] * 100,
             'g-', linewidth=2)
    ax2.set_xlabel('Coverage (%)')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title('Accuracy vs Coverage')
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Sharpe Ratio
    ax3 = axes[1, 0]
    ax3.plot(results_df['coverage'] * 100, results_df['sharpe'],
             'purple', linewidth=2)
    ax3.set_xlabel('Coverage (%)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns vs Coverage')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Bottom-right: EV per sample
    ax4 = axes[1, 1]
    ax4.plot(results_df['coverage'] * 100, results_df['ev_per_sample'],
             'orange', linewidth=2)
    ax4.set_xlabel('Coverage (%)')
    ax4.set_ylabel('EV per Sample (bps)')
    ax4.set_title('Expected Value vs Coverage')
    ax4.grid(True, alpha=0.3)
    
    # Find optimal points
    best_profit_idx = results_df['avg_profit_bps'].idxmax()
    best_ev_idx = results_df['ev_per_sample'].idxmax()
    
    # Mark optimal points
    for ax, metric, idx in [
        (ax1, 'avg_profit_bps', best_profit_idx),
        (ax4, 'ev_per_sample', best_ev_idx)
    ]:
        cov = results_df.loc[idx, 'coverage'] * 100
        val = results_df.loc[idx, metric]
        ax.scatter([cov], [val], color='red', s=100, zorder=5, 
                  label=f'Optimal (tau={results_df.loc[idx, "tau"]:.3f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comprehensive_metrics.png', dpi=150)
    plt.close()
    
    # 3. Tau vs Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['tau'], results_df['avg_profit_bps'], 
            'b-', linewidth=2, label='Avg Profit (bps)')
    ax.plot(results_df['tau'], results_df['coverage'] * 100, 
            'g--', linewidth=2, label='Coverage (%) × 1')
    ax.set_xlabel('Confidence Threshold (tau)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Metrics vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'tau_vs_metrics.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze profit/coverage tradeoffs without retraining'
    )
    parser.add_argument('--predictions', required=True, type=Path,
                       help='Path to test_predictions_two_class.csv')
    parser.add_argument('--cost_bps', type=float, default=1.0,
                       help='Trading cost in bps')
    parser.add_argument('--tau_min', type=float, default=0.50,
                       help='Minimum tau to test')
    parser.add_argument('--tau_max', type=float, default=0.95,
                       help='Maximum tau to test')
    parser.add_argument('--tau_step', type=float, default=0.01,
                       help='Step size for tau')
    parser.add_argument('--out_dir', type=Path, default=None,
                       help='Output directory (default: predictions_dir/tau_analysis)')
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}")
    df = load_predictions(args.predictions)
    print(f"Loaded {len(df)} predictions")
    
    # Output directory
    if args.out_dir is None:
        args.out_dir = args.predictions.parent / 'tau_analysis'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sweep tau values
    print(f"\nScanning tau from {args.tau_min} to {args.tau_max} (step={args.tau_step})")
    tau_values = np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step)
    
    results = []
    for tau in tau_values:
        metrics = compute_metrics_at_tau(df, tau, args.cost_bps)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = args.out_dir / 'tau_sweep_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Find optimal points
    best_profit = results_df.loc[results_df['avg_profit_bps'].idxmax()]
    best_ev = results_df.loc[results_df['ev_per_sample'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    
    # Print summary
    print("\n" + "="*70)
    print("OPTIMAL OPERATING POINTS")
    print("="*70)
    
    print("\n1. MAX PROFIT PER TRADE:")
    print(f"   Tau: {best_profit['tau']:.3f}")
    print(f"   Coverage: {best_profit['coverage']:.1%}")
    print(f"   Avg Profit: {best_profit['avg_profit_bps']:.2f} bps")
    print(f"   Win Rate: {best_profit['win_rate']:.1%}")
    print(f"   Direction Acc: {best_profit['direction_accuracy']:.1%}")
    
    print("\n2. MAX EXPECTED VALUE:")
    print(f"   Tau: {best_ev['tau']:.3f}")
    print(f"   Coverage: {best_ev['coverage']:.1%}")
    print(f"   Avg Profit: {best_ev['avg_profit_bps']:.2f} bps")
    print(f"   EV per Sample: {best_ev['ev_per_sample']:.2f} bps")
    print(f"   Win Rate: {best_ev['win_rate']:.1%}")
    
    print("\n3. MAX SHARPE RATIO:")
    print(f"   Tau: {best_sharpe['tau']:.3f}")
    print(f"   Coverage: {best_sharpe['coverage']:.1%}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.3f}")
    print(f"   Avg Profit: {best_sharpe['avg_profit_bps']:.2f} bps")
    
    print("\n" + "="*70)
    
    # Create plots
    print("\nGenerating plots...")
    plot_curves(results_df, args.out_dir)
    print(f"Plots saved to {args.out_dir}/")
    
    # Save optimal points
    optimal_points = {
        'max_profit': best_profit.to_dict(),
        'max_ev': best_ev.to_dict(),
        'max_sharpe': best_sharpe.to_dict(),
    }
    
    with open(args.out_dir / 'optimal_points.json', 'w') as f:
        json.dump(optimal_points, f, indent=2)
    
    print(f"\nOptimal points saved to {args.out_dir}/optimal_points.json")
    print("\nDone!")

if __name__ == '__main__':
    main()