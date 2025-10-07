#!/usr/bin/env python3
"""
Universal Model Comparator - Compare Centralized vs Federated models
Allows testing different tau thresholds on both models without retraining.

Usage:
  python compare_models.py \
    --centralized_pred experiments/centralized/.../test_predictions_two_class.csv \
    --federated_pred experiments/federated/.../predictions_on_centralized_test.csv \
    --cost_bps 1.0 \
    --tau_min 0.50 --tau_max 0.90 --tau_step 0.005 \
    --out_dir model_comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_predictions(pred_path: Path, model_name: str) -> pd.DataFrame:
    """Load predictions and validate required columns."""
    df = pd.read_csv(pred_path)
    
    # Check for required columns
    if 'proba_up' in df.columns:
        proba_col = 'proba_up'
    elif 'prob_class_1' in df.columns:
        proba_col = 'prob_class_1'
    else:
        raise ValueError(f"{model_name}: Missing probability column (proba_up or prob_class_1)")
    
    # Standardize column names
    df['proba_up'] = df[proba_col]
    
    # Direction: either pred_direction or y_pred
    if 'pred_direction' not in df.columns:
        if 'y_pred' in df.columns:
            df['pred_direction'] = df['y_pred']
        else:
            # Derive from proba
            df['pred_direction'] = (df['proba_up'] >= 0.5).astype(int)
    
    # True labels
    if 'y_direction' not in df.columns:
        if 'y_true' in df.columns:
            df['y_direction'] = df['y_true']
        else:
            raise ValueError(f"{model_name}: Missing y_direction or y_true column")
    
    required = {'timestamp', 'symbol', 'y_direction', 'ret_bps', 'proba_up'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{model_name}: Missing columns: {missing}")
    
    return df


def align_predictions(cent_df: pd.DataFrame, fed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two prediction dataframes on same samples."""
    # Ensure timestamps are comparable
    cent_df['timestamp'] = pd.to_datetime(cent_df['timestamp'], utc=True)
    fed_df['timestamp'] = pd.to_datetime(fed_df['timestamp'], utc=True)
    
    # Create merge key
    cent_df['_key'] = cent_df['timestamp'].astype(str) + '_' + cent_df['symbol'].astype(str)
    fed_df['_key'] = fed_df['timestamp'].astype(str) + '_' + fed_df['symbol'].astype(str)
    
    # Find common samples
    common_keys = set(cent_df['_key']) & set(fed_df['_key'])
    
    cent_aligned = cent_df[cent_df['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    fed_aligned = fed_df[fed_df['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    
    # Verify alignment
    assert (cent_aligned['_key'] == fed_aligned['_key']).all(), "Alignment failed"
    
    print(f"Aligned on {len(common_keys):,} common samples")
    
    return cent_aligned.drop('_key', axis=1), fed_aligned.drop('_key', axis=1)


def compute_metrics_at_tau(df: pd.DataFrame, tau: float, cost_bps: float) -> Dict[str, float]:
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
            'median_profit_bps': -cost_bps,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'ev_per_sample': 0.0,
            'total_profit_bps': 0.0,
        }
    
    # Executed subset
    y_true = df['y_direction'].values[execute]
    pred_dir = (proba_up[execute] >= 0.5).astype(int)
    ret_bps = df['ret_bps'].values[execute]
    
    # Direction accuracy
    dir_acc = float((pred_dir == y_true).mean())
    
    # Profit
    gross = np.where(pred_dir == 1, ret_bps, -ret_bps)
    net = gross - cost_bps
    
    win_rate = float((net > 0).mean())
    avg_profit = float(net.mean())
    median_profit = float(np.median(net))
    sharpe = float(avg_profit / (net.std() + 1e-8))
    ev_per_sample = avg_profit * coverage
    total_profit = float(net.sum())
    
    return {
        'tau': tau,
        'coverage': coverage,
        'n_executed': n_exec,
        'direction_accuracy': dir_acc,
        'avg_profit_bps': avg_profit,
        'median_profit_bps': median_profit,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'ev_per_sample': ev_per_sample,
        'total_profit_bps': total_profit,
    }


def create_comparison_plots(cent_results: pd.DataFrame, fed_results: pd.DataFrame, out_dir: Path):
    """Create comprehensive comparison visualizations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Main comparison: Profit vs Coverage
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cent_results['coverage']*100, cent_results['avg_profit_bps'], 
            'b-', linewidth=2.5, label='Centralized', marker='o', markersize=3, alpha=0.7)
    ax.plot(fed_results['coverage']*100, fed_results['avg_profit_bps'], 
            'r-', linewidth=2.5, label='Federated', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Coverage (%)', fontsize=13)
    ax.set_ylabel('Avg Profit per Trade (bps)', fontsize=13)
    ax.set_title('Profit vs Coverage Trade-off Curve', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'profit_vs_coverage.png', dpi=150)
    plt.close()
    
    # 2. Four-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 1: Profit & Win Rate vs Coverage
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(cent_results['coverage']*100, cent_results['avg_profit_bps'], 
             'b-', linewidth=2, label='Cent Profit')
    ax1.plot(fed_results['coverage']*100, fed_results['avg_profit_bps'], 
             'r-', linewidth=2, label='Fed Profit')
    ax1_twin.plot(cent_results['coverage']*100, cent_results['win_rate']*100, 
                  'b--', linewidth=2, alpha=0.6, label='Cent WR')
    ax1_twin.plot(fed_results['coverage']*100, fed_results['win_rate']*100, 
                  'r--', linewidth=2, alpha=0.6, label='Fed WR')
    ax1.set_xlabel('Coverage (%)')
    ax1.set_ylabel('Avg Profit (bps)', color='black')
    ax1_twin.set_ylabel('Win Rate (%)', color='gray')
    ax1.set_title('Profit & Win Rate vs Coverage')
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Direction Accuracy vs Coverage
    ax2 = axes[0, 1]
    ax2.plot(cent_results['coverage']*100, cent_results['direction_accuracy']*100,
             'b-', linewidth=2, label='Centralized')
    ax2.plot(fed_results['coverage']*100, fed_results['direction_accuracy']*100,
             'r-', linewidth=2, label='Federated')
    ax2.set_xlabel('Coverage (%)')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title('Accuracy vs Coverage')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Sharpe Ratio vs Coverage
    ax3 = axes[1, 0]
    ax3.plot(cent_results['coverage']*100, cent_results['sharpe'],
             'b-', linewidth=2, label='Centralized')
    ax3.plot(fed_results['coverage']*100, fed_results['sharpe'],
             'r-', linewidth=2, label='Federated')
    ax3.set_xlabel('Coverage (%)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns vs Coverage')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Panel 4: EV per Sample vs Coverage
    ax4 = axes[1, 1]
    ax4.plot(cent_results['coverage']*100, cent_results['ev_per_sample'],
             'b-', linewidth=2, label='Centralized')
    ax4.plot(fed_results['coverage']*100, fed_results['ev_per_sample'],
             'r-', linewidth=2, label='Federated')
    ax4.set_xlabel('Coverage (%)')
    ax4.set_ylabel('Expected Value per Sample (bps)')
    ax4.set_title('EV vs Coverage')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comprehensive_comparison.png', dpi=150)
    plt.close()
    
    # 3. Tau-based view
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Profit vs Tau
    ax = axes[0]
    ax.plot(cent_results['tau'], cent_results['avg_profit_bps'], 
            'b-', linewidth=2, label='Centralized')
    ax.plot(fed_results['tau'], fed_results['avg_profit_bps'], 
            'r-', linewidth=2, label='Federated')
    ax.set_xlabel('Confidence Threshold (tau)', fontsize=12)
    ax.set_ylabel('Avg Profit (bps)', fontsize=12)
    ax.set_title('Profit vs Tau', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Coverage vs Tau
    ax = axes[1]
    ax.plot(cent_results['tau'], cent_results['coverage']*100, 
            'b-', linewidth=2, label='Centralized')
    ax.plot(fed_results['tau'], fed_results['coverage']*100, 
            'r-', linewidth=2, label='Federated')
    ax.set_xlabel('Confidence Threshold (tau)', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage vs Tau', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'tau_based_comparison.png', dpi=150)
    plt.close()
    
    # 4. Difference heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate differences (Federated - Centralized)
    diff_profit = fed_results['avg_profit_bps'].values - cent_results['avg_profit_bps'].values
    diff_coverage = fed_results['coverage'].values - cent_results['coverage'].values
    diff_accuracy = fed_results['direction_accuracy'].values - cent_results['direction_accuracy'].values
    
    # Create difference summary at key tau values
    key_taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    diff_data = []
    for tau in key_taus:
        idx = np.argmin(np.abs(cent_results['tau'].values - tau))
        diff_data.append([
            diff_profit[idx],
            diff_coverage[idx] * 100,  # as percentage
            diff_accuracy[idx] * 100,  # as percentage
        ])
    
    diff_array = np.array(diff_data).T
    im = ax.imshow(diff_array, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    ax.set_xticks(range(len(key_taus)))
    ax.set_xticklabels([f'{t:.2f}' for t in key_taus])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Profit Diff (bps)', 'Coverage Diff (%)', 'Accuracy Diff (%)'])
    ax.set_xlabel('Tau', fontsize=12)
    ax.set_title('Federated - Centralized Differences', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(len(key_taus)):
            text = ax.text(j, i, f'{diff_array[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / 'difference_heatmap.png', dpi=150)
    plt.close()


def find_optimal_points(results_df: pd.DataFrame, model_name: str) -> Dict[str, Dict]:
    """Find optimal operating points for a model."""
    return {
        'max_profit': results_df.loc[results_df['avg_profit_bps'].idxmax()].to_dict(),
        'max_ev': results_df.loc[results_df['ev_per_sample'].idxmax()].to_dict(),
        'max_sharpe': results_df.loc[results_df['sharpe'].idxmax()].to_dict(),
    }


def print_summary(cent_results: pd.DataFrame, fed_results: pd.DataFrame):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("OPTIMAL OPERATING POINTS COMPARISON")
    print("="*80)
    
    cent_optimal = find_optimal_points(cent_results, 'Centralized')
    fed_optimal = find_optimal_points(fed_results, 'Federated')
    
    for strategy in ['max_profit', 'max_ev', 'max_sharpe']:
        strategy_name = strategy.replace('_', ' ').upper()
        print(f"\n{strategy_name}:")
        print("-" * 80)
        
        cent = cent_optimal[strategy]
        fed = fed_optimal[strategy]
        
        print(f"{'Metric':<25} {'Centralized':>15} {'Federated':>15} {'Winner':>15}")
        print("-" * 80)
        
        metrics = [
            ('Tau', 'tau', '{:.3f}'),
            ('Coverage', 'coverage', '{:.1%}'),
            ('Avg Profit (bps)', 'avg_profit_bps', '{:.2f}'),
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Direction Accuracy', 'direction_accuracy', '{:.1%}'),
            ('Sharpe', 'sharpe', '{:.3f}'),
        ]
        
        for label, key, fmt in metrics:
            cent_val = cent[key]
            fed_val = fed[key]
            
            if key == 'tau':
                winner = '-'
            else:
                winner = 'Federated' if fed_val > cent_val else 'Centralized' if cent_val > fed_val else 'Tie'
            
            print(f"{label:<25} {fmt.format(cent_val):>15} {fmt.format(fed_val):>15} {winner:>15}")


def main():
    parser = argparse.ArgumentParser(
        description='Universal Model Comparator - Compare Centralized vs Federated'
    )
    parser.add_argument('--centralized_pred', required=True, type=Path,
                       help='Path to centralized test_predictions_two_class.csv')
    parser.add_argument('--federated_pred', required=True, type=Path,
                       help='Path to federated predictions CSV (from analyze_federated_model.py)')
    parser.add_argument('--cost_bps', type=float, default=1.0,
                       help='Trading cost in bps')
    parser.add_argument('--tau_min', type=float, default=0.50,
                       help='Minimum tau to test')
    parser.add_argument('--tau_max', type=float, default=0.90,
                       help='Maximum tau to test')
    parser.add_argument('--tau_step', type=float, default=0.005,
                       help='Step size for tau sweep')
    parser.add_argument('--out_dir', type=Path, default=None,
                       help='Output directory (default: model_comparison)')
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = Path('model_comparison')
    
    print("="*80)
    print("UNIVERSAL MODEL COMPARATOR")
    print("="*80)
    
    # Load predictions
    print("\nLoading predictions...")
    cent_df = load_predictions(args.centralized_pred, "Centralized")
    fed_df = load_predictions(args.federated_pred, "Federated")
    
    print(f"Centralized: {len(cent_df):,} samples")
    print(f"Federated: {len(fed_df):,} samples")
    
    # Align on common samples
    print("\nAligning predictions on common samples...")
    cent_df, fed_df = align_predictions(cent_df, fed_df)
    
    # Sweep tau values
    print(f"\nScanning tau from {args.tau_min} to {args.tau_max} (step={args.tau_step})")
    tau_values = np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step)
    
    cent_results = []
    fed_results = []
    
    for tau in tau_values:
        cent_metrics = compute_metrics_at_tau(cent_df, tau, args.cost_bps)
        fed_metrics = compute_metrics_at_tau(fed_df, tau, args.cost_bps)
        
        cent_results.append(cent_metrics)
        fed_results.append(fed_metrics)
    
    cent_results_df = pd.DataFrame(cent_results)
    fed_results_df = pd.DataFrame(fed_results)
    
    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cent_results_df.to_csv(args.out_dir / 'centralized_tau_sweep.csv', index=False)
    fed_results_df.to_csv(args.out_dir / 'federated_tau_sweep.csv', index=False)
    print(f"\nResults saved to {args.out_dir}/")
    
    # Create plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(cent_results_df, fed_results_df, args.out_dir)
    print(f"Plots saved to {args.out_dir}/")
    
    # Print summary
    print_summary(cent_results_df, fed_results_df)
    
    # Save optimal points
    optimal_comparison = {
        'centralized': find_optimal_points(cent_results_df, 'Centralized'),
        'federated': find_optimal_points(fed_results_df, 'Federated'),
    }
    
    with open(args.out_dir / 'optimal_points_comparison.json', 'w') as f:
        json.dump(optimal_comparison, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results in: {args.out_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()