#!/usr/bin/env python3
"""
Universal Multi-Model Comparator - Compare any number of models
Allows testing different tau thresholds on multiple models without retraining.

Usage:
  python compare_multiple_models.py \
    --model centralized=experiments/centralized/.../test_predictions_two_class.csv \
    --model federated=experiments/federated/.../predictions_on_centralized_test.csv \
    --model privacy=experiments/privacy/.../predictions_on_centralized_test.csv \
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
from collections import OrderedDict

sns.set_style("whitegrid")

# Color palette for plots (up to 10 models)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def parse_model_args(model_specs: List[str]) -> OrderedDict:
    """Parse model arguments in format name=path."""
    models = OrderedDict()
    
    for i, spec in enumerate(model_specs):
        if '=' in spec:
            name, path = spec.split('=', 1)
            name = name.strip()
            path = Path(path.strip())
        else:
            # If name not specified, generate automatically
            name = f"Model_{i+1}"
            path = Path(spec.strip())
        
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        
        models[name] = path
    
    return models


def load_predictions(pred_path: Path, model_name: str) -> pd.DataFrame:
    """Load predictions and validate required columns."""
    df = pd.read_csv(pred_path)
    
    # Check for probability columns
    if 'proba_up' in df.columns:
        proba_col = 'proba_up'
    elif 'prob_class_1' in df.columns:
        proba_col = 'prob_class_1'
    else:
        raise ValueError(f"{model_name}: Missing probability column (proba_up or prob_class_1)")
    
    # Standardize column names
    df['proba_up'] = df[proba_col]
    
    # Direction: pred_direction or y_pred
    if 'pred_direction' not in df.columns:
        if 'y_pred' in df.columns:
            df['pred_direction'] = df['y_pred']
        else:
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


def align_all_predictions(models_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align all models on common samples."""
    print("\nAligning predictions on common samples...")
    
    # Standardize timestamps
    for name, df in models_data.items():
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['_key'] = df['timestamp'].astype(str) + '_' + df['symbol'].astype(str)
    
    # Find intersection of all sets
    common_keys = set(models_data[list(models_data.keys())[0]]['_key'])
    for name, df in list(models_data.items())[1:]:
        common_keys &= set(df['_key'])
    
    print(f"Found {len(common_keys):,} common samples")
    
    if len(common_keys) == 0:
        raise ValueError("No common samples between models!")
    
    # Align each model
    aligned_data = OrderedDict()
    for name, df in models_data.items():
        aligned = df[df['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
        aligned_data[name] = aligned.drop('_key', axis=1)
    
    # Verify alignment
    keys_list = [df['timestamp'].astype(str) + '_' + df['symbol'].astype(str) 
                 for df in aligned_data.values()]
    for i in range(1, len(keys_list)):
        assert (keys_list[0] == keys_list[i]).all(), f"Alignment failed for model {i}"
    
    return aligned_data


def compute_metrics_at_tau(df: pd.DataFrame, tau: float, cost_bps: float) -> Dict[str, float]:
    """Compute metrics for a given tau threshold."""
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


def create_comparison_plots(all_results: Dict[str, pd.DataFrame], out_dir: Path):
    """Create comparison visualizations for N models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = list(all_results.keys())
    n_models = len(model_names)
    
    # 1. Main comparison: Profit vs Coverage
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, results) in enumerate(all_results.items()):
        ax.plot(results['coverage']*100, results['avg_profit_bps'], 
                linewidth=2.5, label=name, marker=MARKERS[i % len(MARKERS)], 
                markersize=4, alpha=0.8, color=COLORS[i % len(COLORS)])
    
    ax.set_xlabel('Coverage (%)', fontsize=14)
    ax.set_ylabel('Avg Profit per Trade (bps)', fontsize=14)
    ax.set_title('Profit vs Coverage Trade-off Curve', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'profit_vs_coverage.png', dpi=150)
    plt.close()
    
    # 2. Four-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Profit
    ax1 = axes[0, 0]
    for i, (name, results) in enumerate(all_results.items()):
        ax1.plot(results['coverage']*100, results['avg_profit_bps'], 
                 linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax1.set_xlabel('Coverage (%)', fontsize=11)
    ax1.set_ylabel('Avg Profit (bps)', fontsize=11)
    ax1.set_title('Profit vs Coverage', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Win Rate
    ax2 = axes[0, 1]
    for i, (name, results) in enumerate(all_results.items()):
        ax2.plot(results['coverage']*100, results['win_rate']*100,
                 linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax2.set_xlabel('Coverage (%)', fontsize=11)
    ax2.set_ylabel('Win Rate (%)', fontsize=11)
    ax2.set_title('Win Rate vs Coverage', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Direction Accuracy
    ax3 = axes[1, 0]
    for i, (name, results) in enumerate(all_results.items()):
        ax3.plot(results['coverage']*100, results['direction_accuracy']*100,
                 linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax3.set_xlabel('Coverage (%)', fontsize=11)
    ax3.set_ylabel('Direction Accuracy (%)', fontsize=11)
    ax3.set_title('Accuracy vs Coverage', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Sharpe Ratio
    ax4 = axes[1, 1]
    for i, (name, results) in enumerate(all_results.items()):
        ax4.plot(results['coverage']*100, results['sharpe'],
                 linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax4.set_xlabel('Coverage (%)', fontsize=11)
    ax4.set_ylabel('Sharpe Ratio', fontsize=11)
    ax4.set_title('Risk-Adjusted Returns vs Coverage', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'comprehensive_comparison.png', dpi=150)
    plt.close()
    
    # 3. Expected Value per Sample
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, results) in enumerate(all_results.items()):
        ax.plot(results['coverage']*100, results['ev_per_sample'],
                linewidth=2.5, label=name, marker=MARKERS[i % len(MARKERS)],
                markersize=4, alpha=0.8, color=COLORS[i % len(COLORS)])
    ax.set_xlabel('Coverage (%)', fontsize=14)
    ax.set_ylabel('Expected Value per Sample (bps)', fontsize=14)
    ax.set_title('Expected Value vs Coverage', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'ev_comparison.png', dpi=150)
    plt.close()
    
    # 4. Tau-based view
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Profit vs Tau
    ax = axes[0]
    for i, (name, results) in enumerate(all_results.items()):
        ax.plot(results['tau'], results['avg_profit_bps'], 
                linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax.set_xlabel('Confidence Threshold (tau)', fontsize=13)
    ax.set_ylabel('Avg Profit (bps)', fontsize=13)
    ax.set_title('Profit vs Tau', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Coverage vs Tau
    ax = axes[1]
    for i, (name, results) in enumerate(all_results.items()):
        ax.plot(results['tau'], results['coverage']*100, 
                linewidth=2, label=name, color=COLORS[i % len(COLORS)])
    ax.set_xlabel('Confidence Threshold (tau)', fontsize=13)
    ax.set_ylabel('Coverage (%)', fontsize=13)
    ax.set_title('Coverage vs Tau', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'tau_based_comparison.png', dpi=150)
    plt.close()
    
    # 5. Heatmap comparison (only if few models)
    if n_models <= 5:
        key_taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        metrics_to_compare = ['avg_profit_bps', 'coverage', 'direction_accuracy', 'win_rate']
        metric_labels = ['Profit (bps)', 'Coverage', 'Accuracy', 'Win Rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data for heatmap
            data = []
            for name, results in all_results.items():
                row = []
                for tau in key_taus:
                    tau_idx = np.argmin(np.abs(results['tau'].values - tau))
                    val = results.iloc[tau_idx][metric]
                    if metric in ['coverage', 'direction_accuracy', 'win_rate']:
                        val *= 100  # to percentage
                    row.append(val)
                data.append(row)
            
            data_array = np.array(data)
            
            # Heatmap
            im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto')
            
            ax.set_xticks(range(len(key_taus)))
            ax.set_xticklabels([f'{t:.2f}' for t in key_taus])
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(model_names)
            ax.set_xlabel('Tau', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            
            # Annotations
            for i in range(n_models):
                for j in range(len(key_taus)):
                    text = ax.text(j, i, f'{data_array[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(out_dir / 'metrics_heatmap.png', dpi=150)
        plt.close()


def find_optimal_points(results_df: pd.DataFrame) -> Dict[str, Dict]:
    """Find optimal operating points for a model."""
    return {
        'max_profit': results_df.loc[results_df['avg_profit_bps'].idxmax()].to_dict(),
        'max_ev': results_df.loc[results_df['ev_per_sample'].idxmax()].to_dict(),
        'max_sharpe': results_df.loc[results_df['sharpe'].idxmax()].to_dict(),
    }


def print_summary(all_results: Dict[str, pd.DataFrame]):
    """Print detailed comparison summary."""
    print("\n" + "="*100)
    print("OPTIMAL OPERATING POINTS COMPARISON")
    print("="*100)
    
    # Find optimal points for each model
    all_optimal = OrderedDict()
    for name, results in all_results.items():
        all_optimal[name] = find_optimal_points(results)
    
    strategies = [
        ('max_profit', 'MAXIMUM PROFIT PER TRADE'),
        ('max_ev', 'MAXIMUM EXPECTED VALUE'),
        ('max_sharpe', 'MAXIMUM SHARPE RATIO'),
    ]
    
    for strategy_key, strategy_name in strategies:
        print(f"\n{strategy_name}:")
        print("-" * 100)
        
        # Table header
        header = f"{'Metric':<25}"
        for name in all_optimal.keys():
            header += f"{name:>18}"
        header += f"{'Best':<15}"
        print(header)
        print("-" * 100)
        
        metrics = [
            ('Tau', 'tau', '{:.3f}', False),
            ('Coverage', 'coverage', '{:.1%}', True),
            ('Avg Profit (bps)', 'avg_profit_bps', '{:.2f}', True),
            ('Win Rate', 'win_rate', '{:.1%}', True),
            ('Direction Accuracy', 'direction_accuracy', '{:.1%}', True),
            ('Sharpe', 'sharpe', '{:.3f}', True),
            ('EV per Sample', 'ev_per_sample', '{:.3f}', True),
        ]
        
        for label, key, fmt, compare in metrics:
            row = f"{label:<25}"
            
            values = []
            for name in all_optimal.keys():
                val = all_optimal[name][strategy_key][key]
                values.append(val)
                row += f"{fmt.format(val):>18}"
            
            # Determine best value
            if compare and values:
                best_idx = np.argmax(values)
                best_name = list(all_optimal.keys())[best_idx]
                row += f"{best_name:<15}"
            else:
                row += f"{'-':<15}"
            
            print(row)
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Universal Model Comparator - Compare any number of models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Compare three models
  python compare_multiple_models.py \\
    --model centralized=cent/predictions.csv \\
    --model federated=fed/predictions.csv \\
    --model privacy=privacy/predictions.csv \\
    --cost_bps 1.0

  # Compare with automatic names
  python compare_multiple_models.py \\
    --model cent/predictions.csv \\
    --model fed/predictions.csv \\
    --cost_bps 1.0
        """
    )
    
    parser.add_argument('--model', action='append', required=True,
                       help='Model in format name=path or just path (can specify multiple times)')
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
    
    print("="*100)
    print("UNIVERSAL MODEL COMPARATOR")
    print("="*100)
    
    # Parse models
    print("\nParsing model specifications...")
    models = parse_model_args(args.model)
    
    if len(models) < 2:
        print("ERROR: Need at least 2 models for comparison")
        return 1
    
    print(f"\nFound {len(models)} models:")
    for name, path in models.items():
        print(f"  - {name}: {path}")
    
    # Load predictions
    print("\nLoading predictions...")
    models_data = OrderedDict()
    for name, path in models.items():
        df = load_predictions(path, name)
        models_data[name] = df
        print(f"  {name}: {len(df):,} samples")
    
    # Align on common samples
    aligned_data = align_all_predictions(models_data)
    
    # Sweep tau values
    print(f"\nScanning tau from {args.tau_min} to {args.tau_max} (step={args.tau_step})")
    tau_values = np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step)
    
    all_results = OrderedDict()
    
    for name, df in aligned_data.items():
        results = []
        for tau in tau_values:
            metrics = compute_metrics_at_tau(df, tau, args.cost_bps)
            results.append(metrics)
        
        all_results[name] = pd.DataFrame(results)
    
    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, results in all_results.items():
        safe_name = name.replace(' ', '_').replace('/', '_')
        results.to_csv(args.out_dir / f'{safe_name}_tau_sweep.csv', index=False)
    
    print(f"\nResults saved to {args.out_dir}/")
    
    # Create plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, args.out_dir)
    print(f"Plots saved to {args.out_dir}/")
    
    # Print summary
    print_summary(all_results)
    
    # Save optimal points
    optimal_comparison = OrderedDict()
    for name, results in all_results.items():
        optimal_comparison[name] = find_optimal_points(results)
    
    with open(args.out_dir / 'optimal_points_all_models.json', 'w', encoding='utf-8') as f:
        json.dump(optimal_comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*100}")
    print(f"Analysis complete! Results in: {args.out_dir}/")
    print(f"{'='*100}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())