#!/usr/bin/env python3
"""
Two-Class Sweep Results Aggregator
=================================

Aggregates and analyzes results from sweep_two_class.py experiments.
Creates comprehensive tables and visualizations for performance analysis.

Usage:
    python sweep_results_aggregator.py --base_dir experiments/centralized
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SweepResultsAggregator:
    """Aggregates and analyzes sweep experiment results."""
    
    def __init__(self, base_dir: Path, output_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        # Create aggregated results as sibling directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_dir.parent / f"{self.base_dir.name}_aggregated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results_df = None
        
        # Configure matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"Aggregator initialized: {self.base_dir} -> {self.output_dir}")
    
    def find_experiment_dirs(self) -> List[Path]:
        """Find all experiment directories."""
        patterns = [
            "*_mlp_*_2class*",
            "*_mlp_*_cal", 
            "20*_mlp_*",
        ]
        
        experiment_dirs = []
        for pattern in patterns:
            for path in self.base_dir.glob(pattern):
                if path.is_dir():
                    required_files = ['metrics_two_class.json']
                    if any((path / f).exists() for f in required_files + ['config.yaml']):
                        experiment_dirs.append(path)
        
        # Remove duplicates and sort
        experiment_dirs = sorted(list(set(experiment_dirs)))
        logger.info(f"Found {len(experiment_dirs)} experiment directories")
        
        if not experiment_dirs:
            logger.warning(f"No experiment directories found in {self.base_dir}")
            # Show what directories exist
            all_dirs = [d for d in self.base_dir.glob("*") if d.is_dir()]
            logger.info(f"Available directories: {[d.name for d in all_dirs[:10]]}")
        
        return experiment_dirs
    
    def parse_experiment_name(self, exp_dir: Path) -> Dict[str, Any]:
        """Parse experiment directory name to extract parameters."""
        name = exp_dir.name
        parts = name.split('_')
        
        params = {'exp_dir': str(exp_dir), 'exp_name': name}
        
        try:
            # Parse timestamp
            if len(parts) >= 1 and '-' in parts[0]:
                timestamp_part = parts[0]
                if len(parts) >= 2:
                    timestamp_part += '_' + parts[1]
                try:
                    params['timestamp'] = pd.to_datetime(timestamp_part, format='%Y-%m-%d_%H%M%S')
                except:
                    params['timestamp'] = None
            
            # Parse model type
            params['model_type'] = 'mlp' if 'mlp' in name.lower() else 'lstm' if 'lstm' in name.lower() else 'unknown'
            
            # Parse horizon (h[number])
            for part in parts:
                if part.startswith('h') and len(part) > 1:
                    try:
                        params['horizon_min'] = int(float(part[1:]))
                        break
                    except ValueError:
                        pass
            
            # Parse deadband (db[number])
            for part in parts:
                if part.startswith('db') and len(part) > 2:
                    try:
                        params['deadband_bps'] = float(part[2:])
                    except ValueError:
                        pass
            
            # Parse flags
            params['two_class_mode'] = '2class' in name.lower()
            params['calibrated'] = '_cal' in name.lower()
            
        except Exception as e:
            logger.warning(f"Error parsing {name}: {e}")
        
        return params
    
    def load_experiment_results(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """Load results from experiment directory."""
        try:
            params = self.parse_experiment_name(exp_dir)
            
            # Load config
            config_path = exp_dir / 'config.yaml'
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Update with config values
                    for key in ['horizon_min', 'deadband_bps', 'confidence_tau', 'profit_cost_bps']:
                        if key in config:
                            params[key] = config[key]
                            
                except ImportError:
                    logger.warning(f"PyYAML not available for {exp_dir}")
                except Exception as e:
                    logger.warning(f"Config error for {exp_dir}: {e}")
            
            # Load metrics
            metrics_path = exp_dir / 'metrics_two_class.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                params.update(metrics)
            else:
                # Try fallback
                fallback_path = exp_dir / 'metrics_global.json'
                if fallback_path.exists():
                    with open(fallback_path, 'r') as f:
                        metrics = json.load(f)
                    params.update(metrics)
                else:
                    logger.warning(f"No metrics found in {exp_dir}")
                    return None
            
            # Load decision threshold
            threshold_path = exp_dir / 'decision_threshold.json'
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    threshold = json.load(f)
                params.update({
                    'decision_tau': threshold.get('confidence_tau'),
                    'val_profit_bps': threshold.get('val_profit_bps'),
                    'val_coverage': threshold.get('val_coverage'),
                })
            
            # Set defaults
            defaults = {
                'horizon_min': 0, 'deadband_bps': 0.0, 'model_type': 'unknown',
                'coverage': 0.0, 'direction_accuracy': 0.0, 'avg_profit_bps': 0.0,
                'win_rate': 0.0, 'n_executed': 0, 'n_total': 0, 'confidence_tau': 0.8
            }
            
            for key, default in defaults.items():
                if key not in params:
                    params[key] = default
            
            return params
            
        except Exception as e:
            logger.error(f"Error loading {exp_dir}: {e}")
            return None
    
    def aggregate_results(self) -> pd.DataFrame:
        """Aggregate all results."""
        logger.info("Aggregating experiment results...")
        
        experiment_dirs = self.find_experiment_dirs()
        if not experiment_dirs:
            raise ValueError(f"No experiments found in {self.base_dir}")
        
        results = []
        for exp_dir in experiment_dirs:
            result = self.load_experiment_results(exp_dir)
            if result:
                results.append(result)
        
        if not results:
            raise ValueError("No valid results found")
        
        df = pd.DataFrame(results)
        df = self._standardize_dataframe(df)
        
        self.results_df = df
        logger.info(f"Aggregated {len(df)} experiments")
        return df
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame."""
        # Add percentage columns
        if 'coverage' in df.columns:
            df['coverage_pct'] = (df['coverage'] * 100).round(2)
        if 'win_rate' in df.columns:
            df['win_rate_pct'] = (df['win_rate'] * 100).round(2)
        
        # Round profit columns
        profit_cols = [col for col in df.columns if 'profit' in col and 'bps' in col]
        for col in profit_cols:
            df[col] = df[col].round(2)
        
        # Handle infinite Sharpe ratios
        if 'profit_sharpe' in df.columns:
            df['profit_sharpe'] = df['profit_sharpe'].replace([np.inf, -np.inf], np.nan)
        
        # Sort by parameters
        sort_cols = [col for col in ['horizon_min', 'deadband_bps'] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        return df
    
    def create_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """Create summary tables."""
        if self.results_df is None:
            raise ValueError("No results loaded")
        
        df = self.results_df
        summaries = {}
        
        # Overall summary
        key_cols = ['horizon_min', 'deadband_bps', 'coverage_pct', 'direction_accuracy', 
                   'avg_profit_bps', 'win_rate_pct', 'n_executed']
        available_cols = [col for col in key_cols if col in df.columns]
        summaries['overall'] = df[available_cols].round(3)
        
        # Best performers
        if len(df) > 0:
            best_profit_idx = df['avg_profit_bps'].idxmax()
            best_winrate_idx = df['win_rate'].idxmax()
            best_coverage_idx = df['coverage'].idxmax()
            
            best_performers = pd.concat([
                df.loc[[best_profit_idx], available_cols].assign(criterion='Best Profit'),
                df.loc[[best_winrate_idx], available_cols].assign(criterion='Best Win Rate'),
                df.loc[[best_coverage_idx], available_cols].assign(criterion='Highest Coverage')
            ])
            summaries['best_performers'] = best_performers
        
        # By horizon
        if 'horizon_min' in df.columns and len(df['horizon_min'].unique()) > 1:
            horizon_summary = df.groupby('horizon_min')[
                ['avg_profit_bps', 'coverage_pct', 'win_rate_pct', 'direction_accuracy']
            ].agg(['mean', 'std', 'count']).round(3)
            horizon_summary.columns = ['_'.join(col) for col in horizon_summary.columns]
            summaries['by_horizon'] = horizon_summary
        
        # By deadband  
        if 'deadband_bps' in df.columns and len(df['deadband_bps'].unique()) > 1:
            deadband_summary = df.groupby('deadband_bps')[
                ['avg_profit_bps', 'coverage_pct', 'win_rate_pct', 'direction_accuracy']
            ].agg(['mean', 'std', 'count']).round(3)
            deadband_summary.columns = ['_'.join(col) for col in deadband_summary.columns]
            summaries['by_deadband'] = deadband_summary
        
        # Correlations
        numeric_cols = [col for col in ['horizon_min', 'deadband_bps', 'coverage', 
                       'direction_accuracy', 'avg_profit_bps', 'win_rate'] if col in df.columns]
        if len(numeric_cols) > 2:
            summaries['correlations'] = df[numeric_cols].corr().round(3)
        
        return summaries
    
    def create_visualizations(self) -> None:
        """Create all visualizations."""
        if self.results_df is None:
            raise ValueError("No results loaded")
        
        df = self.results_df
        logger.info("Creating visualizations...")
        
        # 1. Performance heatmaps
        self._create_heatmaps(df)
        
        # 2. Parameter sweep plots
        self._create_sweep_plots(df)
        
        # 3. Scatter plots
        self._create_scatter_plots(df)
        
        # 4. Distribution plots
        self._create_distributions(df)
    
    def _create_heatmaps(self, df: pd.DataFrame) -> None:
        """Create performance heatmaps."""
        metrics = ['avg_profit_bps', 'win_rate_pct', 'coverage_pct', 'direction_accuracy']
        available = [m for m in metrics if m in df.columns]
        
        if not available:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(available[:4]):
            pivot = df.pivot_table(
                index='deadband_bps', columns='horizon_min', 
                values=metric, aggfunc='mean'
            )
            
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.2f', ax=axes[i],
                           cmap='RdYlBu_r' if 'profit' in metric else 'RdYlBu')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Horizon (min)')
                axes[i].set_ylabel('Deadband (bps)')
        
        for i in range(len(available), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sweep_plots(self, df: pd.DataFrame) -> None:
        """Create sweep analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Profit by horizon
        if 'horizon_min' in df.columns:
            horizon_stats = df.groupby('horizon_min')['avg_profit_bps'].agg(['mean', 'std'])
            axes[0,0].errorbar(horizon_stats.index, horizon_stats['mean'], 
                              yerr=horizon_stats['std'], marker='o', capsize=5)
            axes[0,0].set_xlabel('Horizon (min)')
            axes[0,0].set_ylabel('Avg Profit (bps)')
            axes[0,0].set_title('Profit by Horizon')
            axes[0,0].grid(True, alpha=0.3)
        
        # Profit by deadband
        if 'deadband_bps' in df.columns:
            db_stats = df.groupby('deadband_bps')['avg_profit_bps'].agg(['mean', 'std'])
            axes[0,1].errorbar(db_stats.index, db_stats['mean'], 
                              yerr=db_stats['std'], marker='s', capsize=5)
            axes[0,1].set_xlabel('Deadband (bps)')
            axes[0,1].set_ylabel('Avg Profit (bps)')
            axes[0,1].set_title('Profit by Deadband')
            axes[0,1].grid(True, alpha=0.3)
        
        # Coverage by horizon
        if 'coverage_pct' in df.columns:
            axes[1,0].scatter(df['horizon_min'], df['coverage_pct'], alpha=0.6)
            axes[1,0].set_xlabel('Horizon (min)')
            axes[1,0].set_ylabel('Coverage (%)')
            axes[1,0].set_title('Coverage vs Horizon')
            axes[1,0].grid(True, alpha=0.3)
        
        # Win rate by deadband
        if 'win_rate_pct' in df.columns:
            axes[1,1].scatter(df['deadband_bps'], df['win_rate_pct'], alpha=0.6, c='red')
            axes[1,1].set_xlabel('Deadband (bps)')
            axes[1,1].set_ylabel('Win Rate (%)')
            axes[1,1].set_title('Win Rate vs Deadband')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sweep_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scatter_plots(self, df: pd.DataFrame) -> None:
        """Create scatter plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Profit vs Coverage
        if 'coverage_pct' in df.columns:
            scatter = axes[0,0].scatter(df['coverage_pct'], df['avg_profit_bps'], 
                                       c=df['horizon_min'], cmap='viridis', alpha=0.7)
            axes[0,0].set_xlabel('Coverage (%)')
            axes[0,0].set_ylabel('Avg Profit (bps)')
            axes[0,0].set_title('Profit vs Coverage')
            plt.colorbar(scatter, ax=axes[0,0], label='Horizon (min)')
        
        # Profit vs Win Rate
        if 'win_rate_pct' in df.columns:
            scatter = axes[0,1].scatter(df['win_rate_pct'], df['avg_profit_bps'],
                                       c=df['deadband_bps'], cmap='plasma', alpha=0.7)
            axes[0,1].set_xlabel('Win Rate (%)')
            axes[0,1].set_ylabel('Avg Profit (bps)')
            axes[0,1].set_title('Profit vs Win Rate')
            plt.colorbar(scatter, ax=axes[0,1], label='Deadband (bps)')
        
        # Direction accuracy vs Profit
        axes[1,0].scatter(df['direction_accuracy'], df['avg_profit_bps'], alpha=0.7)
        axes[1,0].set_xlabel('Direction Accuracy')
        axes[1,0].set_ylabel('Avg Profit (bps)')
        axes[1,0].set_title('Direction Accuracy vs Profit')
        
        # Trades vs Profit
        if 'n_executed' in df.columns:
            axes[1,1].scatter(df['n_executed'], df['avg_profit_bps'], alpha=0.7)
            axes[1,1].set_xlabel('Number of Trades')
            axes[1,1].set_ylabel('Avg Profit (bps)')
            axes[1,1].set_title('Trade Volume vs Profit')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distributions(self, df: pd.DataFrame) -> None:
        """Create distribution plots."""
        metrics = ['avg_profit_bps', 'coverage_pct', 'win_rate_pct', 'direction_accuracy']
        available = [m for m in metrics if m in df.columns]
        
        if not available:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(available[:4]):
            df[metric].hist(bins=15, alpha=0.7, ax=axes[i])
            axes[i].set_xlabel(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {metric.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, summaries: Dict[str, pd.DataFrame]) -> None:
        """Save all results."""
        logger.info(f"Saving results to {self.output_dir}")
        
        # Save main results
        if self.results_df is not None:
            self.results_df.to_csv(self.output_dir / 'all_experiments.csv', index=False)
        
        # Save summaries
        for name, df in summaries.items():
            df.to_csv(self.output_dir / f'summary_{name}.csv')
        
        # Create report
        self._create_report(summaries)
    
    def _create_report(self, summaries: Dict[str, pd.DataFrame]) -> None:
        """Create analysis report."""
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write("Two-Class Sweep Results Analysis\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(self.results_df) if self.results_df is not None else 0}\n\n")
            
            if self.results_df is not None:
                df = self.results_df
                f.write(f"Parameter ranges:\n")
                f.write(f"  Horizons: {sorted(df['horizon_min'].unique())}\n")
                f.write(f"  Deadbands: {sorted(df['deadband_bps'].unique())}\n\n")
                
                f.write("Best configuration by profit:\n")
                best_idx = df['avg_profit_bps'].idxmax()
                best = df.loc[best_idx]
                f.write(f"  Horizon: {best['horizon_min']} min\n")
                f.write(f"  Deadband: {best['deadband_bps']} bps\n")
                f.write(f"  Profit: {best['avg_profit_bps']:.2f} bps\n")
                f.write(f"  Coverage: {best.get('coverage_pct', 0):.1f}%\n")
                f.write(f"  Win Rate: {best.get('win_rate_pct', 0):.1f}%\n")
    
    def run_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run complete analysis."""
        logger.info("Starting sweep results analysis...")
        
        # Aggregate results
        df = self.aggregate_results()
        
        # Create summaries
        summaries = self.create_summary_tables()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save everything
        self.save_results(summaries)
        
        logger.info("Analysis completed successfully")
        return summaries


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep experiment results")
    parser.add_argument('--base_dir', required=True, help='Experiments base directory')
    parser.add_argument('--output_dir', help='Output directory (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise ValueError(f"Directory not found: {base_dir}")
    
    try:
        aggregator = SweepResultsAggregator(base_dir, args.output_dir)
        summaries = aggregator.run_analysis()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED")
        print("="*50)
        print(f"Results: {aggregator.output_dir}")
        print(f"Experiments: {len(aggregator.results_df) if aggregator.results_df is not None else 0}")
        
        if aggregator.results_df is not None and len(aggregator.results_df) > 0:
            df = aggregator.results_df
            best_idx = df['avg_profit_bps'].idxmax()
            best = df.loc[best_idx]
            
            print(f"\nBest profit configuration:")
            print(f"  Horizon: {best['horizon_min']} min")
            print(f"  Deadband: {best['deadband_bps']} bps")
            print(f"  Profit: {best['avg_profit_bps']:.2f} bps")
            print(f"  Coverage: {best.get('coverage_pct', 0):.1f}%")
            print(f"  Win Rate: {best.get('win_rate_pct', 0):.1f}%")
        
        print("\nFiles created:")
        print("  - all_experiments.csv")
        print("  - summary_*.csv files") 
        print("  - performance_heatmaps.png")
        print("  - sweep_analysis.png")
        print("  - scatter_plots.png")
        print("  - analysis_report.txt")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
