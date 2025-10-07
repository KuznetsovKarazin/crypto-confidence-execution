#!/usr/bin/env python3
"""
Federated Learning Runner

"""

import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

# Import refactored modules
from federated import (
    FederatedConfig, 
    create_federated_trainer,
    setup_federated_logging
)
from federated.preprocessing import FederatedPreprocessor


def validate_setup_simple(federated_data_dir: Path, centralized_artifacts_dir: Path) -> Tuple[bool, List[str]]:
    """Simple setup validation without external dependencies."""
    issues = []
    
    # Check federated data directory
    if not federated_data_dir.exists():
        issues.append(f"Federated data directory not found: {federated_data_dir}")
    else:
        meta_path = federated_data_dir / 'meta.json'
        if not meta_path.exists():
            issues.append(f"Meta file not found: {meta_path}")
    
    # Check centralized artifacts
    if not centralized_artifacts_dir.exists():
        issues.append(f"Centralized artifacts directory not found: {centralized_artifacts_dir}")
    else:
        required_artifacts = ['imputer.joblib', 'scaler.joblib', 'feature_schema.json']
        for artifact in required_artifacts:
            artifact_path = centralized_artifacts_dir / artifact
            if not artifact_path.exists():
                issues.append(f"Missing centralized artifact: {artifact}")
    
    return len(issues) == 0, issues


def setup_console_logging():
    """Setup console logging with UTF-8 encoding to avoid Unicode errors."""
    # Setup console logging with more verbose output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Use simple ASCII-only formatter to avoid encoding issues
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # Set encoding for console output
    import sys
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--federated_data', type=str, required=True,
                       help='Path to federated data directory')
    parser.add_argument('--centralized_artifacts', type=str, required=True,
                       help='Path to centralized preprocessing artifacts')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    # Core parameters
    parser.add_argument('--aggregation', 
                       choices=['fedavg', 'delta_fedavg', 'fedavgm', 'coordinate_median', 'trimmed_mean'],
                       default='delta_fedavg', help='Aggregation method')
    parser.add_argument('--rounds', type=int, default=15, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Local epochs per round')
    parser.add_argument('--server_lr', type=float, default=0.5, help='Server learning rate for delta aggregation')
    parser.add_argument('--client_lr', type=float, default=0.01, help='Client learning rate')
    parser.add_argument('--server_momentum', type=float, default=0.9, help='Server momentum for FedAvgM')
    
    # Model parameters
    parser.add_argument('--horizon', type=int, default=600, help='Prediction horizon (minutes)')
    parser.add_argument('--deadband', type=float, default=10.0, help='Deadband (bps)')
    parser.add_argument('--two_class', action='store_true', help='Enable two-class mode')
    parser.add_argument('--confidence_tau', type=float, default=None, help='Confidence threshold for selective execution (overrides artifacts)')
    
    # Optional parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='Local batch size')
    parser.add_argument('--mlp_hidden_sizes', nargs='+', type=int, default=[256, 128, 64],
                       help='Hidden layer sizes for MLP')
    
    # Performance parameters
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--min_improvement', type=float, default=1e-4,
                       help='Minimum improvement threshold')
    
    args = parser.parse_args()
    
    # Setup console logging first
    setup_console_logging()
    
    # Setup file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{timestamp}_federated_{args.aggregation}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'federated_training.log'
    setup_federated_logging(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Federated Learning")
    logger.info(f"Aggregation: {args.aggregation}")
    logger.info(f"Horizon: {args.horizon} minutes")
    logger.info(f"Deadband: {args.deadband} bps")
    logger.info(f"Server LR: {args.server_lr}")
    
    try:
        # Validate setup
        federated_data_dir = Path(args.federated_data)
        centralized_artifacts_dir = Path(args.centralized_artifacts)
        
        valid, issues = validate_setup_simple(federated_data_dir, centralized_artifacts_dir)
        if not valid:
            logger.error("Setup validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return 1
        
        logger.info("Setup validation passed")
        
        # Create configuration
        config = FederatedConfig(
            federated_data_dir=federated_data_dir,
            centralized_artifacts_dir=centralized_artifacts_dir,
            output_dir=output_dir,
            
            # Model parameters matching centralized training
            horizon_min=args.horizon,
            deadband_bps=args.deadband,
            two_class_mode=args.two_class,
            
            # Federated parameters
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            local_batch_size=args.batch_size,
            aggregation_method=args.aggregation,
            server_learning_rate=args.server_lr,
            client_learning_rate=args.client_lr, 
            max_gradient_norm=5.0,  
            
            # MLP parameters
            mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
            
            # Training parameters
            early_stopping_patience=args.early_stopping_patience,
            min_improvement=args.min_improvement,
            
            # Other parameters
            seed=args.seed,
            confidence_tau=args.confidence_tau if args.confidence_tau is not None else 0.8,
            
            # Save settings
            save_round_snapshots=True,
            save_best_models=True
        )
        if args.confidence_tau is not None:
            setattr(config, '_confidence_tau_from_cli', True)
        
        # Add server_momentum to config if using FedAvgM
        if args.aggregation == 'fedavgm':
            config.server_momentum = args.server_momentum
            logger.info(f"Server momentum: {args.server_momentum}")
        
        # Load preprocessor
        logger.info("Loading preprocessing artifacts...")
        preprocessor = FederatedPreprocessor(centralized_artifacts_dir)
        if not preprocessor.load_centralized_artifacts():
            raise RuntimeError("Failed to load preprocessor")
        
        logger.info("Preprocessor loaded successfully")
        
        # Create and run trainer
        logger.info("Creating federated trainer...")
        trainer = create_federated_trainer(config, preprocessor)
        
        logger.info("Starting federated training...")
        start_time = time.time()
        results = trainer.run_federated_training()
        total_time = time.time() - start_time
        
        # Save results
        from federated.utils import ModelStateManager, ConfigurationHelper
        
        logger.info("Saving results...")
        
        # Save configuration
        ConfigurationHelper.save_config(config, output_dir / 'config.json')
        
        # Save model and history
        if 'global_parameters' in results and results['global_parameters']:
            ModelStateManager.save_parameters(
                results['global_parameters'], 
                output_dir / 'global_parameters.joblib'
            )
            logger.info("Global parameters saved")
        
        if 'training_history' in results and results['training_history']:
            ModelStateManager.save_training_history(
                results['training_history'],
                output_dir / 'training_history.csv'
            )
            logger.info("Training history saved")
        
        # Save global model if available
        if 'final_results' in results and 'global_model' in results['final_results']:
            global_model = results['final_results']['global_model']
            if global_model:
                ModelStateManager.save_global_model(
                    global_model,
                    output_dir / 'global_model.joblib'
                )
                logger.info("Global model saved")
        
        # Print comprehensive summary
        logger.info("="*60)
        logger.info("FEDERATED TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        logger.info(f"Aggregation method: {args.aggregation}")
        logger.info(f"Rounds completed: {len(results.get('training_history', []))}")
        
        # Print final metrics if available
        if 'final_results' in results and 'global_metrics' in results['final_results']:
            metrics = results['final_results']['global_metrics']
            logger.info("FINAL PERFORMANCE:")
            logger.info(f"  Test F1: {metrics.get('test_f1_macro', 0):.4f}")
            logger.info(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"  Total samples: {metrics.get('n_samples', 0):,}")
            
            if 'global_net_profit_bps' in metrics:
                logger.info(f"  Net Profit: {metrics['global_net_profit_bps']:.2f} bps")
                logger.info(f"  Win Rate: {metrics['global_win_rate']:.3f}")
                logger.info(f"  Sharpe Ratio: {metrics['global_sharpe_ratio']:.3f}")
        
        # Print training progress summary
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            if len(history) >= 2:
                initial = history[0]
                final = history[-1]
                
                acc_improvement = (final.get('avg_test_accuracy', 0) - 
                                 initial.get('avg_test_accuracy', 0))
                f1_improvement = (final.get('avg_test_f1', 0) - 
                                initial.get('avg_test_f1', 0))
                
                logger.info("TRAINING PROGRESS:")
                logger.info(f"  Accuracy improvement: {acc_improvement:+.4f}")
                logger.info(f"  F1 improvement: {f1_improvement:+.4f}")
                
                if config.two_class_mode:
                    profit_improvement = (final.get('avg_test_net_profit_bps', 0) - 
                                        initial.get('avg_test_net_profit_bps', 0))
                    logger.info(f"  Profit improvement: {profit_improvement:+.2f} bps")
        
        logger.info(f"All results saved to: {output_dir}")
        logger.info("="*60)
        
        # Success indicators
        if 'final_results' in results and 'global_metrics' in results['final_results']:
            final_f1 = results['final_results']['global_metrics'].get('test_f1_macro', 0)
            if final_f1 > 0.65:
                logger.info("EXCELLENT: F1 > 0.65 - Strong performance achieved!")
            elif final_f1 > 0.60:
                logger.info("GOOD: F1 > 0.60 - Solid performance")
            elif final_f1 > 0.55:
                logger.info("MODERATE: F1 > 0.55 - Room for improvement")
            else:
                logger.info("NEEDS WORK: F1 < 0.55 - Consider parameter tuning")
        
        logger.info("="*60)
        
        # Compare with centralized model
        try:
            centralized_artifacts_dir = Path(args.centralized_artifacts)
            centralized_metrics_path = centralized_artifacts_dir / 'metrics_two_class.json'
            
            if centralized_metrics_path.exists():
                import json
                with open(centralized_metrics_path, 'r') as f:
                    cent_metrics = json.load(f)
                
                logger.info("="*60)
                logger.info("COMPARISON: FEDERATED vs CENTRALIZED")
                logger.info("="*60)
                
                if 'final_results' in results and 'global_metrics' in results['final_results']:
                    fed_metrics = results['final_results']['global_metrics']

                    # --- Validation of comparability of sets ---
                    cent_tau  = cent_metrics.get('confidence_tau', None)
                    fed_tau   = fed_metrics.get('confidence_tau', None)
                    cent_cost = cent_metrics.get('profit_cost_bps', None)
                    fed_cost  = fed_metrics.get('profit_cost_bps', None)

                    logger.info(f"TAU CHECK: federated={fed_tau} vs centralized={cent_tau}")
                    logger.info(f"COST CHECK (bps): federated={fed_cost} vs centralized={cent_cost}")

                    mismatch_reasons = []
                    if (cent_tau is not None and fed_tau is not None) and abs(float(fed_tau) - float(cent_tau)) > 1e-9:
                        mismatch_reasons.append("confidence_tau differs")
                    if (cent_cost is not None and fed_cost is not None) and abs(float(fed_cost) - float(cent_cost)) > 1e-9:
                        mismatch_reasons.append("profit_cost_bps differs")

                    cent_total = cent_metrics.get('n_total', None)
                    fed_total  = fed_metrics.get('n_total', None)
                    if (cent_total is not None and fed_total is not None) and int(cent_total) != int(fed_total):
                        mismatch_reasons.append(f"n_total differs (fed={fed_total}, cent={cent_total})")

                    if mismatch_reasons:
                        logger.warning("COMPARISON INVALID: " + "; ".join(mismatch_reasons))
                        logger.warning("Tip: Fix the overall test index (symbol,timestamp) and recalculate the metrics.")

                    
                    fed_executed = fed_metrics.get('n_executed', 0)
                    fed_total = fed_metrics.get('n_total', 0)
                    
                    if fed_executed == 0:
                        logger.warning("="*60)
                        logger.warning("FEDERATED MODEL DID NOT EXECUTE ANY TRADES")
                        logger.warning("="*60)
                        logger.warning("Cannot perform meaningful comparison - federated model is non-functional")
                        logger.warning("")
                        logger.warning("Root cause: confidence_tau is too high or model probabilities are poor")
                        logger.warning("")
                        logger.warning("DIAGNOSTIC STEPS:")
                        logger.warning("1. Check tau_optimization_results.csv in output directory")
                        logger.warning("2. Inspect confidence distribution:")
                        logger.warning(f"   - Avg confidence: {fed_metrics.get('avg_confidence', 0):.3f}")
                        logger.warning(f"   - Current tau: {fed_metrics.get('confidence_tau', 'N/A')}")
                        logger.warning("")
                        logger.warning("SUGGESTED FIXES:")
                        logger.warning("1. Lower --confidence_tau (try 0.55-0.65)")
                        logger.warning("2. Use --optimize_tau_by ev --min_coverage 0.01")
                        logger.warning("3. Check model calibration (--calibrate_probabilities)")
                        logger.warning("4. Increase training rounds (--rounds 25)")
                        logger.warning("="*60)
                        
                        logger.info("")
                        logger.info("CENTRALIZED MODEL (for reference):")
                        logger.info(f"  Direction Accuracy: {cent_metrics.get('direction_accuracy', 0):.4f}")
                        logger.info(f"  Coverage: {cent_metrics.get('coverage', 0):.4f}")
                        logger.info(f"  Avg Profit: {cent_metrics.get('avg_profit_bps', 0):.2f} bps")
                        logger.info(f"  Executed: {cent_metrics.get('n_executed', 0):,}/{cent_metrics.get('n_total', 0):,}")
                        
                        return 0  
                    
                    logger.info(f"Federated model executed {fed_executed:,}/{fed_total:,} trades")
                    logger.info("")
                    
                    comparison_metrics = [
                        ('Direction Accuracy', 'direction_accuracy', 'direction_accuracy', '{:.4f}'),
                        ('Coverage', 'coverage', 'coverage', '{:.4f}'),
                        ('Avg Profit (bps)', 'avg_profit_bps', 'avg_profit_bps', '{:.2f}'),
                        ('Median Profit (bps)', 'median_profit_bps', 'median_profit_bps', '{:.2f}'),
                        ('Win Rate', 'win_rate', 'win_rate', '{:.4f}'),
                        ('Avg Confidence', 'avg_confidence', 'avg_confidence', '{:.4f}'),
                        ('Sharpe Ratio', 'profit_sharpe', 'profit_sharpe', '{:.3f}'),
                    ]
                    
                    logger.info(f"{'Metric':<25} {'Federated':>12} {'Centralized':>12} {'Difference':>12} {'Change':>10}")
                    logger.info("-" * 75)
                    
                    for metric_name, fed_key, cent_key, fmt in comparison_metrics:
                        fed_val = fed_metrics.get(fed_key, 0.0)
                        cent_val = cent_metrics.get(cent_key, 0.0)
                        
                        diff = fed_val - cent_val
                        pct_diff = (diff / abs(cent_val)) * 100 if cent_val != 0 else 0.0
                        
                        symbol = "=" if abs(diff) < 0.0001 else ("Up" if diff > 0 else "Down")
                        
                        fed_str = fmt.format(fed_val)
                        cent_str = fmt.format(cent_val)
                        diff_str = ('+' if diff >= 0 else '') + fmt.format(diff)
                        pct_str = f"{pct_diff:+.1f}% {symbol}"
                        
                        logger.info(f"{metric_name:<25} {fed_str:>12} {cent_str:>12} {diff_str:>12} {pct_str:>10}")
                    
                    logger.info("-" * 75)
                    
                    cent_executed = cent_metrics.get('n_executed', 0)
                    cent_total = cent_metrics.get('n_total', 0)
                    logger.info(f"Samples executed: Fed={fed_executed:,}/{fed_total:,}, Cent={cent_executed:,}/{cent_total:,}")
                    
                    logger.info("="*60)
                    logger.info("KEY INSIGHTS:")
                    
                    # Automated insights
                    if fed_metrics.get('coverage', 0) > cent_metrics.get('coverage', 0) * 1.5:
                        logger.info("• Federated model executes MORE trades (higher coverage)")
                    
                    if fed_metrics.get('direction_accuracy', 0) < cent_metrics.get('direction_accuracy', 0):
                        acc_gap = cent_metrics.get('direction_accuracy', 0) - fed_metrics.get('direction_accuracy', 0)
                        logger.info(f"• Centralized model has BETTER direction accuracy ({acc_gap:.1%} higher)")
                    
                    if fed_metrics.get('avg_profit_bps', 0) > cent_metrics.get('avg_profit_bps', 0):
                        logger.info("• Federated model achieves HIGHER average profit per trade")
                    
                    fed_sharpe = fed_metrics.get('profit_sharpe', 0)
                    cent_sharpe = cent_metrics.get('profit_sharpe', 0)
                    if fed_sharpe > 0 and cent_sharpe > 0:
                        if abs(fed_sharpe - cent_sharpe) < 0.1:
                            logger.info(f"• Sharpe ratios are SIMILAR (Fed={fed_sharpe:.2f}, Cent={cent_sharpe:.2f})")
                        elif fed_sharpe > cent_sharpe:
                            logger.info(f"• Federated has BETTER risk-adjusted returns (Sharpe: {fed_sharpe:.2f} vs {cent_sharpe:.2f})")
                        else:
                            logger.info(f"• Centralized has BETTER risk-adjusted returns (Sharpe: {cent_sharpe:.2f} vs {fed_sharpe:.2f})")
                    
                    logger.info("="*60)
                
            else:
                logger.info("Centralized metrics not found for comparison")
                logger.info(f"Expected path: {centralized_metrics_path}")
        
        except Exception as e:
            logger.warning(f"Could not compare with centralized model: {e}")
        
        return 0
    

        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Enhanced debugging guidance
        error_str = str(e).lower()
        if "delta" in error_str:
            logger.error("DEBUG: Delta aggregation issue - check parameter extraction and passing")
        elif "aggregation" in error_str:
            logger.error("DEBUG: Try different aggregation method (--aggregation fedavg)")
        elif "preprocessor" in error_str:
            logger.error("DEBUG: Check centralized artifacts path and files")
        elif "data" in error_str:
            logger.error("DEBUG: Check federated data directory and meta.json")
        else:
            logger.error("DEBUG: Check all paths and configuration parameters")
        
        return 1


if __name__ == "__main__":
    exit(main())