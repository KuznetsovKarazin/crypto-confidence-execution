#!/usr/bin/env python3
"""
Privacy-Preserving Federated Learning Runner
============================================

Integrates Shamir's Secret Sharing and Differential Privacy
with the existing federated learning pipeline.

Usage:
    python run_privacy_federated.py \\
        --federated_data data/federated \\
        --centralized_artifacts artifacts/centralized \\
        --output experiments/privacy \\
        --privacy --shamir --dp \\
        --epsilon 1.0 --threshold 3
"""

import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import privacy-preserving federated learning
from federated_privacy import (
    PrivacyFederatedConfig,
    create_privacy_preserving_trainer
)
from federated.preprocessing import FederatedPreprocessor
from federated.utils import ModelStateManager, ConfigurationHelper


def setup_safe_logging(log_file: Path):
    """Setup logging without Unicode issues."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler with UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler with ASCII-safe formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Try to configure UTF-8 for console (Windows compatibility)
    try:
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Silently ignore if not supported


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument('--federated_data', type=str, required=True,
                       help='Path to federated data directory')
    parser.add_argument('--centralized_artifacts', type=str, required=True,
                       help='Path to centralized preprocessing artifacts')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    # Privacy flags
    parser.add_argument('--privacy', action='store_true',
                       help='Enable privacy-preserving features')
    parser.add_argument('--shamir', action='store_true',
                       help='Enable Shamir Secret Sharing')
    parser.add_argument('--dp', action='store_true',
                       help='Enable Differential Privacy')
    
    # Shamir parameters
    parser.add_argument('--threshold', type=int, default=3,
                       help='Shamir threshold (t-of-n)')
    parser.add_argument('--prime_bits', type=int, default=61,
                       help='Shamir prime size in bits')
    
    # Differential privacy parameters
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='DP epsilon (privacy budget)')
    parser.add_argument('--delta', type=float, default=1e-6,
                       help='DP delta')
    parser.add_argument('--clip_norm', type=float, default=1.0,
                       help='DP L2 clipping norm')
    parser.add_argument('--per_layer_dp', action='store_true',
                       help='Enable per-layer DP calibration')
    
    # Federated learning parameters
    parser.add_argument('--aggregation', 
                       choices=['fedavg', 'delta_fedavg', 'fedavgm'],
                       default='delta_fedavg',
                       help='Aggregation method')
    parser.add_argument('--rounds', type=int, default=15,
                       help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=3,
                       help='Local epochs per round')
    parser.add_argument('--server_lr', type=float, default=0.5,
                       help='Server learning rate')
    parser.add_argument('--client_lr', type=float, default=0.01,
                       help='Client learning rate')
    parser.add_argument('--server_momentum', type=float, default=0.9,
                       help='Server momentum for FedAvgM')    
    
    # Model parameters
    parser.add_argument('--horizon', type=int, default=600,
                       help='Prediction horizon (minutes)')
    parser.add_argument('--deadband', type=float, default=10.0,
                       help='Deadband (bps)')
    parser.add_argument('--two_class', action='store_true',
                       help='Enable two-class mode')
    parser.add_argument('--confidence_tau', type=float, default=None,
                       help='Confidence threshold for selective execution (overrides artifacts)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Local batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    privacy_suffix = "_privacy" if args.privacy else ""
    output_dir = Path(args.output) / f"{timestamp}_federated_{args.aggregation}{privacy_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'training.log'
    setup_safe_logging(log_file)
    
    logger = logging.getLogger(__name__)
    
    # Print header
    logger.info("=" * 80)
    logger.info("PRIVACY-PRESERVING FEDERATED LEARNING")
    logger.info("=" * 80)
    logger.info(f"Aggregation: {args.aggregation}")
    logger.info(f"Rounds: {args.rounds}, Local Epochs: {args.local_epochs}")
    logger.info(f"Horizon: {args.horizon} min, Deadband: {args.deadband} bps")
    logger.info(f"Server LR: {args.server_lr}") 
    
    if args.privacy:
        logger.info("PRIVACY FEATURES ENABLED:")
        if args.shamir:
            logger.info(f"  - Shamir Secret Sharing: {args.threshold}-threshold, {args.prime_bits}-bit prime")
        if args.dp:
            logger.info(f"  - Differential Privacy: epsilon={args.epsilon}, delta={args.delta}")
            if args.per_layer_dp:
                logger.info(f"  - Per-layer DP calibration enabled")
    else:
        logger.info("Privacy features DISABLED (standard federated learning)")
    
    logger.info("=" * 80)
    
    try:
        # Validate paths
        federated_data_dir = Path(args.federated_data)
        centralized_artifacts_dir = Path(args.centralized_artifacts)
        
        if not federated_data_dir.exists():
            raise FileNotFoundError(f"Federated data directory not found: {federated_data_dir}")
        
        if not centralized_artifacts_dir.exists():
            raise FileNotFoundError(f"Centralized artifacts not found: {centralized_artifacts_dir}")
        
        # Load preprocessor
        logger.info("Loading preprocessing artifacts...")
        preprocessor = FederatedPreprocessor(centralized_artifacts_dir)
        if not preprocessor.load_centralized_artifacts():
            raise RuntimeError("Failed to load preprocessor")
        logger.info("Preprocessor loaded successfully")
        
        # Create configuration
        config = PrivacyFederatedConfig(
            # Paths
            federated_data_dir=federated_data_dir,
            centralized_artifacts_dir=centralized_artifacts_dir,
            output_dir=output_dir,
            
            # Model parameters
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
            
            # Privacy parameters
            enable_privacy=args.privacy,
            enable_shamir=args.shamir if args.privacy else False,
            enable_differential_privacy=args.dp if args.privacy else False,
            
            shamir_threshold=args.threshold,
            shamir_prime_bits=args.prime_bits,
            
            dp_epsilon_total=args.epsilon,
            dp_delta=args.delta,
            dp_l2_clip_norm=args.clip_norm,
            dp_per_layer_calibration=args.per_layer_dp,
            
            # Other
            seed=args.seed,
            save_round_snapshots=True,
            save_best_models=True
        )
        if args.confidence_tau is not None:
            config.confidence_tau = args.confidence_tau
            setattr(config, '_confidence_tau_from_cli', True)     

        if args.aggregation == 'fedavgm':
            config.server_momentum = args.server_momentum
            logger.info(f"Server momentum: {args.server_momentum}")   
        
        # Create trainer
        logger.info("Creating privacy-preserving federated trainer...")
        trainer = create_privacy_preserving_trainer(config, preprocessor)
        
        # Run training
        logger.info("Starting federated training...")
        import time
        start_time = time.time()
        
        results = trainer.run_federated_training()
        
        total_time = time.time() - start_time
        
        # Save results
        logger.info("Saving results...")
        
        # Save configuration
        ConfigurationHelper.save_config(config, output_dir / 'config.json')
        
        # Save global parameters
        if 'global_parameters' in results and results['global_parameters']:
            ModelStateManager.save_parameters(
                results['global_parameters'],
                output_dir / 'global_parameters.joblib'
            )
            logger.info("Global parameters saved")
        
        # Save training history
        if 'training_history' in results and results['training_history']:
            ModelStateManager.save_training_history(
                results['training_history'],
                output_dir / 'training_history.csv'
            )
            logger.info("Training history saved")
        
        # Save global model
        if 'final_results' in results and 'global_model' in results['final_results']:
            global_model = results['final_results']['global_model']
            if global_model:
                ModelStateManager.save_global_model(
                    global_model,
                    output_dir / 'global_model.joblib'
                )
                logger.info("Global model saved")
        
        # Print summary
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        logger.info(f"Aggregation: {args.aggregation}")
        logger.info(f"Rounds completed: {len(results.get('training_history', []))}")
        
        if args.privacy:
            logger.info("Privacy features used:")
            if args.shamir:
                logger.info(f"  - Shamir Secret Sharing ({args.threshold}-threshold)")
            if args.dp:
                logger.info(f"  - Differential Privacy (epsilon={args.epsilon})")
        
        # Print final metrics
        if 'final_results' in results and 'global_metrics' in results['final_results']:
            metrics = results['final_results']['global_metrics']
            
            logger.info("")
            logger.info("FINAL PERFORMANCE:")
            logger.info(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"  Test F1 (macro): {metrics.get('test_f1_macro', 0):.4f}")
            logger.info(f"  Balanced Accuracy: {metrics.get('test_balanced_accuracy', 0):.4f}")
            logger.info(f"  Total samples: {metrics.get('n_samples', 0):,}")
            
            # Selective execution metrics
            if 'direction_accuracy' in metrics:
                logger.info("")
                logger.info("SELECTIVE EXECUTION:")
                logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
                logger.info(f"  Coverage: {metrics.get('coverage', 0):.4f}")
                logger.info(f"  Executed: {metrics.get('n_executed', 0):,}/{metrics.get('n_total', 0):,} samples")
                logger.info(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
            
            # Trading metrics
            if 'avg_profit_bps' in metrics:
                logger.info("")
                logger.info("TRADING PERFORMANCE:")
                logger.info(f"  Avg Profit: {metrics['avg_profit_bps']:.2f} bps")
                logger.info(f"  Median Profit: {metrics.get('median_profit_bps', 0):.2f} bps")
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.3f}")
                logger.info(f"  Sharpe Ratio: {metrics.get('profit_sharpe', 0):.3f}")
        
        # Training progress
        if 'training_history' in results and len(results['training_history']) >= 2:
            history = results['training_history']
            initial = history[0]
            final = history[-1]
            
            acc_improvement = (final.get('avg_test_accuracy', 0) - 
                             initial.get('avg_test_accuracy', 0))
            f1_improvement = (final.get('avg_test_f1', 0) - 
                            initial.get('avg_test_f1', 0))
            
            logger.info("")
            logger.info("TRAINING PROGRESS:")
            logger.info(f"  Accuracy improvement: {acc_improvement:+.4f}")
            logger.info(f"  F1 improvement: {f1_improvement:+.4f}")
            
            if config.two_class_mode and 'avg_test_net_profit_bps' in final:
                profit_improvement = (final.get('avg_test_net_profit_bps', 0) - 
                                    initial.get('avg_test_net_profit_bps', 0))
                logger.info(f"  Profit improvement: {profit_improvement:+.2f} bps")
        
        logger.info("")
        logger.info(f"All results saved to: {output_dir}")
        logger.info("=" * 80)
        
        # Performance assessment
        if 'final_results' in results and 'global_metrics' in results['final_results']:
            final_f1 = results['final_results']['global_metrics'].get('test_f1_macro', 0)
            
            logger.info("")
            if final_f1 > 0.65:
                logger.info("PERFORMANCE: EXCELLENT (F1 > 0.65)")
            elif final_f1 > 0.60:
                logger.info("PERFORMANCE: GOOD (F1 > 0.60)")
            elif final_f1 > 0.55:
                logger.info("PERFORMANCE: MODERATE (F1 > 0.55)")
            else:
                logger.info("PERFORMANCE: NEEDS IMPROVEMENT (F1 < 0.55)")
        
        # Privacy-utility tradeoff analysis
        if args.privacy:
            logger.info("")
            logger.info("PRIVACY-UTILITY TRADEOFF:")
            
            # Try to compare with non-privacy baseline if available
            try:
                centralized_metrics_path = centralized_artifacts_dir / 'metrics_two_class.json'
                if centralized_metrics_path.exists():
                    import json
                    with open(centralized_metrics_path, 'r') as f:
                        cent_metrics = json.load(f)
                    
                    if 'final_results' in results and 'global_metrics' in results['final_results']:
                        fed_metrics = results['final_results']['global_metrics']
                        
                        # Calculate utility degradation
                        acc_degradation = cent_metrics.get('direction_accuracy', 0) - fed_metrics.get('direction_accuracy', 0)
                        profit_degradation = cent_metrics.get('avg_profit_bps', 0) - fed_metrics.get('avg_profit_bps', 0)
                        
                        logger.info(f"  Accuracy degradation: {acc_degradation:.4f} ({acc_degradation/cent_metrics.get('direction_accuracy', 1)*100:.1f}%)")
                        logger.info(f"  Profit degradation: {profit_degradation:.2f} bps ({profit_degradation/abs(cent_metrics.get('avg_profit_bps', 1))*100:.1f}%)")
                        
                        # Privacy gain
                        if args.dp:
                            logger.info(f"  Privacy guarantee: (epsilon={args.epsilon}, delta={args.delta})-DP")
                        if args.shamir:
                            logger.info(f"  Secure aggregation: {args.threshold}-out-of-{config.num_clients} threshold")
                        
                        # Assess tradeoff
                        if acc_degradation < 0.05:
                            logger.info("  Assessment: EXCELLENT privacy-utility tradeoff")
                        elif acc_degradation < 0.10:
                            logger.info("  Assessment: GOOD privacy-utility tradeoff")
                        else:
                            logger.info("  Assessment: HIGH privacy cost, consider tuning parameters")
                
            except Exception as e:
                logger.warning(f"Could not compare with centralized baseline: {e}")
        
        logger.info("")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Debugging guidance
        error_str = str(e).lower()
        logger.error("")
        logger.error("DEBUGGING GUIDANCE:")
        
        if "shamir" in error_str or "secret sharing" in error_str:
            logger.error("  - Check Shamir configuration (threshold <= num_clients)")
            logger.error("  - Verify privacy module is properly installed")
        elif "differential privacy" in error_str or "dp" in error_str:
            logger.error("  - Check DP parameters (epsilon > 0, delta > 0)")
            logger.error("  - Try disabling per-layer calibration")
        elif "aggregation" in error_str:
            logger.error("  - Try standard aggregation (--aggregation fedavg)")
            logger.error("  - Check if privacy features are compatible with aggregation method")
        elif "preprocessor" in error_str:
            logger.error("  - Verify centralized artifacts exist and are valid")
        elif "data" in error_str:
            logger.error("  - Check federated data directory and meta.json")
        else:
            logger.error("  - Verify all paths are correct")
            logger.error("  - Check that required dependencies are installed")
            logger.error("  - Try running without privacy features first (remove --privacy)")
        
        return 1


if __name__ == "__main__":
    exit(main())