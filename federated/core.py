#!/usr/bin/env python3
"""
Federated Learning Core Module 
===============================================

Core federated training logic.
Maintains delta aggregation, best model selection, and robust parameter handling.

"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from copy import deepcopy

# Import our refactored modules
from .client import FederatedClient
from .aggregation import ParameterAggregator
from .utils import FederatedDataManager, validate_parameter_shapes, deterministic_seed

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Streamlined federated configuration preserving essential parameters."""
    
    # Core paths
    federated_data_dir: Path
    centralized_artifacts_dir: Path
    output_dir: Path
    
    # Training parameters 
    horizon_min: int = 5
    deadband_bps: float = 2.0
    two_class_mode: bool = True
    
    # Federated parameters 
    num_rounds: int = 20  # Increased from 15
    local_epochs: int = 3  # Reduced from 5 for faster global updates
    local_batch_size: int = 256  # Increased from 256 for stability
    aggregation_method: str = 'fedavg'  # Start with simple FedAvg
    server_learning_rate: float = 1.0  # Increased from 0.5
    
    # Client-specific learning rate
    client_learning_rate: float = 0.001  # 10x increase from default 0.001
    
    # Gradient clipping
    max_gradient_norm: float = 5.0
    
    # Model parameters - preserved architecture
    mlp_hidden_sizes: Tuple[int, ...] = (256, 128, 64)
    use_class_weights: bool = True
    early_stopping_patience: int = 3
    min_improvement: float = 1e-4
    
    # Key innovations preserved 
    use_shared_init: bool = True
    model_seed: int = 42
    max_delta_norm: float = 10.0
    
    # Simplified configuration
    confidence_tau: float = 0.8
    profit_cost_bps: float = 1.0
    seed: int = 42
    
    # Streamlined logging/saving
    save_round_snapshots: bool = True
    save_best_models: bool = True
    
    use_centralized_init: bool = True
    centralized_model_path: Optional[Path] = None

class FederatedTrainer:
    """
    Streamlined federated trainer 
    """
    
    def __init__(self, config: FederatedConfig, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.clients: Dict[str, FederatedClient] = {}
        self.data_manager = FederatedDataManager(config.federated_data_dir)
        self.aggregator = ParameterAggregator(config)
        
        # Core state
        self.global_parameters: Optional[Dict[str, np.ndarray]] = None
        self.previous_global_parameters: Optional[Dict[str, np.ndarray]] = None
        self.training_history: List[Dict[str, Any]] = []
        self.round_snapshots: List[Dict[str, Any]] = []

        # Track best global model
        self.best_global_accuracy = -np.inf
        self.best_global_parameters = None
        self.best_round = -1
        
        # Set seeds
        np.random.seed(config.seed)
        
        logger.info(f"Initialized FederatedTrainer with {config.aggregation_method} aggregation")
    
    def initialize_clients(self) -> bool:
        """Initialize federated clients with preserved data loading logic."""
        try:
            client_assignments = self.data_manager.load_client_assignments()
            logger.info(f"Found {len(client_assignments)} federated clients")
            
            for client_id in client_assignments.keys():
                client = FederatedClient(
                    client_id=client_id,
                    config=self.config,
                    preprocessor=self.preprocessor
                )
                self.clients[client_id] = client
                
            logger.info(f"Initialized {len(self.clients)} clients successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            return False
    
    def run_federated_training(self) -> Dict[str, Any]:
        """
        Main training loop.
        """
        logger.info("="*60)
        logger.info("STARTING FEDERATED TRAINING")
        logger.info(f"Aggregation: {self.config.aggregation_method}")
        logger.info(f"Rounds: {self.config.num_rounds}")
        logger.info(f"Local epochs: {self.config.local_epochs}")
        logger.info(f"Confidence tau (selective): {self.config.confidence_tau:.4f}")  
        if self.config.aggregation_method == 'delta_fedavg':
            logger.info(f"Server learning rate: {self.config.server_learning_rate}")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Prepare client data with progress
        logger.info("PREPARING CLIENT DATA...")
        active_clients = self._prepare_client_data()
        if len(active_clients) < 2:
            raise ValueError(f"Need at least 2 active clients, got {len(active_clients)}")
        
        logger.info(f"{len(active_clients)} clients ready for training")
        
        # Training rounds with detailed progress
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            logger.info("="*50)
            logger.info(f"ROUND {round_num + 1}/{self.config.num_rounds}")
            logger.info("="*50)
            
            # Preserve previous global parameters for delta computation
            if self.global_parameters is not None:
                self.previous_global_parameters = deepcopy(self.global_parameters)
            
            # Execute round with enhanced logging
            round_results = self._execute_federated_round(active_clients, round_num)
            
            if not round_results:
                logger.error(f"Round {round_num + 1} failed")
                break
            
            round_time = time.time() - round_start
            round_results['round_time'] = round_time

            # Enhanced round progress logging
            self._log_enhanced_round_progress(round_results, round_num, round_time)  

            self.training_history.append(round_results)
            
            # Save round snapshot
            if self.config.save_round_snapshots:
                self._save_round_snapshot(round_num, active_clients)
           
            # Show overall progress
            elapsed = time.time() - start_time
            estimated_total = elapsed * self.config.num_rounds / (round_num + 1)
            remaining = estimated_total - elapsed
            
            logger.info(f"Round time: {round_time:.1f}s | "
                    f"Elapsed: {elapsed/60:.1f}min | "
                    f"ETA: {remaining/60:.1f}min")
        
        total_time = time.time() - start_time
        
        # Final evaluation and results
        logger.info("="*50)
        logger.info("FINAL EVALUATION")
        logger.info("="*50)
        
        final_results = self._evaluate_final_performance(active_clients)
        final_results['total_training_time'] = total_time
        final_results['active_clients'] = active_clients
        
        # Enhanced completion summary
        logger.info("="*60)
        logger.info("FEDERATED TRAINING COMPLETED!")
        logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        logger.info(f"Completed {len(self.training_history)} rounds")
        logger.info(f"Active clients: {len(active_clients)}")
        
        # Show final metrics if available
        if 'global_metrics' in final_results:
            metrics = final_results['global_metrics']
            logger.info("FINAL METRICS:")
            logger.info(f"   Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"   Test F1 (macro): {metrics.get('test_f1_macro', 0):.4f}")
            logger.info(f"   Test F1 (weighted): {metrics.get('test_f1_weighted', 0):.4f}")
            logger.info(f"   Balanced Accuracy: {metrics.get('test_balanced_accuracy', 0):.4f}")
            
            # Per-class metrics
            for i in range(2):  # 0=Down, 1=Up
                if f'test_f1_class_{i}' in metrics:
                    logger.info(f"   Class {i} - Precision: {metrics.get(f'test_precision_class_{i}', 0):.4f}, "
                            f"Recall: {metrics.get(f'test_recall_class_{i}', 0):.4f}, "
                            f"F1: {metrics.get(f'test_f1_class_{i}', 0):.4f}")
            
            # Selective execution metrics
            if 'direction_accuracy' in metrics:
                logger.info(f"   Direction Accuracy: {metrics['direction_accuracy']:.4f}")
                logger.info(f"   Coverage: {metrics.get('coverage', 0):.4f} ({metrics.get('n_executed', 0)}/{metrics.get('n_total', 0)} samples)")
                logger.info(f"   Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
            
            # Trading metrics
            if 'avg_profit_bps' in metrics:
                logger.info(f"   Avg Profit: {metrics['avg_profit_bps']:.2f} bps")
                logger.info(f"   Median Profit: {metrics.get('median_profit_bps', 0):.2f} bps")
                logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.3f}")
                logger.info(f"   Sharpe: {metrics.get('profit_sharpe', 0):.3f}")
                logger.info(f"   Profit Range: [{metrics.get('min_profit_bps', 0):.1f}, {metrics.get('max_profit_bps', 0):.1f}] bps")

        logger.info("="*60)
        
        return {
            'training_history': self.training_history,
            'round_snapshots': self.round_snapshots,
            'final_results': final_results,
            'global_parameters': self.global_parameters
        }

    def _log_enhanced_round_progress(self, round_results: Dict[str, Any], round_num: int, round_time: float) -> None:
        """Enhanced round progress logging with accuracy focus."""
        avg_val_accuracy = round_results.get('avg_val_accuracy', 0.0)  
        avg_test_accuracy = round_results.get('avg_test_accuracy', 0.0)  
        avg_best_accuracy = round_results.get('avg_best_val_accuracy', 0.0) 
       
        if len(self.training_history) > 1:  
            prev_round = self.training_history[-1]  
            prev_val_acc = prev_round.get('avg_val_accuracy', 0.0)
            prev_test_acc = prev_round.get('avg_test_accuracy', 0.0)
            
            val_improvement = avg_val_accuracy - prev_val_acc
            test_improvement = avg_test_accuracy - prev_test_acc

            val_trend = "UP" if val_improvement > 0.001 else "DOWN" if val_improvement < -0.001 else "FLAT"
            test_trend = "UP" if test_improvement > 0.001 else "DOWN" if test_improvement < -0.001 else "FLAT"
            
            logger.info(f"{val_trend} Val Accuracy:  {avg_val_accuracy:.4f} (D={val_improvement:+.4f})")
            logger.info(f"{test_trend} Test Accuracy: {avg_test_accuracy:.4f} (D={test_improvement:+.4f})")
        else:
            logger.info(f"INITIAL Val Accuracy:  {avg_val_accuracy:.4f}")
            logger.info(f"INITIAL Test Accuracy: {avg_test_accuracy:.4f}")
        
        logger.info(f"Best Accuracy: {avg_best_accuracy:.4f}")
        
        # Trading metrics
        if self.config.two_class_mode:
            avg_profit = round_results.get('avg_test_net_profit_bps', 0.0)
            avg_winrate = round_results.get('avg_test_win_rate', 0.0)
            
            profit_status = "PROFIT" if avg_profit > 0 else "LOSS" if avg_profit < -1 else "BREAK_EVEN"
            logger.info(f"{profit_status}: {avg_profit:.2f} bps | Win Rate: {avg_winrate:.3f}")
            # Directional selective metrics 
            avg_dir_acc_val  = round_results.get('avg_val_direction_accuracy', 0.0)
            avg_cov_val      = round_results.get('avg_val_coverage', 0.0)
            avg_conf_val     = round_results.get('avg_val_avg_confidence', 0.0)

            avg_dir_acc_test = round_results.get('avg_test_direction_accuracy', 0.0)
            avg_cov_test     = round_results.get('avg_test_coverage', 0.0)
            avg_conf_test    = round_results.get('avg_test_avg_confidence', 0.0)

            logger.info(
                f"VAL   selective: DirAcc={avg_dir_acc_val:.4f} | Coverage={avg_cov_val:.4f} | AvgConf={avg_conf_val:.4f}"
            )
            logger.info(
                f"TEST  selective: DirAcc={avg_dir_acc_test:.4f} | Coverage={avg_cov_test:.4f} | AvgConf={avg_conf_test:.4f}"
            )
        
        # Client participation
        participating = round_results.get('participating_clients', [])
        logger.info(f"Clients: {len(participating)} participating")
        
        logger.info(f"Round completed in {round_time:.1f}s")

    def _prepare_client_data(self) -> List[str]:
        """Prepare client data with enhanced progress logging."""
        active_clients = []
        total_clients = len(self.clients)
        
        logger.info(f"Loading data for {total_clients} clients...")
        
        for i, (client_id, client) in enumerate(self.clients.items(), 1):
            logger.info(f"[{i}/{total_clients}] Processing {client_id}...")
            
            try:
                success = client.load_and_prepare_data()
                if success:
                    active_clients.append(client_id)
                    info = client.get_data_info()
                    logger.info(f" {client_id}: {info}")
                else:
                    logger.warning(f" {client_id}: Data preparation failed")
            except Exception as e:
                logger.error(f" {client_id}: Error - {e}")
        
        logger.info(f"Data preparation complete: {len(active_clients)}/{total_clients} clients ready")
        return active_clients
    
    def _execute_federated_round(self, active_clients: List[str], round_num: int) -> Dict[str, Any]:
        """Execute federated round with detailed client progress logging."""
        client_best_params = []
        client_delta_params = []
        client_metrics = {}
        client_weights = []
        
        total_clients = len(active_clients)
        logger.info(f"Training {total_clients} clients locally...")
        
        # Local training phase with progress
        for i, client_id in enumerate(active_clients, 1):
            client_start = time.time()
            logger.info(f"[{i}/{total_clients}] Training {client_id}...")
            
            try:
                client = self.clients[client_id]

                final_params, delta_params, metrics = client.train_local(
                    data=None,  
                    global_params=self.global_parameters,
                    round_num=round_num
                )

                client_time = time.time() - client_start
                
                if final_params:
                    client_best_params.append(final_params)
                    client_delta_params.append(delta_params)  
                    client_metrics[client_id] = metrics
                    client_weights.append(metrics.get('train_samples', 1))
                    
                    best_accuracy = metrics.get('best_val_accuracy', 0.0)  
                    test_f1 = metrics.get('test_f1', 0.0)
                    train_samples = metrics.get('train_samples', 0)

                    logger.info(f"  {client_id}: Best Accuracy={best_accuracy:.4f}, Test F1={test_f1:.4f}, "
                            f"Samples={train_samples:,}, Time={client_time:.1f}s")
                    
                    if self.config.two_class_mode and 'test_net_profit_bps' in metrics:
                        profit = metrics['test_net_profit_bps']
                        winrate = metrics.get('test_win_rate', 0)
                        logger.info(f"     Profit: {profit:.2f} bps, Win Rate: {winrate:.3f}")
                else:
                    logger.warning(f"  {client_id}: No valid parameters")
                    
            except Exception as e:
                logger.error(f"  {client_id}: Training failed - {e}")
                continue
        
        if not client_best_params:
            logger.error("No valid client parameters to aggregate")
            return {}
        
        logger.info(f"Local training complete: {len(client_best_params)}/{total_clients} clients succeeded")
        
        # Parameter aggregation with progress
        logger.info(f"Aggregating parameters using {self.config.aggregation_method}...")
        agg_start = time.time()
        
        self.global_parameters = self.aggregator.aggregate_parameters(
            client_best_params=client_best_params,
            client_delta_params=client_delta_params,
            client_weights=client_weights,
            previous_global_params=self.previous_global_parameters,
            round_num=round_num
        )
        
        agg_time = time.time() - agg_start
        logger.info(f"Aggregation complete in {agg_time:.2f}s")
        
        # Create round results
        round_results = self._create_round_results(
            round_num, client_metrics, client_weights, active_clients
        )
        # Track best model
        if 'avg_test_accuracy' in round_results:
            current_accuracy = round_results['avg_test_accuracy']
            if current_accuracy > self.best_global_accuracy:
                self.best_global_accuracy = current_accuracy
                self.best_global_parameters = deepcopy(self.global_parameters)
                self.best_round = round_num
                logger.info(f"New best model: accuracy={current_accuracy:.4f} at round {round_num + 1}")

        return round_results
    
    def _create_round_results(self, round_num: int, client_metrics: Dict[str, Dict],
                            client_weights: List[float], active_clients: List[str]) -> Dict[str, Any]:
        """Create round results with preserved metrics calculation."""
        
        # Calculate average metrics 
        metrics_keys = [
            'val_accuracy', 'test_accuracy', 'best_val_accuracy',
            'val_f1', 'test_f1',
            'val_direction_accuracy', 'test_direction_accuracy',
            'val_coverage', 'test_coverage',
            'val_avg_confidence', 'test_avg_confidence',
        ]

        if self.config.two_class_mode:
            metrics_keys.extend([
                'test_net_profit_bps', 'test_win_rate'
            ])

        
        avg_metrics = {}
        for key in metrics_keys:
            values = [m.get(key, 0.0) for m in client_metrics.values() if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)

        # Weighted averages (preserved calculation)
        total_weight = sum(client_weights)
        if total_weight > 0:
            weights_normalized = [w / total_weight for w in client_weights]
            for key in ['val_accuracy', 'test_accuracy', 'best_val_accuracy']:
                values = [m.get(key, 0.0) for m in client_metrics.values()]
                if values and len(values) == len(weights_normalized):
                    weighted_avg = sum(v * w for v, w in zip(values, weights_normalized))
                    avg_metrics[f'weighted_{key}'] = weighted_avg
        
        return {
            'round': round_num,
            'participating_clients': active_clients,
            'client_weights': dict(zip(active_clients, client_weights)),
            'aggregation_method': self.config.aggregation_method,
            'server_lr': self.config.server_learning_rate,
            **avg_metrics
        }
    
    def _save_round_snapshot(self, round_num: int, active_clients: List[str]) -> None:
        """Save round snapshot with preserved metadata."""
        if self.global_parameters:
            snapshot = {
                'round': round_num,
                'global_parameters': deepcopy(self.global_parameters),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'participating_clients': len(active_clients),  # Privacy-conscious
                'aggregation_method': self.config.aggregation_method
            }
            self.round_snapshots.append(snapshot)
    
    def _log_round_progress(self, round_results: Dict[str, Any], round_num: int) -> None:
        """Simplified but informative round progress logging."""
        avg_val_f1 = round_results.get('avg_val_f1', 0.0)
        avg_test_f1 = round_results.get('avg_test_f1', 0.0)
        avg_best_f1 = round_results.get('avg_best_val_f1', 0.0)
        
        # Calculate improvement if not first round
        if len(self.training_history) > 0:
            prev_val_f1 = self.training_history[-1].get('avg_val_f1', 0.0)
            improvement = avg_val_f1 - prev_val_f1
            trend = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            
            logger.info(f"Round {round_num + 1} {trend}: "
                       f"Val F1={avg_val_f1:.4f} (Δ{improvement:+.4f}), "
                       f"Test F1={avg_test_f1:.4f}, "
                       f"Best F1={avg_best_f1:.4f}")
        else:
            logger.info(f"Round {round_num + 1}: "
                       f"Val F1={avg_val_f1:.4f}, "
                       f"Test F1={avg_test_f1:.4f}, "
                       f"Best F1={avg_best_f1:.4f}")
        
        # Trading metrics if available
        if self.config.two_class_mode:
            avg_profit = round_results.get('avg_test_net_profit_bps', 0.0)
            avg_winrate = round_results.get('avg_test_win_rate', 0.0)
            logger.info(f"         Trading: Profit={avg_profit:.2f} bps, "
                       f"Win Rate={avg_winrate:.3f}")
            avg_dir_acc = round_results.get('avg_test_direction_accuracy', 0.0)
            avg_cov     = round_results.get('avg_test_coverage', 0.0)
            logger.info(f"Direction Acc: {avg_dir_acc:.4f} | Coverage: {avg_cov:.4f}")
    
    def _calculate_comprehensive_global_metrics(self, aggregated: Dict) -> Dict[str, float]:
        """Calculate comprehensive global metrics matching centralized format."""
        
        y_true = np.array(aggregated['all_y_true'])
        y_pred = np.array(aggregated['all_y_pred'])
        returns_bps = np.array(aggregated['all_returns_bps']) if aggregated['all_returns_bps'] else None
        
        if len(y_true) == 0:
            logger.warning("No test data for global metrics calculation")
            return {}
        
        # Concatenate probabilities if available
        all_proba = None
        if aggregated['all_proba']:
            all_proba = np.vstack(aggregated['all_proba'])
        
        metrics = {}
        
        # Basic classification metrics
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score,
            precision_recall_fscore_support, roc_auc_score
        )
        
        metrics['test_accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['test_balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        metrics['test_f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['test_f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            metrics[f'test_precision_class_{i}'] = float(p)
            metrics[f'test_recall_class_{i}'] = float(r)
            metrics[f'test_f1_class_{i}'] = float(f)
            metrics[f'test_support_class_{i}'] = int(s)
        
        # AUC metrics if probabilities available
        if all_proba is not None and all_proba.shape[1] == 2:
            try:
                metrics['test_auc'] = float(roc_auc_score(y_true, all_proba[:, 1]))
            except:
                pass
        
        # Two-class specific metrics (direction accuracy, coverage, confidence)
        if all_proba is not None and all_proba.shape[1] == 2:
            p_up = all_proba[:, 1]
            confidence = np.maximum(p_up, 1.0 - p_up)
            
            # Use loaded tau from config
            tau = getattr(self.config, 'confidence_tau', 0.8)
            execute = confidence >= tau
            
            metrics['coverage'] = float(execute.mean())
            metrics['n_executed'] = int(execute.sum())
            metrics['n_total'] = int(len(y_true))
            
            if execute.sum() > 0:
                pred_direction = (p_up >= 0.5).astype(int)
                direction_correct = (pred_direction[execute] == y_true[execute])
                
                metrics['direction_accuracy'] = float(direction_correct.mean())
                metrics['avg_confidence'] = float(confidence[execute].mean())
                
                # Calculate profits if returns available
                if returns_bps is not None and len(returns_bps) == len(y_true):
                    executed_returns = returns_bps[execute]
                    executed_pred = pred_direction[execute]
                    
                    gross_profit = np.where(executed_pred == 1, executed_returns, -executed_returns)
                    net_profit = gross_profit - self.config.profit_cost_bps
                    
                    metrics['avg_profit_bps'] = float(net_profit.mean())
                    metrics['median_profit_bps'] = float(np.median(net_profit))
                    metrics['win_rate'] = float((net_profit > 0).mean())
                    metrics['profit_std_bps'] = float(net_profit.std())
                    metrics['profit_sharpe'] = float(net_profit.mean() / (net_profit.std() + 1e-8))
                    metrics['max_profit_bps'] = float(net_profit.max())
                    metrics['min_profit_bps'] = float(net_profit.min())
                    metrics['profit_p10_bps'] = float(np.percentile(net_profit, 10))
                    metrics['profit_p25_bps'] = float(np.percentile(net_profit, 25))
                    metrics['profit_p75_bps'] = float(np.percentile(net_profit, 75))
                    metrics['profit_p90_bps'] = float(np.percentile(net_profit, 90))
            else:
                metrics['direction_accuracy'] = 0.0
                metrics['avg_confidence'] = float(confidence.mean())
                metrics['avg_profit_bps'] = -self.config.profit_cost_bps
        
        metrics['n_samples'] = int(len(y_true))
        
        return metrics
    
    def _evaluate_final_performance(self, active_clients: List[str]) -> Dict[str, Any]:
        """Final performance evaluation with comprehensive metrics."""
        if not self.global_parameters:
            return {}
        
        # Create global model for evaluation
        global_model = self._create_global_model()
        if not global_model:
            return {}
        
        # Collect metrics from all clients
        client_results = {}
        aggregated_metrics = {
            'all_y_true': [],
            'all_y_pred': [],
            'all_proba': [],
            'all_returns_bps': [],
            'all_symbols': [],
            'n_total_samples': 0
        }
        
        for client_id in active_clients:
            try:
                client = self.clients[client_id]
                
                # Get predictions using evaluate_global_model
                test_metrics = client.evaluate_global_model(global_model)
                
                if test_metrics:
                    client_results[client_id] = test_metrics
                    
                    # Accumulate data for global metrics
                    aggregated_metrics['all_y_true'].extend(test_metrics.get('true_labels', []))
                    aggregated_metrics['all_y_pred'].extend(test_metrics.get('predictions', []))
                    
                    if 'returns_bps' in test_metrics:
                        aggregated_metrics['all_returns_bps'].extend(test_metrics['returns_bps'])
                    else:
                        logger.warning("Client metrics missing 'returns_bps'; skipping returns for global profit calc")
                    
                    aggregated_metrics['n_total_samples'] += test_metrics.get('test_samples', 0)
                    
                    # Accumulate proba if available
                    proba = global_model.predict_proba(client.data['X_test'])
                    aggregated_metrics['all_proba'].append(proba)
                        
            except Exception as e:
                logger.error(f"Failed to evaluate {client_id}: {e}")
        
        # Calculate comprehensive global metrics
        global_metrics = self._calculate_comprehensive_global_metrics(aggregated_metrics)
        
        return {
            'client_results': client_results,
            'global_metrics': global_metrics,
            'global_model': global_model
        }
        
    def _create_global_model(self):
        """Create global model """
        # This will be implemented using the client's model creation logic
        # Preserving the proper sklearn initialization
        if not self.clients:
            return None
            
        # Get template client for model creation
        template_client = next(iter(self.clients.values()))
        return template_client.create_global_model(self.global_parameters)
    
    def _calculate_global_metrics(self, client_metrics_list) -> Dict[str, float]:

        all_executed_flags: list = []
        all_direction_correct: list = []
        all_confidences: list = []
        all_profits_bps: list = []
        n_total = 0  

        for cm in client_metrics_list:
            executed = cm.get("executed_flags", [])
            direction_correct = cm.get("direction_correct", [])
            confidences = cm.get("confidences", [])
            profits_bps = cm.get("profits_bps", [])
            n_test = int(cm.get("n_test", 0))

            L = min(len(executed), len(direction_correct), len(confidences), len(profits_bps))
            if L > 0:
                all_executed_flags.extend(executed[:L])
                all_direction_correct.extend(direction_correct[:L])
                all_confidences.extend(confidences[:L])
                all_profits_bps.extend(profits_bps[:L])

            n_total += n_test

        if n_total <= 0 or len(all_executed_flags) == 0:
            return {
                "coverage": 0.0,
                "n_executed": 0,
                "n_total": int(n_total),
                "direction_accuracy": 0.0,
                "avg_profit_bps": 0.0,
                "median_profit_bps": 0.0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "profit_std_bps": 0.0,
                "profit_sharpe": 0.0,
                "max_profit_bps": 0.0,
                "min_profit_bps": 0.0,
                "profit_p25_bps": 0.0,
                "profit_p75_bps": 0.0,
                "profit_p90_bps": 0.0,
                "profit_p10_bps": 0.0,
            }

        executed_mask = np.array(all_executed_flags, dtype=bool)
        n_executed = int(executed_mask.sum())
        coverage = float(n_executed / max(n_total, 1))

        if n_executed == 0:
            return {
                "coverage": coverage,
                "n_executed": 0,
                "n_total": int(n_total),
                "direction_accuracy": 0.0,
                "avg_profit_bps": 0.0,
                "median_profit_bps": 0.0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "profit_std_bps": 0.0,
                "profit_sharpe": 0.0,
                "max_profit_bps": 0.0,
                "min_profit_bps": 0.0,
                "profit_p25_bps": 0.0,
                "profit_p75_bps": 0.0,
                "profit_p90_bps": 0.0,
                "profit_p10_bps": 0.0,      
                "confidence_tau": float(self.config.confidence_tau) if self.config.confidence_tau is not None else None,
                "profit_cost_bps": float(self.config.profit_cost_bps),                         
            }

        dir_correct = np.array(all_direction_correct, dtype=bool)[executed_mask]
        confidences = np.array(all_confidences, dtype=float)[executed_mask]
        profits = np.array(all_profits_bps, dtype=float)[executed_mask]

        direction_accuracy = float(dir_correct.mean()) if dir_correct.size else 0.0
        win_rate = float((profits > 0).mean()) if profits.size else 0.0
        avg_confidence = float(confidences.mean()) if confidences.size else 0.0

        if profits.size:
            avg_profit = float(profits.mean())
            median_profit = float(np.median(profits))
            std_profit = float(profits.std())
            sharpe = float(avg_profit / (std_profit + 1e-8))
            p10 = float(np.percentile(profits, 10))
            p25 = float(np.percentile(profits, 25))
            p75 = float(np.percentile(profits, 75))
            p90 = float(np.percentile(profits, 90))
            pmin = float(profits.min())
            pmax = float(profits.max())
        else:
            avg_profit = median_profit = std_profit = sharpe = 0.0
            p10 = p25 = p75 = p90 = pmin = pmax = 0.0

        return {
            "coverage": coverage,
            "n_executed": int(n_executed),
            "n_total": int(n_total),
            "direction_accuracy": direction_accuracy,
            "avg_profit_bps": avg_profit,
            "median_profit_bps": median_profit,
            "win_rate": win_rate,
            "avg_confidence": avg_confidence,
            "profit_std_bps": std_profit,
            "profit_sharpe": sharpe,
            "max_profit_bps": pmax,
            "min_profit_bps": pmin,
            "profit_p25_bps": p25,
            "profit_p75_bps": p75,
            "profit_p90_bps": p90,
            "profit_p10_bps": p10,
            "confidence_tau": float(self.config.confidence_tau) if self.config.confidence_tau is not None else None,
            "profit_cost_bps": float(self.config.profit_cost_bps),              
        }



def create_federated_trainer(config: FederatedConfig, preprocessor) -> FederatedTrainer:
    try:
        if getattr(config, '_confidence_tau_from_cli', False):
            logger.info(f"Using CLI confidence_tau={config.confidence_tau:.4f} (overrides artifacts)")
        elif hasattr(preprocessor, "decision_threshold") and preprocessor.decision_threshold is not None:
            config.confidence_tau = float(preprocessor.decision_threshold)
            logger.info(f"Using centralized decision threshold: confidence_tau={config.confidence_tau:.4f}")
        else:
            logger.info(f"Using default/configured confidence_tau={config.confidence_tau}")
    except Exception as e:
        logger.warning(f"Could not set confidence_tau: {e}")

    trainer = FederatedTrainer(config, preprocessor)
 
    if not trainer.initialize_clients():
        raise ValueError("Failed to initialize federated clients")

    return trainer
