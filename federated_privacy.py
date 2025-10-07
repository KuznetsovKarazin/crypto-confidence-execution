#!/usr/bin/env python3
"""
Privacy-Preserving Federated Learning Module
==============================================

Integrates Shamir's Secret Sharing and Differential Privacy 
with existing federated learning infrastructure.

Key Features:
- Secure aggregation with Shamir's Secret Sharing
- Per-layer differential privacy calibration
- Dropout-tolerant mask recovery
- Compatible with existing federated training pipeline
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
from dataclasses import dataclass

# Import existing federated modules
from federated.core import FederatedTrainer, FederatedConfig
from federated.aggregation import ParameterAggregator

# Import privacy module
from utils.enhanced_shamir_privacy import (
    EnhancedShamirSecretSharing,
    SecureAggregationProtocol,
    ShamirConfig,
    DifferentialPrivacyConfig,
    SecretShare
)

logger = logging.getLogger(__name__)


@dataclass
class PrivacyFederatedConfig(FederatedConfig):
    """Extended configuration with privacy parameters."""
    
    # Privacy settings
    enable_privacy: bool = True
    enable_shamir: bool = True
    enable_differential_privacy: bool = True
    
    # Shamir parameters
    shamir_threshold: int = 3
    shamir_prime_bits: int = 61
    shamir_precision_factor: int = 1000000
    
    # Differential privacy parameters
    dp_epsilon_total: float = 1.0
    dp_delta: float = 1e-6
    dp_l2_clip_norm: float = 1.0
    dp_per_layer_calibration: bool = True
    
    # Secure aggregation parameters
    mask_amplitude: float = 0.1

    num_clients: Optional[int] = None  # Dynamically set from federated data    
    
    def __post_init__(self):
        """Validate privacy configuration."""
        if self.enable_privacy:
            if self.shamir_threshold < 2:
                raise ValueError("Shamir threshold must be at least 2")
            
            # Infer num_clients from federated_data_dir if not set
            if not hasattr(self, 'num_clients'):
                # Will be set during trainer initialization
                pass
            
            if self.dp_epsilon_total <= 0:
                raise ValueError("DP epsilon must be positive")
        
        logger.info(f"Privacy Config: Shamir={self.enable_shamir}, DP={self.enable_differential_privacy}")


class PrivacyPreservingAggregator(ParameterAggregator):
    """Enhanced aggregator with privacy-preserving secure aggregation."""
    
    def __init__(self, config: PrivacyFederatedConfig):
        super().__init__(config)
        self.privacy_config = config
        
        # Initialize privacy components if enabled
        if config.enable_privacy:
            # Determine number of clients (will be set properly during initialization)
            num_clients = getattr(config, 'num_clients', 5)
            
            if num_clients is None:  
                raise ValueError(
                    "num_clients must be set before creating PrivacyPreservingAggregator. "
                    "This should be done automatically in PrivacyPreservingTrainer.__init__"
                )

            # Shamir configuration
            shamir_config = ShamirConfig(
                threshold=config.shamir_threshold,
                num_participants=num_clients,
                prime_bits=config.shamir_prime_bits,
                precision_factor=config.shamir_precision_factor,
                enable_vectorization=True
            )
            
            # Differential privacy configuration
            dp_config = DifferentialPrivacyConfig(
                epsilon_total=config.dp_epsilon_total,
                delta=config.dp_delta,
                l2_clip_norm=config.dp_l2_clip_norm,
                num_rounds=config.num_rounds,
                num_clients=num_clients
            ) if config.enable_differential_privacy else None
            
            # Create Shamir instance
            self.shamir = EnhancedShamirSecretSharing(shamir_config) if config.enable_shamir else None
            
            # Create secure aggregation protocol
            self.secure_agg = SecureAggregationProtocol(
                self.shamir, 
                dp_config
            ) if config.enable_shamir else None
            
            # Set mask amplitude
            if self.secure_agg:
                self.secure_agg.mask_amplitude = config.mask_amplitude
            
            logger.info("Privacy-preserving aggregation initialized")
            logger.info(f"  Shamir: {config.shamir_threshold}-of-{num_clients}")
            if dp_config:
                logger.info(f"  DP: ε={dp_config.epsilon_total}, σ={dp_config.noise_multiplier_per_client:.4f}")
        else:
            self.shamir = None
            self.secure_agg = None
            logger.info("Privacy-preserving features disabled")
    
    def _calibrate_layer_dp(self, param_name: str, param_shape: Tuple[int, ...]) -> None:
        """Calibrate differential privacy parameters for specific layer."""
        if (self.privacy_config.enable_differential_privacy and 
            self.privacy_config.dp_per_layer_calibration and
            self.secure_agg and 
            self.secure_agg.layer_dp_manager):
            
            self.secure_agg.layer_dp_manager.calibrate_layer_parameters(param_name, param_shape)
    
    def aggregate_parameters_with_privacy(
        self,
        client_best_params: List[Dict[str, np.ndarray]],
        client_delta_params: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        previous_global_params: Optional[Dict[str, np.ndarray]] = None,
        round_num: int = 0,
        active_client_ids: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate parameters with privacy preservation.
        
        If privacy is disabled, falls back to standard aggregation.
        If privacy is enabled, uses secure aggregation with Shamir + DP.
        """
        if not self.privacy_config.enable_privacy or not self.secure_agg:
            # Fallback to standard aggregation
            return self.aggregate_parameters(
                client_best_params,
                client_delta_params,
                client_weights,
                previous_global_params,
                round_num
            )
        
        logger.info(f"Round {round_num + 1}: Privacy-preserving aggregation")
        
        # Generate round-specific salts
        round_salt = f"round_{round_num}".encode()
        
        # Normalize client weights
        total_weight = sum(client_weights)
        if total_weight <= 0:
            logger.warning("All client weights are zero, using equal weights")
            normalized_weights = [1.0 / len(client_best_params)] * len(client_best_params)
        else:
            normalized_weights = [w / total_weight for w in client_weights]
        
        # Determine which parameters to aggregate
        if self.config.aggregation_method in {'delta_fedavg', 'delta', 'delta-avg'} and round_num > 0:
            params_to_aggregate = client_delta_params
            is_delta_mode = True
        else:
            params_to_aggregate = client_best_params
            is_delta_mode = False
        
        # Get all parameter names
        param_names = list(params_to_aggregate[0].keys()) if params_to_aggregate else []
        
        # Initialize result dictionary
        aggregated_params = {}
        
        # Setup active client IDs
        if active_client_ids is None:
            active_client_ids = list(range(len(params_to_aggregate)))
        
        # Process each parameter separately with secure aggregation
        for param_name in param_names:
            try:
                param_salt = param_name.encode()
                
                # Collect parameter updates from all clients
                client_param_updates = []
                valid_client_ids = []
                valid_owner_indices = []  

                for i, params in enumerate(params_to_aggregate):
                    if param_name in params and params[param_name] is not None:
                        param_array = params[param_name]
                        if param_array.size > 0:
                            client_param_updates.append(param_array.flatten())  
                            valid_client_ids.append(active_client_ids[i])
                            valid_owner_indices.append(i)

                if not client_param_updates:
                    logger.warning(f"No valid updates for {param_name}")
                    continue

                param_weights = np.array([normalized_weights[j] for j in valid_owner_indices], dtype=float)
                sw = float(param_weights.sum())
                if sw <= 0:
                    param_weights[:] = 1.0 / len(param_weights)
                else:
                    param_weights /= sw

                for k in range(len(client_param_updates)):
                    client_param_updates[k] = client_param_updates[k] * param_weights[k]
                
                # Get parameter shape for reconstruction
                original_shape = params_to_aggregate[0][param_name].shape
                
                # Calibrate per-layer DP if enabled
                if self.privacy_config.dp_per_layer_calibration:
                    self._calibrate_layer_dp(param_name, original_shape)
                
                if self.shamir and hasattr(self.privacy_config, "shamir_threshold"):
                    t = int(self.privacy_config.shamir_threshold)
                    n_active = len(valid_client_ids)
                    if t > n_active:
                        raise ValueError(
                            f"Shamir threshold {t} exceeds active clients {n_active} "
                            f"for param '{param_name}'"
                        )                
                
                # Apply secure aggregation with privacy
                masked_updates = []
                all_seed_shares = {}
                
                for i, client_id in enumerate(valid_client_ids):
                    masked_update, seed_shares = self.secure_agg.create_masked_update(
                        client_id=client_id,
                        update=client_param_updates[i],
                        round_salt=round_salt,
                        param_salt=param_salt,
                        active_clients=valid_client_ids,
                        layer_name=param_name
                    )
                    masked_updates.append(masked_update)
                    all_seed_shares[client_id] = seed_shares
                
                # Aggregate masked updates
                aggregated_flat = self.secure_agg.aggregate_updates(
                    masked_updates=masked_updates,
                    active_client_ids=valid_client_ids,
                    all_seed_shares=all_seed_shares,
                    round_salt=round_salt,
                    param_salt=param_salt
                )
                
                # Reshape back to original shape
                aggregated_params[param_name] = aggregated_flat.reshape(original_shape)
                
                logger.debug(f"  {param_name}: aggregated with privacy from {len(valid_client_ids)} clients")
                
            except Exception as e:
                logger.error(f"Privacy aggregation failed for {param_name}: {e}")
                # Fallback to standard aggregation for this parameter
                weighted_arrays = []
                for i, params in enumerate(params_to_aggregate):
                    if param_name in params and params[param_name] is not None:
                        weighted_array = params[param_name] * normalized_weights[i]
                        weighted_arrays.append(weighted_array)
                
                if weighted_arrays:
                    aggregated_params[param_name] = np.sum(weighted_arrays, axis=0)
        
        # Apply delta aggregation if in delta mode
        if is_delta_mode and previous_global_params:
            server_lr = self.config.server_learning_rate
            
            # Apply learning rate decay
            if round_num <= 5:
                actual_lr = server_lr
            elif round_num <= 10:
                actual_lr = server_lr * 0.8
            else:
                actual_lr = server_lr * 0.5
            
            # Gradient clipping
            max_grad_norm = getattr(self.config, 'max_gradient_norm', 5.0)
            
            global_new = {}
            for param_name in param_names:
                if param_name in aggregated_params and param_name in previous_global_params:
                    delta = aggregated_params[param_name]
                    
                    # Clip gradient
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > max_grad_norm:
                        delta = delta * (max_grad_norm / delta_norm)
                    
                    # Apply server learning rate
                    global_new[param_name] = previous_global_params[param_name] + actual_lr * delta
            
            aggregated_params = global_new
        
        # Clean parameters
        aggregated_params = self._clean_aggregated_parameters(aggregated_params)
        
        # Log aggregation statistics
        total_params = sum(p.size for p in aggregated_params.values())
        total_norm = sum(np.linalg.norm(p) for p in aggregated_params.values())
        
        logger.info(f"Privacy-preserving aggregation complete: "
                   f"{len(aggregated_params)} parameters, "
                   f"{total_params:,} total elements, "
                   f"norm={total_norm:.6f}")
        
        return aggregated_params


class PrivacyPreservingTrainer(FederatedTrainer):
    """Enhanced federated trainer with privacy preservation."""
    
    def __init__(self, config: PrivacyFederatedConfig, preprocessor):
        # Update config with actual number of clients before initialization
        if not hasattr(config, 'num_clients') or config.num_clients is None:
            from federated.utils import FederatedDataManager
            data_manager = FederatedDataManager(config.federated_data_dir)
            client_assignments = data_manager.load_client_assignments()
            config.num_clients = len(client_assignments)
            logger.info(f"Detected {config.num_clients} clients from data directory")
        
        # Initialize base trainer
        super().__init__(config, preprocessor)
        
        # Replace aggregator with privacy-preserving version
        self.aggregator = PrivacyPreservingAggregator(config)
        
        logger.info("Privacy-preserving trainer initialized")
    
    def _execute_federated_round(self, active_clients: List[str], round_num: int) -> Dict[str, Any]:
        """Execute federated round with privacy preservation."""
        
        client_best_params = []
        client_delta_params = []
        client_metrics = {}
        client_weights = []
        
        total_clients = len(active_clients)
        logger.info(f"Training {total_clients} clients locally...")
        
        # Local training phase
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
                    
                    logger.info(f"  {client_id}: Best Acc={best_accuracy:.4f}, "
                              f"Test F1={test_f1:.4f}, Time={client_time:.1f}s")
                else:
                    logger.warning(f"  {client_id}: No valid parameters")
            
            except Exception as e:
                logger.error(f"  {client_id}: Training failed - {e}")
                continue
        
        if not client_best_params:
            logger.error("No valid client parameters to aggregate")
            return {}
        
        logger.info(f"Local training complete: {len(client_best_params)}/{total_clients} clients succeeded")
        
        # Privacy-preserving aggregation
        logger.info(f"Aggregating with privacy preservation...")
        agg_start = time.time()
        
        # Preserve previous global parameters
        if self.global_parameters is not None:
            self.previous_global_parameters = deepcopy(self.global_parameters)
        
        # Map client_id strings to integer indices
        active_client_ids = list(range(len(active_clients)))
        
        # Use privacy-preserving aggregation
        if isinstance(self.aggregator, PrivacyPreservingAggregator):
            self.global_parameters = self.aggregator.aggregate_parameters_with_privacy(
                client_best_params=client_best_params,
                client_delta_params=client_delta_params,
                client_weights=client_weights,
                previous_global_params=self.previous_global_parameters,
                round_num=round_num,
                active_client_ids=active_client_ids
            )
        else:
            # Fallback to standard aggregation
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
        
        round_results['privacy_enabled'] = self.config.enable_privacy
        round_results['aggregation_time'] = agg_time
        
        # Track best model
        if 'avg_test_accuracy' in round_results:
            current_accuracy = round_results['avg_test_accuracy']
            if current_accuracy > self.best_global_accuracy:
                self.best_global_accuracy = current_accuracy
                self.best_global_parameters = deepcopy(self.global_parameters)
                self.best_round = round_num
                logger.info(f"New best model: accuracy={current_accuracy:.4f} at round {round_num + 1}")
        
        return round_results


def create_privacy_preserving_trainer(config: PrivacyFederatedConfig, preprocessor) -> PrivacyPreservingTrainer:
    """Factory function for creating privacy-preserving trainer."""
    
    # Load decision threshold with proper priority (CLI > centralized > default)
    try:
        if getattr(config, '_confidence_tau_from_cli', False):
            logger.info(f"Using CLI confidence_tau={config.confidence_tau:.4f} (overrides artifacts)")
        elif hasattr(preprocessor, "decision_threshold") and preprocessor.decision_threshold is not None:
            config.confidence_tau = float(preprocessor.decision_threshold)
            logger.info(f"Using centralized decision threshold: {config.confidence_tau:.4f}")
        else:
            logger.info(f"Using default/configured confidence_tau={config.confidence_tau}")
    except Exception as e:
        logger.warning(f"Could not set confidence_tau: {e}")
    
    # Create trainer
    trainer = PrivacyPreservingTrainer(config, preprocessor)
    
    if not trainer.initialize_clients():
        raise ValueError("Failed to initialize federated clients")
    
    return trainer


# Example usage function
def example_privacy_federated_training():
    """Example of how to use privacy-preserving federated training."""
    
    from pathlib import Path
    from federated.preprocessing import FederatedPreprocessor
    
    # Configuration
    config = PrivacyFederatedConfig(
        # Existing federated parameters
        federated_data_dir=Path("data/federated"),
        centralized_artifacts_dir=Path("artifacts/centralized"),
        output_dir=Path("experiments/privacy_federated"),
        
        horizon_min=600,
        deadband_bps=10.0,
        two_class_mode=True,
        
        num_rounds=15,
        local_epochs=3,
        aggregation_method='delta_fedavg',
        server_learning_rate=0.5,
        
        # Privacy parameters
        enable_privacy=True,
        enable_shamir=True,
        enable_differential_privacy=True,
        
        shamir_threshold=3,
        shamir_prime_bits=61,
        
        dp_epsilon_total=1.0,
        dp_delta=1e-6,
        dp_l2_clip_norm=1.0,
        dp_per_layer_calibration=True,
        
        seed=42
    )
    
    # Load preprocessor
    preprocessor = FederatedPreprocessor(config.centralized_artifacts_dir)
    if not preprocessor.load_centralized_artifacts():
        raise RuntimeError("Failed to load preprocessor")
    
    # Create trainer
    trainer = create_privacy_preserving_trainer(config, preprocessor)
    
    # Run training
    results = trainer.run_federated_training()
    
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Privacy-Preserving Federated Learning Module")
    logger.info("This module integrates Shamir's Secret Sharing and Differential Privacy")
    logger.info("Use create_privacy_preserving_trainer() to create a privacy-enabled trainer")