#!/usr/bin/env python3
"""
Parameter Aggregation Module
==============================================

Centralized parameter aggregation methods:
- Delta-based aggregation with server learning rate
- FedAvgM with server momentum
- Robust aggregation methods (coordinate median, trimmed mean)
- Enhanced parameter validation and cleaning

"""

import logging
import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class ParameterAggregator:
    """
    Centralized parameter aggregation 
    """
    
    def __init__(self, config):
        self.config = config
        self.server_velocity = None
        self.momentum_buffer = {}  
        self.round_counter = 0  
        
    def aggregate_parameters(
        self,
        client_best_params: List[Dict[str, np.ndarray]],
        client_delta_params: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        previous_global_params: Optional[Dict[str, np.ndarray]] = None,
        round_num: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Main aggregation router 
        """
        if not client_best_params:
            return {}
    
        # Adaptive server learning rate based on round
        self.round_counter = round_num
        base_lr = float(self.config.server_learning_rate)
        if round_num < 5:
            lr_multiplier = 1.0
        elif round_num < 10:
            lr_multiplier = 0.8
        else:
            lr_multiplier = 0.5
        self._current_server_lr = base_lr * lr_multiplier


        # Normalize weights
        total_weight = float(sum(client_weights))
        if total_weight <= 0:
            logger.warning("All client weights are zero, using equal weights")
            weights = [1.0 / len(client_best_params)] * len(client_best_params)
        else:
            weights = [w / total_weight for w in client_weights]
        
        # Route to appropriate aggregation method
        method = self.config.aggregation_method.lower().strip()

        if method in {'delta_fedavg', 'delta', 'delta-avg'}:
            if round_num == 0 or previous_global_params is None:
                logger.info(f"Round {round_num + 1}: Delta mode warm-start via FedAvg")
                result = self._federated_average(client_best_params, weights)
            else:
                result = self._delta_federated_average(
                    previous_global_params, client_delta_params, weights, round_num
                )
        elif method in {'fedavgm', 'server_momentum', 'momentum'}:
            result = self._fedavg_with_momentum(client_best_params, weights)
        elif method in {'fedavg', 'avg', 'weighted_avg'}:
            result = self._federated_average(client_best_params, weights)
        elif method in {'coordinate_median', 'coord_median', 'median'}:
            result = self._coordinate_median(client_best_params)
        elif method in {'trimmed_mean', 'trim_mean', 'tmean'}:
            result = self._trimmed_mean(client_best_params, self.config.trimmed_mean_ratio)
        else:
            logger.error(f"Unknown aggregation method: {method}")
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        # Post-process and validate
        result = self._clean_aggregated_parameters(result)
        self._log_aggregation_statistics(result, method)
        
        return result
    
    def _delta_federated_average(
        self,
        global_prev: Optional[Dict[str, np.ndarray]],
        client_deltas: List[Dict[str, np.ndarray]],
        weights: List[float],
        round_num: int
    ) -> Dict[str, np.ndarray]:
        """
        Delta-based federated averaging with server learning rate 
        global_new = global_prev + server_lr * weighted_avg(client_deltas)
        """
        if not global_prev:
            logger.warning("No previous global found in delta aggregation — returning empty dict")
            return {}
        
        if not client_deltas or all(not delta for delta in client_deltas):
            logger.warning("No valid deltas for aggregation, using previous global parameters")
            return deepcopy(global_prev) if global_prev else {}
        
        logger.info(f"Round {round_num + 1}: Delta aggregation with server LR={self.config.server_learning_rate}")
        
        # Aggregate deltas using weighted average
        aggregated_deltas = {}
        param_names = list(global_prev.keys())
        
        for param_name in param_names:
            weighted_deltas = []
            valid_clients = 0
            
            for i, delta_params in enumerate(client_deltas):
                if param_name in delta_params and delta_params[param_name] is not None:
                    delta = delta_params[param_name]
                    if delta.shape == global_prev[param_name].shape:
                        weighted_delta = delta * weights[i]
                        weighted_deltas.append(weighted_delta)
                        valid_clients += 1
            
            if weighted_deltas:
                avg_delta = np.sum(weighted_deltas, axis=0)
                aggregated_deltas[param_name] = avg_delta
                
                delta_norm = np.linalg.norm(avg_delta)
                logger.debug(f"Delta {param_name}: norm={delta_norm:.6f} from {valid_clients} clients")
            else:
                logger.warning(f"No valid deltas for {param_name}")
                aggregated_deltas[param_name] = np.zeros_like(global_prev[param_name])
        
        # Apply server learning rate with decay and gradient clipping
        base_server_lr = self.config.server_learning_rate
        
        # Learning rate decay schedule
        if round_num <= 5:
            server_lr = getattr(self, "_current_server_lr", self.config.server_learning_rate)
        elif round_num <= 10:
            server_lr = base_server_lr * 0.8
        else:
            server_lr = base_server_lr * 0.5
        
        global_new = {}
        total_delta_norm = 0.0
        max_grad_norm = getattr(self.config, 'max_gradient_norm', 5.0)
        
        for param_name in param_names:
            if param_name in aggregated_deltas:
                delta = aggregated_deltas[param_name]
                
                # Gradient clipping per parameter
                delta_norm = np.linalg.norm(delta)
                if delta_norm > max_grad_norm:
                    delta = delta * (max_grad_norm / delta_norm)
                    logger.debug(f"Clipped {param_name}: {delta_norm:.2f} -> {max_grad_norm:.2f}")
                
                delta_scaled = server_lr * delta
                global_new[param_name] = global_prev[param_name] + delta_scaled
                
                total_delta_norm += np.linalg.norm(delta_scaled)
        
        logger.info(f"Delta aggregation: round={round_num}, server_lr={server_lr:.3f}, "
                    f"total_delta_norm={total_delta_norm:.6f}")
        return global_new
    
    def _fedavg_with_momentum(
        self,
        client_params: List[Dict[str, np.ndarray]],
        weights: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        FedAvgM: Server momentum aggregation (ИСПРАВЛЕНО).
        """
        logger.info("Using FedAvgM (server momentum) aggregation")
        
        # First compute standard FedAvg
        fedavg_result = self._federated_average(client_params, weights)
        
        if not fedavg_result:
            return {}

        if not hasattr(self, '_previous_global_params') or self._previous_global_params is None:
            self._previous_global_params = deepcopy(fedavg_result)
            if self.server_velocity is None:
                self.server_velocity = {k: np.zeros_like(v) for k, v in fedavg_result.items()}
            return fedavg_result

        momentum = getattr(self.config, 'server_momentum', 0.9)
        result = {}
        
        for param_name in fedavg_result.keys():
            if param_name in self._previous_global_params and param_name in self.server_velocity:
                update = fedavg_result[param_name] - self._previous_global_params[param_name]

                self.server_velocity[param_name] = (
                    momentum * self.server_velocity[param_name] + update
                )

                result[param_name] = self._previous_global_params[param_name] + self.server_velocity[param_name]
            else:
                result[param_name] = fedavg_result[param_name]
                if param_name not in self.server_velocity:
                    self.server_velocity[param_name] = np.zeros_like(fedavg_result[param_name])

        self._previous_global_params = deepcopy(result)
        
        logger.debug(f"Applied server momentum (β={momentum})")
        return result
    
    def _federated_average(
        self,
        client_params: List[Dict[str, np.ndarray]],
        weights: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Standard FedAvg: weighted average of client parameters 
        """
        if not client_params:
            return {}
        
        logger.debug("Using standard FedAvg aggregation")
        
        aggregated_params = {}
        param_names = list(client_params[0].keys())
        
        for param_name in param_names:
            weighted_arrays = []
            valid_clients = 0
            
            for i, params in enumerate(client_params):
                if param_name in params and params[param_name] is not None:
                    param_array = params[param_name]
                    if param_array.size > 0:
                        # Clean invalid values before weighting
                        if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                            param_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        weighted_array = param_array * weights[i]
                        weighted_arrays.append(weighted_array)
                        valid_clients += 1
            
            if weighted_arrays:
                aggregated_params[param_name] = np.sum(weighted_arrays, axis=0)
                logger.debug(f"Aggregated {param_name} from {valid_clients}/{len(client_params)} clients")
            else:
                logger.warning(f"No valid arrays for parameter {param_name}")
        
        return aggregated_params
    
    def _coordinate_median(
        self,
        client_params: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate-wise median aggregation
        """
        if not client_params:
            return {}
        
        logger.debug("Using coordinate median aggregation")
        
        aggregated_params = {}
        param_names = list(client_params[0].keys())
        
        for param_name in param_names:
            arrays = []
            
            for params in client_params:
                if param_name in params and params[param_name] is not None:
                    param_array = params[param_name]
                    # Clean invalid values
                    if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                        param_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
                    arrays.append(param_array)
            
            if arrays:
                stacked = np.stack(arrays, axis=0)
                aggregated_params[param_name] = np.median(stacked, axis=0)
                logger.debug(f"Coordinate median for {param_name} from {len(arrays)} clients")
        
        return aggregated_params
    
    def _trimmed_mean(
        self,
        client_params: List[Dict[str, np.ndarray]],
        trim_ratio: float
    ) -> Dict[str, np.ndarray]:
        """
        Trimmed mean aggregation
        """
        if not client_params:
            return {}
        
        logger.debug(f"Using trimmed mean aggregation (trim_ratio={trim_ratio})")
        
        n_clients = len(client_params)
        trim_count = int(n_clients * trim_ratio)
        
        # Ensure we don't trim too much
        if trim_count * 2 >= n_clients:
            logger.warning(f"Trim ratio {trim_ratio} too large for {n_clients} clients, using standard mean")
            trim_count = 0
        
        aggregated_params = {}
        param_names = list(client_params[0].keys())
        
        for param_name in param_names:
            arrays = []
            
            for params in client_params:
                if param_name in params and params[param_name] is not None:
                    param_array = params[param_name]
                    # Clean invalid values
                    if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                        param_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
                    arrays.append(param_array)
            
            if arrays:
                stacked = np.stack(arrays, axis=0)
                
                if trim_count > 0:
                    # Sort along client axis and trim extremes
                    sorted_array = np.sort(stacked, axis=0)
                    trimmed = sorted_array[trim_count:-trim_count]
                    aggregated_params[param_name] = np.mean(trimmed, axis=0)
                    logger.debug(f"Trimmed mean for {param_name}: trimmed {trim_count*2}/{n_clients} clients")
                else:
                    aggregated_params[param_name] = np.mean(stacked, axis=0)
        
        return aggregated_params
    
    def _create_shared_template(self, template_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Create shared global template to avoid averaging random initializations 
        """
        if not template_params:
            return {}
        
        logger.info("Creating shared global template with consistent initialization")
        
        global_template = {}
        rng = np.random.RandomState(self.config.model_seed)
        
        for param_name, param_array in template_params.items():
            is_weight = ('weights' in param_name) or ('coefs' in param_name)
            is_bias   = ('bias' in param_name) or ('intercepts' in param_name)

            if is_weight or (not is_bias):
                # Xavier/Glorot initialization
                if len(param_array.shape) >= 2:
                    fan_in, fan_out = param_array.shape[0], param_array.shape[1]
                    limit = np.sqrt(6.0 / (fan_in + fan_out))
                    global_template[param_name] = rng.uniform(
                        -limit, limit, param_array.shape
                    ).astype(param_array.dtype)
                else:
                    std = np.sqrt(2.0 / max(param_array.shape[0], 1))
                    global_template[param_name] = rng.normal(
                        0, std, param_array.shape
                    ).astype(param_array.dtype)
            else:
                global_template[param_name] = np.zeros_like(param_array)

        
        return global_template
    
    def _clean_aggregated_parameters(self, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clean aggregated parameters 
        """
        cleaned_params = {}
        
        for param_name, param_array in params.items():
            if param_array is None:
                continue
                
            # Clean invalid values
            if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                logger.warning(f"Cleaning invalid values in aggregated parameter {param_name}")
                cleaned_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
                cleaned_params[param_name] = cleaned_array
            else:
                cleaned_params[param_name] = param_array
        
        return cleaned_params
    
    def _log_aggregation_statistics(self, params: Dict[str, np.ndarray], method: str) -> None:
        """Log aggregation statistics for monitoring."""
        if not params:
            return
        
        total_params = sum(p.size for p in params.values())
        total_norm = sum(np.linalg.norm(p) for p in params.values())
        
        logger.info(f"Aggregation ({method}): {len(params)} parameter tensors, "
                   f"{total_params:,} total parameters, "
                   f"total norm: {total_norm:.6f}")
        
        # Log per-parameter statistics
        for param_name, param_array in params.items():
            param_norm = np.linalg.norm(param_array)
            param_mean = np.mean(param_array)
            param_std = np.std(param_array)
            
            logger.debug(f"  {param_name}: shape={param_array.shape}, "
                        f"norm={param_norm:.6f}, "
                        f"mean={param_mean:.6f}, "
                        f"std={param_std:.6f}")


def create_parameter_aggregator(config) -> ParameterAggregator:
    """Factory function for creating parameter aggregator."""
    return ParameterAggregator(config)


# Standalone aggregation functions for direct use
def federated_average(client_params: List[Dict[str, np.ndarray]], 
                     weights: List[float]) -> Dict[str, np.ndarray]:
    """Standalone FedAvg function."""
    class DummyConfig:
        aggregation_method = 'fedavg'
        
    aggregator = ParameterAggregator(DummyConfig())
    return aggregator._federated_average(client_params, weights)


def delta_federated_average(previous_global: Dict[str, np.ndarray],
                           client_deltas: List[Dict[str, np.ndarray]],
                           weights: List[float],
                           server_lr: float = 1.0) -> Dict[str, np.ndarray]:
    """Standalone delta FedAvg function."""
    class DummyConfig:
        aggregation_method = 'delta_fedavg'
        server_learning_rate = server_lr
        model_seed = 42
        
    aggregator = ParameterAggregator(DummyConfig())
    return aggregator._delta_federated_average(previous_global, client_deltas, weights, 1)


def coordinate_median(client_params: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Standalone coordinate median function."""
    class DummyConfig:
        aggregation_method = 'coordinate_median'
        
    aggregator = ParameterAggregator(DummyConfig())
    return aggregator._coordinate_median(client_params)


def trimmed_mean(client_params: List[Dict[str, np.ndarray]], 
                trim_ratio: float = 0.1) -> Dict[str, np.ndarray]:
    """Standalone trimmed mean function."""
    class DummyConfig:
        aggregation_method = 'trimmed_mean'
        trimmed_mean_ratio = trim_ratio
        
    aggregator = ParameterAggregator(DummyConfig())
    return aggregator._trimmed_mean(client_params, trim_ratio)