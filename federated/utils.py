#!/usr/bin/env python3
"""
Federated Learning Utilities Module
====================================================

Utility functions and data management:
- Federated data loading and validation
- Parameter shape validation and cleaning
- Deterministic seed generation
- Configuration helpers
- File I/O utilities

"""

import logging
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib

logger = logging.getLogger(__name__)


def deterministic_seed(s: str, base_seed: int) -> int:
    """
    Generate deterministic seed from string using SHA256
    Ensures reproducible randomization across federated clients.
    """
    h = hashlib.sha256((s + str(base_seed)).encode()).hexdigest()
    return int(h[:8], 16)


def validate_parameter_shapes(params1: Dict[str, np.ndarray], 
                             params2: Dict[str, np.ndarray]) -> bool:
    """
    Validate that two parameter dictionaries have matching shapes
    """
    try:
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1.keys():
            if params1[key].shape != params2[key].shape:
                return False
        
        return True
        
    except Exception:
        return False


def clean_parameters(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Clean parameters by removing NaN/inf values 
    """
    cleaned = {}
    
    for param_name, param_array in params.items():
        if param_array is None:
            continue
            
        if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
            logger.warning(f"Cleaning invalid values in parameter {param_name}")
            cleaned_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
            cleaned[param_name] = cleaned_array
        else:
            cleaned[param_name] = param_array.copy()
    
    return cleaned


def calculate_parameter_stats(params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Calculate parameter statistics for monitoring and debugging."""
    if not params:
        return {}
    
    stats = {
        'total_parameters': sum(p.size for p in params.values()),
        'total_norm': float(sum(np.linalg.norm(p) for p in params.values())),
        'parameter_count': len(params),
        'per_parameter_stats': {}
    }
    
    for param_name, param_array in params.items():
        param_stats = {
            'shape': param_array.shape,
            'size': int(param_array.size),
            'norm': float(np.linalg.norm(param_array)),
            'mean': float(np.mean(param_array)),
            'std': float(np.std(param_array)),
            'min': float(np.min(param_array)),
            'max': float(np.max(param_array))
        }
        stats['per_parameter_stats'][param_name] = param_stats
    
    return stats


class FederatedDataManager:
    """
    Manages federated data loading and validation
    """
    
    def __init__(self, federated_data_dir: Path):
        self.federated_data_dir = Path(federated_data_dir)
        self.meta_info: Optional[Dict[str, Any]] = None
        
    def load_client_assignments(self) -> Dict[str, List[str]]:
        """Load client assignments from meta.json."""
        meta_path = self.federated_data_dir / 'meta.json'
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            self.meta_info = meta
            client_assignments = meta.get('assignment', {})
            
            logger.info(f"Loaded federated meta: {len(client_assignments)} clients")
            
            # Log data distribution for non-IID analysis
            if 'loads' in meta:
                total_samples = sum(meta['loads'].values())
                logger.info("Client data distribution:")
                for client_id, load in meta['loads'].items():
                    percentage = (load / total_samples) * 100 if total_samples > 0 else 0
                    logger.info(f"  {client_id}: {load:,} samples ({percentage:.1f}%)")
            
            return client_assignments
            
        except Exception as e:
            raise RuntimeError(f"Failed to load client assignments: {e}")
    
    def validate_client_data(self, client_id: str) -> bool:
        """Validate that client data exists and is readable."""
        client_data_path = self.federated_data_dir / client_id / 'unified_dataset.parquet'
        
        if not client_data_path.exists():
            logger.error(f"Client {client_id} data not found: {client_data_path}")
            return False
        
        try:
            df = pd.read_parquet(client_data_path).head(1)  
            if len(df) == 0:
                logger.error(f"Client {client_id} data is empty")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Client {client_id} data validation failed: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of federated data distribution."""
        if not self.meta_info:
            return {}
        
        summary = {
            'total_clients': len(self.meta_info.get('assignment', {})),
            'data_distribution': self.meta_info.get('loads', {}),
            'symbols_per_client': self.meta_info.get('symbols_per_client', {}),
            'split_method': self.meta_info.get('split_method', 'unknown')
        }
        
        # Calculate statistics
        if 'loads' in self.meta_info:
            loads = list(self.meta_info['loads'].values())
            summary.update({
                'total_samples': sum(loads),
                'avg_samples_per_client': np.mean(loads),
                'std_samples_per_client': np.std(loads),
                'min_samples_per_client': min(loads),
                'max_samples_per_client': max(loads)
            })
        
        return summary


class ConfigurationHelper:
    """Helper for configuration validation and management."""
    
    @staticmethod
    def validate_config(config) -> List[str]:
        """Validate federated configuration and return list of issues."""
        issues = []
        
        # Check required paths
        required_paths = ['federated_data_dir', 'centralized_artifacts_dir', 'output_dir']
        for path_attr in required_paths:
            if not hasattr(config, path_attr):
                issues.append(f"Missing required path: {path_attr}")
            else:
                path_value = getattr(config, path_attr)
                if not isinstance(path_value, Path):
                    issues.append(f"{path_attr} must be a Path object")
        
        # Check numeric parameters
        numeric_checks = [
            ('num_rounds', 1, 100),
            ('local_epochs', 1, 20),
            ('server_learning_rate', 0.01, 2.0),
            ('horizon_min', 1, 60),
            ('deadband_bps', 0.1, 20.0)
        ]
        
        for param_name, min_val, max_val in numeric_checks:
            if hasattr(config, param_name):
                value = getattr(config, param_name)
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    issues.append(f"{param_name} must be between {min_val} and {max_val}")
        
        # Check aggregation method
        valid_methods = ['fedavg', 'delta_fedavg', 'fedavgm', 'coordinate_median', 'trimmed_mean']
        if hasattr(config, 'aggregation_method'):
            if config.aggregation_method not in valid_methods:
                issues.append(f"aggregation_method must be one of {valid_methods}")
        
        # Check trimmed mean ratio if applicable
        if (hasattr(config, 'aggregation_method') and 
            config.aggregation_method == 'trimmed_mean' and
            hasattr(config, 'trimmed_mean_ratio')):
            ratio = config.trimmed_mean_ratio
            if not (0.0 <= ratio <= 0.4):
                issues.append("trimmed_mean_ratio must be between 0.0 and 0.4")
        
        return issues
    
    @staticmethod
    def create_default_mlp_config() -> Dict[str, Any]:
        """Create default MLP configuration """
        return {
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'max_iter': 1,
            'early_stopping': False,
            'warm_start': True
        }
    
    @staticmethod
    def save_config(config, output_path: Path) -> None:
        """Save configuration to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        if hasattr(config, '__dict__'):
            config_dict = {}
            for key, value in config.__dict__.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif isinstance(value, (tuple, list)):
                    config_dict[key] = list(value)
                elif isinstance(value, np.ndarray):
                    config_dict[key] = value.tolist()
                else:
                    config_dict[key] = value
        else:
            config_dict = dict(config)
        
        # Add metadata
        config_dict['version'] = '4.0_refactored'
        config_dict['timestamp'] = pd.Timestamp.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")


class ModelStateManager:
    """Manages model state saving/loading for federated training."""
    
    @staticmethod
    def save_global_model(model, output_path: Path) -> None:
        """Save global model to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logger.info(f"Global model saved to {output_path}")
    
    @staticmethod
    def save_parameters(params: Dict[str, np.ndarray], output_path: Path) -> None:
        """Save parameters dictionary to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(params, output_path)
        logger.debug(f"Parameters saved to {output_path}")
    
    @staticmethod
    def load_parameters(input_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Load parameters dictionary from file."""
        if not input_path.exists():
            logger.warning(f"Parameters file not found: {input_path}")
            return None
        
        try:
            params = joblib.load(input_path)
            logger.debug(f"Parameters loaded from {input_path}")
            return params
        except Exception as e:
            logger.error(f"Failed to load parameters from {input_path}: {e}")
            return None
    
    @staticmethod
    def save_training_history(history: List[Dict[str, Any]], output_path: Path) -> None:
        """Save training history to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if history:
            df = pd.DataFrame(history)
            df.to_csv(output_path, index=False)
            logger.info(f"Training history saved to {output_path}")


class MetricsCalculator:
    """Utility class for metrics calculation """
    
    @staticmethod
    def calculate_improvement_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate improvement metrics from training history."""
        if len(history) < 2:
            return {}
        
        initial = history[0]
        final = history[-1]
        
        metrics = {}
        
        # F1 improvements
        for metric_name in ['avg_val_f1', 'avg_test_f1', 'avg_best_val_f1']:
            if metric_name in initial and metric_name in final:
                improvement = final[metric_name] - initial[metric_name]
                metrics[f'{metric_name}_improvement'] = float(improvement)
        
        # Accuracy improvements
        for metric_name in ['avg_val_accuracy', 'avg_test_accuracy']:
            if metric_name in initial and metric_name in final:
                improvement = final[metric_name] - initial[metric_name]
                metrics[f'{metric_name}_improvement'] = float(improvement)
        
        # Trading improvements (if available)
        if 'avg_test_net_profit_bps' in initial and 'avg_test_net_profit_bps' in final:
            profit_improvement = final['avg_test_net_profit_bps'] - initial['avg_test_net_profit_bps']
            metrics['profit_improvement_bps'] = float(profit_improvement)
        
        # Calculate convergence indicators
        if len(history) >= 3:
            recent_f1 = [h.get('avg_test_f1', 0) for h in history[-3:]]
            f1_variance = float(np.var(recent_f1))
            metrics['recent_f1_variance'] = f1_variance
            metrics['converged'] = f1_variance < 0.001  # Low variance indicates convergence
        
        return metrics
    
    @staticmethod
    def calculate_client_contribution_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate client contribution statistics."""
        if not history:
            return {}
        
        # Extract client participation info
        all_clients = set()
        client_weights_over_time = {}
        
        for round_data in history:
            participating = round_data.get('participating_clients', [])
            all_clients.update(participating)
            
            client_weights = round_data.get('client_weights', {})
            for client_id in all_clients:
                if client_id not in client_weights_over_time:
                    client_weights_over_time[client_id] = []
                
                weight = client_weights.get(client_id, 0.0)
                client_weights_over_time[client_id].append(weight)
        
        # Calculate statistics
        stats = {
            'total_unique_clients': len(all_clients),
            'avg_participation_per_round': np.mean([len(h.get('participating_clients', [])) for h in history]),
            'client_weight_variance': {}
        }
        
        for client_id, weights in client_weights_over_time.items():
            if weights:
                stats['client_weight_variance'][client_id] = float(np.var(weights))
        
        # Find most consistent and dominant clients
        if client_weights_over_time:
            avg_weights = {cid: np.mean(weights) for cid, weights in client_weights_over_time.items() if weights}
            weight_vars = {cid: np.var(weights) for cid, weights in client_weights_over_time.items() if weights}
            
            if avg_weights:
                stats['dominant_client'] = max(avg_weights, key=avg_weights.get)
                stats['dominant_client_avg_weight'] = float(max(avg_weights.values()))
            
            if weight_vars:
                stats['most_consistent_client'] = min(weight_vars, key=weight_vars.get)
                stats['most_consistent_client_variance'] = float(min(weight_vars.values()))
        
        return stats


def create_experiment_directory(base_dir: Path, experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{timestamp}_{experiment_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'config').mkdir(exist_ok=True)
    (exp_dir / 'rounds').mkdir(exist_ok=True)
    (exp_dir / 'artifacts').mkdir(exist_ok=True)
    (exp_dir / 'analysis').mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def setup_federated_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Setup logging for federated training."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Federated logging setup complete: {log_file}")


# Convenience functions for common operations
def load_federated_preprocessor(artifacts_dir: Path):
    """Load federated preprocessor from centralized artifacts."""
    try:
        from .preprocessing import FederatedPreprocessor
        
        preprocessor = FederatedPreprocessor(artifacts_dir)
        success = preprocessor.load_centralized_artifacts()
        
        if success:
            logger.info("Federated preprocessor loaded successfully")
            return preprocessor
        else:
            raise RuntimeError("Failed to load centralized artifacts")
            
    except ImportError as e:
        logger.error(f"Cannot import FederatedPreprocessor: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load federated preprocessor: {e}")
        return None


def validate_federated_setup(federated_data_dir: Path, 
                            centralized_artifacts_dir: Path) -> Tuple[bool, List[str]]:
    """Validate federated learning setup and return issues."""
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