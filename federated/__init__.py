"""
Federated Learning Package 
"""

from .core import FederatedConfig, FederatedTrainer, create_federated_trainer
from .client import FederatedClient
from .aggregation import ParameterAggregator, create_parameter_aggregator
from .preprocessing import FederatedPreprocessor 
from .utils import (
    FederatedDataManager, 
    ConfigurationHelper, 
    ModelStateManager,
    deterministic_seed,
    validate_parameter_shapes,
    validate_federated_setup,
    setup_federated_logging
)

__version__ = "4.0.0"
__all__ = [
    'FederatedConfig',
    'FederatedTrainer', 
    'FederatedClient',
    'ParameterAggregator',
    'FederatedPreprocessor',
    'FederatedDataManager',
    'create_federated_trainer',
    'create_parameter_aggregator',
    'setup_federated_logging',
    'validate_federated_setup'
]