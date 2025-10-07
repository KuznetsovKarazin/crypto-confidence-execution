#!/usr/bin/env python3
"""
Test Privacy-Preserving Federated Learning Integration
=======================================================

Comprehensive tests for privacy module integration.
"""

import logging
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from federated_privacy import (
    PrivacyFederatedConfig,
    PrivacyPreservingAggregator,
    PrivacyPreservingTrainer
)
from utils.enhanced_shamir_privacy import (
    EnhancedShamirSecretSharing,
    SecureAggregationProtocol,
    ShamirConfig,
    DifferentialPrivacyConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_shamir_basic():
    """Test 1: Basic Shamir Secret Sharing"""
    print("\n" + "="*80)
    print("TEST 1: Basic Shamir Secret Sharing")
    print("="*80)
    
    config = ShamirConfig(threshold=3, num_participants=5)
    shamir = EnhancedShamirSecretSharing(config)
    
    # Test various values
    test_values = [0.0, 1.0, -1.0, 3.14159, 1000.0, -500.0]
    
    passed = 0
    for val in test_values:
        shares = shamir.create_shares(val)
        reconstructed = shamir.reconstruct_secret(shares)
        error = abs(reconstructed[0] - val)
        
        status = "PASS" if error < 1e-4 else "FAIL"
        print(f"  Value: {val:10.4f} | Reconstructed: {reconstructed[0]:10.4f} | Error: {error:.2e} | {status}")
        
        if error < 1e-4:
            passed += 1
    
    print(f"\nResult: {passed}/{len(test_values)} tests passed")
    return passed == len(test_values)


def test_differential_privacy():
    """Test 2: Differential Privacy Mechanism"""
    print("\n" + "="*80)
    print("TEST 2: Differential Privacy")
    print("="*80)
    
    config = ShamirConfig(threshold=3, num_participants=5)
    dp_config = DifferentialPrivacyConfig(epsilon_total=1.0, num_rounds=10, num_clients=5)
    
    shamir = EnhancedShamirSecretSharing(config)
    secure_agg = SecureAggregationProtocol(shamir, dp_config)
    
    # Test DP noise application
    update = np.random.normal(0, 0.1, 100)
    
    # Apply DP without layer name (uses global parameters)
    dp_update = secure_agg.apply_differential_privacy(update)
    
    noise = dp_update - update
    noise_norm = np.linalg.norm(noise)
    
    # Expected noise level (approximately)
    expected_noise = dp_config.noise_multiplier_per_client * np.sqrt(len(update))
    
    print(f"  Original update norm: {np.linalg.norm(update):.4f}")
    print(f"  DP update norm: {np.linalg.norm(dp_update):.4f}")
    print(f"  Added noise norm: {noise_norm:.4f}")
    print(f"  Expected noise level: {expected_noise:.4f}")
    
    # Check if noise is in reasonable range (within 5x of expected)
    status = "PASS" if noise_norm < expected_noise * 5 else "FAIL"
    print(f"\nResult: {status}")
    
    return noise_norm < expected_noise * 5


def test_secure_aggregation():
    """Test 3: Secure Aggregation with Mask Cancellation"""
    print("\n" + "="*80)
    print("TEST 3: Secure Aggregation")
    print("="*80)
    
    num_clients = 5
    update_size = 50
    
    config = ShamirConfig(threshold=3, num_participants=num_clients)
    dp_config = DifferentialPrivacyConfig(epsilon_total=1.0, num_rounds=10, num_clients=num_clients)
    
    shamir = EnhancedShamirSecretSharing(config)
    secure_agg = SecureAggregationProtocol(shamir, dp_config)
    
    # Create client updates
    rng = np.random.default_rng(42)
    client_updates = [rng.normal(0, 0.1, update_size) for _ in range(num_clients)]
    
    round_salt = b"test_round"
    param_salt = b"test_param"
    
    # Create masked updates
    masked_updates = []
    all_seed_shares = {}
    
    for client_id in range(num_clients):
        masked_update, seed_shares = secure_agg.create_masked_update(
            client_id=client_id,
            update=client_updates[client_id],
            round_salt=round_salt,
            param_salt=param_salt,
            active_clients=list(range(num_clients)),
            layer_name="test_layer"
        )
        masked_updates.append(masked_update)
        all_seed_shares[client_id] = seed_shares
    
    # Aggregate
    aggregated = secure_agg.aggregate_updates(
        masked_updates=masked_updates,
        active_client_ids=list(range(num_clients))
    )
    
    # Compare with direct sum
    direct_sum = np.sum(client_updates, axis=0)
    aggregation_error = np.linalg.norm(aggregated - direct_sum)
    
    # Expected noise from DP (L2 formula)
    expected_noise = dp_config.noise_multiplier_per_client * np.sqrt(num_clients * update_size)
    
    print(f"  Direct sum norm: {np.linalg.norm(direct_sum):.4f}")
    print(f"  Aggregated norm: {np.linalg.norm(aggregated):.4f}")
    print(f"  Aggregation error: {aggregation_error:.4f}")
    print(f"  Expected DP noise: {expected_noise:.4f}")
    
    # Error should be comparable to DP noise (within 10x)
    status = "PASS" if aggregation_error < expected_noise * 10 else "FAIL"
    print(f"\nResult: {status}")
    
    return aggregation_error < expected_noise * 10


def test_dropout_recovery():
    """Test 4: Dropout Recovery with Seed Reconstruction"""
    print("\n" + "="*80)
    print("TEST 4: Dropout Recovery")
    print("="*80)
    
    num_clients = 5
    update_size = 50
    
    config = ShamirConfig(threshold=3, num_participants=num_clients)
    dp_config = DifferentialPrivacyConfig(epsilon_total=1.0, num_rounds=10, num_clients=num_clients)
    
    shamir = EnhancedShamirSecretSharing(config)
    secure_agg = SecureAggregationProtocol(shamir, dp_config)
    
    # Create client updates
    rng = np.random.default_rng(123)
    client_updates = [rng.normal(0, 0.1, update_size) for _ in range(num_clients)]
    
    round_salt = b"test_round"
    param_salt = b"test_param"
    
    # Create masked updates for all clients
    masked_updates = []
    all_seed_shares = {}
    
    for client_id in range(num_clients):
        masked_update, seed_shares = secure_agg.create_masked_update(
            client_id=client_id,
            update=client_updates[client_id],
            round_salt=round_salt,
            param_salt=param_salt,
            active_clients=list(range(num_clients)),
            layer_name="test_layer"
        )
        masked_updates.append(masked_update)
        all_seed_shares[client_id] = seed_shares
    
    # Simulate 2 clients dropping out
    surviving_clients = [0, 1, 2]
    surviving_masked_updates = [masked_updates[i] for i in surviving_clients]
    surviving_original_updates = [client_updates[i] for i in surviving_clients]
    
    # Aggregate with dropout recovery
    partial_aggregated = secure_agg.aggregate_updates(
        masked_updates=surviving_masked_updates,
        active_client_ids=surviving_clients,
        all_seed_shares=all_seed_shares,
        round_salt=round_salt,
        param_salt=param_salt
    )
    
    # Compare with partial direct sum
    partial_direct_sum = np.sum(surviving_original_updates, axis=0)
    partial_error = np.linalg.norm(partial_aggregated - partial_direct_sum)
    
    # Expected noise for 3 clients
    expected_partial_noise = dp_config.noise_multiplier_per_client * np.sqrt(len(surviving_clients) * update_size)
    
    print(f"  Surviving clients: {surviving_clients}")
    print(f"  Dropped clients: [3, 4]")
    print(f"  Partial sum norm: {np.linalg.norm(partial_direct_sum):.4f}")
    print(f"  Partial aggregated norm: {np.linalg.norm(partial_aggregated):.4f}")
    print(f"  Partial error: {partial_error:.4f}")
    print(f"  Expected noise (3 clients): {expected_partial_noise:.4f}")
    
    # Error should be comparable to expected noise
    status = "PASS" if partial_error < expected_partial_noise * 10 else "FAIL"
    print(f"\nResult: {status}")
    
    return partial_error < expected_partial_noise * 10


def test_privacy_aggregator():
    """Test 5: Privacy-Preserving Aggregator"""
    print("\n" + "="*80)
    print("TEST 5: Privacy-Preserving Aggregator")
    print("="*80)
    
    # Create minimal config
    config = PrivacyFederatedConfig(
        federated_data_dir=Path("data/federated"),
        centralized_artifacts_dir=Path("artifacts/centralized"),
        output_dir=Path("experiments/test"),
        
        enable_privacy=True,
        enable_shamir=True,
        enable_differential_privacy=True,
        
        shamir_threshold=3,
        num_clients=5,
        dp_epsilon_total=1.0,
        num_rounds=10,
        
        aggregation_method='fedavg'
    )
    
    try:
        aggregator = PrivacyPreservingAggregator(config)
        
        # Verify components initialized
        assert aggregator.shamir is not None, "Shamir not initialized"
        assert aggregator.secure_agg is not None, "Secure aggregation not initialized"
        
        print("  Aggregator initialized successfully")
        print(f"  Shamir: {config.shamir_threshold}-of-{config.num_clients}")
        print(f"  DP enabled: {config.enable_differential_privacy}")
        
        # Test parameter aggregation
        num_clients = 5
        param_shape = (10, 5)
        
        # Create dummy client parameters
        client_params = []
        for _ in range(num_clients):
            params = {
                'layer_0_weights': np.random.normal(0, 0.1, param_shape),
                'layer_0_bias': np.random.normal(0, 0.01, param_shape[1])
            }
            client_params.append(params)
        
        # Create dummy deltas (same as params for this test)
        client_deltas = [p.copy() for p in client_params]
        
        # Client weights
        client_weights = [1.0] * num_clients
        
        # Aggregate
        aggregated = aggregator.aggregate_parameters_with_privacy(
            client_best_params=client_params,
            client_delta_params=client_deltas,
            client_weights=client_weights,
            previous_global_params=None,
            round_num=0,
            active_client_ids=list(range(num_clients))
        )
        
        # Verify output
        assert 'layer_0_weights' in aggregated, "Missing weights in aggregated params"
        assert 'layer_0_bias' in aggregated, "Missing bias in aggregated params"
        assert aggregated['layer_0_weights'].shape == param_shape, "Wrong shape for weights"
        
        print("  Parameter aggregation successful")
        print(f"  Aggregated parameters: {list(aggregated.keys())}")
        print(f"  Weights shape: {aggregated['layer_0_weights'].shape}")
        print(f"  Bias shape: {aggregated['layer_0_bias'].shape}")
        
        print("\nResult: PASS")
        return True
        
    except Exception as e:
        print(f"\nResult: FAIL - {e}")
        return False


def test_config_validation():
    """Test 6: Configuration Validation"""
    print("\n" + "="*80)
    print("TEST 6: Configuration Validation")
    print("="*80)
    
    # Test valid configuration
    try:
        config = PrivacyFederatedConfig(
            federated_data_dir=Path("data/federated"),
            centralized_artifacts_dir=Path("artifacts/centralized"),
            output_dir=Path("experiments/test"),
            
            enable_privacy=True,
            shamir_threshold=3,
            num_clients=5,
            dp_epsilon_total=1.0
        )
        print("  Valid config: PASS")
        valid_config_pass = True
    except Exception as e:
        print(f"  Valid config: FAIL - {e}")
        valid_config_pass = False
    
    # Test invalid threshold (too low)
    try:
        config = PrivacyFederatedConfig(
            federated_data_dir=Path("data/federated"),
            centralized_artifacts_dir=Path("artifacts/centralized"),
            output_dir=Path("experiments/test"),
            
            enable_privacy=True,
            shamir_threshold=1,  # Invalid
            num_clients=5
        )
        print("  Invalid threshold detection: FAIL (should have raised error)")
        threshold_test_pass = False
    except ValueError:
        print("  Invalid threshold detection: PASS")
        threshold_test_pass = True
    
    # Test invalid epsilon
    try:
        config = PrivacyFederatedConfig(
            federated_data_dir=Path("data/federated"),
            centralized_artifacts_dir=Path("artifacts/centralized"),
            output_dir=Path("experiments/test"),
            
            enable_privacy=True,
            enable_differential_privacy=True,
            dp_epsilon_total=0.0,  # Invalid
            num_clients=5
        )
        print("  Invalid epsilon detection: FAIL (should have raised error)")
        epsilon_test_pass = False
    except ValueError:
        print("  Invalid epsilon detection: PASS")
        epsilon_test_pass = True
    
    all_pass = valid_config_pass and threshold_test_pass and epsilon_test_pass
    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_per_layer_dp_calibration():
    """Test 7: Per-Layer DP Calibration"""
    print("\n" + "="*80)
    print("TEST 7: Per-Layer DP Calibration")
    print("="*80)
    
    config = ShamirConfig(threshold=3, num_participants=5)
    dp_config = DifferentialPrivacyConfig(
        epsilon_total=1.0,
        num_rounds=10,
        num_clients=5
    )
    
    shamir = EnhancedShamirSecretSharing(config)
    secure_agg = SecureAggregationProtocol(shamir, dp_config)
    
    # Test calibration for different layer types
    layers = {
        'embedding_weights': (1000, 50),
        'hidden_weights': (50, 100),
        'hidden_bias': (100,),
        'output_weights': (100, 1),
        'output_bias': (1,)
    }
    
    print("  Per-layer DP calibration:")
    for layer_name, shape in layers.items():
        secure_agg.layer_dp_manager.calibrate_layer_parameters(layer_name, shape)
        clip_norm, noise_mult = secure_agg.layer_dp_manager.get_layer_parameters(layer_name)
        
        print(f"    {layer_name:20s}: clip={clip_norm:.4f}, noise={noise_mult:.6f}")
    
    # Verify that different layers have different calibrations
    clip_norms = []
    for layer_name in layers:
        clip_norm, _ = secure_agg.layer_dp_manager.get_layer_parameters(layer_name)
        clip_norms.append(clip_norm)
    
    unique_calibrations = len(set(clip_norms))
    print(f"\n  Unique calibrations: {unique_calibrations}/{len(layers)}")
    
    status = "PASS" if unique_calibrations > 1 else "FAIL"
    print(f"\nResult: {status}")
    
    return unique_calibrations > 1


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("PRIVACY-PRESERVING FEDERATED LEARNING - INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Basic Shamir", test_shamir_basic),
        ("Differential Privacy", test_differential_privacy),
        ("Secure Aggregation", test_secure_aggregation),
        ("Dropout Recovery", test_dropout_recovery),
        ("Privacy Aggregator", test_privacy_aggregator),
        ("Config Validation", test_config_validation),
        ("Per-Layer DP", test_per_layer_dp_calibration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[X]"
        print(f"  {symbol} {test_name:30s}: {status}")
    
    print(f"\n  Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n  STATUS: ALL TESTS PASSED - INTEGRATION READY")
    elif passed_tests >= total_tests * 0.8:
        print("\n  STATUS: MOSTLY PASSING - MINOR ISSUES TO ADDRESS")
    else:
        print("\n  STATUS: SIGNIFICANT ISSUES - REVIEW REQUIRED")
    
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)