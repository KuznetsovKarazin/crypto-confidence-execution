#!/usr/bin/env python3
"""
Federated Client Module 
=========================================

Individual client logic:
- Best validation model selection and return
- Proper initialization order (global params -> warm-up)
- Enhanced class-balanced batching with per-epoch shuffling
- Robust parameter handling and validation

"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from .utils import deterministic_seed

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Enhanced federated client 
    """
    
    def __init__(self, client_id: str, config, preprocessor):
        self.client_id = client_id
        self.config = config
        self.preprocessor = preprocessor

        # load decision threshold for selective execution 
        self.decision_threshold = getattr(self.config, "confidence_tau", 0.8)

        # Only override if explicitly loading from centralized artifacts
        if getattr(self.config, 'use_centralized_threshold', False):
            try:
                thr_path = self.config.centralized_artifacts_dir / "decision_threshold.json"
                if thr_path.exists():
                    with open(thr_path, "r", encoding="utf-8") as f:
                        thr = json.load(f)
                    centralized_tau = float(thr.get("confidence_tau", self.decision_threshold))
                    logger.info(f"{self.client_id}: Centralized tau={centralized_tau:.3f}, "
                            f"Config tau={self.decision_threshold:.3f}")
                    # Use centralized value only if config didn't override
                    if not hasattr(self.config, '_confidence_tau_from_cli'):
                        self.decision_threshold = centralized_tau
            except Exception as e:
                logger.warning(f"{self.client_id}: Failed to load centralized tau: {e}")
        
        # Training data
        self.data: Optional[Dict[str, Any]] = None
        self.data_info: Dict[str, Any] = {}
        
        # Model state
        self.local_model: Optional[MLPClassifier] = None
        self.previous_global_params: Optional[Dict[str, np.ndarray]] = None
        self.best_model_params: Optional[Dict[str, np.ndarray]] = None
        self.best_validation_f1: float = -np.inf
        self.best_validation_accuracy: float = -np.inf  
        
        # Training history
        self.training_metrics: Dict[str, Any] = {}
        
        logger.debug(f"Initialized client {client_id}")
    
    def load_and_prepare_data(self) -> bool:
        try:
            # Load client data
            client_data_path = self.config.federated_data_dir / self.client_id / 'unified_dataset.parquet'
            
            if not client_data_path.exists():
                logger.error(f"Client data not found: {client_data_path}")
                return False
            
            df = pd.read_parquet(client_data_path)
            logger.info(f"{self.client_id}: Loaded {len(df):,} samples")
            
            feature_cols = [c for c in df.columns if c not in ['symbol', 'timestamp', 'y', 'y_direction', 'y_binary', 'ret_bps', 'ret', 'future_mid']]
            logger.info(f"{self.client_id}: Raw features before selection: {len(feature_cols)}")
            
            # Create targets (preserved from centralized logic)
            from centralized_training_two_classes import TargetEncoder, TemporalSplitter
            
            target_encoder = TargetEncoder(
                horizon_min=self.config.horizon_min,
                deadband_bps=self.config.deadband_bps,
                task='classification'
            )
            df_with_targets = target_encoder.create_targets(df)
            
            # Temporal split (preserved logic)
            splitter = TemporalSplitter(
                val_split=0.15,  # Fixed for consistency
                test_split=0.15
            )
            
            train_df, val_df, test_df = splitter.split_data_symbol_wise(df_with_targets)
            
            # Filter for two-class mode if enabled (preserved logic)
            if self.config.two_class_mode:
                train_df, val_df, test_df = splitter.filter_trade_only_samples(train_df, val_df, test_df)
                
                if len(train_df) < 50:
                    logger.error(f"{self.client_id}: Insufficient trade samples: {len(train_df)}")
                    return False
            
            # Apply preprocessing (preserved logic)

            X_train, feature_names = self.preprocessor.select_and_transform(train_df)
            X_val, _ = self.preprocessor.select_and_transform(val_df)
            X_test, _ = self.preprocessor.select_and_transform(test_df)

            import hashlib
            def _cols_hash(cols): return hashlib.md5("|".join(cols).encode("utf-8")).hexdigest()

            assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
                f"{self.client_id}: feature dim mismatch train/val/test -> "
                f"{X_train.shape[1]}/{X_val.shape[1]}/{X_test.shape[1]}"
            )

            logger.info(f"{self.client_id}: n_features={X_train.shape[1]}, feature_names_hash={_cols_hash(list(feature_names))}")

            X_train = X_train.astype(np.float32, copy=False)
            X_val   = X_val.astype(np.float32, copy=False)
            X_test  = X_test.astype(np.float32, copy=False)

            
            # Extract targets
            if self.config.two_class_mode:
                y_train = train_df['y_direction'].values
                y_val = val_df['y_direction'].values
                y_test = test_df['y_direction'].values
            else:
                y_train = train_df['y'].values
                y_val = val_df['y'].values
                y_test = test_df['y'].values
            
            # Validate binary targets for two-class mode
            if self.config.two_class_mode:
                for y_array, name in [(y_train, 'train'), (y_val, 'val'), (y_test, 'test')]:
                    unique_vals = np.unique(y_array)
                    if not np.array_equal(np.sort(unique_vals), np.array([0, 1])):
                        logger.error(f"{self.client_id}: Invalid {name} targets: {unique_vals}")
                        return False
            
            # Store data
            self.data = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
                'feature_names': feature_names
            }
            
            # Store data info
            train_class_counts = np.bincount(y_train, minlength=2 if self.config.two_class_mode else 3)
            self.data_info = {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'train_class_distribution': train_class_counts.tolist(),
                'min_class_count': int(train_class_counts.min()),
                'symbols': train_df['symbol'].nunique() if 'symbol' in train_df.columns else 1
            }
            
            logger.info(f"{self.client_id}: Prepared data - "
                       f"Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
            logger.info(f"{self.client_id}: Class distribution: {train_class_counts.tolist()}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.client_id}: Data preparation failed: {e}")
            return False
    
    def train_local_model(self, global_params: Optional[Dict[str, np.ndarray]] = None,
                         round_num: int = 0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
        """
        Enhanced local training with ACCURACY-based best model selection (matching centralized).
        """
        if self.data is None:
            logger.error(f"{self.client_id}: No data loaded")
            return {}, {}, {}
        
        try:
            start_time = time.time()
            
            # Deterministic seeds 
            model_seed = self.config.model_seed + round_num
            shuffle_seed = deterministic_seed(f"{self.client_id}_shuffle", self.config.seed + round_num)
            
            required_classes = np.array([0, 1] if self.config.two_class_mode else [0, 1, 2])
            
            # Build and initialize model 
            model = self._build_model(model_seed)
            
            if round_num == 0 and global_params is None:
                model = self._initialize_model_round_0(model, required_classes, shuffle_seed)
            else:
                model = self._initialize_model_with_global_params(
                    model, global_params, required_classes, shuffle_seed
                )
            
            # Extract initial parameters for delta computation
            initial_params = self._extract_parameters(model)
            if not initial_params:
                logger.error(f"{self.client_id}: Failed to extract initial parameters")
                return {}, {}, {}
            
            # Training loop 
            self.best_validation_accuracy = -np.inf
            self.best_model_params = None

            X_train_all = self.data['X_train']
            y_train_all = self.data['y_train']

            eff_batch_size = max(256, getattr(self.config, "local_batch_size", getattr(self.config, "batch_size", 1024)))
            cls_counts = np.bincount(y_train_all, minlength=2 if self.config.two_class_mode else 3)
            logger.info(f"{self.client_id}: training with local_batch_size={eff_batch_size}, "
                        f"epochs={self.config.local_epochs}, class_counts={cls_counts.tolist()}")

            for epoch in range(self.config.local_epochs):
                batches = self._create_balanced_batches(
                    X_train_all, y_train_all,
                    base_seed=self.config.seed,
                    round_num=round_num,
                    epoch=epoch
                )

                for Xb, yb in batches:
                    model.partial_fit(Xb, yb)

                val_metrics = self._evaluate_model(model, self.data['X_val'], self.data['y_val'])
                val_accuracy = val_metrics['accuracy']
                if val_accuracy > self.best_validation_accuracy:
                    self.best_validation_accuracy = val_accuracy
                    self.best_model_params = self._extract_parameters(model)


            # Use best parameters if available
            final_params = self.best_model_params if self.best_model_params else self._extract_parameters(model)

            
            # Compute delta parameters 
            delta_params = self._compute_delta_parameters(initial_params, final_params)

            def _l2_norm(param_dict: Dict[str, np.ndarray]) -> float:
                return float(np.sqrt(sum(float((v**2).sum()) for v in param_dict.values())))

            delta_norm = _l2_norm(delta_params)
            logger.info(f"{self.client_id}: local delta L2-norm = {delta_norm:.6f}")

            if delta_norm == 0.0:
                logger.warning(f"{self.client_id}: zero delta — recomputing from current model params instead of best")
                final_params = self._extract_parameters(model)
                delta_params = self._compute_delta_parameters(initial_params, final_params)
                delta_norm = _l2_norm(delta_params)
                logger.info(f"{self.client_id}: recomputed delta L2-norm = {delta_norm:.6f}")

            # Final evaluation
            metrics = self._calculate_comprehensive_metrics(model)
            metrics['training_time'] = time.time() - start_time
            metrics['best_val_accuracy'] = float(self.best_validation_accuracy)  
            metrics['epochs_completed'] = epoch + 1
            metrics['delta_norm'] = delta_norm  
            
            self.local_model = model
            self.training_metrics = metrics
            
            logger.info(f"{self.client_id}: Training completed - Best Val Accuracy={self.best_validation_accuracy:.4f}")  
            
            return final_params, delta_params, metrics
            
        except Exception as e:
            logger.error(f"{self.client_id}: Training failed: {e}")
            return {}, {}, {}

    def train_local(self, data: Dict[str, Any] = None, 
                global_params: Optional[Dict[str, np.ndarray]] = None,
                local_epochs: Optional[int] = None, 
                round_num: int = 0, 
                **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
        """
        Wrapper for backward compatibility
        """
        # Call the main training method
        final_params, delta_params, metrics = self.train_local_model(
            global_params=global_params,
            round_num=round_num
        )

        return final_params, delta_params, metrics
    
    def _build_model(self, seed: int) -> MLPClassifier:
        client_lr = getattr(self.config, 'client_learning_rate', 0.001)  
        
        return MLPClassifier(
            hidden_layer_sizes=self.config.mlp_hidden_sizes,
            activation='relu',
            solver='sgd',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=client_lr,
            momentum=0.9,  
            max_iter=1,
            early_stopping=False,
            warm_start=True,
            random_state=seed
        )
    
    def _initialize_model_round_0(self, model: MLPClassifier, required_classes: np.ndarray,
                                 seed: int) -> MLPClassifier:
        """Standard initialization for round 0 (preserved logic)."""
        init_size = min(64, len(self.data['y_train']))
        rng = np.random.RandomState(seed)
        init_indices = rng.choice(len(self.data['y_train']), size=init_size, replace=False)
        
        X_init = self.data['X_train'][init_indices]
        y_init = self.data['y_train'][init_indices]
        
        model.partial_fit(X_init, y_init, classes=required_classes)
        return model
    
    def _initialize_model_with_global_params(self, model: MLPClassifier, 
                                        global_params: Optional[Dict[str, np.ndarray]],
                                        required_classes: np.ndarray,
                                        seed: int) -> MLPClassifier:
        """
        Initialize with global parameters first, then minimal warm-up 
        """
        if not global_params:
            return self._initialize_model_round_0(model, required_classes, seed)

        n_features = self.data['X_train'].shape[1]
        n_classes = len(required_classes)
  
        X_dummy = np.zeros((n_classes, n_features), dtype=np.float32)
        y_dummy = required_classes

        model.partial_fit(X_dummy, y_dummy, classes=required_classes)

        if self._validate_parameter_shapes(model, global_params):
            try:
                for layer_idx in range(len(model.coefs_)):
                    weights_key = f'layer_{layer_idx}_weights'
                    bias_key = f'layer_{layer_idx}_bias'
                    
                    if weights_key in global_params:
                        model.coefs_[layer_idx] = global_params[weights_key].copy()
                    if bias_key in global_params:
                        model.intercepts_[layer_idx] = global_params[bias_key].copy()                                      
                
                logger.debug(f"{self.client_id}: Global parameters loaded successfully (round > 0)")
            except Exception as e:
                logger.warning(f"{self.client_id}: Failed to load global parameters: {e}")
                # Fallback to standard initialization
                model = self._initialize_model_round_0(model, required_classes, seed)
        else:
            logger.warning(f"{self.client_id}: Parameter shapes don't match, using random init")
            model = self._initialize_model_round_0(model, required_classes, seed)
        
        return model
    
    def _create_balanced_batches(self, X_train: np.ndarray, y_train: np.ndarray,
                            base_seed: int, round_num: int, epoch: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Enhanced balanced batching with per-epoch shuffling 
        """
        # Create epoch-specific seed for proper randomization
        epoch_seed = deterministic_seed(f"{self.client_id}_{round_num}_{epoch}", base_seed)
        rng = np.random.default_rng(epoch_seed)

        min_batch_size = 256
        cfg_bs = getattr(self.config, "local_batch_size", getattr(self.config, "batch_size", 1024))
        batch_size = int(max(min_batch_size, cfg_bs))

        if self.config.two_class_mode:
            unique_classes = np.array([0, 1])
            class_counts = np.bincount(y_train, minlength=2)
            if min(class_counts) < len(y_train) * 0.1:
                return self._create_stratified_batches(X_train, y_train, batch_size, rng)

        indices = rng.permutation(len(X_train))
        batches = []
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            idx = indices[start:end]
            if len(idx) == 0:
                continue
            batches.append((X_train[idx], y_train[idx]))

        
        return batches

    def _create_stratified_batches(self, X_train: np.ndarray, y_train: np.ndarray,
                                batch_size: int, rng) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Helper method for stratified batch creation."""
        class_0_idx = np.where(y_train == 0)[0]
        class_1_idx = np.where(y_train == 1)[0]
        
        rng.shuffle(class_0_idx)
        rng.shuffle(class_1_idx)
        
        # Ensure at least 20% of minority class in each batch
        min_class_samples = max(1, int(batch_size * 0.2))

        batches = []
        idx_0, idx_1 = 0, 0
        n0, n1 = len(class_0_idx), len(class_1_idx)

        while idx_0 < n0 or idx_1 < n1:
            take1 = min(min_class_samples, n1 - idx_1)
            take0 = min(batch_size - take1, n0 - idx_0)
            if take1 == 0 and take0 == 0:
                break
            cur = []
            if take0 > 0:
                cur.extend(class_0_idx[idx_0:idx_0 + take0]); idx_0 += take0
            if take1 > 0:
                cur.extend(class_1_idx[idx_1:idx_1 + take1]); idx_1 += take1
            cur = np.array(cur)
            rng.shuffle(cur)
            batches.append((X_train[cur], y_train[cur]))
        
        return batches
    
    def _compute_delta_parameters(self, initial_params: Dict[str, np.ndarray],
                                final_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute delta parameters with L2 norm clipping """
        delta_params = {}
        
        for param_name in initial_params.keys():
            if param_name in final_params:
                initial_param = initial_params[param_name]
                final_param = final_params[param_name]
                
                if initial_param.shape == final_param.shape:
                    delta = final_param - initial_param
                    
                    # L2 norm clipping per parameter tensor
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > self.config.max_delta_norm:
                        delta = delta * (self.config.max_delta_norm / delta_norm)
                    
                    delta_params[param_name] = delta
        
        return delta_params
    
    def _extract_parameters(self, model: MLPClassifier) -> Dict[str, np.ndarray]:
        """Safely extract model parameters """
        params = {}
        
        try:
            if not hasattr(model, 'coefs_') or not hasattr(model, 'intercepts_'):
                return params
            
            if not model.coefs_ or not model.intercepts_:
                return params
            
            for i, coef in enumerate(model.coefs_):
                if coef is not None and coef.size > 0:
                    # Clean invalid values
                    if np.any(np.isnan(coef)) or np.any(np.isinf(coef)):
                        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
                    params[f'layer_{i}_weights'] = coef.copy()
            
            for i, intercept in enumerate(model.intercepts_):
                if intercept is not None and intercept.size > 0:
                    # Clean invalid values
                    if np.any(np.isnan(intercept)) or np.any(np.isinf(intercept)):
                        intercept = np.nan_to_num(intercept, nan=0.0, posinf=0.0, neginf=0.0)
                    params[f'layer_{i}_bias'] = intercept.copy()
            
        except Exception as e:
            logger.error(f"{self.client_id}: Parameter extraction failed: {e}")
        
        return params
    
    def _validate_parameter_shapes(self, model: MLPClassifier, 
                                  global_params: Dict[str, np.ndarray]) -> bool:
        """Validate global parameter shapes against local model."""
        try:
            if not hasattr(model, 'coefs_') or not model.coefs_:
                return False
            
            for layer_idx, local_weights in enumerate(model.coefs_):
                weights_key = f'layer_{layer_idx}_weights'
                bias_key = f'layer_{layer_idx}_bias'
                
                if weights_key not in global_params or bias_key not in global_params:
                    return False
                
                global_weights = global_params[weights_key]
                global_bias = global_params[bias_key]
                local_bias = model.intercepts_[layer_idx]
                
                if (global_weights.shape != local_weights.shape or 
                    global_bias.shape != local_bias.shape):
                    return False
            
            return True
        except Exception:
            return False
    
    def _evaluate_model(self, model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Quick model evaluation for validation."""
        try:
            y_pred = model.predict(X)
            return {
                'accuracy': accuracy_score(y, y_pred),
                'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        except Exception:
            return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    
    def _compute_direction_metrics_with_threshold(
        self,
        y_true: np.ndarray,
        proba_pos: np.ndarray,
        tau: float
    ) -> Dict[str, float]:
        """
        Selective execution by confidence: execute long if p>=tau, short if p<=1-tau, else skip.
        Returns coverage, n_executed, n_total, direction_accuracy, avg_confidence.
        """
        #exec_long = proba_pos >= tau
        #exec_short = proba_pos <= (1.0 - tau)
        #exec_mask = exec_long | exec_short
        confidence = np.maximum(proba_pos, 1.0 - proba_pos)
        exec_mask = confidence >= tau

        n_total = int(y_true.shape[0])
        n_exec = int(exec_mask.sum())
        coverage = n_exec / n_total if n_total > 0 else 0.0

        if n_exec == 0:
            return {
                "test_coverage": 0.0,
                "test_n_executed": 0,
                "test_n_total": n_total,
                "test_direction_accuracy": 0.0,
                "test_avg_confidence": float(np.mean(np.maximum(proba_pos, 1.0 - proba_pos))) if n_total else 0.0,
                "test_correct_executed": 0.0
            }
        
        # Predicted direction only on executables
        #y_hat_exec = np.empty(n_exec, dtype=int)
        #y_hat_exec[exec_long[exec_mask]] = 1
        #y_hat_exec[exec_short[exec_mask]] = 0
        pred_direction = (proba_pos >= 0.5).astype(int)
        y_hat_exec = pred_direction[exec_mask]

        y_true_exec = y_true[exec_mask]
        correct = float((y_hat_exec == y_true_exec).sum())
        dir_acc = correct / n_exec

        avg_conf = float(np.mean(np.maximum(proba_pos[exec_mask], 1.0 - proba_pos[exec_mask])))

        return {
            "test_coverage": float(coverage),
            "test_n_executed": int(n_exec),
            "test_n_total": int(n_total),
            "test_direction_accuracy": float(dir_acc),
            "test_avg_confidence": avg_conf,
            # for aggregation by round:
            "test_correct_executed": float(correct)
        }



    def _calculate_comprehensive_metrics(self, model: MLPClassifier) -> Dict[str, float]:
        """Calculate comprehensive metrics incl. selective (direction/coverage/confidence)."""
        metrics = {}
        try:
            train_pred = model.predict(self.data['X_train'])
            val_pred   = model.predict(self.data['X_val'])
            test_pred  = model.predict(self.data['X_test'])

            metrics.update({
                'train_accuracy': accuracy_score(self.data['y_train'], train_pred),
                'val_accuracy':   accuracy_score(self.data['y_val'],   val_pred),
                'test_accuracy':  accuracy_score(self.data['y_test'],  test_pred),
                'train_f1': f1_score(self.data['y_train'], train_pred, average='macro', zero_division=0),
                'val_f1':   f1_score(self.data['y_val'],   val_pred,   average='macro', zero_division=0),
                'test_f1':  f1_score(self.data['y_test'],  test_pred,  average='macro', zero_division=0),
            })


            thr = getattr(self.config, 'decision_threshold', None)
            if thr is None and hasattr(self.config, 'decision_thresholds'):
                thr = self.config.decision_thresholds.get('two_class', None)
            if thr is None:
                thr = 0.8  

            # p1 = P(y=1), confidence = max(p1, 1-p1)
            def _selective_block(X, y, returns_bps):
                proba = model.predict_proba(X)[:, 1]
                conf  = np.maximum(proba, 1.0 - proba)
                mask  = conf >= thr
                coverage = float(mask.mean())

                if mask.any():
                    pred_dir = (proba >= 0.5).astype(int)
                    dir_acc  = float(accuracy_score(y[mask], pred_dir[mask]))
                    avg_conf = float(conf[mask].mean())

                    #trade_profit = np.where(pred_dir[mask] == 1, returns_bps[mask], -returns_bps[mask])
                    #net_profit   = trade_profit - self.config.profit_cost_bps
                    gross_profit = np.where(pred_dir[mask] == 1, returns_bps[mask], -returns_bps[mask])
                    net_profit = gross_profit - self.config.profit_cost_bps
                    avg_np       = float(net_profit.mean())
                    win_rate     = float((net_profit > 0).mean())
                    sharpe       = float(net_profit.mean() / (net_profit.std() + 1e-8))
                else:
                    dir_acc = 0.0
                    avg_conf = 0.0
                    avg_np = 0.0
                    win_rate = 0.0
                    sharpe = 0.0

                return {
                    'direction_accuracy': dir_acc,
                    'coverage': coverage,
                    'avg_confidence': avg_conf,
                    'net_profit_bps_sel': avg_np,
                    'win_rate_sel': win_rate,
                    'sharpe_sel': sharpe,
                }

            if self.config.two_class_mode and 'val_df' in self.data and 'test_df' in self.data:
                val_ret  = self.data['val_df']['ret_bps'].values
                test_ret = self.data['test_df']['ret_bps'].values
                y_val    = self.data['y_val']
                y_test   = self.data['y_test']

                val_sel  = _selective_block(self.data['X_val'],  y_val,  val_ret)
                test_sel = _selective_block(self.data['X_test'], y_test, test_ret)

                metrics.update({
                    'val_direction_accuracy':  val_sel['direction_accuracy'],
                    'val_coverage':            val_sel['coverage'],
                    'val_avg_confidence':      val_sel['avg_confidence'],
                    'val_net_profit_bps_sel':  val_sel['net_profit_bps_sel'],
                    'val_win_rate_sel':        val_sel['win_rate_sel'],
                    'val_sharpe_sel':          val_sel['sharpe_sel'],

                    'test_direction_accuracy': test_sel['direction_accuracy'],
                    'test_coverage':           test_sel['coverage'],
                    'test_avg_confidence':     test_sel['avg_confidence'],
                    'test_net_profit_bps_sel': test_sel['net_profit_bps_sel'],
                    'test_win_rate_sel':       test_sel['win_rate_sel'],
                    'test_sharpe_sel':         test_sel['sharpe_sel'],
                })

            if self.config.two_class_mode and 'test_df' in self.data:
                test_returns_bps = self.data['test_df']['ret_bps'].values
                val_returns_bps  = self.data['val_df']['ret_bps'].values

                test_direction_profit = np.where(test_pred == 1, test_returns_bps, -test_returns_bps)
                test_net_profit = test_direction_profit - self.config.profit_cost_bps

                val_direction_profit = np.where(val_pred == 1, val_returns_bps, -val_returns_bps)
                val_net_profit = val_direction_profit - self.config.profit_cost_bps

                metrics.update({
                    'test_net_profit_bps': float(test_net_profit.mean()),
                    'test_win_rate': float((test_net_profit > 0).mean()),
                    'test_sharpe_ratio': float(test_net_profit.mean() / (test_net_profit.std() + 1e-8)),
                    'val_net_profit_bps': float(val_net_profit.mean()),
                    'val_win_rate': float((val_net_profit > 0).mean()),
                })

            metrics.update(self.data_info)

        except Exception as e:
            logger.error(f"{self.client_id}: Metrics calculation failed: {e}")

        return metrics


    
    def evaluate_global_model(self, global_model) -> Optional[Dict[str, Any]]:
        """Evaluate global model on client's test data."""
        if self.data is None or global_model is None:
            return None
        
        try:
            test_pred = global_model.predict(self.data['X_test'])
            test_proba = global_model.predict_proba(self.data['X_test'])
            
            metrics = {
                'test_accuracy': accuracy_score(self.data['y_test'], test_pred),
                'test_f1': f1_score(self.data['y_test'], test_pred, average='macro', zero_division=0),
                'test_samples': len(self.data['y_test']),
                'predictions': test_pred.tolist(),
                'true_labels': self.data['y_test'].tolist()
            }
            
            # AUC if binary
            if test_proba.shape[1] == 2:
                metrics['test_auc'] = roc_auc_score(self.data['y_test'], test_proba[:, 1])
            
            # Trading metrics with selective execution
            if self.config.two_class_mode and 'test_df' in self.data:
                returns_bps = self.data['test_df']['ret_bps'].values
                
                # Get confidence and selective execution mask
                proba_pos = test_proba[:, 1]
                confidences = np.maximum(proba_pos, 1.0 - proba_pos)
                tau = float(getattr(self, "decision_threshold", getattr(self.config, "confidence_tau", 0.8)))
                
                exec_mask = confidences >= tau
                
                # Direction: predict 1 if proba >= 0.5, else 0
                pred_direction = (proba_pos >= 0.5).astype(int)
                direction_correct = (pred_direction == self.data['y_test'])
                
                # Calculate profits
                direction_profit = np.where(pred_direction == 1, returns_bps, -returns_bps)
                net_profit = direction_profit - self.config.profit_cost_bps
                
                metrics.update({
                    'test_net_profit_bps': float(net_profit.mean()),
                    'test_win_rate': float((net_profit > 0).mean()),
                    'returns_bps': returns_bps.tolist(),             
                    'pred_direction': pred_direction.tolist(), 
                    'executed_flags': exec_mask.tolist(),
                    'direction_correct': direction_correct.tolist(),
                    'confidences': confidences.tolist(),
                    'profits_bps': net_profit.tolist(),
                    'net_profits': net_profit.tolist()
                })

            
            return metrics
            
        except Exception as e:
            logger.error(f"{self.client_id}: Global model evaluation failed: {e}")
            return None
    
    def create_global_model(self, global_params: Dict[str, np.ndarray]):
        """Create global model with preserved initialization logic."""
        if not global_params or self.data is None:
            return None
        
        try:
            # Create model with same config
            model = self._build_model(self.config.model_seed)
            
            # Initialize sklearn internals
            required_classes = np.array([0, 1] if self.config.two_class_mode else [0, 1, 2])
            model = self._initialize_model_round_0(model, required_classes, self.config.seed)
            
            # Set global parameters
            if self._validate_parameter_shapes(model, global_params):
                layer_idx = 0
                while (f'layer_{layer_idx}_weights' in global_params and 
                       layer_idx < len(model.coefs_)):
                    
                    weights = global_params[f'layer_{layer_idx}_weights']
                    bias = global_params[f'layer_{layer_idx}_bias']
                    
                    if weights.shape == model.coefs_[layer_idx].shape:
                        model.coefs_[layer_idx] = weights.copy()
                    if bias.shape == model.intercepts_[layer_idx].shape:
                        model.intercepts_[layer_idx] = bias.copy()
                    
                    layer_idx += 1
            
            return model
            
        except Exception as e:
            logger.error(f"{self.client_id}: Global model creation failed: {e}")
            return None
    
    def get_data_info(self) -> str:
        """Get formatted data info string."""
        if not self.data_info:
            return "No data loaded"
        
        return (f"Train={self.data_info['train_samples']:,}, "
                f"Features={self.data_info['n_features']}, "
                f"Symbols={self.data_info['symbols']}, "
                f"Classes={self.data_info['train_class_distribution']}")