#!/usr/bin/env python3
"""
Centralized Training Module for Cryptocurrency Price Movement Prediction
=====================================================================

Production-ready module for training centralized models to predict mid-price movements
in cryptocurrency markets using unified microstructure and macroeconomic features.

Enhanced with proper class weighting, two-stage training approach, two-class mode,
and critical bug fixes.

Author: AI Assistant
License: MIT
Version: 1.3 - Added two-class mode support
"""

import warnings
import argparse
import logging
import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import joblib

# ML/DL libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import mutual_info_classif
def mi_score_func(X, y):
    return mutual_info_classif(X, y, n_neighbors=2, random_state=42)

# PyTorch imports (conditional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils import clip_grad_norm_
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MLP will use sklearn MLPClassifier fallback.")

# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Module logger
logger = logging.getLogger(__name__)


def mi_score_func(X, y):
    """Mutual information score function for feature selection (top-level for serializability)."""
    return mutual_info_classif(X, y, n_neighbors=2, random_state=42)


@dataclass
class TrainingConfig:
    """Configuration for centralized training pipeline."""
    
    # Data paths
    unified_data_path: Path
    output_dir: Path
    
    # Task configuration
    task: str = 'classification'  # 'classification' or 'regression'
    horizon_min: int = 5
    deadband_bps: float = 2.0
    
    # Model configuration
    model_type: str = 'mlp'  # 'mlp' or 'lstm'
    sequence_len: int = 60  # For LSTM
    
    # Training parameters
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Model architecture (MLP)
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    
    # LSTM architecture
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    
    # Training optimization
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    gradient_clip_value: float = 1.0
    
    # Class weighting and two-stage training
    use_class_weights: bool = True
    use_two_stage: bool = True
    two_stage_trade_threshold: float = 0.5
    min_class_ratio: float = 0.05
    
    # Two-class mode configuration
    two_class_mode: bool = False
    confidence_tau: float = 0.7
    confidence_grid_start: float = 0.50
    confidence_grid_end: float = 0.95
    confidence_grid_step: float = 0.01
    profit_cost_bps: float = 1.0
    
    # Feature selection and preprocessing
    max_train_size: int = 1_500_000
    top_k_features: Optional[int] = 128
    use_robust_scaler: bool = True
    feature_selection_method: str = 'mutual_info'  # 'variance', 'mutual_info', 'f_classif'
    
    # Calibration and threshold optimization
    calibrate_probabilities: bool = True
    optimize_threshold_for_profit: bool = True
    
    # Misc
    seed: int = 42
    device: str = 'auto'
    save_predictions: bool = True

    optimize_tau_by: str = "profit"          # 'profit' | 'ev' | 'profit_with_min_coverage'
    min_coverage: float = 0.0                # minimum coverage when choosing by profit_with_min_coverage


class TargetEncoder:
    """Enhanced target variable creation with multiple horizons support."""
    
    def __init__(self, horizon_min: int, deadband_bps: float, task: str = 'classification'):
        self.horizon_min = horizon_min
        self.deadband_bps = deadband_bps
        self.task = task
        
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables with proper temporal alignment."""
        logger.info(f"Creating {self.task} targets: horizon={self.horizon_min}min, deadband={self.deadband_bps}bps")
        logger.info(f"Input memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        
        # CRITICAL FIX: Always sort to ensure monotonicity
        df.sort_values(['symbol', 'timestamp'], inplace=True)
        logger.info("Data sorted by symbol and timestamp")
        
        # Initialize target arrays
        n_rows = len(df)
        future_mid_values = np.full(n_rows, np.nan, dtype=np.float32)
        
        # Process each symbol separately
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_indices = np.where(symbol_mask)[0]
            
            if len(symbol_indices) <= self.horizon_min:
                continue
            
            # Get mid_price for this symbol
            symbol_mid_prices = df['mid_price'].iloc[symbol_indices].values
            
            # CRITICAL FIX: Correct future price calculation (no double shift)
            future_prices = np.full(len(symbol_mid_prices), np.nan, dtype=np.float32)
            if len(symbol_mid_prices) > self.horizon_min:
                future_prices[:-self.horizon_min] = symbol_mid_prices[self.horizon_min:]
            
            future_mid_values[symbol_indices] = future_prices
        
        # Create valid mask
        valid_mask = ~np.isnan(future_mid_values)
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            raise ValueError(f"No valid samples found with horizon {self.horizon_min}")
        
        logger.info(f"Valid samples: {n_valid:,} / {n_rows:,} ({n_valid/n_rows:.1%})")
        
        # Calculate returns
        valid_indices = np.where(valid_mask)[0]
        mid_price_valid = df['mid_price'].values[valid_mask]
        future_mid_valid = future_mid_values[valid_mask]
        
        ret = (future_mid_valid / mid_price_valid) - 1
        ret_bps = ret * 10000
        
        # Create targets
        if self.task == 'classification':
            y = np.full(len(ret_bps), 2, dtype=np.int8)  # Default: no-trade
            y[ret_bps > self.deadband_bps] = 1  # Up
            y[ret_bps < -self.deadband_bps] = 0  # Down
            
            # Binary target for two-stage training
            y_binary = (y != 2).astype(np.int8)
            
            # Direction target (only for trade samples)
            y_direction = np.full(len(ret_bps), -1, dtype=np.int8)
            trade_mask = y != 2
            y_direction[trade_mask] = y[trade_mask]
            
        else:  # regression
            y = ret.astype(np.float32)
            y_binary = None
            y_direction = None
        
        # Create result DataFrame
        result_df = df.iloc[valid_indices].copy()
        result_df['y'] = y
        result_df['future_mid'] = future_mid_valid
        result_df['ret'] = ret
        result_df['ret_bps'] = ret_bps
        
        if self.task == 'classification':
            result_df['y_binary'] = y_binary
            result_df['y_direction'] = y_direction
        
        result_df.reset_index(drop=True, inplace=True)
        
        # Log statistics
        self._log_target_statistics(y, ret_bps if self.task == 'classification' else ret)
        
        if self.task == 'classification':
            self._log_binary_and_direction_stats(y_binary, y_direction)
        
        logger.info(f"Output memory usage: {result_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        
        return result_df
    
    def _log_target_statistics(self, y: np.ndarray, ret_values: np.ndarray) -> None:
        """Log target statistics."""
        if self.task == 'classification':
            unique_classes, class_counts = np.unique(y, return_counts=True)
            total = len(y)
            class_props = class_counts / total
            
            logger.info("Classification target distribution:")
            class_names = {0: 'Down', 1: 'Up', 2: 'No-trade'}
            for cls, count, prop in zip(unique_classes, class_counts, class_props):
                name = class_names.get(cls, f'Class-{cls}')
                logger.info(f"  {name}: {count:,} ({prop:.1%})")
            
            if class_props.min() < 0.01:
                logger.warning(f"Severely imbalanced class detected: {class_props.min():.1%}")
        else:
            logger.info("Regression target statistics:")
            logger.info(f"  Mean: {np.mean(y):.4f}")
            logger.info(f"  Std: {np.std(y):.4f}")
            logger.info(f"  Range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    
    def _log_binary_and_direction_stats(self, y_binary: np.ndarray, y_direction: np.ndarray) -> None:
        """Log binary and direction target statistics."""
        binary_counts = np.bincount(y_binary)
        logger.info(f"Binary target distribution (trade vs no-trade):")
        logger.info(f"  No-trade: {binary_counts[0]:,} ({binary_counts[0]/len(y_binary):.1%})")
        logger.info(f"  Trade: {binary_counts[1]:,} ({binary_counts[1]/len(y_binary):.1%})")
        
        direction_trade_mask = y_direction >= 0
        if direction_trade_mask.sum() > 0:
            direction_counts = np.bincount(y_direction[direction_trade_mask])
            logger.info(f"Direction distribution (trade samples only):")
            logger.info(f"  Down: {direction_counts[0]:,} ({direction_counts[0]/direction_trade_mask.sum():.1%})")
            logger.info(f"  Up: {direction_counts[1]:,} ({direction_counts[1]/direction_trade_mask.sum():.1%})")


class FeatureProcessor:
    """Enhanced feature processing with better selection methods and no data leakage."""
    
    def __init__(
        self,
        exclude_cols: Optional[List[str]] = None,
        top_k_features: Optional[int] = None,
        use_robust_scaler: bool = False,
        feature_selection_method: str = 'mutual_info'
    ):
        self.exclude_cols = exclude_cols or [
            'symbol', 'timestamp', 'date', 'future_mid', 'ret', 'ret_bps', 'y',
            'y_binary', 'y_direction', 'mid_price'
        ]
        self.top_k_features = top_k_features
        self.use_robust_scaler = use_robust_scaler
        self.feature_selection_method = feature_selection_method
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.feature_names: Optional[List[str]] = None

    def fit_feature_list(self, df_train: pd.DataFrame, y_train: Union[np.ndarray, pd.Series, None] = None) -> None:
        """Fit feature list and selector using TRAIN ONLY (safe alignment + robust sampling)."""
        rng = np.random.default_rng(getattr(self, "seed", 42))

        # 1) Кандидаты: только числовые (и не исключённые)
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in numeric_cols if c not in getattr(self, "exclude_cols", [])]

        # 2) Удаляем константы/все-NaN
        valid = []
        for c in candidates:
            col = df_train[c]
            if col.notna().sum() > 0 and col.nunique(dropna=True) > 1:
                valid.append(c)
            else:
                logger.warning(f"Removing feature {c}: all NaN or constant")

        if not valid:
            raise ValueError("No valid features found")

        # 3) Подготовим X и y с ЖЁСТКИМ выравниванием по индексам
        X_all = df_train.loc[:, valid]  # важен .loc, чтобы сохранить индекс
        y_series = None
        if y_train is not None:
            if isinstance(y_train, pd.Series):
                y_series = y_train.reindex(df_train.index)
            else:
                # предполагаем, что y_train соответствует df_train по порядку
                if len(y_train) != len(df_train):
                    raise ValueError("y_train length does not match df_train rows")
                y_series = pd.Series(y_train, index=df_train.index)

            # выравниваем X и y на пересечении индексов
            X_all, y_series = X_all.align(y_series, join="inner", axis=0)

        # 4) Если селекция не нужна (меньше top_k или топ-к не задан) — просто зафиксируем имена
        need_selection = bool(self.top_k_features) and (len(valid) > int(self.top_k_features))
        if not need_selection:
            self.feature_names = sorted(valid)
            self._log_feature_categories()
            return

        logger.info(f"Selecting top {self.top_k_features} features using {self.feature_selection_method}")

        # 5) Универсальный безопасный сэмплер позиций (0..n-1), с опциональной стратификацией по symbol
        def sample_positions_stratified(n_max: int) -> np.ndarray:
            n_rows = len(X_all)
            if n_rows == 0:
                raise ValueError("Empty training frame for feature selection")

            n_max = min(n_max, n_rows)
            if n_max == n_rows:
                return np.arange(n_rows, dtype=np.int64)

            if "symbol" in df_train.columns:
                # символы на тех же индексах, что и X_all
                symbol_series = df_train.loc[X_all.index, "symbol"]
                # сгруппируем ПО ПОЗИЦИЯМ, не по меткам индексов
                groups: Dict[Any, List[int]] = {}
                for pos, sym in enumerate(symbol_series.to_numpy()):
                    groups.setdefault(sym, []).append(pos)

                symbols = list(groups.keys())
                if len(symbols) > 0:
                    base = n_max // len(symbols)
                    rem = n_max % len(symbols)
                    picked: List[int] = []
                    for i, sym in enumerate(symbols):
                        pool = groups[sym]
                        quota = base + (1 if i < rem else 0)
                        if quota <= 0:
                            continue
                        if quota >= len(pool):
                            picked.extend(pool)
                        else:
                            picked.extend(rng.choice(pool, size=quota, replace=False).tolist())
                    if len(picked) > n_max:
                        picked = rng.choice(picked, size=n_max, replace=False).tolist()
                    return np.asarray(picked, dtype=np.int64)

            # без символов — просто случайные позиции
            return rng.choice(len(X_all), size=n_max, replace=False).astype(np.int64)

        # 6) Ветви селекции
        selected_features: List[str]

        method = str(getattr(self, "feature_selection_method", "variance")).lower()
        n_max_mi = min(100_000, len(X_all))
        n_max_var = min(200_000, len(X_all))

        if method == "mutual_info" and y_series is not None and y_series.nunique(dropna=True) >= 2:
            import time
            from sklearn.feature_selection import SelectKBest

            t0 = time.perf_counter()
            pos = sample_positions_stratified(n_max_mi)

            # формируем сэмпл
            X_sample = X_all.iloc[pos].replace([np.inf, -np.inf], np.nan)
            y_sample = y_series.iloc[pos]

            # простая импутация + компактный тип
            X_sample = X_sample.fillna(X_sample.mean()).astype(np.float32)
            y_sample = np.asarray(y_sample).ravel()

            # используем заранее определённую top-level функцию mi_score_func (как у тебя было)
            selector = SelectKBest(score_func=mi_score_func, k=int(self.top_k_features))
            selector.fit(X_sample, y_sample)
            selected_features = X_sample.columns[selector.get_support()].tolist()
            self.feature_selector = selector

            logger.info(
                f"mutual_info selection done in {time.perf_counter()-t0:.1f}s "
                f"on sample {len(X_sample):,} rows, picked {len(selected_features)} features"
            )

        elif method == "f_classif" and y_series is not None and y_series.nunique(dropna=True) >= 2:
            from sklearn.feature_selection import SelectKBest, f_classif

            pos = sample_positions_stratified(n_max_mi)
            X_sample = X_all.iloc[pos].replace([np.inf, -np.inf], np.nan)
            y_sample = y_series.iloc[pos]

            X_sample = X_sample.fillna(X_sample.mean()).astype(np.float32)
            y_sample = np.asarray(y_sample).ravel()

            selector = SelectKBest(score_func=f_classif, k=int(self.top_k_features))
            selector.fit(X_sample, y_sample)
            selected_features = X_sample.columns[selector.get_support()].tolist()
            self.feature_selector = selector

            logger.info(f"f_classif selection picked {len(selected_features)} features from {X_sample.shape[1]}")

        else:
            # Фоллбек: дисперсия (работает без y)
            if method not in {"variance", "var"}:
                logger.warning(f"Unknown/unsupported selection method '{method}' with given y; falling back to variance")

            if len(X_all) > n_max_var:
                pos = rng.choice(len(X_all), size=n_max_var, replace=False).astype(np.int64)
                sample = X_all.iloc[pos]
            else:
                sample = X_all

            # заменим inf на NaN, чтобы var() не сломался, и проимпутим
            sample = sample.replace([np.inf, -np.inf], np.nan).fillna(sample.mean())
            vars_ = sample.var().sort_values(ascending=False)
            selected_features = vars_.index[: int(self.top_k_features)].tolist()

            logger.info(f"variance selection picked {len(selected_features)} features from {sample.shape[1]}")

        # 7) Финал
        if not selected_features:
            raise ValueError("Feature selection produced empty feature set")

        self.feature_names = sorted(selected_features)
        self._log_feature_categories()

    def _log_feature_categories(self) -> None:
        """Log feature categories."""
        cats = {
            'price': [f for f in self.feature_names if any(p in f.lower() for p in ['close','open','high','low','mid_price'])],
            'returns': [f for f in self.feature_names if ('return' in f.lower() or 'momentum' in f.lower())],
            'volatility': [f for f in self.feature_names if ('volatility' in f.lower() or 'std' in f.lower())],
            'volume': [f for f in self.feature_names if 'volume' in f.lower()],
            'spread': [f for f in self.feature_names if 'spread' in f.lower()],
            'imbalance': [f for f in self.feature_names if 'imbalance' in f.lower()],
            'technical': [f for f in self.feature_names if any(t in f.lower() for t in ['ma_','rsi','atr'])],
        }
        used = set().union(*cats.values())
        cats['other'] = [f for f in self.feature_names if f not in used]
        
        logger.info(f"Selected {len(self.feature_names)} features")
        for k, v in cats.items():
            if v:
                logger.info(f"  {k}: {len(v)} features")

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features using fitted feature list."""
        if self.feature_names is None:
            raise ValueError("fit_feature_list() must be called first")
        return df[self.feature_names]

    def fit_preprocessors(self, X_train: pd.DataFrame) -> None:
        """Fit imputer and scaler on training data."""
        logger.info("Fitting preprocessors on training data")
        
        # Fit imputer with strategy aligned to scaler choice
        imputer_strategy = "median" if self.use_robust_scaler else "mean"
        logger.info(f"Using imputer strategy: {imputer_strategy}")
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.imputer.fit(X_train)
        
        # Transform training data for scaler fitting
        X_imputed = self.imputer.transform(X_train)
        
        # Fit scaler
        scaler_name = 'RobustScaler' if self.use_robust_scaler else 'StandardScaler'
        logger.info(f"Fitting {scaler_name}")
        self.scaler = RobustScaler() if self.use_robust_scaler else StandardScaler()
        self.scaler.fit(X_imputed)
        
        # Log scaler statistics
        if hasattr(self.scaler, 'mean_'):
            center_min, center_max = float(np.min(self.scaler.mean_)), float(np.max(self.scaler.mean_))
        elif hasattr(self.scaler, 'center_'):
            center_min, center_max = float(np.min(self.scaler.center_)), float(np.max(self.scaler.center_))
        else:
            center_min = center_max = float('nan')
        
        scale_min, scale_max = float(np.min(self.scaler.scale_)), float(np.max(self.scaler.scale_))
        logger.info(f"Scaler fitted: center range [{center_min:.3f}, {center_max:.3f}]")
        logger.info(f"Scale range: [{scale_min:.3f}, {scale_max:.3f}]")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with fitted preprocessors."""
        if self.imputer is None or self.scaler is None:
            raise ValueError("Preprocessors not fitted. Call fit_preprocessors() first.")
        
        # Handle infinite values
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Apply imputer and scaler
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed).astype(np.float32)
        
        return X_scaled

    def save_artifacts(self, output_dir: Path) -> None:
        """Save preprocessing artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessors
        joblib.dump(self.imputer, output_dir / 'imputer.joblib')
        joblib.dump(self.scaler, output_dir / 'scaler.joblib')
        
        logger.info(f"Saved preprocessors to {output_dir}")

        # Save feature schema
        if hasattr(self.scaler, 'mean_'):
            center_min, center_max = float(np.min(self.scaler.mean_)), float(np.max(self.scaler.mean_))
        elif hasattr(self.scaler, 'center_'):
            center_min, center_max = float(np.min(self.scaler.center_)), float(np.max(self.scaler.center_))
        else:
            center_min = center_max = float('nan')
        
        scale_min, scale_max = float(np.min(self.scaler.scale_)), float(np.max(self.scaler.scale_))

        schema = {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'exclude_cols': self.exclude_cols,
            'feature_selection_method': self.feature_selection_method,
            'scaler_center_range': [center_min, center_max],
            'scaler_scale_range': [scale_min, scale_max],
            'scaler_type': 'robust' if self.use_robust_scaler else 'standard'
        }
        
        with open(output_dir / 'feature_schema.json', 'w') as f:
            json.dump(schema, f, indent=2)


class TemporalSplitter:
    """Symbol-aware temporal data splitting."""
    
    def __init__(self, val_split: float = 0.15, test_split: float = 0.15):
        self.val_split = val_split
        self.test_split = test_split
        self.train_split = 1.0 - val_split - test_split
        
    def split_data_symbol_wise(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create symbol-wise temporal splits to avoid data leakage."""
        logger.info(f"Creating symbol-wise temporal splits: train={self.train_split:.1%}, val={self.val_split:.1%}, test={self.test_split:.1%}")
        
        parts = []
        for symbol, group in df.sort_values('timestamp').groupby('symbol', sort=False):
            # Explicit sorting within group for robustness
            group = group.sort_values('timestamp')
            n = len(group)
            n_train = int(n * self.train_split)
            n_val = int(n * self.val_split)
            
            train_part = group.iloc[:n_train]
            val_part = group.iloc[n_train:n_train + n_val]
            test_part = group.iloc[n_train + n_val:]
            
            parts.append((train_part, val_part, test_part))
        
        # Concatenate all parts
        train_data = pd.concat([part[0] for part in parts if len(part[0]) > 0])
        val_data = pd.concat([part[1] for part in parts if len(part[1]) > 0])
        test_data = pd.concat([part[2] for part in parts if len(part[2]) > 0])
        
        # Reset indices
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        
        self._log_split_statistics(train_data, val_data, test_data)
        return train_data, val_data, test_data
    
    def filter_trade_only_samples(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter to keep only trade samples (y_binary == 1) for two-class mode."""
        logger.info("Filtering for trade-only samples (two-class mode)")
        
        # Filter each split independently
        mask_tr = train_df['y_binary'].values == 1
        mask_va = val_df['y_binary'].values == 1
        mask_te = test_df['y_binary'].values == 1

        train_filtered = train_df.loc[mask_tr]
        val_filtered   = val_df.loc[mask_va]
        test_filtered  = test_df.loc[mask_te]

        
        logger.info(f"Trade-only filtering results:")
        logger.info(f"  Train: {len(train_df):,} -> {len(train_filtered):,} ({len(train_filtered)/len(train_df):.1%})")
        logger.info(f"  Val: {len(val_df):,} -> {len(val_filtered):,} ({len(val_filtered)/len(val_df):.1%})")
        logger.info(f"  Test: {len(test_df):,} -> {len(test_filtered):,} ({len(test_filtered)/len(test_df):.1%})")
        
        # Check that we have sufficient samples
        if len(train_filtered) < 1000:
            logger.warning(f"Very few training trade samples: {len(train_filtered)}. Consider reducing deadband_bps.")
        if len(val_filtered) < 100:
            logger.warning(f"Very few validation trade samples: {len(val_filtered)}. May affect threshold optimization.")
        if len(test_filtered) < 100:
            logger.warning(f"Very few test trade samples: {len(test_filtered)}. Results may be unreliable.")
        
        return train_filtered, val_filtered, test_filtered
        
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/validation/test splits."""
        return self.split_data_symbol_wise(df)  # Use symbol-wise by default
        
    def cap_train_size(self, train_df: pd.DataFrame, cap: int = 1_500_000) -> pd.DataFrame:
        """Cap training size with symbol-aware sampling using random sampling instead of stride."""
        if len(train_df) <= cap:
            return train_df
        
        logger.info(f"Capping training data: {len(train_df):,} -> {cap:,} rows")
        
        # Calculate sampling ratio
        sample_ratio = cap / len(train_df)
        
        # Sample from each symbol proportionally with randomization to avoid autocorrelations
        result_parts = []
        for symbol, group in train_df.groupby('symbol'):
            n_sample = max(1, int(len(group) * sample_ratio))
            n_sample = min(n_sample, len(group))
            
            if n_sample < len(group):
                # Use random sampling instead of stride to avoid periodic patterns
                sampled = group.sample(n=n_sample, random_state=42)
            else:
                sampled = group
            result_parts.append(sampled)
        
        result = pd.concat(result_parts).sort_values(['symbol', 'timestamp'])
        result.reset_index(drop=True, inplace=True)
        
        logger.info(f"After capping: {len(result):,} rows using proportional random sampling")
        return result
    
    def _log_split_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Log detailed split statistics."""
        splits = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
        
        logger.info("Symbol-wise temporal split statistics:")
        for name, df in splits.items():
            if len(df) > 0:
                symbols = df['symbol'].nunique()
                date_range = (df['timestamp'].min(), df['timestamp'].max())
                memory_mb = df.memory_usage(deep=True).sum() / 1e6
                logger.info(f"  {name}: {len(df):,} samples, {symbols} symbols, {memory_mb:.1f} MB")
                logger.info(f"    Date range: {date_range[0]} to {date_range[1]}")


# PyTorch Models
if TORCH_AVAILABLE:
    class MLPClassifier(nn.Module):
        """Enhanced MLP with configurable architecture and dropout."""
        
        def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.2):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            
            layers = []
            prev_size = input_dim
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, num_classes))
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        def forward(self, x):
            return self.network(x)
    
    
    class LSTMClassifier(nn.Module):
        """Enhanced LSTM for sequence classification."""
        
        def __init__(self, input_dim: int, hidden_size: int, num_layers: int, 
                     num_classes: int, dropout_rate: float = 0.2):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_classes = num_classes
            
            self.lstm = nn.LSTM(
                input_dim, hidden_size, num_layers,
                dropout=dropout_rate if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_size, num_classes)
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.constant_(param, 0)
        
        def forward(self, x):
            """Forward pass. All sequences have the same length in our implementation."""
            lstm_out, (hidden, _) = self.lstm(x)
            # Use last hidden state
            last_hidden = hidden[-1]
            output = self.dropout(last_hidden)
            output = self.classifier(output)
            return output


class SequenceDataGenerator:
    """Generate sequences for LSTM training with corrected temporal alignment."""
    
    def __init__(self, sequence_len: int, horizon_min: int):
        self.sequence_len = sequence_len
        self.horizon_min = horizon_min
    
    def create_sequences(self, df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with CORRECTED target alignment."""
        sequences = []
        targets = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < self.sequence_len:
                continue
            
            features = symbol_data[feature_names].values
            symbol_targets = symbol_data['y'].values
            
            # CRITICAL FIX: Correct target alignment (no double shift)
            for i in range(len(symbol_data) - self.sequence_len + 1):
                seq_features = features[i:i + self.sequence_len]
                seq_target = symbol_targets[i + self.sequence_len - 1]  # FIXED: removed horizon_min
                
                sequences.append(seq_features)
                targets.append(seq_target)
        
        if not sequences:
            raise ValueError("No valid sequences generated")
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets)
        
        logger.info(f"Generated {len(sequences)} sequences of length {self.sequence_len}")
        return sequences, targets


class TwoClassConfidenceClassifier:
    """Two-class classifier with confidence-based execution threshold."""
    
    def __init__(self, confidence_tau: float = 0.7):
        self.confidence_tau = confidence_tau
        self.model = None
        self.is_calibrated = False
        
    def fit(self, X_train: np.ndarray, y_direction_train: np.ndarray, 
            model_class, model_params: Dict[str, Any], use_class_weights: bool = True) -> None:
        """Train binary direction classifier (Up vs Down)."""
        logger.info("Training two-class direction classifier (Up vs Down)")
        
        # Verify we have binary targets
        unique_classes = np.unique(y_direction_train)
        if not np.array_equal(unique_classes, [0, 1]):
            raise ValueError(f"Expected binary classes [0, 1], got {unique_classes}")
        
        # Log class distribution
        class_counts = np.bincount(y_direction_train)
        logger.info(f"Direction distribution: Down={class_counts[0]}, Up={class_counts[1]}")
        logger.info(f"Class balance: {class_counts[1]/(class_counts[0]+class_counts[1]):.3f} Up ratio")
        
        self.model = model_class(**model_params)
        
        if use_class_weights and hasattr(self.model, 'fit'):
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_direction_train)
            logger.info(f"Sample weights: mean={sample_weights.mean():.4f}, std={sample_weights.std():.4f}")
            self.model.fit(X_train, y_direction_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_direction_train)
    
    def calibrate_probabilities(self, X_val: np.ndarray, y_direction_val: np.ndarray, 
                              method: str = 'isotonic') -> None:
        """Calibrate probabilities using validation set."""
        logger.info(f"Calibrating two-class probabilities using {method} method")
        
        self.model = CalibratedClassifierCV(
            self.model, method=method, cv='prefit'
        )
        self.model.fit(X_val, y_direction_val)
        self.is_calibrated = True
        logger.info("Two-class probability calibration completed")
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence-based execution logic."""
        # Get probabilities
        proba = self.model.predict_proba(X)
        p_up = proba[:, 1]
        p_down = proba[:, 0]
        
        # Calculate confidence
        confidence = np.maximum(p_up, p_down)
        
        # Execution decision based on confidence threshold
        execute = confidence >= self.confidence_tau
        
        # Direction prediction (only meaningful when executing)
        direction = (p_up > p_down).astype(np.int8)  # 1=Up, 0=Down
        
        return direction, confidence, execute
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def optimize_confidence_tau_for_profit(
        self,
        X_val: np.ndarray,
        y_direction_val: np.ndarray,
        returns_bps: np.ndarray,
        cost_bps: float = 1.0,
        tau_grid_start: float = 0.5,
        tau_grid_end: float = 0.95,
        tau_grid_step: float = 0.01,
        select_by: str = "profit",                 # 'profit' | 'ev' | 'profit_with_min_coverage'
        min_coverage: float | None = None,
        output_dir: "Path | None" = None           #optional: where to save the CSV
    ) -> float:
        """
        Optimize confidence threshold tau.
        select_by:
            'profit' -> maximum average profit per trade (as before)
            'ev' -> maximum EV_per_obs = profit_bps * coverage
            'profit_with_min_coverage' -> maximum profit with coverage >= min_coverage
        """
        logger.info(f"Optimizing confidence tau (cost={cost_bps} bps, select_by={select_by}, "
                    f"min_coverage={min_coverage})")

        tau_values = np.arange(tau_grid_start, tau_grid_end + tau_grid_step, tau_grid_step)

        proba = self.model.predict_proba(X_val)
        p_up = proba[:, 1]
        p_down = proba[:, 0]

        results = []

        for tau in tau_values:
            confidence = np.maximum(p_up, p_down)
            execute = confidence >= tau

            # --- ZERO-COVERAGE SAFETY BLOCK: paste right after `execute = (confidence >= tau)` ---
            n_total = int(len(y_test))
            n_exec = int(execute.sum())

            if n_exec == 0:
                
                two_class_metrics = {
                    "coverage": 0.0,
                    "n_executed": 0,
                    "n_total": n_total,
                    "direction_accuracy": 0.0,
                    "avg_profit_bps": -self.config.profit_cost_bps,   
                    "median_profit_bps": -self.config.profit_cost_bps,
                    "win_rate": 0.0,
                    "avg_confidence": float(np.maximum(p_up, p_down).mean()),
                    "profit_std_bps": 0.0,
                    "profit_sharpe": 0.0,
                    "max_profit_bps": 0.0,
                    "min_profit_bps": 0.0,
                    "profit_p25_bps": 0.0,
                    "profit_p75_bps": 0.0,
                    "profit_p90_bps": 0.0,
                    "profit_p10_bps": 0.0,
                }

                
                preds_cols = ["symbol","timestamp","p_up","p_down","confidence","y_true","return_bps","executed"]
                pd.DataFrame(columns=preds_cols).to_csv(output_dir / "test_predictions_two_class.csv",
                                                        index=False, encoding="utf-8")

                
                self._log_two_class_metrics(two_class_metrics)
                (output_dir / "metrics_two_class.json").write_text(
                    json.dumps(two_class_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                return two_class_metrics
            # --- end ZERO-COVERAGE SAFETY BLOCK ---


            if execute.sum() == 0:
                profit = -cost_bps          # no trades -> empty penalty (or 0.0 if you prefer)
                coverage = 0.0
                hit_rate = 0.0
                n_trades = 0
            else:
                # Direction and facts
                direction_pred = (p_up > p_down)[execute]           # 1=Up, 0=Down
                direction_true = y_direction_val[execute]           # 1=Up, 0=Down
                trade_returns = returns_bps[execute]                # future bps

                # PnL by direction, then subtract costs
                direction_profit = np.where(direction_pred == 1, trade_returns, -trade_returns)
                net_profit = direction_profit - cost_bps

                profit = float(net_profit.mean())                   # average profit per transaction
                coverage = float(execute.mean())                    # share of executables
                hit_rate = float((direction_pred == direction_true).mean())
                n_trades = int(execute.sum())

            ev_per_obs = profit * coverage

            results.append({
                "tau": float(tau),
                "profit_bps": profit,
                "coverage": coverage,
                "hit_rate": hit_rate,
                "n_trades": n_trades,
                "EV_per_obs": ev_per_obs
            })

        # The best by different criteria
        best_by_profit = max(results, key=lambda r: r["profit_bps"])
        best_by_ev     = max(results, key=lambda r: r["EV_per_obs"])

        if select_by == "ev":
            chosen = best_by_ev
        elif select_by == "profit_with_min_coverage":
            thr = 0.0 if min_coverage is None else float(min_coverage)
            eligible = [r for r in results if r["coverage"] >= thr]
            chosen = max(eligible, key=lambda r: r["profit_bps"]) if len(eligible) else best_by_profit
        else:  # 'profit' by default
            chosen = best_by_profit

        # Landmark Logs
        logger.info(f"Best by profit: tau={best_by_profit['tau']:.3f}, "
                    f"profit={best_by_profit['profit_bps']:.2f} bps, "
                    f"cov={best_by_profit['coverage']:.3%}, trades={best_by_profit['n_trades']}")
        logger.info(f"Best by EV:     tau={best_by_ev['tau']:.3f}, "
                    f"EV={best_by_ev['EV_per_obs']:.2f}, "
                    f"cov={best_by_ev['coverage']:.3%}, profit={best_by_ev['profit_bps']:.2f}")

        logger.info(f"Chosen tau ({select_by}): {chosen['tau']:.3f} | "
                    f"profit={chosen['profit_bps']:.2f} bps, cov={chosen['coverage']:.3%}, "
                    f"EV={chosen['EV_per_obs']:.2f}, trades={chosen['n_trades']}")

        # Save CSV (if there is a place)
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            if output_dir is not None:
                (output_dir / "tau_optimization_results.csv").parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / "tau_optimization_results.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to save tau_optimization_results.csv: {e}")

        # Also keep in mind
        self.tau_optimization_results = results

        return float(chosen["tau"])

    
    def save_model(self, output_dir: Path) -> None:
        """Save model and configuration."""
        joblib.dump(self.model, output_dir / 'two_class_model.joblib')
        
        config = {
            'confidence_tau': self.confidence_tau,
            'model_type': 'two_class',
            'is_calibrated': self.is_calibrated
        }
        
        with open(output_dir / 'two_class_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save tau optimization results if available
        if hasattr(self, 'tau_optimization_results'):
            tau_df = pd.DataFrame(self.tau_optimization_results)
            tau_df.to_csv(output_dir / 'tau_optimization_results.csv', index=False)
        
        logger.info(f"Saved two-class model to {output_dir}")


class TwoStageClassifier:
    """Enhanced two-stage classifier with calibration and threshold optimization."""
    
    def __init__(self, trade_threshold: float = 0.5):
        self.trade_threshold = trade_threshold
        self.trade_model = None
        self.direction_model = None
        self.is_calibrated = False
        
    def fit(self, X_train: np.ndarray, y_binary_train: np.ndarray, y_direction_train: np.ndarray,
            model_class, model_params: Dict[str, Any], use_class_weights: bool = True) -> None:
        """Train both stages of the classifier."""
        logger.info("Training two-stage classifier")
        
        # Stage 1: Trade vs No-trade
        logger.info("Stage 1: Training trade vs no-trade model")
        self.trade_model = model_class(**model_params)
        
        if use_class_weights and hasattr(self.trade_model, 'fit'):
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_binary_train)
            logger.info(f"Stage 1 sample weights: mean={sample_weights.mean():.4f}, std={sample_weights.std():.4f}")
            self.trade_model.fit(X_train, y_binary_train, sample_weight=sample_weights)
        else:
            self.trade_model.fit(X_train, y_binary_train)
        
        # Stage 2: Up vs Down (only on trade samples)
        trade_mask = y_direction_train >= 0
        if trade_mask.sum() == 0:
            raise ValueError("No trade samples found for direction training")
        
        X_trade = X_train[trade_mask]
        y_direction_trade = y_direction_train[trade_mask]
        
        logger.info(f"Stage 2: Training direction model on {trade_mask.sum():,} trade samples")
        logger.info(f"Direction distribution: Up={np.sum(y_direction_trade == 1)}, Down={np.sum(y_direction_trade == 0)}")
        
        self.direction_model = model_class(**model_params)
        
        if use_class_weights and hasattr(self.direction_model, 'fit'):
            sample_weights_dir = compute_sample_weight(class_weight='balanced', y=y_direction_trade)
            logger.info(f"Stage 2 sample weights: mean={sample_weights_dir.mean():.4f}, std={sample_weights_dir.std():.4f}")
            self.direction_model.fit(X_trade, y_direction_trade, sample_weight=sample_weights_dir)
        else:
            self.direction_model.fit(X_trade, y_direction_trade)
    
    def calibrate_probabilities(self, X_val: np.ndarray, y_binary_val: np.ndarray, 
                              y_direction_val: np.ndarray, method: str = 'isotonic') -> None:
        """Calibrate probabilities using validation set."""
        logger.info(f"Calibrating probabilities using {method} method")
        
        # Calibrate trade model
        self.trade_model = CalibratedClassifierCV(
            self.trade_model, method=method, cv='prefit'
        )
        self.trade_model.fit(X_val, y_binary_val)
        
        # Calibrate direction model (only on trade samples)
        trade_mask = y_direction_val >= 0
        if trade_mask.sum() > 0:
            X_val_trade = X_val[trade_mask]
            y_val_direction_trade = y_direction_val[trade_mask]
            
            self.direction_model = CalibratedClassifierCV(
                self.direction_model, method=method, cv='prefit'
            )
            self.direction_model.fit(X_val_trade, y_val_direction_trade)
        
        self.is_calibrated = True
        logger.info("Probability calibration completed")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using two-stage approach with optimized efficiency."""
        # Stage 1: Predict trade probability
        trade_proba = self.trade_model.predict_proba(X)[:, 1]
        
        # Stage 2: Predict direction only for likely trades (OPTIMIZED)
        direction_proba = np.zeros((len(X), 2), dtype=np.float32)
        trade_mask = trade_proba >= self.trade_threshold
        
        if trade_mask.any():
            direction_proba[trade_mask] = self.direction_model.predict_proba(X[trade_mask])
        
        # Combine predictions
        final_pred = np.full(len(X), 2, dtype=np.int8)  # Default: no-trade
        final_proba = np.zeros((len(X), 3), dtype=np.float32)
        
        # Set probabilities
        final_proba[:, 2] = 1.0 - trade_proba  # No-trade probability
        
        if trade_mask.sum() > 0:
            # Up probability = P(trade) * P(up|trade)
            final_proba[trade_mask, 1] = trade_proba[trade_mask] * direction_proba[trade_mask, 1]
            # Down probability = P(trade) * P(down|trade)
            final_proba[trade_mask, 0] = trade_proba[trade_mask] * direction_proba[trade_mask, 0]
            # Update no-trade probability
            final_proba[trade_mask, 2] = 1.0 - trade_proba[trade_mask]
            
            # Final prediction: argmax
            final_pred[trade_mask] = np.argmax(final_proba[trade_mask, :2], axis=1)
        
        return final_pred, final_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        _, proba = self.predict(X)
        return proba
    
    def optimize_threshold_for_profit(self, X_val: np.ndarray, y_val: np.ndarray,
                                    returns_bps: np.ndarray, cost_bps: float = 1.0) -> float:
        """Optimize threshold for expected profit instead of balanced accuracy."""
        logger.info("Optimizing trade threshold for expected profit")
        
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_profit = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            self.trade_threshold = threshold
            val_pred, _ = self.predict(X_val)
            
            # Calculate expected profit
            profit = self._calculate_expected_profit(y_val, val_pred, returns_bps, cost_bps)
            
            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} (Expected profit: {best_profit:.2f} bps)")
        return best_threshold
    
    def _calculate_expected_profit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 returns_bps: np.ndarray, cost_bps: float) -> float:
        """Calculate expected profit for threshold optimization."""
        # Only consider trade predictions
        trade_mask = y_pred != 2
        if trade_mask.sum() == 0:
            return -cost_bps  # No trades, only costs
        
        # Calculate profit for trades
        trade_returns = returns_bps[trade_mask]
        trade_pred = y_pred[trade_mask]
        
        # Direction profit (up=1, down=0)
        direction_profit = np.where(trade_pred == 1, trade_returns, -trade_returns)
        
        # Subtract trading costs
        net_profit = direction_profit - cost_bps
        
        return float(net_profit.mean())
    
    def save_models(self, output_dir: Path) -> None:
        """Save both models and configuration."""
        joblib.dump(self.trade_model, output_dir / 'trade_model.joblib')
        joblib.dump(self.direction_model, output_dir / 'direction_model.joblib')
        
        config = {
            'trade_threshold': self.trade_threshold,
            'model_type': 'two_stage',
            'is_calibrated': self.is_calibrated
        }
        with open(output_dir / 'two_stage_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved two-stage models to {output_dir}")


class MetricsCalculator:
    """Enhanced metrics calculation with trading-specific metrics."""
    
    def __init__(self, task: str):
        self.task = task
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        if self.task == 'classification':
            return self._calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        else:
            return self._calculate_regression_metrics(y_true, y_pred)
    
    def calculate_two_class_metrics(self, y_direction_true: np.ndarray, direction_pred: np.ndarray,
                                  confidence: np.ndarray, execute: np.ndarray, 
                                  returns_bps: np.ndarray, cost_bps: float = 1.0) -> Dict[str, float]:
        """Calculate metrics specific to two-class mode."""
        metrics = {}
        
        # Basic coverage and execution stats
        metrics['coverage'] = float(execute.mean())
        metrics['n_executed'] = int(execute.sum())
        metrics['n_total'] = len(execute)
        
        if execute.sum() == 0:
            # No trades executed
            metrics['direction_accuracy'] = 0.0
            metrics['avg_profit_bps'] = -cost_bps
            metrics['median_profit_bps'] = -cost_bps
            metrics['win_rate'] = 0.0
            metrics['avg_confidence'] = float(confidence.mean())
            return metrics
        
        # Metrics on executed trades only
        executed_direction_pred = direction_pred[execute]
        executed_direction_true = y_direction_true[execute]
        executed_returns = returns_bps[execute]
        executed_confidence = confidence[execute]
        
        # Direction accuracy
        metrics['direction_accuracy'] = float((executed_direction_pred == executed_direction_true).mean())
        
        # Calculate profits
        # If pred=1 (Up), take return; if pred=0 (Down), take -return
        gross_profits = np.where(executed_direction_pred == 1, executed_returns, -executed_returns)
        net_profits = gross_profits - cost_bps
        
        metrics['avg_profit_bps'] = float(net_profits.mean())
        metrics['median_profit_bps'] = float(np.median(net_profits))
        metrics['win_rate'] = float((net_profits > 0).mean())
        metrics['avg_confidence'] = float(executed_confidence.mean())
        
        # Risk metrics
        metrics['profit_std_bps'] = float(net_profits.std())
        metrics['profit_sharpe'] = float(net_profits.mean() / (net_profits.std() + 1e-8))
        metrics['max_profit_bps'] = float(net_profits.max())
        metrics['min_profit_bps'] = float(net_profits.min())
        
        # Percentiles
        metrics['profit_p25_bps'] = float(np.percentile(net_profits, 25))
        metrics['profit_p75_bps'] = float(np.percentile(net_profits, 75))
        metrics['profit_p90_bps'] = float(np.percentile(net_profits, 90))
        metrics['profit_p10_bps'] = float(np.percentile(net_profits, 10))
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate enhanced classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(np.mean(y_true == y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        
        # Trading-specific metrics (conditional on trades only)
        trade_mask_true = y_true != 2
        trade_mask_pred = y_pred != 2
        
        if trade_mask_true.sum() > 0 and trade_mask_pred.sum() > 0:
            # Hit rate among predicted trades
            predicted_trades = y_pred[trade_mask_pred]
            actual_trades = y_true[trade_mask_pred]
            trade_hit_rate = float(np.mean(predicted_trades == actual_trades))
            metrics['trade_hit_rate'] = trade_hit_rate
            
            # Direction accuracy among actual trades
            if trade_mask_true.sum() > 0:
                direction_acc = float(np.mean(y_true[trade_mask_true] == y_pred[trade_mask_true]))
                metrics['direction_accuracy'] = direction_acc
        
        # Multi-class metrics
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_pred_proba, 
                                                               multi_class='ovr', average='macro'))
                else:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            except ValueError:
                metrics['roc_auc_ovr'] = np.nan
            
            # Per-class PR-AUC
            pr_aucs = []
            for class_idx in range(y_pred_proba.shape[1]):
                y_true_binary = (y_true == class_idx).astype(int)
                if y_true_binary.sum() > 0:  # Only if class exists
                    pr_auc = average_precision_score(y_true_binary, y_pred_proba[:, class_idx])
                    pr_aucs.append(pr_auc)
                    metrics[f'pr_auc_class_{class_idx}'] = float(pr_auc)
            
            if pr_aucs:
                metrics['pr_auc_macro'] = float(np.mean(pr_aucs))
        
        # F1 scores and per-class metrics
        from sklearn.metrics import f1_score
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Confusion matrix and per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        precision, recall, _, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        
        for i, (p, r, s) in enumerate(zip(precision, recall, support)):
            metrics[f'precision_class_{i}'] = float(p)
            metrics[f'recall_class_{i}'] = float(r)
            metrics[f'support_class_{i}'] = int(s)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = float(1 - (ss_res / (ss_tot + 1e-10)))
        
        # Sign accuracy
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        metrics['sign_accuracy'] = float(np.mean(y_true_sign == y_pred_sign))
        
        # Quantile-based metrics
        errors = np.abs(y_true - y_pred)
        metrics['mae_q50'] = float(np.percentile(errors, 50))
        metrics['mae_q90'] = float(np.percentile(errors, 90))
        metrics['mae_q95'] = float(np.percentile(errors, 95))
        
        return metrics
    
    def calculate_per_symbol_metrics(self, df: pd.DataFrame, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> pd.DataFrame:
        """Calculate metrics per symbol."""
        df_results = df.copy()
        df_results['y_true'] = y_true
        df_results['y_pred'] = y_pred
        
        symbol_metrics = []
        
        for symbol in df_results['symbol'].unique():
            symbol_mask = df_results['symbol'] == symbol
            symbol_y_true = y_true[symbol_mask]
            symbol_y_pred = y_pred[symbol_mask]
            
            if len(symbol_y_true) == 0:
                continue
            
            if self.task == 'classification':
                accuracy = np.mean(symbol_y_true == symbol_y_pred)
                balanced_acc = balanced_accuracy_score(symbol_y_true, symbol_y_pred)
                
                # Trading-specific metrics per symbol
                trade_mask = symbol_y_true != 2
                trade_hit_rate = np.nan
                if trade_mask.sum() > 0:
                    trade_hit_rate = np.mean(symbol_y_true[trade_mask] == symbol_y_pred[trade_mask])
                
                symbol_metrics.append({
                    'symbol': symbol,
                    'n_samples': len(symbol_y_true),
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'trade_hit_rate': trade_hit_rate,
                })
            else:
                mae = mean_absolute_error(symbol_y_true, symbol_y_pred)
                mse = mean_squared_error(symbol_y_true, symbol_y_pred)
                sign_acc = np.mean(np.sign(symbol_y_true) == np.sign(symbol_y_pred))
                
                symbol_metrics.append({
                    'symbol': symbol,
                    'n_samples': len(symbol_y_true),
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'sign_accuracy': sign_acc,
                })
        
        return pd.DataFrame(symbol_metrics)
    
    def calculate_two_class_per_symbol_metrics(self, df: pd.DataFrame, direction_pred: np.ndarray,
                                             confidence: np.ndarray, execute: np.ndarray,
                                             returns_bps: np.ndarray, cost_bps: float = 1.0) -> pd.DataFrame:
        """Calculate per-symbol metrics for two-class mode."""
        df_results = df.copy()
        df_results['direction_pred'] = direction_pred
        df_results['confidence'] = confidence
        df_results['execute'] = execute
        
        symbol_metrics = []
        
        for symbol in df_results['symbol'].unique():
            symbol_mask = df_results['symbol'] == symbol
            symbol_data = df_results[symbol_mask]
            
            if len(symbol_data) == 0:
                continue
            
            # Basic stats
            coverage = symbol_data['execute'].mean()
            n_executed = symbol_data['execute'].sum()
            
            if n_executed == 0:
                symbol_metrics.append({
                    'symbol': symbol,
                    'n_samples': len(symbol_data),
                    'coverage': coverage,
                    'n_executed': n_executed,
                    'direction_accuracy': 0.0,
                    'avg_profit_bps': -cost_bps,
                    'win_rate': 0.0,
                    'avg_confidence': symbol_data['confidence'].mean()
                })
                continue
            
            # Metrics on executed trades
            executed_data = symbol_data[symbol_data['execute']]
            symbol_returns = returns_bps[symbol_mask][symbol_data['execute']]
            
            # Direction accuracy
            direction_acc = (executed_data['direction_pred'] == executed_data['y_direction']).mean()
            
            # Profit calculation
            gross_profits = np.where(executed_data['direction_pred'] == 1, 
                                   symbol_returns, -symbol_returns)
            net_profits = gross_profits - cost_bps
            
            symbol_metrics.append({
                'symbol': symbol,
                'n_samples': len(symbol_data),
                'coverage': coverage,
                'n_executed': n_executed,
                'direction_accuracy': direction_acc,
                'avg_profit_bps': net_profits.mean(),
                'win_rate': (net_profits > 0).mean(),
                'avg_confidence': executed_data['confidence'].mean()
            })
        
        return pd.DataFrame(symbol_metrics)


class CentralizedTrainer:
    """Enhanced main training class with two-class mode support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.training_history = []
        self.train_df_capped = None  # Store capped training data
        
        # Setup logging
        self._setup_logging()
        
        # Set random seeds
        self._set_seeds()
        
    def _setup_device(self) -> str:
        """Setup compute device."""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        if self.config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config.device
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_path = self.config.output_dir / 'logs.txt'
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        logger.info(f"Logging to {log_path}")
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.seed)
        
        logger.info(f"Random seed set to {self.config.seed}")
    
    def train(self) -> Dict[str, Any]:
        """Main training pipeline with two-class mode support."""
        logger.info("Starting centralized training pipeline")
        
        if self.config.two_class_mode:
            logger.info("TWO-CLASS MODE ENABLED: Training binary direction classifier only")
        
        # Load and prepare data
        logger.info("="*60)
        logger.info("LOADING AND PREPARING DATA")
        logger.info("="*60)
        
        data = self._load_data()
        data_with_targets = self._create_targets(data)
        
        # Check class balance
        if self.config.two_class_mode:
            # For two-class mode, check balance of direction targets among trades
            trade_mask = data_with_targets['y_binary'] == 1
            if trade_mask.sum() == 0:
                raise ValueError("No trade samples found. Reduce deadband_bps.")
            direction_targets = data_with_targets.loc[trade_mask, 'y_direction'].values
            check_class_balance(direction_targets, task=self.config.task, mode="two_class", 
                              min_class_ratio=self.config.min_class_ratio)
        else:
            check_class_balance(data_with_targets['y'].values, task=self.config.task, 
                              min_class_ratio=self.config.min_class_ratio)
        
        train_df, val_df, test_df = self._split_data(data_with_targets)
        
        # Prepare features
        logger.info("="*60)
        logger.info("FEATURE PROCESSING")
        logger.info("="*60)
        
        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_features(
            train_df, val_df, test_df
        )
        
        # Train model
        logger.info("="*60)
        logger.info("MODEL TRAINING")
        logger.info("="*60)
        
        if self.config.model_type == 'lstm':
            results = self._train_lstm_model(
                train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test
            )
        else:
            results = self._train_mlp_model(X_train, y_train, X_val, y_val, X_test, y_test,
                                          train_df, val_df, test_df)
        
        results['test_df'] = test_df
        
        # Evaluate and save results
        logger.info("="*60)
        logger.info("EVALUATION AND SAVING")
        logger.info("="*60)
        
        self._evaluate_and_save_results(results)
        
        logger.info("Training pipeline completed successfully")
        return results
    
    def _load_data(self) -> pd.DataFrame:
        """Load unified dataset."""
        logger.info(f"Loading data from {self.config.unified_data_path}")
        
        if not self.config.unified_data_path.exists():
            raise FileNotFoundError(f"Unified dataset not found: {self.config.unified_data_path}")
        
        df = pd.read_parquet(self.config.unified_data_path)
        logger.info(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Basic validation
        required_cols = ['symbol', 'timestamp', 'mid_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables."""
        target_encoder = TargetEncoder(
            horizon_min=self.config.horizon_min,
            deadband_bps=self.config.deadband_bps,
            task=self.config.task
        )
        return target_encoder.create_targets(df)
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create symbol-wise temporal splits."""
        splitter = TemporalSplitter(
            val_split=self.config.validation_split,
            test_split=self.config.test_split
        )
        
        # Create basic temporal splits first
        train_df, val_df, test_df = splitter.split_data_symbol_wise(df)
        
        # Filter for trade-only samples in two-class mode
        if self.config.two_class_mode:
            train_df, val_df, test_df = splitter.filter_trade_only_samples(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                         test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features with two-class mode support."""
        # Cap training size if needed
        splitter = TemporalSplitter()
        train_df_capped = splitter.cap_train_size(train_df, cap=self.config.max_train_size)
        
        # Store capped training data
        self.train_df_capped = train_df_capped.copy()
        
        # Initialize feature processor
        processor = FeatureProcessor(
            top_k_features=self.config.top_k_features,
            use_robust_scaler=self.config.use_robust_scaler,
            feature_selection_method=self.config.feature_selection_method
        )
        
        # Target for feature selection
        if self.config.two_class_mode:
            # Use direction targets for feature selection in two-class mode
            feature_selection_target = train_df_capped['y_direction'].values
        else:
            feature_selection_target = train_df_capped['y'].values
        
        # Fit feature list with appropriate target
        if self.config.task == 'classification' and self.config.feature_selection_method in ['mutual_info', 'f_classif']:
            processor.fit_feature_list(train_df_capped, feature_selection_target)
        else:
            processor.fit_feature_list(train_df_capped)
        
        # Select features consistently across splits
        X_train_df = processor.select_features(train_df_capped)
        X_val_df = processor.select_features(val_df)
        X_test_df = processor.select_features(test_df)
        
        logger.info(f"Feature matrix shapes:")
        logger.info(f"  Train: {X_train_df.shape}")
        logger.info(f"  Val: {X_val_df.shape}")
        logger.info(f"  Test: {X_test_df.shape}")
        
        # Fit preprocessors on training data only
        processor.fit_preprocessors(X_train_df)
        
        # Transform all splits
        X_train = processor.transform(X_train_df)
        X_val = processor.transform(X_val_df)
        X_test = processor.transform(X_test_df)
        
        # Extract targets based on mode
        if self.config.two_class_mode:
            # Use direction targets (0=Down, 1=Up)
            y_train = train_df_capped['y_direction'].values
            y_val = val_df['y_direction'].values
            y_test = test_df['y_direction'].values
        else:
            # Use standard targets (0=Down, 1=Up, 2=No-trade)
            y_train = train_df_capped['y'].values
            y_val = val_df['y'].values
            y_test = test_df['y'].values
        
        # Save preprocessing artifacts
        processor.save_artifacts(self.config.output_dir)
        self.feature_processor = processor
        
        logger.info(f"Final feature shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        logger.info(f"Target mode: {'direction (0/1)' if self.config.two_class_mode else 'standard (0/1/2)'}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _train_mlp_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Train MLP model with two-class mode support."""
        logger.info(f"Training MLP model with architecture: {self.config.mlp_hidden_sizes}")
        
        if self.config.two_class_mode:
            # Two-class mode: train single binary classifier
            return self._train_two_class_sklearn(
                X_train, y_train, X_val, y_val, X_test, y_test,
                train_df, val_df, test_df
            )
        elif (self.config.task == 'classification' 
              and self.config.use_two_stage 
              and hasattr(self, 'train_df_capped')):
            # Two-stage mode
            return self._train_two_stage_sklearn(
                X_train, y_train, X_val, y_val, X_test, y_test,
                train_df, val_df, test_df
            )
        else:
            # Standard single-stage sklearn
            return self._train_sklearn_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                                         train_df, val_df, test_df)
    
    def _train_two_class_sklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Train two-class mode classifier."""
        logger.info("Using two-class mode: binary direction classifier (Up vs Down)")
        
        from sklearn.neural_network import MLPClassifier
        
        # Verify targets are binary
        unique_classes = np.unique(y_train)
        if not np.array_equal(unique_classes, [0, 1]):
            raise ValueError(f"Two-class mode expects binary targets [0, 1], got {unique_classes}")
        
        # Model parameters with early stopping enabled
        model_params = {
            'hidden_layer_sizes': tuple(self.config.mlp_hidden_sizes),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': min(200, max(50, len(X_train)//100)),
            'learning_rate_init': self.config.learning_rate,
            'max_iter': self.config.epochs,
            'early_stopping': True,  # Enable early stopping for better performance
            'n_iter_no_change': self.config.early_stopping_patience,
            'validation_fraction': 0.1,  # Use 10% of training for early stopping validation
            'random_state': self.config.seed
        }
        
        # Create and train two-class classifier
        two_class_classifier = TwoClassConfidenceClassifier(confidence_tau=self.config.confidence_tau)
        two_class_classifier.fit(X_train, y_train, MLPClassifier, model_params, self.config.use_class_weights)
        
        # Calibrate probabilities if enabled
        if self.config.calibrate_probabilities:
            two_class_classifier.calibrate_probabilities(X_val, y_val)
        
        # Optimize confidence threshold
        val_returns_bps = val_df['ret_bps'].values
        '''
        best_tau = two_class_classifier.optimize_confidence_tau_for_profit(
            X_val, y_val, val_returns_bps, 
            cost_bps=self.config.profit_cost_bps,
            tau_grid_start=self.config.confidence_grid_start,
            tau_grid_end=self.config.confidence_grid_end,
            tau_grid_step=self.config.confidence_grid_step
        )
        
        two_class_classifier.confidence_tau = best_tau
        logger.info(f"Optimized confidence tau: {best_tau:.3f}")
        '''
        if self.config.optimize_threshold_for_profit:
            chosen_tau = two_class_classifier.optimize_confidence_tau_for_profit(
                X_val, y_val, val_returns_bps,
                cost_bps=self.config.profit_cost_bps,
                tau_grid_start=self.config.confidence_grid_start,
                tau_grid_end=self.config.confidence_grid_end,
                tau_grid_step=self.config.confidence_grid_step,
                select_by=self.config.optimize_tau_by,             # <-- НОВОЕ
                min_coverage=self.config.min_coverage,             # <-- НОВОЕ (может быть 0.0)
                output_dir=self.config.output_dir                  # <-- чтобы сохранить CSV рядом с артефактами
            )
            logger.info(f"Optimized confidence tau: {chosen_tau:.3f}")
        else:
            if self.config.confidence_tau is None:
                raise ValueError("Pass --confidence_tau or enable --optimize_threshold_for_profit.")
            chosen_tau = float(self.config.confidence_tau)
            logger.info(f"Using provided confidence tau: {chosen_tau:.3f}")

        two_class_classifier.confidence_tau = chosen_tau
        self._save_decision_threshold_config(chosen_tau, val_df, two_class_classifier)


        # Generate test predictions
        test_direction, test_confidence, test_execute = two_class_classifier.predict_with_confidence(X_test)
        test_proba = two_class_classifier.predict_proba(X_test)
        
        # Save model
        two_class_classifier.save_model(self.config.output_dir)
        
        self.model = two_class_classifier
        
        return {
            'model': two_class_classifier,
            'test_pred': test_direction,
            'test_proba': test_proba,
            'test_confidence': test_confidence,
            'test_execute': test_execute,
            'y_test': y_test,
            'training_history': [],  # Two-class doesn't have unified history
            'mode': 'two_class'
        }
    
    def _save_decision_threshold_config(self, chosen_tau: float, val_df: pd.DataFrame,
                                        classifier: TwoClassConfidenceClassifier) -> None:
        X_val = self.feature_processor.transform(
            self.feature_processor.select_features(val_df)
        )
        val_direction, val_confidence, val_execute = classifier.predict_with_confidence(X_val)
        val_returns_bps = val_df['ret_bps'].values
        if val_execute.sum() > 0:
            val_gross = np.where(val_direction[val_execute] == 1,
                                val_returns_bps[val_execute],
                                -val_returns_bps[val_execute])
            val_net = val_gross - self.config.profit_cost_bps
            val_profit_mean = float(val_net.mean())
        else:
            val_profit_mean = -self.config.profit_cost_bps
        
        decision_config = {
            "mode": "two_class",
            "confidence_tau": float(chosen_tau),
            "cost_bps": float(self.config.profit_cost_bps),
            "val_profit_bps": val_profit_mean,
            "val_coverage": float(val_execute.mean()),
            "val_n_executed": int(val_execute.sum()),
            "tau_grid_start": float(self.config.confidence_grid_start),
            "tau_grid_end": float(self.config.confidence_grid_end),
            "tau_grid_step": float(self.config.confidence_grid_step)
        }
        
        with open(self.config.output_dir / 'decision_threshold.json', 'w') as f:
            json.dump(decision_config, f, indent=2)
        
        logger.info(f"Saved decision threshold config to {self.config.output_dir}/decision_threshold.json")
        logger.info(f"Validation profit with tau={chosen_tau:.3f}: {val_profit_mean:.2f} bps "
                   f"(coverage: {val_execute.mean():.1%})")
    
    def _train_sklearn_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced sklearn MLP training."""
        logger.info("Using sklearn MLPClassifier")
        
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        
        if self.config.task == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config.mlp_hidden_sizes),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=min(200, max(50, len(X_train)//100)),
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.epochs,
                early_stopping=True,  # Enable early stopping for better performance
                n_iter_no_change=self.config.early_stopping_patience,
                validation_fraction=0.1,
                random_state=self.config.seed
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=tuple(self.config.mlp_hidden_sizes),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=min(200, max(50, len(X_train)//100)),
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.epochs,
                early_stopping=False,
                n_iter_no_change=self.config.early_stopping_patience,
                random_state=self.config.seed
            )
        
        # Apply class weighting
        sample_weights = None
        if self.config.task == 'classification' and self.config.use_class_weights:
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train).astype(np.float32)
            logger.info(f"Sample weights: mean={sample_weights.mean():.4f}, std={sample_weights.std():.4f}")
        
        # Train model
        logger.info("Training sklearn MLP...")
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        # Generate predictions
        test_pred = model.predict(X_test)
        
        if self.config.task == 'classification':
            test_proba = model.predict_proba(X_test)
        else:
            test_proba = None
        
        # Save model
        joblib.dump(model, self.config.output_dir / 'model.joblib')
        
        # Training history
        self.training_history = [
            {'epoch': i+1, 'train_loss': loss} 
            for i, loss in enumerate(model.loss_curve_)
        ]
        
        self.model = model
        
        return {
            'model': model,
            'test_pred': test_pred,
            'test_proba': test_proba,
            'y_test': y_test,
            'training_history': self.training_history,
            'mode': 'standard'
        }
    
    def _train_two_stage_sklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced two-stage sklearn training."""
        logger.info("Using enhanced two-stage sklearn approach")
        
        from sklearn.neural_network import MLPClassifier
        
        # Extract targets from capped training data
        y_binary_train = self.train_df_capped['y_binary'].values
        y_direction_train = self.train_df_capped['y_direction'].values
        
        # For validation (threshold optimization)
        y_binary_val = val_df['y_binary'].values
        y_direction_val = val_df['y_direction'].values
        
        # Model parameters with early stopping enabled
        model_params = {
            'hidden_layer_sizes': tuple(self.config.mlp_hidden_sizes),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'batch_size': min(200, max(50, len(X_train)//100)),
            'learning_rate_init': self.config.learning_rate,
            'max_iter': self.config.epochs,
            'early_stopping': True,  # Enable early stopping for better performance
            'n_iter_no_change': self.config.early_stopping_patience,
            'validation_fraction': 0.1,
            'random_state': self.config.seed
        }
        
        # Create and train two-stage classifier
        two_stage = TwoStageClassifier(trade_threshold=self.config.two_stage_trade_threshold)
        two_stage.fit(X_train, y_binary_train, y_direction_train, 
                     MLPClassifier, model_params, self.config.use_class_weights)
        
        # Calibrate probabilities if enabled
        if self.config.calibrate_probabilities:
            two_stage.calibrate_probabilities(X_val, y_binary_val, y_direction_val)
        
        # Optimize threshold
        if self.config.optimize_threshold_for_profit and 'ret_bps' in val_df.columns:
            val_returns_bps = val_df['ret_bps'].values
            best_threshold = two_stage.optimize_threshold_for_profit(
                X_val, y_val, val_returns_bps, cost_bps=self.config.profit_cost_bps
            )
        else:
            best_threshold = self._optimize_trade_threshold(
                two_stage, X_val, y_val, y_binary_val, y_direction_val
            )
        
        two_stage.trade_threshold = best_threshold
        logger.info(f"Optimized trade threshold: {best_threshold:.3f}")
        
        # Generate test predictions
        test_pred, test_proba = two_stage.predict(X_test)
        
        # Save models
        two_stage.save_models(self.config.output_dir)
        
        self.model = two_stage
        
        return {
            'model': two_stage,
            'test_pred': test_pred,
            'test_proba': test_proba,
            'y_test': y_test,
            'training_history': [],
            'mode': 'two_stage'
        }
    
    def _optimize_trade_threshold(self, two_stage: TwoStageClassifier, X_val: np.ndarray, 
                                y_val: np.ndarray, y_binary_val: np.ndarray, 
                                y_direction_val: np.ndarray) -> float:
        """Optimize trade threshold on validation set."""
        logger.info("Optimizing trade threshold using balanced accuracy")
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = -1.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            two_stage.trade_threshold = threshold
            val_pred, _ = two_stage.predict(X_val)
            score = balanced_accuracy_score(y_val, val_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} (Balanced Acc: {best_score:.4f})")
        return best_threshold

    def _train_lstm_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                         X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model with corrected sequence generation."""
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch required for LSTM training")
        
        logger.info(f"Training LSTM model with sequence_len={self.config.sequence_len}")
        
        # For LSTM, we need to transform the dataframes first, then create sequences
        # This ensures proper scaling/imputation before sequence creation
        
        # Transform dataframes to scaled arrays first
        train_df_scaled = self.train_df_capped.copy()
        train_features_scaled = self.feature_processor.transform(
            self.feature_processor.select_features(train_df_scaled)
        )
        
        val_df_scaled = val_df.copy()  
        val_features_scaled = self.feature_processor.transform(
            self.feature_processor.select_features(val_df_scaled)
        )
        
        test_df_scaled = test_df.copy()
        test_features_scaled = self.feature_processor.transform(
            self.feature_processor.select_features(test_df_scaled)
        )
        
        # Add scaled features back to dataframes for sequence generation
        feature_cols = self.feature_processor.feature_names
        train_df_scaled[feature_cols] = train_features_scaled
        val_df_scaled[feature_cols] = val_features_scaled
        test_df_scaled[feature_cols] = test_features_scaled
        
        # Create sequences with corrected generator
        seq_generator = SequenceDataGenerator(self.config.sequence_len, self.config.horizon_min)
        
        X_train_seq, y_train_seq = seq_generator.create_sequences(train_df_scaled, feature_cols)
        X_val_seq, y_val_seq = seq_generator.create_sequences(val_df_scaled, feature_cols)
        X_test_seq, y_test_seq = seq_generator.create_sequences(test_df_scaled, feature_cols)
        
        logger.info(f"Sequence shapes: Train {X_train_seq.shape}, Val {X_val_seq.shape}, Test {X_test_seq.shape}")
        
        # Create LSTM model
        if self.config.task == 'classification':
            if self.config.two_class_mode:
                num_classes = 2  # Binary classification
                logger.info("LSTM Two-class mode: 2 classes")
            else:
                num_classes = len(np.unique(y_train_seq))
                logger.info(f"LSTM Classification with {num_classes} classes")
        else:
            num_classes = 1
            logger.info("LSTM Regression")
        
        model = LSTMClassifier(
            input_dim=X_train_seq.shape[2],
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            num_classes=num_classes,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        logger.info(f"LSTM Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup training
        if self.config.task == 'classification':
            class_weights = None
            if self.config.use_class_weights:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train_seq)
                class_weights_values = compute_class_weight('balanced', classes=classes, y=y_train_seq)
                class_weights = torch.FloatTensor(class_weights_values).to(self.device)
                logger.info(f"LSTM Class weights: {class_weights_values}")
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.config.reduce_lr_patience, factor=0.5
        )
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq).to(self.device),
            torch.LongTensor(y_train_seq).to(self.device) if self.config.task == 'classification'
            else torch.FloatTensor(y_train_seq).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq).to(self.device),
            torch.LongTensor(y_val_seq).to(self.device) if self.config.task == 'classification'
            else torch.FloatTensor(y_val_seq).to(self.device)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                              shuffle=False, num_workers=0, pin_memory=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Enable AMP for faster training
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        if self.config.task == 'classification':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.config.gradient_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    if self.config.task == 'classification':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.config.gradient_clip_value)
                    optimizer.step()
                
                train_loss += loss.item()
                train_total += batch_y.size(0)
                
                if self.config.task == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_X)
                            if self.config.task == 'classification':
                                loss = criterion(outputs, batch_y)
                            else:
                                loss = criterion(outputs.squeeze(), batch_y)
                    else:
                        outputs = model(batch_X)
                        if self.config.task == 'classification':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y)
                    
                    val_loss += loss.item()
                    val_total += batch_y.size(0)
                    
                    if self.config.task == 'classification':
                        _, predicted = torch.max(outputs.data, 1)
                        val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics and log progress
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            if self.config.task == 'classification':
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                progress_pct = (epoch + 1) / self.config.epochs * 100
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} ({progress_pct:.1f}%): "
                           f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                progress_pct = (epoch + 1) / self.config.epochs * 100
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} ({progress_pct:.1f}%): "
                           f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling and early stopping
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.config.output_dir / 'model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Store training history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            if self.config.task == 'classification':
                history_entry['train_acc'] = train_acc
                history_entry['val_acc'] = val_acc
            
            self.training_history.append(history_entry)
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.config.output_dir / 'model.pth'))
        model.eval()
        
        # Generate test predictions
        with torch.no_grad():
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_seq).to(self.device),
                torch.LongTensor(test_lengths).to(self.device)
            )
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                                   shuffle=False, num_workers=0, pin_memory=True)
            
            test_predictions = []
            test_probabilities = []
            
            for batch_X, batch_lengths in test_loader:
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X, batch_lengths)
                else:
                    outputs = model(batch_X, batch_lengths)
                
                if self.config.task == 'classification':
                    proba = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                    
                    test_predictions.extend(pred.cpu().numpy())
                    test_probabilities.extend(proba.cpu().numpy())
                else:
                    pred = outputs.squeeze()
                    test_predictions.extend(pred.cpu().numpy())
            
            test_pred = np.array(test_predictions)
            test_proba = np.array(test_probabilities) if test_probabilities else None
        
        self.model = model
        
        return {
            'model': model,
            'test_pred': test_pred,
            'test_proba': test_proba,
            'y_test': y_test_seq,
            'training_history': self.training_history,
            'mode': 'two_class' if self.config.two_class_mode else 'standard'
        }
    
    def _evaluate_and_save_results(self, results: Dict[str, Any]) -> None:
        """Evaluate model performance and save results."""
        logger.info("Evaluating model performance...")
        
        # Calculate metrics based on mode
        if results.get('mode') == 'two_class':
            self._evaluate_two_class_results(results)
        else:
            self._evaluate_standard_results(results)
        
        logger.info("Results saved successfully")
    
    def _evaluate_two_class_results(self, results: Dict[str, Any]) -> None:
        """Evaluate and save results for two-class mode."""
        test_df = results['test_df']
        
        # Extract predictions
        direction_pred = results['test_pred']
        confidence = results.get('test_confidence', np.ones(len(direction_pred)))
        execute = results.get('test_execute', np.ones(len(direction_pred), dtype=bool))
        returns_bps = test_df['ret_bps'].values
        y_direction_true = results['y_test']
        
        # Calculate two-class specific metrics
        metrics_calc = MetricsCalculator(self.config.task)
        two_class_metrics = metrics_calc.calculate_two_class_metrics(
            y_direction_true, direction_pred, confidence, execute, 
            returns_bps, self.config.profit_cost_bps
        )
        
        # Calculate per-symbol metrics
        per_symbol_metrics = metrics_calc.calculate_two_class_per_symbol_metrics(
            test_df, direction_pred, confidence, execute, returns_bps, self.config.profit_cost_bps
        )
        
        # Log key metrics
        self._log_two_class_metrics(two_class_metrics)
        
        # Save results
        self._save_two_class_artifacts(two_class_metrics, per_symbol_metrics, results)
    
    def _evaluate_standard_results(self, results: Dict[str, Any]) -> None:
        """Evaluate and save results for standard mode."""
        # Calculate global metrics
        metrics_calc = MetricsCalculator(self.config.task)
        global_metrics = metrics_calc.calculate_metrics(
            results['y_test'], results['test_pred'], results.get('test_proba')
        )
        
        # Calculate per-symbol metrics
        if 'test_df' in results:
            per_symbol_metrics = metrics_calc.calculate_per_symbol_metrics(
                results['test_df'], results['y_test'], results['test_pred']
            )
        else:
            per_symbol_metrics = pd.DataFrame()
        
        # Log key metrics
        self._log_final_metrics(global_metrics)
        
        # Save all results
        self._save_training_artifacts(global_metrics, per_symbol_metrics, results)
    

    def _log_two_class_metrics(self, metrics: dict):
        logger.info("Two-class mode test metrics:")
        logger.info(f"  Coverage: {metrics.get('coverage', 0.0):.1%} ({metrics.get('n_executed', 0)}/{metrics.get('n_total', 0)} trades)")
        logger.info(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0.0):.4f}")
        logger.info(f"  Average Profit: {metrics.get('avg_profit_bps', 0.0):.2f} bps")
        logger.info(f"  Median Profit: {metrics.get('median_profit_bps', 0.0):.2f} bps")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0.0):.1%}")
        logger.info(f"  Average Confidence: {metrics.get('avg_confidence', 0.0):.3f}")
        logger.info(f"  Profit Sharpe: {metrics.get('profit_sharpe', 0.0):.3f}")
        logger.info(f"  Profit Range: [{metrics.get('min_profit_bps', 0.0):.1f}, {metrics.get('max_profit_bps', 0.0):.1f}] bps")

    
    def _log_final_metrics(self, global_metrics: Dict[str, float]) -> None:
        """Log final performance metrics."""
        logger.info("Global test metrics:")
        if self.config.task == 'classification':
            logger.info(f"  Accuracy: {global_metrics['accuracy']:.4f}")
            logger.info(f"  Balanced Accuracy: {global_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  F1 (macro): {global_metrics['f1_macro']:.4f}")
            
            # Trading-specific metrics
            if 'trade_hit_rate' in global_metrics:
                logger.info(f"  Trade Hit Rate: {global_metrics['trade_hit_rate']:.4f}")
            if 'direction_accuracy' in global_metrics:
                logger.info(f"  Direction Accuracy: {global_metrics['direction_accuracy']:.4f}")
            
            if 'roc_auc' in global_metrics:
                logger.info(f"  ROC-AUC: {global_metrics['roc_auc']:.4f}")
            if 'pr_auc_macro' in global_metrics:
                logger.info(f"  PR-AUC (macro): {global_metrics['pr_auc_macro']:.4f}")
        else:
            logger.info(f"  MAE: {global_metrics['mae']:.4f}")
            logger.info(f"  RMSE: {global_metrics['rmse']:.4f}")
            logger.info(f"  R²: {global_metrics['r2']:.4f}")
            logger.info(f"  Sign Accuracy: {global_metrics['sign_accuracy']:.4f}")

    def _save_two_class_artifacts(self, metrics: Dict[str, float], 
                                per_symbol_metrics: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Save artifacts specific to two-class mode."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration (enhanced for two-class)
        config_dict = self._create_base_config_dict()
        config_dict.update({
            'two_class_mode': True,
            'confidence_tau': self.config.confidence_tau,
            'confidence_grid_start': self.config.confidence_grid_start,
            'confidence_grid_end': self.config.confidence_grid_end,
            'confidence_grid_step': self.config.confidence_grid_step,
            'profit_cost_bps': self.config.profit_cost_bps,
        })
        
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, indent=2)
        
        # Save two-class specific metrics
        with open(output_dir / 'metrics_two_class.json', 'w') as f:
            serializable_metrics = self._make_json_serializable(metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save per-symbol metrics
        if not per_symbol_metrics.empty:
            per_symbol_metrics.to_csv(output_dir / 'metrics_per_symbol_two_class.csv', index=False)
        
        # Save training history
        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(output_dir / 'training_history.csv', index=False)
        
        # Save predictions with two-class format
        if self.config.save_predictions:
            self._save_two_class_predictions(results, output_dir)
        
        logger.info(f"Two-class artifacts saved to {output_dir}")
    
    def _save_two_class_predictions(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save predictions in two-class format."""
        test_df = results['test_df']
        direction_pred = results['test_pred']
        proba = results.get('test_proba')
        confidence = results.get('test_confidence', np.ones(len(direction_pred)))
        execute = results.get('test_execute', np.ones(len(direction_pred), dtype=bool))
        
        # Create predictions dataframe
        predictions_df = test_df[['timestamp', 'symbol', 'y_direction', 'ret_bps']].copy()
        predictions_df['pred_direction'] = direction_pred
        predictions_df['confidence'] = confidence
        predictions_df['execute'] = execute
        
        if proba is not None:
            predictions_df['proba_down'] = proba[:, 0]
            predictions_df['proba_up'] = proba[:, 1]
        
        # Calculate actual profits only where we trade
        gross = np.where(direction_pred == 1,
                        test_df['ret_bps'].values,
                        -test_df['ret_bps'].values)

        predictions_df['gross_profit_bps'] = np.where(execute, gross, 0.0)
        predictions_df['net_profit_bps']   = np.where(
            execute, gross - self.config.profit_cost_bps, 0.0
        )
      
       
        predictions_df.to_csv(output_dir / 'test_predictions_two_class.csv', index=False)
    
    def _save_training_artifacts(self, global_metrics: Dict[str, float],
                               per_symbol_metrics: pd.DataFrame,
                               results: Dict[str, Any]) -> None:
        """Save all training artifacts and results."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = self._create_base_config_dict()
        
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_dict, f, indent=2)
        
        # Save global metrics
        with open(output_dir / 'metrics_global.json', 'w') as f:
            serializable_metrics = self._make_json_serializable(global_metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save per-symbol metrics
        if not per_symbol_metrics.empty:
            per_symbol_metrics.to_csv(output_dir / 'metrics_per_symbol.csv', index=False)
        
        # Save training history
        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(output_dir / 'training_history.csv', index=False)
        
        # Save predictions if requested
        if self.config.save_predictions and 'test_df' in results:
            predictions_df = results['test_df'][['symbol', 'timestamp', 'y']].copy()
            predictions_df['y_pred'] = results['test_pred']
            
            if results.get('test_proba') is not None:
                proba_df = pd.DataFrame(results['test_proba'], 
                                      columns=[f'prob_class_{i}' for i in range(results['test_proba'].shape[1])])
                predictions_df = pd.concat([predictions_df, proba_df], axis=1)
            
            predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
        
        logger.info(f"All artifacts saved to {output_dir}")
    
    def _create_base_config_dict(self) -> Dict[str, Any]:
        """Create base configuration dictionary."""
        return {
            'unified_data_path': str(self.config.unified_data_path),
            'output_dir': str(self.config.output_dir),
            'task': self.config.task,
            'horizon_min': self.config.horizon_min,
            'deadband_bps': self.config.deadband_bps,
            'model_type': self.config.model_type,
            'sequence_len': self.config.sequence_len,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'mlp_hidden_sizes': self.config.mlp_hidden_sizes,
            'lstm_hidden_size': self.config.lstm_hidden_size,
            'lstm_num_layers': self.config.lstm_num_layers,
            'dropout_rate': self.config.dropout_rate,
            'use_class_weights': self.config.use_class_weights,
            'use_two_stage': self.config.use_two_stage,
            'two_stage_trade_threshold': self.config.two_stage_trade_threshold,
            'calibrate_probabilities': self.config.calibrate_probabilities,
            'optimize_threshold_for_profit': self.config.optimize_threshold_for_profit,
            'max_train_size': self.config.max_train_size,
            'top_k_features': self.config.top_k_features,
            'use_robust_scaler': self.config.use_robust_scaler,
            'feature_selection_method': self.config.feature_selection_method,
            'seed': self.config.seed,
            'training_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _make_json_serializable(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to JSON serializable types."""
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                serializable_metrics[k] = v.item()
            else:
                serializable_metrics[k] = v
        return serializable_metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced options including two-class mode."""
    parser = argparse.ArgumentParser(
        description="Enhanced Centralized Training for Cryptocurrency Price Movement Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--unified', type=str, required=True,
                       help='Path to unified dataset parquet file')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for experiments')
    
    # Task configuration
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification',
                       help='Prediction task type')
    parser.add_argument('--horizon_min', type=int, default=5,
                       help='Prediction horizon in minutes')
    parser.add_argument('--deadband_bps', type=float, default=2.0,
                       help='Dead zone threshold in basis points')
    
    # Model configuration
    parser.add_argument('--model', choices=['mlp', 'lstm'], default='mlp',
                       help='Model architecture')
    parser.add_argument('--sequence_len', type=int, default=60,
                       help='Sequence length for LSTM (minutes)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    
    # Data splits
    parser.add_argument('--validation_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    
    # Model architecture
    parser.add_argument('--mlp_hidden_sizes', nargs='+', type=int, default=[256, 128, 64],
                       help='Hidden layer sizes for MLP')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                       help='Hidden size for LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    
    # Training optimization
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--reduce_lr_patience', type=int, default=5,
                       help='ReduceLROnPlateau patience')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Class weighting and two-stage arguments
    parser.add_argument('--no_class_weights', action='store_true',
                       help='Disable class weighting')
    parser.add_argument('--use_two_stage', action='store_true',
                       help='Enable two-stage training')
    parser.add_argument('--two_stage_threshold', type=float, default=0.5,
                       help='Initial trade threshold for two-stage approach')
    
    # Two-class mode arguments
    parser.add_argument('--two_class_mode', action='store_true',
                       help='Enable two-class mode (Up vs Down only)')
    parser.add_argument('--confidence_tau', type=float, default=0.7,
                       help='Initial confidence threshold for two-class mode')
    parser.add_argument('--confidence_grid', type=str, default='0.50:0.95:0.01',
                       help='Confidence grid for optimization (start:end:step)')
    parser.add_argument('--profit_cost_bps', type=float, default=1.0,
                       help='Trading cost in basis points')
    
    # Enhanced preprocessing arguments
    parser.add_argument('--max_train_size', type=int, default=1_500_000,
                       help='Maximum training samples')
    parser.add_argument('--top_k_features', type=int, default=128,
                       help='Top-k features to select (0 = no selection)')
    parser.add_argument('--use_robust_scaler', action='store_true',
                       help='Use RobustScaler instead of StandardScaler')
    parser.add_argument('--feature_selection_method', 
                       choices=['variance', 'mutual_info', 'f_classif'], 
                       default='mutual_info',
                       help='Feature selection method')
    
    # Calibration and threshold optimization
    parser.add_argument('--calibrate_probabilities', action='store_true',
                       help='Enable probability calibration')
    parser.add_argument('--optimize_threshold_for_profit', action='store_true',
                       help='Optimize threshold for profit instead of balanced accuracy')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Compute device')
    parser.add_argument('--no_save_predictions', action='store_true',
                       help='Disable saving predictions')
    

    parser.add_argument(
        "--optimize_tau_by",
        choices=["profit", "ev", "profit_with_min_coverage"],
        default="profit",
        help="Criterion to select tau when optimizing: profit (per trade), ev (EV_per_obs = profit*coverage), or profit_with_min_coverage"
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.0,
        help="Minimum coverage constraint for profit_with_min_coverage (e.g., 0.005 for 0.5%)"
    )

    
    return parser.parse_args()


def create_experiment_directory(base_dir: str, config: TrainingConfig) -> Path:
    """Create experiment directory with timestamp and configuration info."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = f"{timestamp}_{config.model_type}_h{config.horizon_min}_db{config.deadband_bps}"
    
    if config.model_type == 'lstm':
        exp_name += f"_seq{config.sequence_len}"
    
    if config.two_class_mode:
        exp_name += "_2class"
    elif config.use_two_stage:
        exp_name += "_2stage"
        
    if config.calibrate_probabilities:
        exp_name += "_cal"
    
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def validate_config(config: TrainingConfig) -> None:
    """Validate configuration parameters."""
    if not config.unified_data_path.exists():
        raise ValueError(f"Unified dataset not found: {config.unified_data_path}")
    
    if config.horizon_min <= 0:
        raise ValueError("horizon_min must be positive")
    
    if config.deadband_bps < 0:
        raise ValueError("deadband_bps must be non-negative")
    
    if config.validation_split + config.test_split >= 1.0:
        raise ValueError("validation_split + test_split must be < 1.0")
    
    if config.model_type == 'lstm' and config.sequence_len <= 0:
        raise ValueError("sequence_len must be positive for LSTM")
    
    if not TORCH_AVAILABLE and config.model_type == 'lstm':
        raise ValueError("PyTorch required for LSTM training")
    
    # Two-class mode validations
    if config.two_class_mode:
        if config.task != 'classification':
            raise ValueError("Two-class mode only supports classification task")
        if config.use_two_stage:
            logger.warning("Two-stage training disabled in two-class mode")
            config.use_two_stage = False
        if config.confidence_tau <= 0.5 or config.confidence_tau >= 1.0:
            raise ValueError("confidence_tau must be in (0.5, 1.0)")
    
    # Confidence grid validation
    if config.two_class_mode:
        if (config.confidence_grid_start >= config.confidence_grid_end or 
            config.confidence_grid_step <= 0):
            raise ValueError("Invalid confidence grid parameters")


def check_class_balance(y: np.ndarray, min_class_ratio: float = 0.05, 
                       task: str = 'classification', mode: str = 'standard') -> None:
    """Check class balance and warn about severe imbalance."""
    if task != 'classification':
        return
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total = len(y)
    class_ratios = class_counts / total
    
    logger.info(f"Class distribution ({mode} mode):")
    
    if mode == 'two_class':
        class_names = {0: 'Down', 1: 'Up'}
    else:
        class_names = {0: 'Down', 1: 'Up', 2: 'No-trade'}
        
    for cls, count, ratio in zip(unique_classes, class_counts, class_ratios):
        name = class_names.get(cls, f'Class-{cls}')
        logger.info(f"  {name}: {count:,} ({ratio:.1%})")
    
    # Check for severely imbalanced classes using config-consistent threshold
    min_ratio = class_ratios.min()
    if min_ratio < min_class_ratio:
        logger.warning(f"Severely imbalanced classes detected!")
        logger.warning(f"Minimum class ratio: {min_ratio:.1%} < {min_class_ratio:.1%}")
        logger.warning("Consider:")
        logger.warning("  - Adjusting deadband_bps")
        logger.warning("  - Using class weighting (enabled by default)")
        if mode != 'two_class':
            logger.warning("  - Using two-stage approach (--use_two_stage)")
            logger.warning("  - Using two-class mode (--two_class_mode)")
        logger.warning("  - Collecting more data")
        
        if min_ratio < 0.01:
            raise ValueError(f"Class too small: {min_ratio:.1%} < 1%. Cannot train reliably.")


def parse_confidence_grid(grid_str: str) -> Tuple[float, float, float]:
    """Parse confidence grid string format: start:end:step"""
    try:
        parts = grid_str.split(':')
        if len(parts) != 3:
            raise ValueError
        start, end, step = map(float, parts)
        if start >= end or step <= 0:
            raise ValueError
        return start, end, step
    except (ValueError, TypeError):
        raise ValueError(f"Invalid confidence grid format: {grid_str}. Expected start:end:step")


def main():
    """Main execution function with enhanced error handling and two-class mode support."""
    # Parse arguments
    args = parse_arguments()
    
    # Parse confidence grid
    if hasattr(args, 'confidence_grid'):
        conf_start, conf_end, conf_step = parse_confidence_grid(args.confidence_grid)
    else:
        conf_start, conf_end, conf_step = 0.50, 0.95, 0.01
    
    # Create experiment directory
    exp_dir = create_experiment_directory(args.out, TrainingConfig(
        unified_data_path=Path(args.unified),
        output_dir=Path(args.out),
        model_type=args.model,
        horizon_min=args.horizon_min,
        deadband_bps=args.deadband_bps,
        use_two_stage=args.use_two_stage,
        two_class_mode=args.two_class_mode
    ))
    
    # Create enhanced configuration
    config = TrainingConfig(
        unified_data_path=Path(args.unified),
        output_dir=exp_dir,
        task=args.task,
        horizon_min=args.horizon_min,
        deadband_bps=args.deadband_bps,
        model_type=args.model,
        sequence_len=args.sequence_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        test_split=args.test_split,
        mlp_hidden_sizes=args.mlp_hidden_sizes,
        dropout_rate=args.dropout_rate,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        gradient_clip_value=args.gradient_clip_value,
        use_class_weights=not args.no_class_weights,
        use_two_stage=args.use_two_stage,
        two_stage_trade_threshold=args.two_stage_threshold,
        # Two-class mode parameters
        two_class_mode=args.two_class_mode,
        confidence_tau=args.confidence_tau,
        confidence_grid_start=conf_start,
        confidence_grid_end=conf_end,
        confidence_grid_step=conf_step,
        profit_cost_bps=args.profit_cost_bps,
        # Other parameters
        max_train_size=args.max_train_size,
        top_k_features=args.top_k_features if args.top_k_features > 0 else None,
        use_robust_scaler=args.use_robust_scaler,
        feature_selection_method=args.feature_selection_method,
        calibrate_probabilities=args.calibrate_probabilities,
        optimize_threshold_for_profit=args.optimize_threshold_for_profit,
        seed=args.seed,
        device=args.device,
        save_predictions=not args.no_save_predictions


    )
    
    # Validate configuration
    validate_config(config)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger.info("="*80)
    if config.two_class_mode:
        logger.info("ENHANCED CENTRALIZED TRAINING - TWO-CLASS MODE")
    else:
        logger.info("ENHANCED CENTRALIZED TRAINING FOR CRYPTOCURRENCY PRICE MOVEMENT PREDICTION")
    logger.info("="*80)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration:")
    logger.info(f"  Task: {config.task}")
    logger.info(f"  Model: {config.model_type}")
    logger.info(f"  Mode: {'Two-class (Up/Down only)' if config.two_class_mode else 'Standard (Up/Down/No-trade)'}")
    logger.info(f"  Horizon: {config.horizon_min} minutes")
    logger.info(f"  Deadband: {config.deadband_bps} bps")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Class weights: {config.use_class_weights}")
    
    if config.two_class_mode:
        logger.info(f"  Confidence tau: {config.confidence_tau}")
        logger.info(f"  Confidence grid: {config.confidence_grid_start}:{config.confidence_grid_end}:{config.confidence_grid_step}")
        logger.info(f"  Trading cost: {config.profit_cost_bps} bps")
    else:
        logger.info(f"  Two-stage training: {config.use_two_stage}")
        logger.info(f"  Calibrate probabilities: {config.calibrate_probabilities}")
        logger.info(f"  Optimize for profit: {config.optimize_threshold_for_profit}")
    
    logger.info(f"  Feature selection: {config.feature_selection_method}")
    logger.info(f"  Robust scaler: {config.use_robust_scaler}")
    logger.info(f"  Max train size: {config.max_train_size:,}")
    logger.info(f"  Top-k features: {config.top_k_features}")
    logger.info(f"  Seed: {config.seed}")
    
    if config.model_type == 'lstm':
        logger.info(f"  Sequence length: {config.sequence_len}")
        logger.info(f"  LSTM hidden size: {config.lstm_hidden_size}")
        logger.info(f"  LSTM layers: {config.lstm_num_layers}")
    else:
        logger.info(f"  MLP architecture: {config.mlp_hidden_sizes}")
    
    try:
        # Initialize trainer
        trainer = CentralizedTrainer(config)
        
        # Run training
        results = trainer.train()
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results saved to: {exp_dir}")
        logger.info("Key files:")
        logger.info(f"  - Model: {exp_dir}/model.pth (or model.joblib)")
        
        if config.two_class_mode:
            logger.info(f"  - Two-class metrics: {exp_dir}/metrics_two_class.json")
            logger.info(f"  - Two-class model: {exp_dir}/two_class_model.joblib")
            logger.info(f"  - Decision threshold: {exp_dir}/decision_threshold.json")
            if config.save_predictions:
                logger.info(f"  - Predictions: {exp_dir}/test_predictions_two_class.csv")
        else:
            logger.info(f"  - Metrics: {exp_dir}/metrics_global.json")
            if config.use_two_stage:
                logger.info(f"  - Two-stage models: {exp_dir}/trade_model.joblib, {exp_dir}/direction_model.joblib")
            if config.save_predictions:
                logger.info(f"  - Predictions: {exp_dir}/test_predictions.csv")
        
        logger.info(f"  - Config: {exp_dir}/config.yaml")
        logger.info(f"  - Logs: {exp_dir}/logs.txt")
        logger.info(f"  - Preprocessors: {exp_dir}/imputer.joblib, {exp_dir}/scaler.joblib")
        
        # Final performance summary
        if config.two_class_mode and 'test_confidence' in results:
            logger.info("Final two-class test performance:")
            # Calculate final metrics quickly
            direction_pred = results['test_pred']
            confidence = results['test_confidence']
            execute = results['test_execute']
            returns_bps = results['test_df']['ret_bps'].values
            y_direction_true = results['y_test']
            
            coverage = execute.mean()
            n_executed = execute.sum()
            
            if n_executed > 0:
                executed_direction_pred = direction_pred[execute]
                executed_direction_true = y_direction_true[execute]
                executed_returns = returns_bps[execute]
                
                direction_acc = (executed_direction_pred == executed_direction_true).mean()
                gross_profits = np.where(executed_direction_pred == 1, executed_returns, -executed_returns)
                net_profits = gross_profits - config.profit_cost_bps
                avg_profit = net_profits.mean()
                win_rate = (net_profits > 0).mean()
                
                logger.info(f"  Coverage: {coverage:.1%} ({n_executed} trades)")
                logger.info(f"  Direction Accuracy: {direction_acc:.4f}")
                logger.info(f"  Average Profit: {avg_profit:.2f} bps")
                logger.info(f"  Win Rate: {win_rate:.1%}")
            else:
                logger.info(f"  Coverage: {coverage:.1%} (no trades executed)")
        
        elif 'test_df' in results and not config.two_class_mode:
            metrics_calc = MetricsCalculator(config.task)
            final_metrics = metrics_calc.calculate_metrics(
                results['y_test'], results['test_pred'], results.get('test_proba')
            )
            
            logger.info("Final test performance:")
            if config.task == 'classification':
                logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
                logger.info(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
                logger.info(f"  F1 (macro): {final_metrics['f1_macro']:.4f}")
                
                if 'trade_hit_rate' in final_metrics:
                    logger.info(f"  Trade Hit Rate: {final_metrics['trade_hit_rate']:.4f}")
                if 'direction_accuracy' in final_metrics:
                    logger.info(f"  Direction Accuracy: {final_metrics['direction_accuracy']:.4f}")
                
                if 'roc_auc' in final_metrics:
                    logger.info(f"  ROC-AUC: {final_metrics['roc_auc']:.4f}")
                if 'pr_auc_macro' in final_metrics:
                    logger.info(f"  PR-AUC (macro): {final_metrics['pr_auc_macro']:.4f}")
            else:
                logger.info(f"  MAE: {final_metrics['mae']:.4f}")
                logger.info(f"  RMSE: {final_metrics['rmse']:.4f}")
                logger.info(f"  R²: {final_metrics['r2']:.4f}")
                logger.info(f"  Sign Accuracy: {final_metrics['sign_accuracy']:.4f}")
        
        logger.info("="*80)
        logger.info("RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
        logger.info("1. Try different deadband_bps values (3-8 bps) for better class balance")
        logger.info("2. Experiment with feature engineering (rolling windows, cross-asset signals)")
        logger.info("3. Consider ensemble methods combining multiple models")
        logger.info("4. Implement cross-validation for more robust evaluation")
        logger.info("5. Add market regime detection for adaptive thresholds")
        
        if config.two_class_mode:
            logger.info("6. Try different confidence thresholds for risk/return tradeoff")
            logger.info("7. Experiment with different cost assumptions")
        else:
            if not config.use_two_stage:
                logger.info("6. Enable two-stage training with --use_two_stage for better performance")
            if not config.calibrate_probabilities:
                logger.info("7. Enable probability calibration with --calibrate_probabilities")
            logger.info("8. Try two-class mode with --two_class_mode for direct direction prediction")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Check logs for detailed error information")
        
        # Provide specific debugging guidance
        if "No valid sequences generated" in str(e):
            logger.error("DEBUGGING: Increase dataset size or reduce sequence_len for LSTM")
        elif "Memory" in str(e) or "CUDA out of memory" in str(e):
            logger.error("DEBUGGING: Reduce batch_size, max_train_size, or top_k_features")
        elif "No trade samples found" in str(e):
            logger.error("DEBUGGING: Reduce deadband_bps to get more trade samples")
        elif "Class too small" in str(e):
            logger.error("DEBUGGING: Adjust deadband_bps or collect more data")
        elif "Two-class mode expects binary targets" in str(e):
            logger.error("DEBUGGING: Check data filtering - should only have Up/Down samples")
        
        raise


if __name__ == "__main__":
    main()
