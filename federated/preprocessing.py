#!/usr/bin/env python3
"""
Federated Preprocessing Module
===================================

Preprocessing logic
Preserves all functionality for loading and applying centralized artifacts.
"""

import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class FederatedPreprocessor:
    """
    Enhanced preprocessor that loads and applies centralized preprocessing artifacts
    with strict feature alignment to prevent dimension mismatches.
    
    """
    
    def __init__(self, centralized_artifacts_dir: Path):
        self.centralized_artifacts_dir = Path(centralized_artifacts_dir)
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        self.feature_schema = None
        self.is_fitted = False
        self.top_k_features = None
        self.decision_threshold = None
        
        # Setup local logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def load_centralized_artifacts(self) -> bool:
        try:
            # Load imputer
            imputer_path = self.centralized_artifacts_dir / 'imputer.joblib'
            if imputer_path.exists():
                self.imputer = joblib.load(imputer_path)
                self.logger.info(f"Loaded imputer from {imputer_path}")
            else:
                self.logger.error(f"CRITICAL: Imputer not found at {imputer_path}")
                return False
                
            # Load scaler
            scaler_path = self.centralized_artifacts_dir / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler from {scaler_path}")
            else:
                self.logger.error(f"CRITICAL: Scaler not found at {scaler_path}")
                return False
            
            # Load feature schema 
            schema_path = self.centralized_artifacts_dir / 'feature_schema.json'
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self.feature_schema = json.load(f)
                self.feature_names = self.feature_schema['feature_names']
                self.top_k_features = len(self.feature_names)
                self.logger.info(f"Loaded {self.top_k_features} feature names from schema")

                expected_n_features = self.top_k_features
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != expected_n_features:
                    self.logger.warning(
                        f"Scaler was trained on {self.scaler.n_features_in_} features, "
                        f"but current feature set has {expected_n_features}. "
                        "Check the consistency of centralized provisioning artifacts."
                    )

                
                # Validate dimensions
                expected_n_features = len(self.feature_names)
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != expected_n_features:
                    self.logger.warning(f"Scaler was trained on {self.scaler.n_features_in_} features, "
                                      f"but schema has {expected_n_features}. May cause issues.")
                
            else:
                self.logger.error(f"CRITICAL: Feature schema not found at {schema_path}")
                return False

            #load decision threshold (tau) if present
            tau_path = self.centralized_artifacts_dir / 'decision_threshold.json'
            if tau_path.exists():
                try:
                    with open(tau_path, 'r') as f:
                        tau_obj = json.load(f)
                    self.decision_threshold = float(
                        tau_obj.get('tau', tau_obj.get('threshold', tau_obj.get('confidence_tau', 0.0)))
                    )
                    self.logger.info(f"Loaded decision threshold (tau)={self.decision_threshold:.4f} from {tau_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse decision threshold at {tau_path}: {e}")
                    self.decision_threshold = None
            else:
                self.logger.info("decision_threshold.json not found — will fall back to config.confidence_tau")

            self.is_fitted = True
            self.logger.info("Successfully loaded all centralized preprocessing artifacts")
            self.logger.info(f"Expected feature matrix shape: (N, {len(self.feature_names)})")
            return True

            
        except Exception as e:
            self.logger.error(f"Failed to load centralized artifacts: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def select_and_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Select features and apply preprocessing with STRICT alignment to centralized schema.
        """
        if not self.is_fitted:
            raise ValueError("Must load centralized artifacts first")
        
        try:
            # Reindex to full centralized feature schema, adding NaN for missing features
            self.logger.debug(f"Reindexing DataFrame to centralized schema with {len(self.feature_names)} features")
            
            # Create feature matrix with exact schema alignment
            X = df.reindex(columns=self.feature_names)
            
            # Log missing features for diagnostic purposes
            available_features = [f for f in self.feature_names if f in df.columns]
            missing_features = [f for f in self.feature_names if f not in df.columns]
            
            if missing_features:
                self.logger.info(f"Missing {len(missing_features)} features (will be imputed): "
                               f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
            self.logger.info(f"Feature alignment: {len(available_features)}/{len(self.feature_names)} "
                           f"available, {len(missing_features)} missing (filled with NaN)")
            
            # Handle infinite values before transformation
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Verify dimensions before transformation
            expected_shape = (len(df), len(self.feature_names))
            actual_shape = X.shape
            if actual_shape != expected_shape:
                raise ValueError(f"Feature matrix shape mismatch: expected {expected_shape}, got {actual_shape}")
            
            # Apply imputer and scaler with enhanced error handling
            self.logger.debug(f"Applying imputer to matrix of shape {X.shape}")
            X_imputed = self.imputer.transform(X)
            
            self.logger.debug(f"Applying scaler to matrix of shape {X_imputed.shape}")
            X_scaled = self.scaler.transform(X_imputed).astype(np.float64)
            
            # Enhanced cleaning for federated stability
            # Replace any remaining NaN/inf values
            nan_mask = np.isnan(X_scaled)
            inf_mask = np.isinf(X_scaled)
            if nan_mask.any() or inf_mask.any():
                self.logger.warning(f"Found {nan_mask.sum()} NaN and {inf_mask.sum()} inf values after scaling")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gentle clipping to prevent gradient explosion in federated setting
            extreme_mask = (np.abs(X_scaled) > 25.0)
            if extreme_mask.any():
                self.logger.warning(f"Clipping {extreme_mask.sum()} extreme values")
                np.clip(X_scaled, -25.0, 25.0, out=X_scaled)
            
            # Final verification
            final_shape = X_scaled.shape
            if final_shape[1] != len(self.feature_names):
                raise ValueError(f"Final feature matrix has wrong width: {final_shape[1]} != {len(self.feature_names)}")
            
            self.logger.debug(f"Successfully transformed data to shape {final_shape}")
            return X_scaled, self.feature_names
            
        except Exception as e:
            self.logger.error(f"Feature selection and transformation failed: {e}")
            self.logger.error(f"DataFrame shape: {df.shape}, Expected features: {len(self.feature_names)}")
            raise