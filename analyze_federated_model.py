#!/usr/bin/env python3
"""
Evaluate federated global model on centralized test set for fair comparison.
Saves full predictions with probabilities for tau analysis.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

def load_centralized_test_data(cent_dir: Path):
    """Load centralized test predictions to get the test index."""
    pred_path = cent_dir / "test_predictions_two_class.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Centralized predictions not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    return df[['timestamp', 'symbol', 'y_direction', 'ret_bps']].copy()

def load_federated_model(fed_dir: Path):
    """Load the global federated model."""
    model_path = fed_dir / "global_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Global model not found: {model_path}")
    return joblib.load(model_path)

def load_preprocessor(cent_dir: Path):
    """Load preprocessing artifacts from centralized training."""
    imputer = joblib.load(cent_dir / "imputer.joblib")
    scaler = joblib.load(cent_dir / "scaler.joblib")
    
    with open(cent_dir / "feature_schema.json") as f:
        schema = json.load(f)
    
    with open(cent_dir / "decision_threshold.json") as f:
        threshold_cfg = json.load(f)
    
    return imputer, scaler, schema['feature_names'], threshold_cfg

def prepare_features(df: pd.DataFrame, feature_names, imputer, scaler):
    """Apply same preprocessing as training."""
    # Reindex to match expected features
    X = df.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan)
    
    # Apply preprocessing
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    return X_scaled.astype(np.float32)

def evaluate_two_class(model, X_test, y_true, ret_bps, confidence_tau, cost_bps):
    """Evaluate with selective execution logic."""
    # Get probabilities
    proba = model.predict_proba(X_test)
    p_up = proba[:, 1]
    p_down = proba[:, 0]
    
    # Calculate confidence
    confidence = np.maximum(p_up, p_down)
    
    # Execution mask
    execute = confidence >= confidence_tau
    
    # Direction prediction
    direction = (p_up >= 0.5).astype(int)
    
    # Metrics on executed trades
    n_total = len(y_true)
    n_executed = int(execute.sum())
    coverage = float(n_executed / n_total) if n_total > 0 else 0.0
    
    if n_executed == 0:
        return {
            "coverage": 0.0,
            "n_executed": 0,
            "n_total": n_total,
            "direction_accuracy": 0.0,
            "avg_profit_bps": -cost_bps,
            "median_profit_bps": -cost_bps,
            "win_rate": 0.0,
            "avg_confidence": float(confidence.mean()),
        }, proba
    
    # Executed subset
    dir_exec = direction[execute]
    y_exec = y_true[execute]
    ret_exec = ret_bps[execute]
    conf_exec = confidence[execute]
    
    # Direction accuracy
    dir_acc = float((dir_exec == y_exec).mean())
    
    # Profit calculation
    gross_profit = np.where(dir_exec == 1, ret_exec, -ret_exec)
    net_profit = gross_profit - cost_bps
    
    return {
        "coverage": coverage,
        "n_executed": n_executed,
        "n_total": n_total,
        "direction_accuracy": dir_acc,
        "avg_profit_bps": float(net_profit.mean()),
        "median_profit_bps": float(np.median(net_profit)),
        "win_rate": float((net_profit > 0).mean()),
        "avg_confidence": float(conf_exec.mean()),
        "profit_std_bps": float(net_profit.std()),
        "profit_sharpe": float(net_profit.mean() / (net_profit.std() + 1e-8)),
    }, proba

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--federated_dir", required=True, type=Path,
                       help="Federated experiment directory")
    parser.add_argument("--centralized_dir", required=True, type=Path,
                       help="Centralized experiment directory (for artifacts and test set)")
    parser.add_argument("--unified_data", required=True, type=Path,
                       help="Path to unified dataset to extract test features")
    args = parser.parse_args()
    
    print("Loading centralized test set index...")
    test_index = load_centralized_test_data(args.centralized_dir)
    
    print("Loading federated global model...")
    model = load_federated_model(args.federated_dir)
    
    print("Loading preprocessing artifacts...")
    imputer, scaler, feature_names, threshold_cfg = load_preprocessor(args.centralized_dir)
    
    print("Loading full dataset and extracting test samples...")
    df_full = pd.read_parquet(args.unified_data)
    
    # Merge to get test samples with features
    test_index['timestamp'] = pd.to_datetime(test_index['timestamp'], utc=True)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], utc=True)
    
    df_test = pd.merge(
        test_index,
        df_full,
        on=['timestamp', 'symbol'],
        how='inner'
    )
    
    print(f"Test set: {len(df_test)} samples")
    
    print("Preparing features...")
    X_test = prepare_features(df_test, feature_names, imputer, scaler)
    y_test = df_test['y_direction'].values
    ret_test = df_test['ret_bps'].values
    
    print("Evaluating federated model on centralized test set...")
    metrics, proba = evaluate_two_class(
        model, X_test, y_test, ret_test,
        confidence_tau=threshold_cfg['confidence_tau'],
        cost_bps=threshold_cfg['cost_bps']
    )
    
    # Save metrics
    output_path = args.federated_dir / "metrics_on_centralized_test.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save full predictions with probabilities for tau analysis
    predictions_df = df_test[['timestamp', 'symbol', 'y_direction', 'ret_bps']].copy()
    predictions_df['proba_down'] = proba[:, 0]
    predictions_df['proba_up'] = proba[:, 1]
    predictions_df['pred_direction'] = (proba[:, 1] >= 0.5).astype(int)
    predictions_df['confidence'] = np.maximum(proba[:, 0], proba[:, 1])
    
    # Add execution flag based on default tau
    predictions_df['execute'] = predictions_df['confidence'] >= threshold_cfg['confidence_tau']
    
    # Calculate profits
    gross = np.where(
        predictions_df['pred_direction'] == 1,
        predictions_df['ret_bps'],
        -predictions_df['ret_bps']
    )
    predictions_df['gross_profit_bps'] = gross
    predictions_df['net_profit_bps'] = gross - threshold_cfg['cost_bps']
    
    pred_output_path = args.federated_dir / "predictions_on_centralized_test.csv"
    predictions_df.to_csv(pred_output_path, index=False)
    
    print("\n" + "="*60)
    print("FEDERATED MODEL ON CENTRALIZED TEST SET")
    print("="*60)
    print(f"Coverage: {metrics['coverage']:.1%} ({metrics['n_executed']}/{metrics['n_total']})")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    print(f"Avg Profit: {metrics['avg_profit_bps']:.2f} bps")
    print(f"Median Profit: {metrics['median_profit_bps']:.2f} bps")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Sharpe: {metrics['profit_sharpe']:.3f}")
    print("="*60)
    print(f"\nMetrics saved to: {output_path}")
    print(f"Predictions saved to: {pred_output_path}")
    
    # Load centralized metrics for comparison
    cent_metrics_path = args.centralized_dir / "metrics_two_class.json"
    if cent_metrics_path.exists():
        with open(cent_metrics_path) as f:
            cent_metrics = json.load(f)
        
        print("\nCOMPARISON:")
        print(f"{'Metric':<25} {'Federated':>12} {'Centralized':>12} {'Diff':>12}")
        print("-" * 65)
        
        comparisons = [
            ("Direction Accuracy", "direction_accuracy"),
            ("Coverage", "coverage"),
            ("Avg Profit (bps)", "avg_profit_bps"),
            ("Win Rate", "win_rate"),
        ]
        
        for label, key in comparisons:
            fed_val = metrics.get(key, 0)
            cent_val = cent_metrics.get(key, 0)
            diff = fed_val - cent_val
            
            if key in ["coverage", "win_rate"]:
                print(f"{label:<25} {fed_val:>11.1%} {cent_val:>11.1%} {diff:>+11.1%}")
            else:
                print(f"{label:<25} {fed_val:>12.4f} {cent_val:>12.4f} {diff:>+12.4f}")
    
    print("\n" + "="*60)
    print("Now you can use compare_models.py to analyze different tau values!")
    print("="*60)

if __name__ == "__main__":
    main()