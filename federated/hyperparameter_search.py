#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Federated Learning
"""

import itertools
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedGridSearch:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
        
    def run_experiment(self, params):
        """Run a single experiment with given parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"grid_{timestamp}"
        
        # Team building
        cmd = [
            'python', 'run_federated.py',
            '--federated_data', self.base_config['federated_data'],
            '--centralized_artifacts', self.base_config['centralized_artifacts'],
            '--output', f"experiments/grid_search/{exp_name}",
            '--horizon', str(self.base_config['horizon']),
            '--deadband', str(self.base_config['deadband']),
            '--two_class',
            '--rounds', str(params['rounds']),
            '--local_epochs', str(params['local_epochs']),
            '--batch_size', str(params['batch_size']),
            '--client_lr', str(params['client_lr']),
            '--server_lr', str(params['server_lr']),
            '--aggregation', params['aggregation']
        ]
        
        logger.info(f"Running experiment: {params}")
        
        try:
            # Run an experiment
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parsing results from output
            output_lines = result.stdout.split('\n')
            test_accuracy = None
            profit_bps = None
            
            for line in output_lines:
                if 'Test Accuracy:' in line:
                    test_accuracy = float(line.split(':')[1].strip())
                if 'Profit:' in line and 'bps' in line:
                    profit_bps = float(line.split(':')[1].split('bps')[0].strip())
            
            # Saving results
            exp_result = {
                **params,
                'test_accuracy': test_accuracy,
                'profit_bps': profit_bps,
                'exp_name': exp_name,
                'timestamp': timestamp
            }
            
            self.results.append(exp_result)
            logger.info(f"Results: Accuracy={test_accuracy}, Profit={profit_bps}")
            
            return exp_result
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return None
    
    def grid_search(self, param_grid):
        """Полный перебор по сетке параметров"""
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        total_combinations = 1
        for v in values:
            total_combinations *= len(v)
        
        logger.info(f"Total combinations to test: {total_combinations}")
        
        # Trying all combinations
        for i, combination in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combination))
            
            logger.info(f"Experiment {i+1}/{total_combinations}")
            self.run_experiment(params)
            
            # Saving intermediate results
            self.save_results('experiments/grid_search/intermediate_results.csv')
        
        # Final analysis
        self.analyze_results()
    
    def save_results(self, filepath):
        """Saving results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")
    
    def analyze_results(self):
        """Analysis of results and derivation of the best configurations"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Sort by accuracy
        df_sorted = df.sort_values('test_accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("TOP 5 CONFIGURATIONS BY ACCURACY")
        print("="*60)
        
        for i, row in df_sorted.head(5).iterrows():
            print(f"\n#{i+1}:")
            print(f"  Accuracy: {row['test_accuracy']:.4f}")
            print(f"  Profit: {row['profit_bps']:.2f} bps")
            print(f"  Config: local_epochs={row['local_epochs']}, "
                  f"batch_size={row['batch_size']}, "
                  f"client_lr={row['client_lr']}, "
                  f"aggregation={row['aggregation']}")
        
        # Saving final results
        self.save_results('experiments/grid_search/final_results.csv')
        
        # Saving the best configuration
        best_config = df_sorted.iloc[0].to_dict()
        with open('experiments/grid_search/best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)

def main():
    # Basic configuration
    base_config = {
        'federated_data': 'data/processed_federated_iid',
        'centralized_artifacts': 'experiments/centralized/2025-09-12_215045_mlp_h600_db10.0_2class_cal',
        'horizon': 600,
        'deadband': 10.0
    }
    
    # Grid of parameters for searching
    param_grid = {
        'rounds': [20],  # Fixed for speed
        'local_epochs': [3, 5, 8],
        'batch_size': [256, 512],
        'client_lr': [0.0005, 0.001, 0.005],
        'server_lr': [1.0],  # For FedAvg it is usually 1.0
        'aggregation': ['fedavg', 'fedavgm']
    }
    
    # Starting a search
    searcher = FederatedGridSearch(base_config)
    searcher.grid_search(param_grid)

if __name__ == "__main__":
    main()