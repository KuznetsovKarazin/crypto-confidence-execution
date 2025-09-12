# Cryptocurrency Confidence-Based Execution System

A production-ready machine learning system for cryptocurrency price movement prediction using confidence-based execution thresholds. The system combines high-frequency order book data with daily price data to predict directional movements and execute trades only when model confidence exceeds optimized thresholds.

## 🎯 Overview

This system implements a **two-class confidence-based trading approach** that:

- **Predicts directional price movements** (Up vs Down) using neural networks
- **Executes trades only when model confidence exceeds optimized thresholds** (τ)
- **Combines macro (daily OHLCV) and micro (order book) features**
- **Optimizes for real trading metrics** (profit per trade, coverage, win rate)
- **Provides comprehensive backtesting and analysis tools**

### Key Innovation: Confidence-Based Execution

Unlike traditional classification systems that predict three classes (Up/Down/No-trade), this system:

1. **Trains a binary classifier** on Up vs Down movements only
2. **Uses prediction confidence** to determine trade execution
3. **Optimizes confidence thresholds (τ)** for maximum profitability
4. **Accounts for trading costs** in threshold optimization

## 🏗️ Architecture

```
crypto-confidence-execution/
├── data/                           # Data storage
│   ├── macro/                      # Daily OHLCV data
│   ├── micro/                      # Order book snapshots
│   └── processed/                  # Unified datasets
├── experiments/                    # Training results
│   ├── centralized/               # Single experiments
│   ├── [tau = 0.6]/              # Sweep results (τ=0.6)
│   └── [tau = 0.8]/              # Sweep results (τ=0.8)
├── centralized_training_two_classes.py  # Main training script
├── sweep_two_class.py             # Parameter sweep utility
├── sweep_results_aggregator.py    # Results analysis
└── data_preprocessing.py          # Data pipeline
```

## 🚀 Quick Start

### 1. Data Preparation

Download the required datasets:

- **Macro data**: [Top 100 Cryptocurrency 2020-2025](https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025)
- **Micro data**: [Cryptocurrency Order Book Data](https://www.kaggle.com/datasets/ilyazawilsiv/cryptocurrency-order-book-data-asks-and-bids)

Place files in:
```
data/macro/top_100_cryptos_with_correct_network.csv
data/micro/*.csv  # Individual order book files
```

### 2. Preprocess Data

```bash
python data_preprocessing.py
```

Creates `data/processed/unified_dataset.parquet` combining macro and micro features.

### 3. Train Model

**Single Experiment:**
```bash
python centralized_training_two_classes.py \
    --unified data/processed/unified_dataset.parquet \
    --out experiments/centralized \
    --horizon_min 600 --deadband_bps 10 \
    --two_class_mode --confidence_tau 0.80 \
    --calibrate_probabilities
```

**Parameter Sweep:**
```bash
python sweep_two_class.py \
    --horizons 300 600 900 \
    --deadbands 5 10 15 \
    --optimize_tau_by profit ev
```

### 4. Analyze Results

```bash
python sweep_results_aggregator.py \
    --base_dir experiments/centralized
```

## 📊 Model Architecture

### Two-Class Confidence System

The system uses a **binary neural network classifier** with the following components:

1. **Feature Engineering**:
   - **Macro features**: Moving averages, volatility, RSI, momentum
   - **Micro features**: Spread, order imbalance, depth features
   - **Temporal features**: Session indicators, time-based patterns

2. **Target Creation**:
   - Samples where |future_return| > deadband_bps are labeled as Up (1) or Down (0)
   - No-trade samples are filtered out during training

3. **Confidence-Based Execution**:
   ```python
   confidence = max(P(Up), P(Down))
   execute_trade = confidence >= τ (threshold)
   direction = argmax(P(Up), P(Down))
   ```

4. **Threshold Optimization**:
   - **Profit maximization**: `τ* = argmax E[profit|τ]`
   - **Expected value**: `τ* = argmax E[profit × coverage|τ]` 
   - **Coverage constraints**: `τ* = argmax E[profit|τ, coverage ≥ min_coverage]`

### Performance Metrics

- **Coverage**: Percentage of observations where trades are executed
- **Direction Accuracy**: Accuracy on executed trades only
- **Average Profit**: Mean profit per executed trade (after costs)
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `horizon_min` | Prediction horizon in minutes | 600 |
| `deadband_bps` | Minimum movement threshold (basis points) | 10 |
| `confidence_tau` | Execution confidence threshold | 0.7 |
| `profit_cost_bps` | Trading cost per transaction | 1.0 |
| `optimize_tau_by` | Threshold optimization criterion | "profit" |

### Optimization Criteria

- **`profit`**: Maximize average profit per trade
- **`ev`**: Maximize expected value (profit × coverage)
- **`profit_with_min_coverage`**: Maximize profit subject to minimum coverage

## 📈 Experimental Results

### Current Experiments

The repository contains results from confidence threshold experiments:

- **`[tau = 0.6]`**: Results from sweep with τ=0.6 optimization
- **`[tau = 0.8]`**: Results from sweep with τ=0.8 optimization  
- **`centralized`**: Latest individual experiment results

These directories were generated using the sweep system with different confidence threshold ranges and contain:
- Model artifacts (`*.joblib`)
- Performance metrics (`metrics_two_class.json`)
- Prediction outputs (`test_predictions_two_class.csv`)
- Training configurations (`config.yaml`)

### Example Performance

```
Coverage: 15.2% (1,247/8,203 trades)
Direction Accuracy: 0.6823
Average Profit: 3.47 bps
Win Rate: 58.3%
Sharpe Ratio: 1.24
```

## 🔧 Advanced Usage

### Custom Feature Engineering

Modify feature windows in `data_preprocessing.py`:

```python
config = PreprocessingConfig(
    volatility_windows=[5, 15, 30],
    ma_windows=[5, 10, 20],
    depth_levels=8
)
```

### Multi-Horizon Training

Train models for different time horizons:

```bash
python sweep_two_class.py \
    --horizons 60 300 600 1200 \
    --deadbands 2 5 10 20
```

### Custom Threshold Optimization

```python
# In centralized_training_two_classes.py
best_tau = classifier.optimize_confidence_tau_for_profit(
    X_val, y_val, returns_bps,
    cost_bps=1.0,
    select_by="profit_with_min_coverage",
    min_coverage=0.05  # Minimum 5% coverage
)
```

## 📋 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0  # Optional, for LSTM models
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🔍 Code Structure

### Core Components

- **`TargetEncoder`**: Creates directional labels with deadband filtering
- **`TwoClassConfidenceClassifier`**: Binary classifier with confidence thresholding
- **`FeatureProcessor`**: Feature selection and preprocessing pipeline
- **`MetricsCalculator`**: Trading-specific performance metrics
- **`CentralizedTrainer`**: Main training orchestrator

### Data Pipeline

- **`MacroDataProcessor`**: Daily OHLCV feature engineering
- **`MicroDataProcessor`**: Order book feature extraction
- **`FeatureEngineeringPipeline`**: Unified dataset creation

## 📊 Results Analysis

The `sweep_results_aggregator.py` creates comprehensive analysis:

- **Performance heatmaps** across parameter combinations
- **Scatter plots** of profit vs coverage trade-offs
- **Per-symbol performance** breakdowns
- **Statistical summaries** of sweep results

Output files:
```
experiments/centralized_aggregated/
├── all_experiments.csv
├── performance_heatmaps.png
├── sweep_analysis.png
├── scatter_plots.png
└── analysis_report.txt
```

## 🚨 Risk Management & Implementation Guidelines

### Built-in Safeguards

The system incorporates multiple risk management layers:

1. **Confidence-Based Execution**: Only trade when model confidence exceeds optimized thresholds
2. **Transaction Cost Integration**: All metrics include realistic 1 basis point trading costs  
3. **Temporal Validation**: Strict time-based train/validation/test splits prevent data leakage
4. **Outlier Detection**: Automatic flagging of extreme price movements and data quality issues
5. **Symbol-Wise Splitting**: Independent evaluation per cryptocurrency prevents cross-contamination

### Implementation Risk Considerations

**Data Dependencies:**
- **Microstructure availability**: Limited to 11 symbol pairs with complete order book data
- **Feature computation latency**: ~15ms for 64-feature calculation (compatible with minute-frequency)
- **Multi-stream reliability**: Requires robust handling of macro daily + micro minute data feeds

**Market Impact & Execution:**
- **Transaction cost assumptions**: 1 bps baseline may underestimate institutional-scale impact
- **High confidence margins**: 6 bps cost tolerance provides buffer for realistic execution
- **Position sizing**: Confidence scores enable natural position scaling beyond binary decisions

**Operational Constraints:**
- **Parameter stability**: Fixed τ values may require adaptive adjustment as market regimes shift
- **Coverage variability**: High confidence (3-21%) vs moderate confidence (50-65%) creates different operational profiles
- **11-symbol constraint**: Limited diversification compared to broader cryptocurrency universe

### Recommended Deployment Strategy

**Phase 1: Conservative Implementation**
```python
# Start with high confidence, longer horizons
config = {
    'confidence_tau': 0.80,
    'horizon_min': 600,
    'deadband_bps': 10,
    'profit_cost_bps': 2.0  # Conservative cost assumption
}
```

**Phase 2: Risk Monitoring**
- **Track coverage consistency**: Monitor if actual execution rates match backtest expectations
- **Performance attribution**: Separate prediction accuracy from execution timing effects  
- **Market regime detection**: Watch for periods where confidence calibration degrades

**Phase 3: Strategy Scaling**
- **Portfolio integration**: Use confidence scores for position sizing across multiple symbols
- **Multi-horizon ensemble**: Combine 400min and 600min models for diversification
- **Adaptive thresholds**: Consider dynamic τ adjustment based on recent prediction accuracy

### Production Deployment Checklist

- [ ] **Data Infrastructure**: Reliable macro daily + micro minute feeds for 11 symbols
- [ ] **Feature Pipeline**: Real-time calculation of 64 unified features within 15ms latency
- [ ] **Model Serving**: Calibrated prediction probabilities with confidence threshold logic
- [ ] **Risk Controls**: Position sizing, maximum drawdown limits, correlation monitoring
- [ ] **Performance Monitoring**: Live tracking of coverage, accuracy, and profit metrics
- [ ] **Graceful Degradation**: Fallback behavior when partial data becomes unavailable

### Performance Expectations & Limitations

**Expected Performance Ranges** (based on research results):
- **High Confidence (τ=0.8)**: 150-167 bps profit, 3-21% coverage
- **Moderate Confidence (τ=0.6)**: 90-105 bps profit, 50-65% coverage  
- **Win Rates**: 82-95% (high confidence), 68-83% (moderate confidence)

**Evaluation Period Limitations**:
- **Time scope**: October 2023 - October 2024 market conditions
- **Regulatory environment**: Pre-major institutional adoption phase
- **Symbol constraints**: 11 major pairs, may not represent full crypto market dynamics

**Future Research Directions**:
- **Adaptive confidence thresholds** responsive to changing market volatility
- **Portfolio-level optimization** beyond individual symbol prediction
- **Broader cryptocurrency universe** as microstructure data availability improves
- **Integration with traditional assets** where similar macro-micro relationships exist

## 📄 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Academic Citation

If you use this work in your research, please cite:

```bibtex
@article{lastname2025cryptocurrency,
  title={Multi-Scale Feature Integration for Cryptocurrency Trading: A Confidence-Threshold Approach to Direction Prediction},
  author={Oleksandr Kuznetsov},
  year={2025},
}
```

### Research Data & Code Availability

- **Complete codebase**: Available at [https://github.com/KuznetsovKarazin/crypto-confidence-execution](https://github.com/KuznetsovKarazin/crypto-confidence-execution)
- **Experimental results**: All 80 configuration results included in repository
- **Preprocessing pipeline**: Full data processing and feature engineering code
- **Visualization tools**: Performance analysis and aggregation scripts

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Users are responsible for their own trading decisions and should consult financial advisors before deploying any trading system.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Built with ❤️ for the cryptocurrency research community**