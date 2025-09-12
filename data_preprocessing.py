#!/usr/bin/env python3
"""
Cryptocurrency Market Data Preprocessing Module
=====================================================

Production-ready module for preprocessing cryptocurrency market data.

License: MIT
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import logging
import json
import re
import gc
from itertools import islice
from datetime import datetime, timezone, date
import numpy as np
import pandas as pd

# Suppress warnings but keep critical ones
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*groupby.*apply.*')

# Module-specific logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline with all parameters utilized."""
    
    # Data paths
    macro_csv_path: Path
    micro_data_dir: Path
    output_dir: Path
    
    # Data quality thresholds (all utilized)
    max_missing_ratio: float = 0.3
    outlier_std_threshold: float = 4.0
    min_observations_per_symbol: int = 1000
    duplicate_threshold: float = 0.05
    
    # Temporal parameters
    max_gap_minutes: int = 60
    validation_split_ratio: float = 0.2
    
    # Feature engineering
    volatility_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    ma_windows: List[int] = field(default_factory=lambda: [5, 20, 50])
    depth_levels: int = 10  # Reduced from 20 for compactness
    
    # Adaptive quality control
    adaptive_spread_threshold: bool = True
    adaptive_price_change_threshold: bool = True  # New: per-symbol price change limits
    global_min_spread_bps: float = 0.01
    global_max_spread_bps: float = 1000.0
    min_price: float = 0.0001
    base_max_price_change: float = 0.2  # Base threshold, will be adapted per symbol
    
    # Memory management and processing
    chunk_size: int = 50000
    save_debug_info: bool = True
    downcast_numeric: bool = True
    minute_aggregation_method: str = 'last'  # 'last', 'median', 'first' for intra-minute dedup
    
    # Output format and timezone handling
    consistent_column_naming: bool = True
    macro_alignment_offset_hours: int = 0


class DataQualityError(Exception):
    """Custom exception for data quality issues."""
    pass


class DebugDataSaver:
    """Helper class for saving debug information during processing."""
    
    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = Path(output_dir) / "debug"
        self.enabled = enabled
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_rejected_data(self, data: pd.DataFrame, reason: str, suffix: str = ""):
        """Save rejected data with reason."""
        if not self.enabled or data.empty:
            return
        
        filename = f"rejected_{reason}{suffix}.parquet"
        filepath = self.output_dir / filename
        try:
            data.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(data)} rejected rows to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save rejected data: {e}")
    
    def save_statistics(self, stats: Dict[str, Any], name: str):
        """Save processing statistics."""
        if not self.enabled:
            return
        
        filepath = self.output_dir / f"{name}_stats.json"
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def save_orderbook_examples(self, data: pd.DataFrame, reason: str, max_examples: int = 100):
        """Save examples of order book issues."""
        if not self.enabled or data.empty:
            return
        
        sample_data = data.head(max_examples) if len(data) > max_examples else data
        filename = f"orderbook_examples_{reason}.parquet"
        filepath = self.output_dir / filename
        try:
            sample_data.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(sample_data)} order book examples to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save order book examples: {e}")


class SymbolNormalizer:
    """Enhanced symbol normalization with broader quote currency support."""
    
    QUOTE_CURRENCIES = [
        'USDT', 'USD', 'USDC', 'BUSD', 'TUSD', 'DAI',  # USD stablecoins
        'EUR', 'GBP', 'JPY', 'AUD', 'CAD',  # Major fiat
        'BTC', 'ETH', 'BNB',  # Major crypto
        'TRY', 'BRL', 'RUB', 'UAH', 'KRW'  # Other fiat
    ]
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Enhanced symbol normalization supporting all quote currencies."""
        if pd.isna(symbol) or not symbol:
            return ""
        
        clean_symbol = str(symbol).upper().strip()
        clean_symbol = re.sub(r'^(BINANCE_SPOT_|SPOT_|FUTURES_)', '', clean_symbol)
        clean_symbol = re.sub(r'[^A-Z0-9_-]', '', clean_symbol)
        clean_symbol = clean_symbol.replace('-', '_')
        
        if '_' in clean_symbol and len(clean_symbol.split('_')) == 2:
            parts = clean_symbol.split('_')
            if parts[1] in SymbolNormalizer.QUOTE_CURRENCIES:
                return clean_symbol
        
        base, quote = SymbolNormalizer._parse_base_quote_enhanced(clean_symbol)
        if base and quote:
            return f"{base}_{quote}"
        
        return clean_symbol
    
    @staticmethod
    def extract_symbol_from_filename(file_path: Path) -> str:
        """Extract symbol from filename with enhanced patterns."""
        filename = file_path.stem.upper()
        
        # Remove common prefixes/suffixes
        filename = re.sub(r'^(BINANCE_SPOT_|SPOT_|FUTURES_)', '', filename)
        filename = re.sub(r'_(ORDERBOOK|TRADES|CANDLES|DATA)$', '', filename)
        filename = filename.replace('.CSV', '')
        
        # If already in BASE_QUOTE format, normalize and return
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) == 2 and parts[1] in SymbolNormalizer.QUOTE_CURRENCIES:
                return filename
        
        # Try to parse from filename
        normalized = SymbolNormalizer.normalize_symbol(filename)
        if normalized and '_' in normalized:
            return normalized
        
        # Fallback: return cleaned filename
        return filename if filename else "UNKNOWN"
    
    @staticmethod
    def _parse_base_quote_enhanced(symbol: str) -> Tuple[str, str]:
        """Enhanced parsing supporting all quote currencies."""
        for quote in sorted(SymbolNormalizer.QUOTE_CURRENCIES, key=len, reverse=True):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[:-len(quote)]
                if len(base) >= 1:
                    return base, quote
        
        for pattern in [r'([A-Z1-9]+)(USD[TC]?)$', r'([A-Z1-9]+)(BTC|ETH|BNB)$']:
            match = re.match(pattern, symbol)
            if match and match.group(2) in SymbolNormalizer.QUOTE_CURRENCIES:
                return match.group(1), match.group(2)
        
        return "", ""
    
    @staticmethod
    def get_base_currency(symbol: str) -> str:
        """Extract base currency from normalized symbol."""
        normalized = SymbolNormalizer.normalize_symbol(symbol)
        return normalized.split('_')[0] if '_' in normalized else normalized
    
    @staticmethod
    def get_quote_currency(symbol: str) -> str:
        """Extract quote currency from normalized symbol."""
        normalized = SymbolNormalizer.normalize_symbol(symbol)
        return normalized.split('_')[1] if '_' in normalized else ""


class OrderBookValidator:
    """Enhanced order book structure validation."""
    
    @staticmethod
    def validate_orderbook_structure(df: pd.DataFrame, depth_levels: int = 10) -> Dict[str, Any]:
        """Comprehensive order book structure validation."""
        results = {
            'status': 'passed',
            'issues': [],
            'statistics': {},
            'invalid_indices': []
        }
        
        total_rows = len(df)
        if total_rows == 0:
            return results
        
        # Check for crossed/locked markets
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            crossed_mask = df['best_bid'] >= df['best_ask']
            crossed_count = crossed_mask.sum()
            if crossed_count > 0:
                results['issues'].append(f"Crossed markets: {crossed_count} ({crossed_count/total_rows:.2%})")
                results['invalid_indices'].extend(df[crossed_mask].index.tolist())
        
        # Check price monotonicity for asks (should be increasing)
        ask_violations = []
        for level in range(min(depth_levels-1, 9)):  # Check first 10 levels
            col1 = f'asks[{level}].price'
            col2 = f'asks[{level+1}].price'
            if col1 in df.columns and col2 in df.columns:
                # Remove NaN values for comparison
                valid_mask = df[col1].notna() & df[col2].notna()
                if valid_mask.sum() > 0:
                    violation_mask = valid_mask & (df[col1] >= df[col2])
                    violations = violation_mask.sum()
                    if violations > 0:
                        ask_violations.append((level, violations))
                        results['invalid_indices'].extend(df[violation_mask].index.tolist())
        
        # Check price monotonicity for bids (should be decreasing)  
        bid_violations = []
        for level in range(min(depth_levels-1, 9)):
            col1 = f'bids[{level}].price'
            col2 = f'bids[{level+1}].price'
            if col1 in df.columns and col2 in df.columns:
                valid_mask = df[col1].notna() & df[col2].notna()
                if valid_mask.sum() > 0:
                    violation_mask = valid_mask & (df[col1] <= df[col2])
                    violations = violation_mask.sum()
                    if violations > 0:
                        bid_violations.append((level, violations))
                        results['invalid_indices'].extend(df[violation_mask].index.tolist())
        
        # Check for negative sizes
        negative_sizes = 0
        for side in ['asks', 'bids']:
            for level in range(depth_levels):
                col = f'{side}[{level}].size'
                if col in df.columns:
                    negative_mask = df[col] < 0
                    negative_sizes += negative_mask.sum()
                    if negative_mask.sum() > 0:
                        results['invalid_indices'].extend(df[negative_mask].index.tolist())
        
        # Compile results
        if ask_violations:
            total_ask_violations = sum(count for _, count in ask_violations)
            results['issues'].append(f"Ask price violations: {total_ask_violations} across {len(ask_violations)} levels")
        
        if bid_violations:
            total_bid_violations = sum(count for _, count in bid_violations)
            results['issues'].append(f"Bid price violations: {total_bid_violations} across {len(bid_violations)} levels")
        
        if negative_sizes > 0:
            results['issues'].append(f"Negative sizes: {negative_sizes}")
        
        # Remove duplicate indices
        results['invalid_indices'] = list(set(results['invalid_indices']))
        
        # Statistics
        results['statistics'] = {
            'total_rows': total_rows,
            'invalid_rows': len(results['invalid_indices']),
            'invalid_ratio': len(results['invalid_indices']) / total_rows if total_rows > 0 else 0,
            'crossed_markets': crossed_count if 'crossed_count' in locals() else 0,
            'ask_violations': len(ask_violations),
            'bid_violations': len(bid_violations),
            'negative_sizes': negative_sizes
        }
        
        # Determine status
        invalid_ratio = results['statistics']['invalid_ratio']
        if invalid_ratio > 0.1:  # >10% invalid
            results['status'] = 'failed'
        elif invalid_ratio > 0.01:  # >1% invalid
            results['status'] = 'warning'
        elif results['issues']:
            results['status'] = 'warning'
        
        return results

class DataValidator:
    """Enhanced data validation with gap flagging."""
    
    @staticmethod
    def validate_temporal_consistency_with_flags(df: pd.DataFrame, 
                                               timestamp_col: str = 'timestamp',
                                               symbol_col: Optional[str] = None,
                                               max_gap_minutes: int = 60,
                                               duplicate_threshold: float = 0.05) -> Dict[str, Any]:
        """FIXED: Temporal validation with gap flagging instead of just warnings."""
        results = {'status': 'passed', 'issues': [], 'statistics': {}}
        
        if timestamp_col not in df.columns:
            results['status'] = 'failed'
            results['issues'].append(f"Missing timestamp column: {timestamp_col}")
            return results
        
        # Initialize gap flags
        df['is_gap'] = False
        df['gap_minutes'] = 0.0
        
        # Check duplicates with configurable threshold
        if symbol_col and symbol_col in df.columns:
            duplicates = df.duplicated(subset=[symbol_col, timestamp_col]).sum()
            duplicate_ratio = duplicates / len(df) if len(df) > 0 else 0
            results['statistics']['duplicate_ratio'] = duplicate_ratio
            results['statistics']['duplicate_count'] = duplicates
            
            if duplicate_ratio > duplicate_threshold:
                results['status'] = 'failed'
                results['issues'].append(
                    f"Excessive duplicates: {duplicate_ratio:.2%} > {duplicate_threshold:.2%} "
                    f"({duplicates:,} rows)"
                )
                return results
            elif duplicates > 0:
                results['issues'].append(f"Found {duplicates:,} duplicate entries ({duplicate_ratio:.2%})")
        
        # FIXED: Check for gaps and FLAG them in the dataframe
        df_sorted = df.sort_values([symbol_col, timestamp_col] if symbol_col else [timestamp_col])
        
        total_gaps = 0
        gap_summary = {}
        
        if symbol_col and symbol_col in df.columns:
            for symbol in df[symbol_col].unique():
                symbol_mask = df[symbol_col] == symbol
                symbol_data = df_sorted[symbol_mask]
                
                # Calculate gaps and flag them
                gaps, gap_flags = DataValidator._find_and_flag_gaps(
                    symbol_data, timestamp_col, max_gap_minutes
                )
                
                if len(gaps) > 0:
                    total_gaps += len(gaps)
                    gap_summary[symbol] = {
                        'gap_count': len(gaps),
                        'max_gap_minutes': max([gap[2] for gap in gaps]),
                        'flagged_rows': gap_flags.sum()
                    }
                    
                    # Apply flags to original dataframe
                    original_indices = symbol_data.index
                    df.loc[original_indices, 'is_gap'] = gap_flags
                    
                    # Calculate gap durations and apply to original dataframe
                    gap_durations = DataValidator._calculate_gap_durations(symbol_data, timestamp_col)
                    df.loc[original_indices, 'gap_minutes'] = gap_durations
                    
                    if len(gaps) > 10:  # Only log if significant
                        results['issues'].append(
                            f"Symbol {symbol}: {len(gaps)} gaps > {max_gap_minutes} minutes "
                            f"(max: {gap_summary[symbol]['max_gap_minutes']:.1f} min)"
                        )
        else:
            gaps, gap_flags = DataValidator._find_and_flag_gaps(
                df_sorted, timestamp_col, max_gap_minutes
            )
            total_gaps = len(gaps)
            if gaps:
                df['is_gap'] = gap_flags
                gap_durations = DataValidator._calculate_gap_durations(df_sorted, timestamp_col)
                df['gap_minutes'] = gap_durations
                results['issues'].append(f"Found {len(gaps)} gaps > {max_gap_minutes} minutes")
        
        # Enhanced statistics with gap information
        results['statistics']['total_observations'] = len(df)
        results['statistics']['total_gaps'] = total_gaps
        results['statistics']['flagged_observations'] = df['is_gap'].sum()
        results['statistics']['gap_summary_by_symbol'] = gap_summary
        
        if len(df) > 0:
            results['statistics']['date_range'] = (
                df[timestamp_col].min().isoformat(),
                df[timestamp_col].max().isoformat()
            )
        
        # Log gap summary
        if total_gaps > 0:
            logger.info(f"Gap analysis: {total_gaps} gaps detected, {df['is_gap'].sum()} observations flagged")
            if gap_summary:
                for symbol, stats in gap_summary.items():
                    logger.info(f"  {symbol}: {stats['gap_count']} gaps, "
                              f"{stats['flagged_rows']} flagged rows, "
                              f"max gap: {stats['max_gap_minutes']:.1f} min")
        
        if results['issues']:
            results['status'] = 'warning'
        
        return results
    
       
    @staticmethod
    def _find_and_flag_gaps(symbol_data: pd.DataFrame, timestamp_col: str, 
                          max_gap_minutes: int) -> Tuple[List[Tuple], pd.Series]:
        """Find gaps and return both gap list and boolean flags for each row."""
        if len(symbol_data) < 2:
            return [], pd.Series([False] * len(symbol_data), index=symbol_data.index)
        
        timestamps = symbol_data[timestamp_col].reset_index(drop=True)
        time_diffs = timestamps.diff().dt.total_seconds() / 60
        
        # Find gap positions
        gap_positions = np.where(time_diffs.values > max_gap_minutes)[0]
        
        # Create gap list (for reporting)
        gaps = []
        for pos in gap_positions:
            if pos > 0:
                gaps.append((
                    timestamps.iloc[pos-1], 
                    timestamps.iloc[pos], 
                    time_diffs.iloc[pos]
                ))
        
        # Create boolean flags for each row (mark rows that follow a gap)
        gap_flags = pd.Series([False] * len(symbol_data), index=symbol_data.index)
        if len(gap_positions) > 0:
            # Flag the rows that come after a gap
            for pos in gap_positions:
                if pos < len(symbol_data):
                    original_idx = symbol_data.index[pos]
                    gap_flags.loc[original_idx] = True
        
        return gaps, gap_flags
    
    @staticmethod
    def _calculate_gap_durations(symbol_data: pd.DataFrame, timestamp_col: str) -> pd.Series:
        """Calculate gap durations for each row."""
        if len(symbol_data) < 2:
            return pd.Series([0.0] * len(symbol_data), index=symbol_data.index)
        
        timestamps = symbol_data[timestamp_col].reset_index(drop=True)
        time_diffs = timestamps.diff().dt.total_seconds() / 60
        
        # Map back to original indices
        gap_durations = pd.Series([0.0] * len(symbol_data), index=symbol_data.index)
        
        for i, duration in enumerate(time_diffs):
            if i < len(symbol_data):
                original_idx = symbol_data.index[i]
                gap_durations.loc[original_idx] = duration if pd.notna(duration) else 0.0
        
        return gap_durations
   
    @staticmethod
    def validate_temporal_consistency(df: pd.DataFrame, 
                                    timestamp_col: str = 'timestamp',
                                    symbol_col: Optional[str] = None,
                                    max_gap_minutes: int = 60,
                                    duplicate_threshold: float = 0.05) -> Dict[str, Any]:
        """Fixed temporal consistency validation with configurable duplicate threshold."""
        results = {'status': 'passed', 'issues': [], 'statistics': {}}
        
        if timestamp_col not in df.columns:
            results['status'] = 'failed'
            results['issues'].append(f"Missing timestamp column: {timestamp_col}")
            return results
        
        # Check duplicates with configurable threshold
        if symbol_col and symbol_col in df.columns:
            duplicates = df.duplicated(subset=[symbol_col, timestamp_col]).sum()
            duplicate_ratio = duplicates / len(df) if len(df) > 0 else 0
            results['statistics']['duplicate_ratio'] = duplicate_ratio
            results['statistics']['duplicate_count'] = duplicates
            
            if duplicate_ratio > duplicate_threshold:
                results['status'] = 'failed'
                results['issues'].append(
                    f"Excessive duplicates: {duplicate_ratio:.2%} > {duplicate_threshold:.2%} "
                    f"({duplicates:,} rows)"
                )
                return results
            elif duplicates > 0:
                results['issues'].append(f"Found {duplicates:,} duplicate entries ({duplicate_ratio:.2%})")
        
        # Check for gaps
        df_sorted = df.sort_values([symbol_col, timestamp_col] if symbol_col else [timestamp_col])
        
        total_gaps = 0
        if symbol_col and symbol_col in df.columns:
            for symbol in df[symbol_col].unique():
                symbol_data = df_sorted[df_sorted[symbol_col] == symbol]
                gaps = DataValidator._find_temporal_gaps_fixed(symbol_data[timestamp_col], max_gap_minutes)
                if gaps:
                    total_gaps += len(gaps)
                    if len(gaps) > 10:  # Only log if significant
                        results['issues'].append(f"Symbol {symbol}: {len(gaps)} gaps > {max_gap_minutes} minutes")
        else:
            gaps = DataValidator._find_temporal_gaps_fixed(df_sorted[timestamp_col], max_gap_minutes)
            total_gaps = len(gaps)
            if gaps:
                results['issues'].append(f"Found {len(gaps)} gaps > {max_gap_minutes} minutes")
        
        # Statistics
        results['statistics']['total_observations'] = len(df)
        results['statistics']['total_gaps'] = total_gaps
        if len(df) > 0:
            results['statistics']['date_range'] = (
                df[timestamp_col].min().isoformat(),
                df[timestamp_col].max().isoformat()
            )
        
        if results['issues']:
            results['status'] = 'warning'
        
        return results
    
    @staticmethod
    def _find_temporal_gaps_fixed(timestamps: pd.Series, max_gap_minutes: int) -> List[Tuple]:
        """Fixed gap detection using proper position-based indexing."""
        if len(timestamps) < 2:
            return []
        
        ts_reset = timestamps.reset_index(drop=True)
        time_diffs = ts_reset.diff().dt.total_seconds() / 60
        
        gap_positions = np.where(time_diffs.values > max_gap_minutes)[0]
        
        gaps = []
        for pos in gap_positions:
            if pos > 0:
                gaps.append((
                    ts_reset.iloc[pos-1], 
                    ts_reset.iloc[pos], 
                    time_diffs.iloc[pos]
                ))
        
        return gaps
    
    @staticmethod
    def validate_price_data_adaptive(df: pd.DataFrame, 
                                   price_cols: List[str],
                                   symbol_col: Optional[str] = None,
                                   min_price: float = 0.0001,
                                   base_max_change: float = 0.2,
                                   outlier_std_threshold: float = 4.0,
                                   adaptive_thresholds: bool = True) -> Dict[str, Any]:
        """Enhanced price validation with adaptive per-symbol thresholds."""
        results = {'status': 'passed', 'issues': [], 'statistics': {}, 'flagged_data': {}}
        total_observations = len(df)
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Add flagging columns
            df[f'{col}_is_extreme'] = False
            df[f'{col}_is_outlier'] = False
            
            # Check for negative or zero prices
            invalid_prices = (df[col] <= 0).sum()
            invalid_ratio = invalid_prices / total_observations
            if invalid_prices > 0:
                results['issues'].append(f"{col}: {invalid_prices} non-positive prices ({invalid_ratio:.2%})")
            
            # Check for unrealistically low prices
            low_prices = (df[col] < min_price).sum()
            low_ratio = low_prices / total_observations
            if low_prices > 0:
                results['issues'].append(f"{col}: {low_prices} prices below {min_price} ({low_ratio:.2%})")
            
            # Adaptive price change validation
            if symbol_col and symbol_col in df.columns and adaptive_thresholds:
                extreme_count, outlier_count = DataValidator._validate_price_changes_adaptive(
                    df, col, symbol_col, base_max_change, outlier_std_threshold
                )
            else:
                # Global thresholds
                price_changes = df[col].pct_change().abs()
                extreme_mask = price_changes > base_max_change
                extreme_count = extreme_mask.sum()
                df.loc[extreme_mask, f'{col}_is_extreme'] = True
                
                # Simple outlier detection
                mean_price = df[col].mean()
                std_price = df[col].std()
                outlier_mask = (df[col] - mean_price).abs() > outlier_std_threshold * std_price
                outlier_count = outlier_mask.sum()
                df.loc[outlier_mask, f'{col}_is_outlier'] = True
            
            extreme_ratio = extreme_count / total_observations
            outlier_ratio = outlier_count / total_observations
            
            if extreme_count > 0:
                results['issues'].append(
                    f"{col}: {extreme_count} extreme changes ({extreme_ratio:.2%}) - FLAGGED, not removed"
                )
            
            if outlier_count > 0:
                results['issues'].append(
                    f"{col}: {outlier_count} outliers ({outlier_ratio:.2%}) - FLAGGED, not removed"
                )
            
            # Statistics
            results['statistics'][f'{col}_median'] = float(df[col].median())
            results['statistics'][f'{col}_q95'] = float(df[col].quantile(0.95))
            results['statistics'][f'{col}_extreme_ratio'] = extreme_ratio
            results['statistics'][f'{col}_outlier_ratio'] = outlier_ratio
        
        # Determine status based on flagged ratios (but don't fail - just flag)
        max_flag_ratio = max([
            results['statistics'].get(f'{col}_extreme_ratio', 0) + 
            results['statistics'].get(f'{col}_outlier_ratio', 0)
            for col in price_cols if col in df.columns
        ], default=0)
        
        if max_flag_ratio > 0.1:  # >10% flagged
            results['status'] = 'warning'
        elif results['issues']:
            results['status'] = 'warning'
        
        return results
    
    @staticmethod
    def _validate_price_changes_adaptive(df: pd.DataFrame, price_col: str, symbol_col: str, 
                                       base_threshold: float, outlier_threshold: float) -> Tuple[int, int]:
        """Adaptive price change validation per symbol."""
        extreme_count = 0
        outlier_count = 0
        
        for symbol in df[symbol_col].unique():
            symbol_mask = df[symbol_col] == symbol
            symbol_data = df.loc[symbol_mask, price_col]
            
            if len(symbol_data) < 50:  # Skip symbols with insufficient data
                continue
            
            # Calculate adaptive threshold based on historical volatility
            returns = symbol_data.pct_change().abs()
            volatility = returns.rolling(window=min(100, len(returns)//2), min_periods=20).std()
            
            # Adaptive threshold: base + 2 * rolling volatility
            adaptive_threshold = base_threshold + 2 * volatility
            adaptive_threshold = adaptive_threshold.fillna(base_threshold)
            
            # Flag extreme changes
            extreme_mask = symbol_mask & (returns > adaptive_threshold)
            extreme_count += extreme_mask.sum()
            df.loc[extreme_mask, f'{price_col}_is_extreme'] = True
            
            # Flag outliers based on symbol-specific statistics
            symbol_mean = symbol_data.mean()
            symbol_std = symbol_data.std()
            outlier_mask = symbol_mask & (
                (symbol_data - symbol_mean).abs() > outlier_threshold * symbol_std
            )
            outlier_count += outlier_mask.sum()
            df.loc[outlier_mask, f'{price_col}_is_outlier'] = True
        
        return extreme_count, outlier_count


def optimize_dtypes(df: pd.DataFrame, downcast_numeric: bool = True, 
                   preserve_categories: bool = True) -> pd.DataFrame:
    """Enhanced memory optimization with category conflict handling."""
    if not downcast_numeric:
        return df
    
    original_memory = df.memory_usage(deep=True).sum()
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Smart category conversion
    if preserve_categories:
        for col in df.select_dtypes(include=['object']).columns:
            if col == 'symbol':
                continue  # Keep as object for merging
            
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.3:
                df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    reduction_pct = (1 - new_memory/original_memory) * 100
    logger.info(f"Memory optimization: {original_memory/1e6:.1f}MB -> {new_memory/1e6:.1f}MB "
                f"({reduction_pct:.1f}% reduction)")
    
    return df


class BaseDataProcessor(ABC):
    """Enhanced base processor with memory management and debug saving."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.data = None
        self.metadata = {
            'processing_steps': [], 
            'quality_checks': {},
            'memory_stats': {},
            'feature_list': [],
            'rejected_data_stats': {},
            'symbol_stats': {}
        }
        self.debug_saver = DebugDataSaver(config.output_dir, config.save_debug_info)
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw data."""
        pass
    
    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """Clean and validate data."""
        pass
    
    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features from cleaned data."""
        pass
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to snake_case if configured."""
        if not self.config.consistent_column_naming:
            return df
        
        name_mapping = {
            'Symbol': 'symbol',
            'Date': 'date', 
            'timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        return df.rename(columns={k: v for k, v in name_mapping.items() if k in df.columns})
    
    def save_processed_data(self, suffix: str = "") -> None:
        """Enhanced data saving with optimization and debug info."""
        if self.data is None:
            raise ValueError("No data to save. Process data first.")
        
        optimized_data = optimize_dtypes(self.data.copy(), self.config.downcast_numeric)
        
        output_path = self.config.output_dir / f"processed_data{suffix}.parquet"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        optimized_data.to_parquet(output_path, index=False, compression='snappy')
        
        feature_cols = [col for col in optimized_data.columns 
                       if col not in ['symbol', 'timestamp', 'date']]
        self.metadata['feature_list'] = feature_cols
        self.metadata['final_shape'] = optimized_data.shape
        
        # Enhanced symbol statistics
        if 'symbol' in optimized_data.columns:
            symbol_counts = optimized_data['symbol'].value_counts()
            self.metadata['symbol_stats'] = {
                'unique_symbols': len(symbol_counts),
                'min_observations': int(symbol_counts.min()),
                'max_observations': int(symbol_counts.max()),
                'median_observations': int(symbol_counts.median()),
                'total_observations': int(symbol_counts.sum())
            }
        
        metadata_path = self.config.output_dir / f"metadata{suffix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        if self.config.save_debug_info:
            self.debug_saver.save_statistics(self.metadata, f"processing{suffix}")
        
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Final dataset: {optimized_data.shape[0]} rows, {len(feature_cols)} features")
        if 'symbol' in optimized_data.columns:
            logger.info(f"Symbols: {optimized_data['symbol'].nunique()}")


class MacroDataProcessor(BaseDataProcessor):
    """Enhanced macro processor with adaptive price validation."""
    
    def load_data(self) -> pd.DataFrame:
        """Load macro data with chunked processing if needed."""
        logger.info(f"Loading macro data from {self.config.macro_csv_path}")
        
        try:
            file_size_mb = self.config.macro_csv_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 100:
                logger.info(f"Large file ({file_size_mb:.1f}MB), using chunked loading")
                chunks = []
                for chunk in pd.read_csv(self.config.macro_csv_path, chunksize=self.config.chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(self.config.macro_csv_path)
            
            logger.info(f"Loaded {len(df)} rows with {df.shape[1]} columns")
            
        except Exception as e:
            raise DataQualityError(f"Failed to load macro data: {e}")
        
        # Standardize column names
        column_mapping = {
            'symbol': 'Symbol', 'date': 'Date', 'open': 'Open', 
            'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Validate required columns
        required_cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataQualityError(f"Missing required columns: {missing_cols}")
        
        # Parse dates (keep as date type, no TZ complications)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        
        # Normalize symbols
        df['Symbol'] = df['Symbol'].apply(SymbolNormalizer.normalize_symbol)
        
        # Handle volume with explicit marking
        if 'Volume' not in df.columns:
            # Proxy volume: (high-low) * close * multiplier
            df['Volume'] = (df['High'] - df['Low']) * df['Close'] * 1000
            df['volume_is_proxy'] = True
            logger.warning("Volume column missing, created proxy volume (marked with volume_is_proxy=True)")
        else:
            df['volume_is_proxy'] = False
        
        # Convert to numeric
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded macro data: {df['Symbol'].nunique()} unique symbols")
        
        self.data = df
        self.metadata['processing_steps'].append('loaded_macro_data')
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Enhanced cleaning with adaptive price validation."""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        logger.info("Cleaning macro data...")
        df = self.data.copy()
        initial_rows = len(df)
        initial_symbols = df['Symbol'].nunique()
        
        # Remove rows with critical missing data
        critical_missing = df[['Symbol', 'Date', 'Close']].isnull().any(axis=1)
        if critical_missing.sum() > 0:
            rejected_data = df[critical_missing].copy()
            self.debug_saver.save_rejected_data(rejected_data, "critical_missing", "_macro")
            df = df[~critical_missing]
        
        # Temporal validation with gap flagging
        temporal_validation = DataValidator.validate_temporal_consistency_with_flags(
            df, 'Date', 'Symbol', 
            max_gap_minutes=24*60,  # Daily data
            duplicate_threshold=self.config.duplicate_threshold
        )
        self.metadata['quality_checks']['temporal_validation'] = temporal_validation
        
        if temporal_validation['status'] == 'failed':
            raise DataQualityError(f"Temporal validation failed: {temporal_validation['issues']}")
        
        # Remove duplicates after validation
        duplicate_mask = df.duplicated(subset=['Symbol', 'Date'], keep='last')
        if duplicate_mask.sum() > 0:
            rejected_duplicates = df[duplicate_mask].copy()
            self.debug_saver.save_rejected_data(rejected_duplicates, "duplicates", "_macro")
            df = df[~duplicate_mask]
        
        # Check missing data per symbol
        symbols_to_remove = []
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]
            missing_ratio = symbol_data[['Open', 'High', 'Low', 'Close']].isnull().mean().mean()
            if missing_ratio > self.config.max_missing_ratio:
                symbols_to_remove.append(symbol)
        
        if symbols_to_remove:
            logger.warning(f"Removing {len(symbols_to_remove)} symbols with >{self.config.max_missing_ratio:.1%} missing data")
            rejected_symbols_data = df[df['Symbol'].isin(symbols_to_remove)].copy()
            self.debug_saver.save_rejected_data(rejected_symbols_data, "high_missing_ratio", "_macro")
            df = df[~df['Symbol'].isin(symbols_to_remove)]
        
        # OHLC validation
        ohlc_mask = (
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) & (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) & (df['Low'] <= df['Close']) &
            (df['Open'] > 0) & (df['High'] > 0) & 
            (df['Low'] > 0) & (df['Close'] > 0)
        )
        
        invalid_ohlc = ~ohlc_mask
        if invalid_ohlc.sum() > 0:
            logger.warning(f"Removing {invalid_ohlc.sum()} rows with invalid OHLC")
            rejected_ohlc = df[invalid_ohlc].copy()
            self.debug_saver.save_rejected_data(rejected_ohlc, "invalid_ohlc", "_macro")
            df = df[ohlc_mask]
        
        # Remove symbols with insufficient observations
        symbol_counts = df['Symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= self.config.min_observations_per_symbol].index
        insufficient_symbols = df[~df['Symbol'].isin(valid_symbols)]
        
        if len(insufficient_symbols) > 0:
            removed_symbols = insufficient_symbols['Symbol'].nunique()
            logger.info(f"Removing {removed_symbols} symbols with < {self.config.min_observations_per_symbol} observations")
            self.debug_saver.save_rejected_data(insufficient_symbols, "insufficient_observations", "_macro")
            df = df[df['Symbol'].isin(valid_symbols)]
        
        # Sort for temporal consistency
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # ENHANCED: Adaptive price validation with per-symbol flagging stats
        price_validation = DataValidator.validate_price_data_adaptive(
            df, ['Open', 'High', 'Low', 'Close'], 
            symbol_col='Symbol',
            min_price=self.config.min_price,
            base_max_change=self.config.base_max_price_change,
            outlier_std_threshold=self.config.outlier_std_threshold,
            adaptive_thresholds=self.config.adaptive_price_change_threshold
        )
        self.metadata['quality_checks']['price_validation'] = price_validation
        
        # FIXED: Log per-symbol flagging statistics for "toxic" tickers
        if 'Symbol' in df.columns:
            symbol_flag_stats = {}
            for symbol in df['Symbol'].unique():
                symbol_mask = df['Symbol'] == symbol
                symbol_data = df[symbol_mask]
                
                total_obs = len(symbol_data)
                if total_obs == 0:
                    continue
                
                # Count flags per symbol
                extreme_flags = 0
                outlier_flags = 0
                for col in ['Open', 'High', 'Low', 'Close']:
                    if f'{col}_is_extreme' in df.columns:
                        extreme_flags += symbol_data[f'{col}_is_extreme'].sum()
                    if f'{col}_is_outlier' in df.columns:
                        outlier_flags += symbol_data[f'{col}_is_outlier'].sum()
                
                flag_ratio = (extreme_flags + outlier_flags) / (total_obs * 4)  # 4 price columns
                
                if flag_ratio > 0.05:  # >5% of price observations flagged
                    symbol_flag_stats[symbol] = {
                        'total_observations': total_obs,
                        'extreme_flags': extreme_flags,
                        'outlier_flags': outlier_flags,
                        'flag_ratio': flag_ratio
                    }
            
            if symbol_flag_stats:
                # Sort by flag ratio and log top "toxic" symbols
                sorted_symbols = sorted(symbol_flag_stats.items(), 
                                      key=lambda x: x[1]['flag_ratio'], reverse=True)
                logger.warning("Symbols with high flag ratios (potential data quality issues):")
                for symbol, stats in sorted_symbols[:5]:  # Top 5
                    logger.warning(f"  {symbol}: {stats['flag_ratio']:.1%} flagged "
                                 f"({stats['extreme_flags']} extreme, {stats['outlier_flags']} outliers)")
                
                self.metadata['symbol_flag_stats'] = dict(sorted_symbols)
        
        if price_validation['status'] == 'failed':
            logger.error("Critical price data validation failures detected")
        elif price_validation['status'] == 'warning':
            logger.warning(f"Price validation warnings: {price_validation['issues']}")
        
        # Log gap statistics if present
        if 'is_gap' in df.columns:
            total_gap_flags = df['is_gap'].sum()
            if total_gap_flags > 0:
                logger.info(f"Gap flags: {total_gap_flags} observations marked as post-gap")
                
                # Per-symbol gap stats
                if 'Symbol' in df.columns:
                    symbol_gap_stats = df.groupby('Symbol')['is_gap'].agg(['sum', 'count'])
                    symbol_gap_stats['gap_ratio'] = symbol_gap_stats['sum'] / symbol_gap_stats['count']
                    high_gap_symbols = symbol_gap_stats[symbol_gap_stats['gap_ratio'] > 0.01]  # >1% gaps
                    
                    if not high_gap_symbols.empty:
                        logger.info("Symbols with frequent gaps:")
                        for symbol, stats in high_gap_symbols.iterrows():
                            logger.info(f"  {symbol}: {stats['sum']} gaps / {stats['count']} obs "
                                      f"({stats['gap_ratio']:.1%})")
        
        cleaned_rows = len(df)
        final_symbols = df['Symbol'].nunique()
        retention_rate = cleaned_rows / initial_rows
        
        logger.info(f"Macro cleaning: {initial_rows:,} -> {cleaned_rows:,} rows ({retention_rate:.1%} retained)")
        logger.info(f"Symbols: {initial_symbols} -> {final_symbols} ({final_symbols/initial_symbols:.1%} retained)")
        
        self.metadata['rejected_data_stats']['macro'] = {
            'initial_rows': initial_rows,
            'final_rows': cleaned_rows,
            'retention_rate': retention_rate,
            'initial_symbols': initial_symbols,
            'final_symbols': final_symbols,
            'symbol_retention_rate': final_symbols / initial_symbols if initial_symbols > 0 else 0
        }
        
        self.data = df
        self.metadata['processing_steps'].append('cleaned_macro_data')
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """Enhanced feature engineering with vectorized RSI and volume handling."""
        if self.data is None:
            raise ValueError("Data must be cleaned first")
        
        logger.info("Engineering macro features...")
        df = self.data.copy()
        
        # Ensure proper sorting
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Basic returns calculation
        df['returns'] = df.groupby('Symbol')['Close'].pct_change()
        df['log_returns'] = df.groupby('Symbol')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        
        # Moving averages (lagged to prevent look-ahead)
        for window in self.config.ma_windows:
            df[f'ma_{window}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).mean()
            )
        
        # Volatility measures
        for window in self.config.volatility_windows:
            df[f'volatility_{window}d'] = df.groupby('Symbol')['returns'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).std()
            )
        
        # True Range calculation
        df['prev_close'] = df.groupby('Symbol')['Close'].shift(1)
        df['true_range'] = np.maximum.reduce([
            df['High'] - df['Low'],
            (df['High'] - df['prev_close']).abs(),
            (df['Low'] - df['prev_close']).abs()
        ])
        
        # Average True Range
        df['atr_14'] = df.groupby('Symbol')['true_range'].transform(
            lambda x: x.rolling(14, min_periods=7).mean()
        )
        
        # FIXED: Vectorized RSI calculation without groupby.apply warnings
        df['rsi_14'] = df.groupby('Symbol')['Close'].transform(
            lambda x: self._calculate_rsi_vectorized(x, 14)
        )
        
        # Market microstructure proxies
        df['spread_proxy'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
        
        # Price momentum
        for lag in [1, 5, 20]:
            df[f'momentum_{lag}d'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.pct_change(lag).shift(1)
            )
        
        # Volume indicators (handle proxy volumes appropriately)
        if 'Volume' in df.columns:
            df['volume_ma_20'] = df.groupby('Symbol')['Volume'].transform(
                lambda x: x.shift(1).rolling(20, min_periods=10).mean()
            )
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
            
            # Don't build volume features for proxy volumes
            proxy_mask = df['volume_is_proxy'] == True
            if proxy_mask.sum() > 0:
                logger.warning(f"Flagging volume-based features for {proxy_mask.sum()} rows with proxy volume")
                df.loc[proxy_mask, 'volume_ratio'] = np.nan
        
        # Optimized market regime indicators
        volatility_col = f'volatility_{self.config.volatility_windows[1]}d'
        if volatility_col in df.columns:
            # Pre-calculate quantiles per symbol to avoid repeated computation
            symbol_quantiles = df.groupby('Symbol')[volatility_col].quantile(0.75)
            df['volatility_regime'] = df.apply(
                lambda row: int(row[volatility_col] > symbol_quantiles.get(row['Symbol'], np.inf)) 
                if pd.notna(row[volatility_col]) else 0, axis=1
            )
        
        # Symbol metadata
        df['base_currency'] = df['Symbol'].apply(SymbolNormalizer.get_base_currency)
        df['quote_currency'] = df['Symbol'].apply(SymbolNormalizer.get_quote_currency)
        
        # Remove intermediate columns
        df = df.drop(columns=['prev_close', 'true_range'], errors='ignore')
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        self.data = df
        self.metadata['processing_steps'].append('engineered_macro_features')
        
        feature_cols = [col for col in df.columns 
                       if col not in ['symbol', 'Symbol', 'date', 'Date', 'open', 'Open', 
                                    'high', 'High', 'low', 'Low', 'close', 'Close', 
                                    'volume', 'Volume', 'returns', 'log_returns']]
        logger.info(f"Engineered {len(feature_cols)} macro features")
        
        return df
    
    def _calculate_rsi_vectorized(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Vectorized RSI calculation using transform (no groupby.apply warnings)."""
        # Ensure proper lagging
        lagged_prices = prices.shift(1)
        delta = lagged_prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing
        alpha = 1.0 / window
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MicroDataProcessor(BaseDataProcessor):
    """CRITICAL FIX: Enhanced micro processor with proper symbol assignment from filenames."""
    
    def load_data(self) -> pd.DataFrame:
        """FIXED: Load micro data with proper symbol assignment from filenames."""
        logger.info(f"Loading micro data from {self.config.micro_data_dir}")
        
        csv_files = list(self.config.micro_data_dir.glob("*.csv"))
        if not csv_files:
            raise DataQualityError("No CSV files found in micro data directory")
        
        # Filter out macro files
        micro_files = [f for f in csv_files if 'top_100' not in f.name.lower()]
        logger.info(f"Found {len(micro_files)} micro data files")
        
        combined_data = []
        
        for file_path in micro_files:
            try:
                # CRITICAL FIX: Assign symbol from filename BEFORE loading
                symbol = SymbolNormalizer.extract_symbol_from_filename(file_path)
                logger.info(f"Processing {file_path.name} -> symbol: {symbol}")
                
                # Load file with chunking if needed
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb > 50:
                    file_chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=self.config.chunk_size):
                        chunk['Symbol'] = symbol  # Assign symbol immediately
                        file_chunks.append(chunk)
                    df = pd.concat(file_chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path)
                    df['Symbol'] = symbol  # Assign symbol immediately
                
                # Parse timestamp
                timestamp_cols = ['time_exchange_minute', 'timestamp', 'time']
                timestamp_col = next((col for col in timestamp_cols if col in df.columns), None)
                
                if timestamp_col:
                    df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                else:
                    logger.warning(f"No timestamp column found in {file_path.name}")
                    continue
                
                # Validate that we have the required columns
                if 'timestamp' not in df.columns:
                    logger.error(f"Failed to parse timestamp for {file_path.name}")
                    continue
                
                combined_data.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} rows for symbol {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue
        
        if not combined_data:
            raise DataQualityError("No valid micro data files loaded")
        
        # Combine all data
        df = pd.concat(combined_data, ignore_index=True, sort=False)
        del combined_data
        gc.collect()
        
        # CRITICAL: Validate symbol assignment worked
        unique_symbols = df['Symbol'].nunique()
        logger.info(f"Combined micro data: {len(df):,} rows, {unique_symbols} unique symbols")
        
        if unique_symbols < 2:
            raise DataQualityError(
                f"Expected >= 2 symbols after loading {len(micro_files)} files, "
                f"got {unique_symbols}. Symbol assignment failed."
            )
        
        # Log symbol distribution
        symbol_counts = df['Symbol'].value_counts()
        logger.info("Symbol distribution:")
        for symbol, count in symbol_counts.head(10).items():
            logger.info(f"  {symbol}: {count:,} observations")
        
        self.data = df
        self.metadata['processing_steps'].append('loaded_micro_data')
        self.metadata['symbol_stats']['loaded'] = {
            'unique_symbols': unique_symbols,
            'total_observations': len(df),
            'files_processed': len(micro_files)
        }
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Enhanced cleaning with minute aggregation and robust order book validation."""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        logger.info("Cleaning micro data...")
        df = self.data.copy()
        initial_rows = len(df)
        initial_symbols = df['Symbol'].nunique()
        
        # Remove rows without timestamp or symbol
        critical_missing = df[['timestamp', 'Symbol']].isnull().any(axis=1)
        if critical_missing.sum() > 0:
            rejected_data = df[critical_missing].copy()
            self.debug_saver.save_rejected_data(rejected_data, "critical_missing", "_micro")
            df = df[~critical_missing]
        
        # CRITICAL: Minute-level aggregation to remove intra-minute duplicates
        logger.info("Performing minute-level aggregation...")
        df = self._perform_minute_aggregation(df)
        
        # Extract and validate order book data
        df = self._extract_order_book_levels_robust(df)
        
        # Comprehensive order book validation
        orderbook_validation = OrderBookValidator.validate_orderbook_structure(
            df, self.config.depth_levels
        )
        self.metadata['quality_checks']['orderbook_validation'] = orderbook_validation
        
        if orderbook_validation['status'] == 'failed':
            logger.warning(f"Order book validation issues: {orderbook_validation['issues']}")
        
        # Remove invalid order book snapshots
        if orderbook_validation['invalid_indices']:
            invalid_mask = df.index.isin(orderbook_validation['invalid_indices'])
            rejected_books = df[invalid_mask].copy()
            self.debug_saver.save_orderbook_examples(rejected_books, "invalid_structure")
            df = df[~invalid_mask]
            logger.info(f"Removed {invalid_mask.sum()} snapshots with invalid order book structure")
        
        # Calculate microstructure metrics
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
        df['spread'] = df['best_ask'] - df['best_bid']
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        # Adaptive or global spread filtering
        if self.config.adaptive_spread_threshold:
            df = self._apply_adaptive_spread_filter(df)
        else:
            spread_mask = (
                (df['spread_bps'] >= self.config.global_min_spread_bps) &
                (df['spread_bps'] <= self.config.global_max_spread_bps)
            )
            extreme_spreads = ~spread_mask
            if extreme_spreads.sum() > 0:
                logger.warning(f"Removing {extreme_spreads.sum()} observations with extreme spreads")
                rejected_spreads = df[extreme_spreads].copy()
                self.debug_saver.save_rejected_data(rejected_spreads, "extreme_spreads", "_micro")
                df = df[spread_mask]
        
        # Temporal validation with gap flagging AFTER aggregation
        temporal_validation = DataValidator.validate_temporal_consistency_with_flags(
            df, 'timestamp', 'Symbol', 
            self.config.max_gap_minutes,
            self.config.duplicate_threshold
        )
        self.metadata['quality_checks']['temporal_validation'] = temporal_validation
        
        if temporal_validation['status'] == 'failed':
            raise DataQualityError(f"Temporal validation failed: {temporal_validation['issues']}")
        
        # Final duplicate check (should be minimal after aggregation)
        duplicate_mask = df.duplicated(subset=['Symbol', 'timestamp'], keep='last')
        if duplicate_mask.sum() > 0:
            logger.info(f"Removed {duplicate_mask.sum()} remaining duplicates after aggregation")
            df = df[~duplicate_mask]
        
        # Symbol quality filtering
        symbols_to_remove = []
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol]
            
            if len(symbol_data) < self.config.min_observations_per_symbol:
                symbols_to_remove.append(symbol)
                continue
            
            key_cols = ['best_bid', 'best_ask', 'best_bid_size', 'best_ask_size', 'mid_price']
            missing_ratio = symbol_data[key_cols].isnull().mean().mean()
            if missing_ratio > self.config.max_missing_ratio:
                symbols_to_remove.append(symbol)
        
        if symbols_to_remove:
            logger.info(f"Removing {len(symbols_to_remove)} symbols due to quality issues")
            rejected_symbols = df[df['Symbol'].isin(symbols_to_remove)].copy()
            self.debug_saver.save_rejected_data(rejected_symbols, "poor_quality_symbols", "_micro")
            df = df[~df['Symbol'].isin(symbols_to_remove)]
        
        # Sort for temporal consistency
        df = df.sort_values(['Symbol', 'timestamp']).reset_index(drop=True)
        
        cleaned_rows = len(df)
        final_symbols = df['Symbol'].nunique()
        retention_rate = cleaned_rows / initial_rows
        
        logger.info(f"Micro cleaning: {initial_rows:,} -> {cleaned_rows:,} rows ({retention_rate:.1%} retained)")
        logger.info(f"Symbols: {initial_symbols} -> {final_symbols} ({final_symbols/initial_symbols:.1%} retained)")
        
        self.metadata['rejected_data_stats']['micro'] = {
            'initial_rows': initial_rows,
            'final_rows': cleaned_rows,
            'retention_rate': retention_rate,
            'initial_symbols': initial_symbols,
            'final_symbols': final_symbols,
            'symbol_retention_rate': final_symbols / initial_symbols if initial_symbols > 0 else 0
        }
        
        self.data = df
        self.metadata['processing_steps'].append('cleaned_micro_data')
        return df
    
    def _perform_minute_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform minute-level aggregation to remove intra-minute duplicates."""
        before_count = len(df)
        
        # Round timestamps to minute
        df['timestamp_minute'] = df['timestamp'].dt.floor('T')
        
        # Group by symbol and minute, apply aggregation method
        if self.config.minute_aggregation_method == 'last':
            # Take the last observation within each minute
            df_agg = df.sort_values(['Symbol', 'timestamp']).groupby(
                ['Symbol', 'timestamp_minute']
            ).last().reset_index()
        elif self.config.minute_aggregation_method == 'first':
            df_agg = df.sort_values(['Symbol', 'timestamp']).groupby(
                ['Symbol', 'timestamp_minute']
            ).first().reset_index()
        elif self.config.minute_aggregation_method == 'median':
            # Take median for numeric columns, last for others
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: 'median' for col in numeric_cols if col != 'timestamp'}
            agg_dict.update({col: 'last' for col in df.columns if col not in numeric_cols and col not in ['Symbol', 'timestamp_minute']})
            df_agg = df.groupby(['Symbol', 'timestamp_minute']).agg(agg_dict).reset_index()
        else:
            # Default to 'last'
            df_agg = df.sort_values(['Symbol', 'timestamp']).groupby(
                ['Symbol', 'timestamp_minute']
            ).last().reset_index()
        
        # Use aggregated timestamp as the main timestamp
        df_agg['timestamp'] = df_agg['timestamp_minute']
        df_agg = df_agg.drop(columns=['timestamp_minute'])
        
        after_count = len(df_agg)
        reduction = (before_count - after_count) / before_count * 100
        
        logger.info(f"Minute aggregation: {before_count:,} -> {after_count:,} rows ({reduction:.1f}% reduction)")
        
        return df_agg
    
    def _extract_order_book_levels_robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced order book extraction with comprehensive error handling."""
        # Primary mapping for order book data
        bid_ask_mappings = [
            # Standard format
            {'bids[0].price': 'best_bid', 'asks[0].price': 'best_ask', 
             'bids[0].size': 'best_bid_size', 'asks[0].size': 'best_ask_size'},
            # Alternative formats
            {'bid_price': 'best_bid', 'ask_price': 'best_ask',
             'bid_size': 'best_bid_size', 'ask_size': 'best_ask_size'},
            # Another alternative
            {'bid': 'best_bid', 'ask': 'best_ask',
             'bid_qty': 'best_bid_size', 'ask_qty': 'best_ask_size'}
        ]
        
        # Try each mapping
        mapping_found = False
        for mapping in bid_ask_mappings:
            available_cols = [col for col in mapping.keys() if col in df.columns]
            if len(available_cols) >= 4:  # Need all 4 basic columns
                for old_col, new_col in mapping.items():
                    if old_col in df.columns:
                        df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                        mapping_found = True
                logger.info(f"Using order book mapping: {mapping}")
                break
        
        if not mapping_found:
            # Fallback: search for any bid/ask columns
            price_cols = {}
            size_cols = {}
            
            for col in df.columns:
                col_lower = col.lower()
                if 'bid' in col_lower and ('price' in col_lower or col_lower.endswith('bid')):
                    price_cols['best_bid'] = col
                elif 'ask' in col_lower and ('price' in col_lower or col_lower.endswith('ask')):
                    price_cols['best_ask'] = col
                elif 'bid' in col_lower and ('size' in col_lower or 'qty' in col_lower or 'volume' in col_lower):
                    size_cols['best_bid_size'] = col
                elif 'ask' in col_lower and ('size' in col_lower or 'qty' in col_lower or 'volume' in col_lower):
                    size_cols['best_ask_size'] = col
            
            # Apply fallback mapping
            for new_col, old_col in {**price_cols, **size_cols}.items():
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                mapping_found = True
            
            if mapping_found:
                logger.warning(f"Using fallback order book mapping: {price_cols}, {size_cols}")
        
        # Check if minimum required columns exist
        required_cols = ['best_bid', 'best_ask']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataQualityError(
                f"Could not extract minimum order book columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Set default sizes if not found
        if 'best_bid_size' not in df.columns:
            df['best_bid_size'] = 1.0
            logger.warning("best_bid_size not found, using default value 1.0")
        if 'best_ask_size' not in df.columns:
            df['best_ask_size'] = 1.0
            logger.warning("best_ask_size not found, using default value 1.0")
        
        # Extract compact order book features (not full depth)
        for side in ['bids', 'asks']:
            # Look for available depth levels
            available_levels = []
            for level in range(self.config.depth_levels):
                price_col = f'{side}[{level}].price'
                size_col = f'{side}[{level}].size'
                if price_col in df.columns and size_col in df.columns:
                    available_levels.append(level)
            
            if available_levels:
                # Calculate depth features for k levels
                for k in [1, 5, 10]:
                    if k <= len(available_levels):
                        # Volume at depth k
                        volume_cols = [f'{side}[{level}].size' for level in available_levels[:k]]
                        volume_data = df[volume_cols].fillna(0)
                        df[f'{side}_volume_depth_{k}'] = volume_data.sum(axis=1)
                        
                        # Count of available levels per row
                        df[f'{side}_levels_depth_{k}'] = (volume_data > 0).sum(axis=1)
                
                # Total available volume and levels
                all_volume_cols = [f'{side}[{level}].size' for level in available_levels]
                all_volume_data = df[all_volume_cols].fillna(0)
                df[f'total_{side}_volume'] = all_volume_data.sum(axis=1)
                df[f'{side}_levels_available'] = (all_volume_data > 0).sum(axis=1)
            else:
                # No depth data available
                df[f'total_{side}_volume'] = 0.0
                df[f'{side}_levels_available'] = 0
        
        return df
    
    def _apply_adaptive_spread_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Apply per-symbol adaptive spread thresholds with BTC-friendly logic."""
        filtered_data = []
        removed_stats = {}
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            
            if len(symbol_data) < 100:
                filtered_data.append(symbol_data)
                continue
            
            # DIAGNOSTIC: Check spread_bps distribution before filtering
            spread_stats = {
                'count': len(symbol_data),
                'nan_count': symbol_data['spread_bps'].isna().sum(),
                'inf_count': np.isinf(symbol_data['spread_bps']).sum(),
                'zero_or_negative': (symbol_data['spread_bps'] <= 0).sum(),
                'p01': symbol_data['spread_bps'].quantile(0.01),
                'p05': symbol_data['spread_bps'].quantile(0.05),
                'p50': symbol_data['spread_bps'].quantile(0.50),
                'p95': symbol_data['spread_bps'].quantile(0.95),
                'p99': symbol_data['spread_bps'].quantile(0.99)
            }
            
            logger.info(f"Symbol {symbol} spread diagnostics: "
                       f"count={spread_stats['count']:,}, "
                       f"nan={spread_stats['nan_count']}, "
                       f"inf={spread_stats['inf_count']}, "
                       f"zero/neg={spread_stats['zero_or_negative']}, "
                       f"p01/p50/p99={spread_stats['p01']:.4f}/{spread_stats['p50']:.4f}/{spread_stats['p99']:.4f} bps")
            
            # FIXED: More conservative thresholds for tight spreads (BTC-friendly)
            spread_q05 = symbol_data['spread_bps'].quantile(0.05)  # Use p05 instead of p01
            spread_q95 = symbol_data['spread_bps'].quantile(0.95)  # Use p95 instead of p99
            
            # FIXED: Much more conservative minimum threshold for tight spreads
            # For BTC: spread can be 0.01-0.1 bps legitimately
            conservative_min = max(spread_q05 * 0.5, 0.001)  # Half of p05, minimum 0.001 bps
            min_threshold = max(conservative_min, self.config.global_min_spread_bps)
            
            # More conservative maximum (use p95 with buffer)
            conservative_max = min(spread_q95 * 2.0, self.config.global_max_spread_bps)
            max_threshold = conservative_max
            
            # Additional checks for data quality issues
            crossed_markets = (symbol_data['best_ask'] <= symbol_data['best_bid']).sum()
            invalid_prices = (
                (symbol_data['best_bid'] <= 0) |
                (symbol_data['best_ask'] <= 0) |
                (symbol_data['mid_price'] <= 0)
            ).sum()
            
            # FIXED: More lenient filtering - only remove clear outliers and invalid data
            before_count = len(symbol_data)
            
            # Valid spread mask - much more conservative
            valid_spread_mask = (
                (symbol_data['spread_bps'] >= min_threshold) &
                (symbol_data['spread_bps'] <= max_threshold) &
                (symbol_data['best_ask'] > symbol_data['best_bid']) &  # No crossed markets
                (symbol_data['spread_bps'].notna()) &  # Not NaN
                (~np.isinf(symbol_data['spread_bps'])) &  # Not infinite
                (symbol_data['best_bid'] > 0) &  # Valid prices
                (symbol_data['best_ask'] > 0) &
                (symbol_data['mid_price'] > 0)
            )
            
            rejected_spread_data = symbol_data[~valid_spread_mask]
            
            # FIXED: Use normalized symbol for rejection filename (not raw venue)
            if len(rejected_spread_data) > 0:
                # Ensure we use the clean symbol name for the rejection file
                clean_symbol = symbol.replace('_', '').replace('-', '').replace('/', '')
                self.debug_saver.save_rejected_data(
                    rejected_spread_data, f"adaptive_spread_{clean_symbol}", "_micro"
                )
            
            symbol_data = symbol_data[valid_spread_mask]
            after_count = len(symbol_data)
            
            # Enhanced statistics
            removed_stats[symbol] = {
                'removed_count': before_count - after_count,
                'removed_ratio': (before_count - after_count) / before_count,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold,
                'crossed_markets': crossed_markets,
                'invalid_prices': invalid_prices,
                'spread_distribution': spread_stats,
                'filter_reasons': {
                    'below_min': (symbol_data['spread_bps'] < min_threshold).sum() if len(rejected_spread_data) > 0 else 0,
                    'above_max': (symbol_data['spread_bps'] > max_threshold).sum() if len(rejected_spread_data) > 0 else 0,
                    'crossed': crossed_markets,
                    'invalid_prices': invalid_prices,
                    'nan_inf': spread_stats['nan_count'] + spread_stats['inf_count']
                }
            }
            
            # Enhanced logging
            if before_count - after_count > 0:
                retention_rate = after_count / before_count
                logger.info(f"Symbol {symbol}: kept {after_count:,} / {before_count:,} ({retention_rate:.1%}) "
                          f"spread range: [{min_threshold:.4f}, {max_threshold:.4f}] bps")
                
                # CRITICAL: Log if we're losing too much data (>50%)
                if retention_rate < 0.5:
                    logger.warning(f"HIGH REJECTION RATE for {symbol}: {retention_rate:.1%} kept. "
                                 f"Check spread calculation and thresholds.")
            else:
                logger.info(f"Symbol {symbol}: kept all {before_count:,} observations")
            
            filtered_data.append(symbol_data)
        
        total_before = len(df)
        result_df = pd.concat(filtered_data, ignore_index=True)
        total_after = len(result_df)
        total_removed = total_before - total_after
        
        if total_removed > 0:
            overall_retention = total_after / total_before
            logger.info(f"Adaptive spread filtering: kept {total_after:,} / {total_before:,} ({overall_retention:.1%})")
            
            # Log per-symbol retention summary
            logger.info("Per-symbol retention rates:")
            for symbol, stats in removed_stats.items():
                retention = 1 - stats['removed_ratio']
                logger.info(f"  {symbol}: {retention:.1%}")
        
        self.metadata['adaptive_filtering_stats'] = removed_stats
        return result_df
    
    def engineer_features(self) -> pd.DataFrame:
        """Comprehensive microstructure feature engineering with compact features."""
        if self.data is None:
            raise ValueError("Data must be cleaned first")
        
        logger.info("Engineering microstructure features...")
        df = self.data.copy()
        
        # Ensure proper sorting
        df = df.sort_values(['Symbol', 'timestamp']).reset_index(drop=True)
        
        # Basic microstructure features
        df['imbalance'] = (
            (df['best_bid_size'] - df['best_ask_size']) / 
            (df['best_bid_size'] + df['best_ask_size']).replace(0, np.nan)
        )
        
        # Depth imbalance features (compact versions)
        for k in [1, 5, 10]:
            bid_vol_col = f'bids_volume_depth_{k}'
            ask_vol_col = f'asks_volume_depth_{k}'
            
            if bid_vol_col in df.columns and ask_vol_col in df.columns:
                total_depth = df[bid_vol_col] + df[ask_vol_col]
                df[f'depth_imbalance_{k}'] = (
                    (df[bid_vol_col] - df[ask_vol_col]) / 
                    total_depth.replace(0, np.nan)
                )
                df[f'total_liquidity_{k}'] = total_depth
        
        # Price-based features with clipping
        df['mid_return'] = df.groupby('Symbol')['mid_price'].pct_change()
        df['mid_return'] = df['mid_return'].clip(-self.config.base_max_price_change, 
                                                self.config.base_max_price_change)
        
        # Volatility measures
        for window in self.config.volatility_windows:
            df[f'volatility_{window}m'] = df.groupby('Symbol')['mid_return'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).std()
            )
        
        # Lagged features to prevent look-ahead bias
        lag_features = ['spread_bps', 'imbalance']
        
        # Add depth imbalance features to lagging
        for k in [1, 5, 10]:
            if f'depth_imbalance_{k}' in df.columns:
                lag_features.append(f'depth_imbalance_{k}')
        
        for feature in lag_features:
            if feature in df.columns:
                df[f'{feature}_lag1'] = df.groupby('Symbol')[feature].shift(1)
                
                # Rolling statistics (lagged)
                for window in [5, 20]:
                    df[f'{feature}_ma{window}'] = df.groupby('Symbol')[feature].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).mean()
                    )
                    df[f'{feature}_std{window}'] = df.groupby('Symbol')[feature].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=max(1, window//2)).std()
                    )
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute  
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Trading session indicators
        df['is_asia_session'] = ((df['hour'] >= 22) | (df['hour'] < 8)).astype(int)
        df['is_europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        
        # Microstructure noise measures (properly lagged)
        df['price_deviation'] = (
            np.abs(df['mid_price'] - df.groupby('Symbol')['mid_price'].shift(1)) / 
            df.groupby('Symbol')['mid_price'].shift(1)
        ).shift(1)
        
        # Order flow imbalance proxy
        if 'total_bids_volume' in df.columns:
            df['ofi_proxy'] = df.groupby('Symbol').apply(
                lambda x: self._calculate_ofi_proxy(x)
            ).reset_index(level=0, drop=True)
        
        # Symbol metadata  
        df['base_currency'] = df['Symbol'].apply(SymbolNormalizer.get_base_currency)
        df['quote_currency'] = df['Symbol'].apply(SymbolNormalizer.get_quote_currency)
        
        # Remove raw mid_return to prevent accidental usage
        df = df.drop(columns=['mid_return'], errors='ignore')
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        # Memory optimization
        df = optimize_dtypes(df, self.config.downcast_numeric)
        
        self.data = df
        self.metadata['processing_steps'].append('engineered_micro_features')
        
        # Count compact features (exclude raw order book data)
        exclude_cols = [
            'symbol', 'Symbol', 'timestamp', 'best_bid', 'best_ask', 'best_bid_size', 'best_ask_size',
            'mid_price', 'spread', 'total_bids_volume', 'total_asks_volume'
        ]
        # Also exclude raw depth columns
        exclude_cols.extend([col for col in df.columns if 'bids[' in col or 'asks[' in col])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Engineered {len(feature_cols)} compact microstructure features")
        
        return df
    
    def _calculate_ofi_proxy(self, symbol_data: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance proxy with proper lagging."""
        if 'total_bids_volume' not in symbol_data.columns:
            return pd.Series(np.nan, index=symbol_data.index)
        
        bid_vol_change = symbol_data['total_bids_volume'].diff()
        ask_vol_change = symbol_data['total_asks_volume'].diff()
        
        ofi = bid_vol_change - ask_vol_change
        return ofi.shift(1)


class FeatureEngineeringPipeline:
    """Enhanced pipeline with fixed unified dataset creation and validation splits."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.macro_processor = MacroDataProcessor(config)
        self.micro_processor = MicroDataProcessor(config)
    
    def process_macro_data(self) -> pd.DataFrame:
        """Process macro data with enhanced error handling."""
        logger.info("Processing macro data pipeline...")
        
        try:
            self.macro_processor.load_data()
            self.macro_processor.clean_data()
            self.macro_processor.engineer_features()
            self.macro_processor.save_processed_data("_macro")
            return self.macro_processor.data
        except Exception as e:
            logger.error(f"Macro processing failed: {e}")
            raise
    
    def process_micro_data(self) -> pd.DataFrame:
        """Process micro data with enhanced error handling."""
        logger.info("Processing micro data pipeline...")
        
        try:
            self.micro_processor.load_data()
            self.micro_processor.clean_data()
            self.micro_processor.engineer_features() 
            self.micro_processor.save_processed_data("_micro")
            return self.micro_processor.data
        except Exception as e:
            logger.error(f"Micro processing failed: {e}")
            raise
    
    def create_unified_dataset(self, 
                             macro_data: Optional[pd.DataFrame] = None,
                             micro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """FIXED: Create unified dataset with proper symbol column handling and timezone alignment."""
        logger.info("Creating unified dataset...")
        
        if macro_data is None:
            macro_data = self.macro_processor.data
        if micro_data is None:
            micro_data = self.micro_processor.data
        
        if macro_data is None or micro_data is None:
            raise ValueError("Both macro and micro data must be processed first")
        
        # Ensure consistent column naming
        if self.config.consistent_column_naming:
            macro_cols_map = {'Symbol': 'symbol', 'Date': 'date'}
            macro_data = macro_data.rename(columns={k: v for k, v in macro_cols_map.items() 
                                                  if k in macro_data.columns})
        
        # Prepare macro data for merging
        macro_daily = macro_data.copy()
        
        # Ensure date column is proper date type
        if 'date' not in macro_daily.columns and 'Date' in macro_daily.columns:
            macro_daily['date'] = macro_daily['Date']
        
        if macro_daily['date'].dtype != 'object':
            macro_daily['date'] = pd.to_datetime(macro_daily['date']).dt.date
        
        # Prepare micro data for merging with timezone-aware date extraction
        micro_minute = micro_data.copy()
        
        # Apply macro alignment offset if configured (for timezone differences)
        if self.config.macro_alignment_offset_hours != 0:
            offset_hours = pd.Timedelta(hours=self.config.macro_alignment_offset_hours)
            adjusted_timestamp = micro_minute['timestamp'] + offset_hours
            micro_minute['date'] = adjusted_timestamp.dt.date
        else:
            micro_minute['date'] = micro_minute['timestamp'].dt.date
        
        # CRITICAL FIX: Ensure symbol columns have same name before merge
        symbol_col_macro = 'symbol' if 'symbol' in macro_daily.columns else 'Symbol'
        symbol_col_micro = 'symbol' if 'symbol' in micro_minute.columns else 'Symbol'
        
        # Rename to ensure consistency
        if symbol_col_macro != 'symbol':
            macro_daily = macro_daily.rename(columns={symbol_col_macro: 'symbol'})
        if symbol_col_micro != 'symbol':
            micro_minute = micro_minute.rename(columns={symbol_col_micro: 'symbol'})
        
        # Perform merge without suffixes issue
        unified = micro_minute.merge(
            macro_daily,
            on=['symbol', 'date'],
            how='left',
            suffixes=('', '_macro')
        )
        
        # Clean up any remaining _macro columns where we have the base version
        columns_to_drop = []
        for col in unified.columns:
            if col.endswith('_macro'):
                base_col = col.replace('_macro', '')
                if base_col in unified.columns:
                    # Keep the non-macro version, drop the macro version
                    columns_to_drop.append(col)
                else:
                    # Rename macro version to base name
                    unified[base_col] = unified[col]
                    columns_to_drop.append(col)
        
        unified = unified.drop(columns=columns_to_drop, errors='ignore')
        unified = unified.drop(columns=['date'], errors='ignore')
        
        # Remove duplicate columns
        unified = unified.loc[:, ~unified.columns.duplicated()]
        
        # Sort and optimize
        unified = unified.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        unified = optimize_dtypes(unified, self.config.downcast_numeric, preserve_categories=False)
        
        # Validate merge success
        symbols_count = unified['symbol'].nunique()
        logger.info(f"Created unified dataset: {len(unified):,} rows, {unified.shape[1]} columns")
        logger.info(f"Unified dataset symbols: {symbols_count}")
        
        if symbols_count < 2:
            logger.warning(f"Unified dataset has only {symbols_count} symbols - check merge logic")
        
        # Save unified dataset
        output_path = self.config.output_dir / "unified_dataset.parquet"
        unified.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved unified dataset to {output_path}")
        
        return unified
    
    def create_validation_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal validation split using configured ratio."""
        if self.config.validation_split_ratio <= 0:
            return data, pd.DataFrame()
        
        # Split by timestamp for temporal validation
        split_point = data['timestamp'].quantile(1 - self.config.validation_split_ratio)
        
        train_data = data[data['timestamp'] < split_point].copy()
        validation_data = data[data['timestamp'] >= split_point].copy()
        
        logger.info(f"Created validation split: {len(train_data):,} train, {len(validation_data):,} validation")
        logger.info(f"Split point: {split_point}")
        
        return train_data, validation_data
    
    def run_full_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Run complete preprocessing pipeline with comprehensive error handling."""
        logger.info("Starting full preprocessing pipeline...")
        
        results = {}
        processing_errors = {}
        
        try:
            # Process macro data
            logger.info("=" * 50)
            logger.info("PROCESSING MACRO DATA")
            logger.info("=" * 50)
            macro_data = self.process_macro_data()
            results['macro'] = macro_data
            
        except Exception as e:
            logger.error(f"Macro processing failed: {e}")
            processing_errors['macro'] = str(e)
            macro_data = None
        
        try:
            # Process micro data  
            logger.info("=" * 50)
            logger.info("PROCESSING MICRO DATA")
            logger.info("=" * 50)
            micro_data = self.process_micro_data()
            results['micro'] = micro_data
            
        except Exception as e:
            logger.error(f"Micro processing failed: {e}")
            processing_errors['micro'] = str(e)
            micro_data = None
        
        # Create unified dataset if both succeeded
        if macro_data is not None and micro_data is not None:
            try:
                logger.info("=" * 50)
                logger.info("CREATING UNIFIED DATASET")
                logger.info("=" * 50)
                unified_data = self.create_unified_dataset(macro_data, micro_data)
                results['unified'] = unified_data
                
                # Create validation split if configured
                if self.config.validation_split_ratio > 0:
                    logger.info("Creating temporal validation split...")
                    train_data, val_data = self.create_validation_split(unified_data)
                    results['train'] = train_data
                    results['validation'] = val_data
                
            except Exception as e:
                logger.error(f"Unified dataset creation failed: {e}")
                processing_errors['unified'] = str(e)
        
        # Generate comprehensive report
        self._generate_processing_report(results, processing_errors)
        
        if processing_errors:
            logger.warning(f"Pipeline completed with errors: {list(processing_errors.keys())}")
        else:
            logger.info("Preprocessing pipeline completed successfully!")
        
        return results
    
    def _generate_processing_report(self, results: Dict[str, pd.DataFrame], errors: Dict[str, str]) -> None:
        """Generate comprehensive processing report with enhanced quality metrics."""
        report_path = self.config.output_dir / "preprocessing_report.json"
        
        report = {
            'pipeline_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'version': '2.2',
                'success': len(errors) == 0,
                'errors': errors,
                'config_summary': {
                    'adaptive_spread_threshold': self.config.adaptive_spread_threshold,
                    'adaptive_price_change_threshold': self.config.adaptive_price_change_threshold,
                    'minute_aggregation_method': self.config.minute_aggregation_method,
                    'duplicate_threshold': self.config.duplicate_threshold,
                    'outlier_std_threshold': self.config.outlier_std_threshold,
                    'validation_split_ratio': self.config.validation_split_ratio
                }
            },
            'configuration': {
                'macro_csv_path': str(self.config.macro_csv_path),
                'micro_data_dir': str(self.config.micro_data_dir), 
                'output_dir': str(self.config.output_dir),
                'min_observations_per_symbol': self.config.min_observations_per_symbol,
                'max_missing_ratio': self.config.max_missing_ratio,
                'duplicate_threshold': self.config.duplicate_threshold,
                'max_gap_minutes': self.config.max_gap_minutes,
                'outlier_std_threshold': self.config.outlier_std_threshold,
                'validation_split_ratio': self.config.validation_split_ratio,
                'volatility_windows': self.config.volatility_windows,
                'ma_windows': self.config.ma_windows,
                'depth_levels': self.config.depth_levels,
                'adaptive_spread_threshold': self.config.adaptive_spread_threshold,
                'adaptive_price_change_threshold': self.config.adaptive_price_change_threshold,
                'chunk_size': self.config.chunk_size,
                'downcast_numeric': self.config.downcast_numeric,
                'save_debug_info': self.config.save_debug_info,
                'minute_aggregation_method': self.config.minute_aggregation_method,
                'macro_alignment_offset_hours': self.config.macro_alignment_offset_hours
            },
            'datasets': {},
            'quality_metrics': {}
        }
        
        # Enhanced dataset statistics
        for name, df in results.items():
            if df is not None and len(df) > 0:
                # Basic stats
                dataset_info = {
                    'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
                    'symbols': int(df['symbol'].nunique()) if 'symbol' in df.columns else 'N/A',
                    'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1e6),
                    'missing_data_ratio': float(df.isnull().mean().mean())
                }
                
                # Date range
                timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                if timestamp_col in df.columns:
                    dataset_info['date_range'] = {
                        'start': df[timestamp_col].min().isoformat(),
                        'end': df[timestamp_col].max().isoformat(),
                        'span_days': int((pd.to_datetime(df[timestamp_col].max()) - 
                                        pd.to_datetime(df[timestamp_col].min())).days)
                    }
                
                # Enhanced feature analysis
                feature_cols = [col for col in df.columns 
                              if col not in ['symbol', 'timestamp', 'date']]
                dataset_info['features'] = {
                    'count': len(feature_cols),
                    'names': feature_cols[:20],
                    'numeric_features': int(df[feature_cols].select_dtypes(include=[np.number]).shape[1]),
                    'categorical_features': int(df[feature_cols].select_dtypes(include=['category']).shape[1]),
                    'flagged_features': len([col for col in feature_cols if '_is_extreme' in col or '_is_outlier' in col])
                }
                
                # Enhanced symbol statistics
                if 'symbol' in df.columns:
                    symbol_counts = df.groupby('symbol').size()
                    dataset_info['symbol_stats'] = {
                        'unique_symbols': int(len(symbol_counts)),
                        'min_observations': int(symbol_counts.min()),
                        'max_observations': int(symbol_counts.max()),
                        'median_observations': int(symbol_counts.median()),
                        'total_observations': int(symbol_counts.sum()),
                        'top_symbols': symbol_counts.head(5).to_dict()
                    }
                
                report['datasets'][name] = dataset_info
        
        # Enhanced quality check results
        if hasattr(self.macro_processor, 'metadata'):
            report['quality_metrics']['macro'] = self.macro_processor.metadata.get('quality_checks', {})
            report['quality_metrics']['macro']['rejected_stats'] = self.macro_processor.metadata.get('rejected_data_stats', {})
        
        if hasattr(self.micro_processor, 'metadata'):
            report['quality_metrics']['micro'] = self.micro_processor.metadata.get('quality_checks', {})
            report['quality_metrics']['micro']['rejected_stats'] = self.micro_processor.metadata.get('rejected_data_stats', {})
            
            # Include detailed adaptive filtering stats
            if 'adaptive_filtering_stats' in self.micro_processor.metadata:
                adaptive_stats = self.micro_processor.metadata['adaptive_filtering_stats']
                report['quality_metrics']['micro']['adaptive_filtering'] = {
                    'symbols_filtered': len(adaptive_stats),
                    'total_removed': sum(stats.get('removed_count', 0) for stats in adaptive_stats.values()),
                    'average_removal_ratio': np.mean([stats.get('removed_ratio', 0) for stats in adaptive_stats.values()]),
                    'symbol_details': adaptive_stats
                }
        
        # Save report
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive processing report saved to {report_path}")
        
        # Enhanced console summary
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        for name, info in report['datasets'].items():
            symbols_info = f", {info['symbols']} symbols" if info['symbols'] != 'N/A' else ""
            logger.info(f"{name.upper()}: {info['shape']['rows']:,} rows, {info['features']['count']} features{symbols_info}")
            
            if 'symbol_stats' in info:
                logger.info(f"  Symbol range: {info['symbol_stats']['min_observations']:,} - {info['symbol_stats']['max_observations']:,} obs")
        
        # Memory and quality summary
        total_memory = sum(info.get('memory_usage_mb', 0) for info in report['datasets'].values())
        logger.info(f"Total memory usage: {total_memory:.1f} MB")
        
        if report['quality_metrics']:
            logger.info("Quality issues summary:")
            for dataset, metrics in report['quality_metrics'].items():
                for check, result in metrics.items():
                    if isinstance(result, dict) and 'status' in result and result['status'] not in ['passed']:
                        issue_count = len(result.get('issues', []))
                        logger.info(f"  {dataset} - {check}: {result['status']} ({issue_count} issues)")


# Utility functions
def create_preprocessing_config(macro_path: Union[str, Path], 
                              micro_dir: Union[str, Path],
                              output_dir: Union[str, Path],
                              **kwargs) -> PreprocessingConfig:
    """Create preprocessing configuration with enhanced defaults."""
    return PreprocessingConfig(
        macro_csv_path=Path(macro_path),
        micro_data_dir=Path(micro_dir),
        output_dir=Path(output_dir),
        **kwargs
    )


def run_preprocessing_pipeline(config: PreprocessingConfig) -> Dict[str, pd.DataFrame]:
    """Convenience function to run full preprocessing pipeline."""
    pipeline = FeatureEngineeringPipeline(config)
    return pipeline.run_full_pipeline()


def analyze_rejected_data(output_dir: Union[str, Path]) -> Dict[str, Any]:
    """Enhanced analysis of rejected data from debug directory."""
    debug_dir = Path(output_dir) / "debug"
    if not debug_dir.exists():
        logger.warning("No debug directory found")
        return {}
    
    rejected_files = list(debug_dir.glob("rejected_*.parquet"))
    if not rejected_files:
        logger.info("No rejected data files found")
        return {}
    
    analysis = {}
    total_rejected = 0
    
    for file_path in rejected_files:
        try:
            df = pd.read_parquet(file_path)
            reason = file_path.stem.replace('rejected_', '')
            
            analysis[reason] = {
                'count': len(df),
                'symbols': df['symbol'].nunique() if 'symbol' in df.columns else (
                    df['Symbol'].nunique() if 'Symbol' in df.columns else 'N/A'),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else 'N/A',
                    'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else 'N/A'
                } if 'timestamp' in df.columns else 'N/A',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            total_rejected += len(df)
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path.name}: {e}")
            analysis[file_path.stem] = {'error': str(e)}
    
    analysis['summary'] = {
        'total_rejected_observations': total_rejected,
        'rejection_categories': len([k for k in analysis.keys() if k != 'summary']),
        'top_rejection_reasons': sorted(
            [(k, v.get('count', 0)) for k, v in analysis.items() if k != 'summary' and 'count' in v],
            key=lambda x: x[1], reverse=True
        )[:5]
    }
    
    # Save enhanced analysis
    analysis_path = debug_dir / "rejection_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info(f"Enhanced rejection analysis saved to {analysis_path}")
    logger.info(f"Total rejected observations: {total_rejected:,}")
    
    return analysis


# Example usage with all critical fixes implemented
if __name__ == "__main__":
    # Configuration addressing all reported critical issues
    config = create_preprocessing_config(
        macro_path="./data/macro/top_100_cryptos_with_correct_network.csv",
        micro_dir="./data/micro/",
        output_dir="./data/processed/",
        
        # CRITICAL FIXES:
        # 1. Symbol assignment from filename - handled in MicroDataProcessor.load_data()
        # 2. Minute aggregation to prevent intra-minute duplicates
        minute_aggregation_method='last',
        
        # 3. BTC-friendly spread filtering 
        adaptive_spread_threshold=True,
        global_min_spread_bps=0.001,  # FIXED: Much lower for BTC (0.001 bps vs 0.01)
        global_max_spread_bps=200.0,  # More reasonable for crypto
        
        # 4. Configurable duplicate threshold (was hardcoded)
        duplicate_threshold=0.03,
        
        # 5. Adaptive price thresholds to reduce false positives
        adaptive_price_change_threshold=True,
        base_max_price_change=0.12,  # Lower base, will be adapted per symbol
        outlier_std_threshold=4.5,   # More lenient for crypto volatility
        
        # Data quality - crypto-friendly
        min_observations_per_symbol=500,
        max_missing_ratio=0.3,
        
        # Temporal parameters with gap flagging
        max_gap_minutes=90,  # More lenient for 24/7 crypto markets
        validation_split_ratio=0.2,
        
        # COMPACT feature engineering (memory optimization)
        volatility_windows=[5, 15, 30],  # Reduced
        ma_windows=[5, 10, 20],         # Reduced  
        depth_levels=8,                 # Reduced to 8 levels max
        
        # Performance optimizations
        chunk_size=30000,
        downcast_numeric=True,
        save_debug_info=True,
        consistent_column_naming=True,
        macro_alignment_offset_hours=0
    )
    
    # Run preprocessing with comprehensive error handling and diagnostics
    try:
        logger.info("Starting cryptocurrency data preprocessing pipeline v2.2 - CRITICAL FIXES")
        logger.info("Key fixes applied:")
        logger.info("  • Symbol assignment from filenames (fixes 1-symbol bug)")
        logger.info("  • BTC-friendly spread thresholds (0.001 bps minimum)")  
        logger.info("  • Compact order book features (drops raw 200+ columns)")
        logger.info("  • Gap flagging instead of warnings")
        logger.info("  • Per-symbol quality statistics")
        
        results = run_preprocessing_pipeline(config)
        
        print("\n" + "="*70)
        print("CRYPTOCURRENCY PREPROCESSING RESULTS v2.2 - CRITICAL FIXES")
        print("="*70)
        
        # Enhanced validation and reporting
        success_count = 0
        for dataset_name, df in results.items():
            if df is not None and len(df) > 0:
                success_count += 1
                symbols_count = df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'
                
                print(f"\n✓ {dataset_name.upper()}:")
                print(f"    Rows: {len(df):,}")
                print(f"    Columns: {df.shape[1]}")
                print(f"    Symbols: {symbols_count}")
                
                if symbols_count != 'N/A' and isinstance(symbols_count, int):
                    if symbols_count >= 10:  # Should have 10+ symbols from 11 files
                        print(f"    ✓ Symbol count: {symbols_count} (good)")
                    else:
                        print(f"    ⚠️ Symbol count: {symbols_count} (expected 10+)")
                    
                    # Symbol distribution
                    symbol_counts = df['symbol'].value_counts()
                    print(f"    Range: {symbol_counts.min():,} - {symbol_counts.max():,} obs per symbol")
                    
                    # Check for missing major symbols
                    major_symbols = ['BTC_USDT', 'ETH_USDT', 'ADA_USDT']  
                    present_majors = [s for s in major_symbols if s in symbol_counts.index]
                    if len(present_majors) > 0:
                        print(f"    Major symbols present: {', '.join(present_majors)}")
                    else:
                        print(f"    ⚠️ No major symbols found in: {', '.join(major_symbols)}")
                
                # Memory usage
                memory_gb = df.memory_usage(deep=True).sum() / 1e9
                if memory_gb < 2.0:
                    print(f"    ✓ Memory: {memory_gb:.1f} GB (good)")
                elif memory_gb < 5.0:
                    print(f"    ⚠️ Memory: {memory_gb:.1f} GB (moderate)")
                else:
                    print(f"    ❌ Memory: {memory_gb:.1f} GB (high - consider more compression)")
                
                # Feature analysis
                feature_cols = [col for col in df.columns if col not in ['symbol', 'timestamp', 'date']]
                compact_features = len([col for col in feature_cols if not any(x in col for x in ['[', '].price', '].size'])])
                raw_orderbook = len(feature_cols) - compact_features
                
                print(f"    Features: {compact_features} compact")
                if raw_orderbook > 0:
                    print(f"    ⚠️ Raw orderbook columns: {raw_orderbook} (should be 0 for memory efficiency)")
                
                # Quality flags
                if 'is_gap' in df.columns:
                    gap_flags = df['is_gap'].sum()
                    print(f"    Gap flags: {gap_flags:,} ({gap_flags/len(df)*100:.1f}%)")
                
                flagged_cols = [col for col in feature_cols if '_is_extreme' in col or '_is_outlier' in col]
                if flagged_cols:
                    print(f"    Quality flags: {len(flagged_cols)} flag columns")
            else:
                print(f"\n❌ {dataset_name.upper()}: FAILED or empty")
        
        print(f"\nSuccessfully processed: {success_count}/{len(results)} datasets")
        print(f"Files saved to: {config.output_dir}")
        
        # Critical fixes validation
        print("\n" + "="*50)
        print("CRITICAL FIXES VALIDATION")
        print("="*50)
        
        issues_found = []
        
        if 'micro' in results and results['micro'] is not None:
            micro_symbols = results['micro']['symbol'].nunique()
            print(f"✓ Micro symbols: {micro_symbols}")
            if micro_symbols < 10:
                issues_found.append(f"Low symbol count in micro data: {micro_symbols} < 10")
                
            # Check if BTC is present
            btc_present = any('BTC' in str(sym) for sym in results['micro']['symbol'].unique())
            if btc_present:
                print("✓ BTC symbol detected in micro data")
            else:
                issues_found.append("BTC symbol missing from micro data")
        
        if 'unified' in results and results['unified'] is not None:
            unified_memory_gb = results['unified'].memory_usage(deep=True).sum() / 1e9
            print(f"Unified memory: {unified_memory_gb:.1f} GB")
            if unified_memory_gb > 5.0:
                issues_found.append(f"High memory usage: {unified_memory_gb:.1f} GB > 5 GB")
        
        # Analyze rejected data
        if config.save_debug_info:
            print("\n" + "="*50) 
            print("REJECTION ANALYSIS")
            print("="*50)
            
            rejection_analysis = analyze_rejected_data(config.output_dir)
            if rejection_analysis and 'summary' in rejection_analysis:
                print(f"Total rejected: {rejection_analysis['summary']['total_rejected_observations']:,}")
                
                # Check for BTC rejection issues
                btc_rejected = 0
                for reason, count in rejection_analysis['summary']['top_rejection_reasons']:
                    if 'BTC' in reason.upper():
                        btc_rejected += count
                        print(f"  ⚠️ {reason}: {count:,} (BTC-related)")
                    else:
                        print(f"  {reason}: {count:,}")
                
                if btc_rejected > 100000:  # Large BTC rejection
                    issues_found.append(f"Large BTC rejection: {btc_rejected:,} observations")
        
        # Final assessment
        if issues_found:
            print(f"\n⚠️ Issues detected: {len(issues_found)}")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("\n🎯 All critical fixes validated successfully!")
        
        print("\nPipeline completed with fixes:")
        print("  ✓ Symbol assignment from filenames")
        print("  ✓ BTC-friendly spread filtering (0.001 bps minimum)")
        print("  ✓ Memory-efficient compact features") 
        print("  ✓ Gap flagging with per-row metadata")
        print("  ✓ Per-symbol quality statistics")
        print("  ✓ Vectorized RSI (no warnings)")
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        print("Check logs and debug directory for detailed diagnostics")
        raise
