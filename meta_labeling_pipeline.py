"""
Meta-Labeling Pipeline for BIST VIOP (Turkish Futures) Market
==============================================================

This script implements a complete Meta-Labeling framework using:
1. Volatility-normalized features for regime-adaptive signals
2. Triple Barrier labeling for dynamic profit-taking/stop-loss
3. XGBoost classifier with walk-forward validation
4. Comparison of naive vs ML-filtered strategies

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA GENERATION & PREPROCESSING
# =============================================================================

def generate_sample_ohlcv(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data that mimics BIST VIOP futures behavior.
    Includes regime changes (trending + mean-reverting periods).
    """
    np.random.seed(seed)
    
    # Start price for a typical VIOP contract (e.g., XU030 mini futures)
    initial_price = 5000.0
    
    # Generate returns with regime switching
    # Regime 1: Low volatility trending (60% of time)
    # Regime 2: High volatility mean-reverting (40% of time)
    regimes = np.random.choice([0, 1], size=n_bars, p=[0.6, 0.4])
    
    returns = np.zeros(n_bars)
    for i in range(n_bars):
        if regimes[i] == 0:
            # Low vol trend: small drift + low noise
            returns[i] = 0.0002 + np.random.normal(0, 0.008)
        else:
            # High vol mean-revert: no drift + high noise
            returns[i] = np.random.normal(0, 0.02)
    
    # Generate close prices
    close = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close (realistic intraday ranges)
    atr_pct = 0.015  # ~1.5% average true range
    high = close * (1 + np.abs(np.random.normal(0, atr_pct/2, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, atr_pct/2, n_bars)))
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n_bars)
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    # Generate volume with volatility correlation
    base_volume = 10000
    volume = base_volume * (1 + np.abs(returns) * 50) * np.random.uniform(0.5, 1.5, n_bars)
    
    # Create DataFrame with datetime index
    dates = pd.date_range(start='2022-01-01', periods=n_bars, freq='h')
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int)
    }, index=dates)
    
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) - the key volatility normalizer.
    
    WHY ATR FOR NORMALIZATION?
    --------------------------
    ATR adapts to current market volatility, making our features
    regime-independent. A 1% move means very different things in:
    - Low volatility regime: Strong signal
    - High volatility regime: Just noise
    
    By dividing by ATR, we create "volatility-adjusted" features that
    maintain consistent signal strength across different market regimes.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    Standard momentum oscillator bounded 0-100.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) - trend strength indicator.
    
    ADX > 25: Strong trend (good for trend-following)
    ADX < 20: Weak/no trend (avoid trend-following signals)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # True Range
    tr = compute_atr(df, period=1) * period  # Unnormalized TR
    atr = compute_atr(df, period)
    
    # Smoothed DI
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    
    return adx


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-normalized features for the ML model.
    
    VOLATILITY NORMALIZATION PHILOSOPHY:
    ====================================
    The core insight is that raw price-based features are 
    NON-STATIONARY - their distributions shift with market regimes.
    
    By normalizing with ATR (or realized volatility), we create
    STATIONARY features that:
    1. Have consistent distributions across time
    2. Enable the ML model to learn patterns that generalize
    3. Reduce the need for constant model retraining
    
    Example: "Price dropped 50 points" means nothing alone.
             "Price dropped 2 ATRs" is universally significant.
    """
    features = pd.DataFrame(index=df.index)
    
    # Core volatility measure
    atr = compute_atr(df, period=14)
    
    # ---------------------------------------------------------------------
    # FEATURE 1: Volatility-Normalized Returns
    # WHY: Raw returns are heteroskedastic. Dividing by ATR creates
    #      approximately unit-scale returns regardless of regime.
    # ---------------------------------------------------------------------
    returns = df['close'].pct_change()
    features['norm_return_1'] = returns / (atr / df['close'])
    features['norm_return_5'] = df['close'].pct_change(5) / (atr / df['close'])
    features['norm_return_10'] = df['close'].pct_change(10) / (atr / df['close'])
    
    # ---------------------------------------------------------------------
    # FEATURE 2: Distance from MA Normalized by ATR
    # WHY: "Price is 2% above its MA" varies in meaning by volatility.
    #      "Price is 1.5 ATRs above MA" is consistently meaningful.
    # ---------------------------------------------------------------------
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    
    features['dist_ema20_norm'] = (df['close'] - ema_20) / atr
    features['dist_ema50_norm'] = (df['close'] - ema_50) / atr
    
    # ---------------------------------------------------------------------
    # FEATURE 3: Volume Z-Score
    # WHY: Volume anomalies often precede significant moves.
    #      Z-score provides regime-independent volume signal.
    # ---------------------------------------------------------------------
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    features['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)
    
    # ---------------------------------------------------------------------
    # FEATURE 4: RSI (Already Normalized 0-100)
    # WHY: Momentum oscillator for overbought/oversold conditions.
    #      Used both as feature and base signal generator.
    # ---------------------------------------------------------------------
    features['rsi'] = compute_rsi(df['close'], period=14)
    
    # ---------------------------------------------------------------------
    # FEATURE 5: ADX (Trend Strength, Normalized 0-100)
    # WHY: Meta-model should know if we're in trending or ranging market.
    #      Trend-following signals in ranging markets = false positives.
    # ---------------------------------------------------------------------
    features['adx'] = compute_adx(df, period=14)
    
    # ---------------------------------------------------------------------
    # FEATURE 6: ATR Ratio (Volatility Expansion/Contraction)
    # WHY: Volatility breakouts often signal regime change.
    #      Ratio of short-term to long-term ATR captures this.
    # ---------------------------------------------------------------------
    atr_fast = compute_atr(df, period=7)
    atr_slow = compute_atr(df, period=21)
    features['atr_ratio'] = atr_fast / (atr_slow + 1e-10)
    
    # ---------------------------------------------------------------------
    # FEATURE 7: Higher Highs / Lower Lows Count
    # WHY: Trend continuation pattern. More consecutive HH/LL = stronger trend.
    # ---------------------------------------------------------------------
    hh = (df['high'] > df['high'].shift(1)).astype(int)
    ll = (df['low'] < df['low'].shift(1)).astype(int)
    features['hh_count'] = hh.rolling(5).sum()
    features['ll_count'] = ll.rolling(5).sum()
    
    # Store ATR for barrier calculations
    features['atr'] = atr
    
    return features


# =============================================================================
# SECTION 2: BASE STRATEGY (EVENT GENERATOR)
# =============================================================================

def generate_base_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Generate primary trading signals using a simple trend-following logic.
    
    Strategy: RSI Oversold/Overbought with Trend Confirmation
    - BUY (1):  RSI < 30 AND ADX > 20 (Oversold in trending market)
    - SELL (-1): RSI > 70 AND ADX > 20 (Overbought in trending market)
    
    These are CANDIDATE signals - the meta-model will filter false positives.
    """
    signals = pd.Series(0, index=df.index)
    
    rsi = features['rsi']
    adx = features['adx']
    
    # Buy signal: RSI oversold + trend present
    buy_condition = (rsi < 30) & (adx > 20)
    
    # Sell signal: RSI overbought + trend present
    sell_condition = (rsi > 70) & (adx > 20)
    
    signals[buy_condition] = 1   # Long entry candidates
    signals[sell_condition] = -1  # Short entry candidates
    
    # Remove consecutive signals (only keep first in sequence)
    signals = signals.where(signals != signals.shift(1), 0)
    
    return signals


# =============================================================================
# SECTION 3: TRIPLE BARRIER LABELING
# =============================================================================

def apply_triple_barrier(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.Series,
    pt_mult: float = 2.0,    # Take profit = pt_mult * ATR
    sl_mult: float = 1.0,    # Stop loss = sl_mult * ATR
    max_holding: int = 10    # Max bars to hold
) -> pd.DataFrame:
    """
    Implement Triple Barrier Labeling Method.
    
    THE TRIPLE BARRIER CONCEPT:
    ===========================
    Traditional fixed-time labeling (e.g., "what's return after 10 bars?")
    ignores realistic trading mechanics. Triple Barrier models actual trades:
    
    1. UPPER BARRIER (Take Profit): Close if profit target hit
    2. LOWER BARRIER (Stop Loss): Close if loss limit hit  
    3. VERTICAL BARRIER (Time Stop): Close if neither hit within max_holding
    
    WHY ATR-BASED BARRIERS?
    -----------------------
    Fixed point/percentage barriers fail across volatility regimes:
    - In low vol: TP might never be hit (too wide)
    - In high vol: SL hit immediately (too tight)
    
    ATR-based barriers adapt to current market conditions:
    - Low vol period: Tighter barriers (smaller ATR)
    - High vol period: Wider barriers (larger ATR)
    
    Returns:
    --------
    DataFrame with columns:
    - 'direction': 1 (long), -1 (short), 0 (no signal)
    - 'barrier_hit': 'upper', 'lower', 'vertical'
    - 'label': 1 if upper barrier hit first, 0 otherwise
    - 'return': Actual return of the trade
    - 'bars_held': Number of bars until barrier hit
    """
    
    close = df['close'].values
    atr = features['atr'].values
    signal_values = signals.values
    
    n = len(df)
    labels = np.zeros(n)
    barrier_hit = np.array(['none'] * n, dtype=object)
    trade_returns = np.zeros(n)
    bars_held = np.zeros(n, dtype=int)
    
    for i in range(n):
        if signal_values[i] == 0:
            continue
            
        direction = signal_values[i]  # 1 for long, -1 for short
        entry_price = close[i]
        current_atr = atr[i]
        
        if np.isnan(current_atr) or current_atr == 0:
            continue
        
        # Set barriers based on ATR at entry
        take_profit = entry_price + direction * pt_mult * current_atr
        stop_loss = entry_price - direction * sl_mult * current_atr
        
        # Walk forward to find which barrier is hit first
        for j in range(i + 1, min(i + max_holding + 1, n)):
            current_close = close[j]
            
            # Check upper barrier (take profit)
            if direction == 1:  # Long position
                if current_close >= take_profit:
                    labels[i] = 1
                    barrier_hit[i] = 'upper'
                    trade_returns[i] = (current_close - entry_price) / entry_price
                    bars_held[i] = j - i
                    break
                elif current_close <= stop_loss:
                    labels[i] = 0
                    barrier_hit[i] = 'lower'
                    trade_returns[i] = (current_close - entry_price) / entry_price
                    bars_held[i] = j - i
                    break
            else:  # Short position
                if current_close <= take_profit:
                    labels[i] = 1
                    barrier_hit[i] = 'upper'
                    trade_returns[i] = (entry_price - current_close) / entry_price
                    bars_held[i] = j - i
                    break
                elif current_close >= stop_loss:
                    labels[i] = 0
                    barrier_hit[i] = 'lower'
                    trade_returns[i] = (entry_price - current_close) / entry_price
                    bars_held[i] = j - i
                    break
        else:
            # Vertical barrier hit (max holding period reached)
            if i + max_holding < n:
                exit_price = close[i + max_holding]
                trade_returns[i] = direction * (exit_price - entry_price) / entry_price
                labels[i] = 1 if trade_returns[i] > 0 else 0
                barrier_hit[i] = 'vertical'
                bars_held[i] = max_holding
    
    result = pd.DataFrame({
        'direction': signals,
        'barrier_hit': barrier_hit,
        'label': labels,
        'trade_return': trade_returns,
        'bars_held': bars_held
    }, index=df.index)
    
    return result


# =============================================================================
# SECTION 4: XGBOOST META-MODEL WITH WALK-FORWARD VALIDATION
# =============================================================================

def train_meta_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int = 5
) -> tuple:
    """
    Train XGBoost meta-model using Walk-Forward Validation.
    
    WHY WALK-FORWARD (TIME SERIES SPLIT)?
    =====================================
    Standard K-Fold cross-validation SHUFFLES data, which causes:
    1. LOOK-AHEAD BIAS: Model sees "future" data during training
    2. OVERFITTING: Learns temporal correlations it can't exploit live
    
    Walk-Forward Validation respects temporal ordering:
    - Train on past, validate on future (never the reverse)
    - Each fold moves forward in time
    - Mimics actual trading where you can't see the future
    
    Returns:
    --------
    Tuple of (trained_model, predictions_df, feature_importance)
    """
    
    # Filter to only signal rows (meta-labeling approach)
    signal_mask = labels['direction'] != 0
    X = features[signal_mask].drop(columns=['atr'], errors='ignore')
    y = labels.loc[signal_mask, 'label']
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n{'='*60}")
    print(f"META-MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Total signal events: {len(X)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Handle class imbalance with scale_pos_weight
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_predictions = pd.Series(index=X.index, dtype=float)
    fold_metrics = []
    
    # Ensure labels are integers
    y = y.astype(int)
    
    print(f"\nWalk-Forward Validation ({n_splits} splits):")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Skip fold if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold+1}: Skipped (single class in training set)")
            continue
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        all_predictions.iloc[val_idx] = y_pred_proba
        
        # Calculate metrics
        y_pred = (y_pred_proba > 0.5).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        
        fold_metrics.append({'precision': precision, 'recall': recall, 'auc': auc})
        print(f"Fold {fold+1}: Precision={precision:.3f}, Recall={recall:.3f}, AUC={auc:.3f}")
    
    # Train final model on all data
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "-" * 50)
    print("Average Metrics Across Folds:")
    avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    for metric, value in avg_metrics.items():
        print(f"  {metric.upper()}: {value:.3f}")
    
    return final_model, all_predictions, importance


# =============================================================================
# SECTION 5: EVALUATION & STRATEGY COMPARISON
# =============================================================================

def evaluate_strategies(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: pd.Series,
    threshold: float = 0.6
) -> pd.DataFrame:
    """
    Compare equity curves of:
    A) Naive Strategy: Take ALL primary signals
    B) ML-Filtered Strategy: Take signals only if P(success) > threshold
    
    Returns DataFrame with cumulative returns for both strategies.
    """
    
    print(f"\n{'='*60}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    # Get signal rows with valid predictions
    signal_mask = labels['direction'] != 0
    signal_labels = labels[signal_mask].copy()
    
    # Align predictions with labels
    signal_labels['pred_proba'] = predictions
    signal_labels = signal_labels.dropna(subset=['pred_proba'])
    
    # Naive strategy: Take all signals
    naive_returns = signal_labels['trade_return']
    naive_cumret = (1 + naive_returns).cumprod()
    
    # ML-Filtered strategy: Only take high-probability signals
    ml_mask = signal_labels['pred_proba'] > threshold
    ml_returns = signal_labels.loc[ml_mask, 'trade_return']
    
    # Create full-length cumulative return series
    ml_cumret = pd.Series(1.0, index=signal_labels.index)
    ml_cumret[ml_mask] = (1 + ml_returns).cumprod()
    ml_cumret = ml_cumret.ffill()
    
    # Statistics
    print(f"\nNaive Strategy (all signals):")
    print(f"  Total trades: {len(naive_returns)}")
    print(f"  Win rate: {(naive_returns > 0).mean():.1%}")
    print(f"  Total return: {(naive_cumret.iloc[-1] - 1):.1%}")
    print(f"  Avg return per trade: {naive_returns.mean():.2%}")
    
    print(f"\nML-Filtered Strategy (P > {threshold}):")
    print(f"  Total trades: {ml_mask.sum()}")
    print(f"  Trades filtered out: {(~ml_mask).sum()} ({(~ml_mask).mean():.1%})")
    if ml_mask.sum() > 0:
        print(f"  Win rate: {(ml_returns > 0).mean():.1%}")
        print(f"  Total return: {(ml_cumret.iloc[-1] - 1):.1%}")
        print(f"  Avg return per trade: {ml_returns.mean():.2%}")
    
    # Final metrics on filtered predictions
    if len(signal_labels[signal_labels['pred_proba'].notna()]) > 0:
        y_true = signal_labels['label']
        y_pred_proba = signal_labels['pred_proba']
        y_pred = (y_pred_proba > threshold).astype(int)
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION METRICS (Threshold = {threshold})")
        print(f"{'='*60}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
        if len(np.unique(y_true)) > 1:
            print(f"AUC-ROC: {roc_auc_score(y_true, y_pred_proba):.3f}")
    
    # Return equity curves
    equity = pd.DataFrame({
        'naive_cumret': naive_cumret,
        'ml_cumret': ml_cumret
    })
    
    return equity


def plot_results(equity: pd.DataFrame, importance: pd.DataFrame):
    """
    Plot equity curves and feature importance.
    Optional - requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Equity curves
        ax1 = axes[0]
        equity['naive_cumret'].plot(ax=ax1, label='Naive (All Signals)', alpha=0.7)
        equity['ml_cumret'].plot(ax=ax1, label='ML-Filtered (P > 0.6)', alpha=0.7)
        ax1.set_title('Equity Curves: Naive vs ML-Filtered Strategy')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        ax2 = axes[1]
        importance.head(10).plot(
            kind='barh', 
            x='feature', 
            y='importance', 
            ax=ax2,
            legend=False
        )
        ax2.set_title('Top 10 Feature Importance')
        ax2.set_xlabel('Importance')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('meta_labeling_results.png', dpi=150, bbox_inches='tight')
        print("\n[INFO] Results saved to 'meta_labeling_results.png'")
        plt.show()
        
    except ImportError:
        print("\n[INFO] matplotlib not installed. Skipping plots.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline for Meta-Labeling.
    """
    print("=" * 60)
    print("META-LABELING PIPELINE FOR BIST VIOP")
    print("=" * 60)
    
    # Step 1: Generate/Load Data
    print("\n[1/5] Generating sample OHLCV data...")
    df = generate_sample_ohlcv(n_bars=2000)
    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Compute Features
    print("\n[2/5] Computing volatility-normalized features...")
    features = compute_features(df)
    print(f"  Features: {list(features.columns)}")
    
    # Step 3: Generate Base Signals
    print("\n[3/5] Generating base strategy signals...")
    signals = generate_base_signals(df, features)
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    print(f"  Buy signals: {n_buy}")
    print(f"  Sell signals: {n_sell}")
    print(f"  Total events: {n_buy + n_sell}")
    
    # Step 4: Apply Triple Barrier Labeling
    print("\n[4/5] Applying Triple Barrier labeling...")
    labels = apply_triple_barrier(
        df, features, signals,
        pt_mult=2.0,  # Take profit = 2 ATR
        sl_mult=1.0,  # Stop loss = 1 ATR
        max_holding=10  # Max 10 bars
    )
    
    # Barrier hit distribution
    signal_labels = labels[labels['direction'] != 0]
    barrier_counts = signal_labels['barrier_hit'].value_counts()
    print(f"  Barrier hit distribution:")
    for barrier, count in barrier_counts.items():
        if barrier != 'none':
            print(f"    {barrier}: {count} ({count/len(signal_labels):.1%})")
    
    # Step 5: Train Meta-Model
    print("\n[5/5] Training XGBoost Meta-Model...")
    model, predictions, importance = train_meta_model(features, labels, n_splits=5)
    
    # Feature importance summary
    print("\nTop 5 Most Important Features:")
    print(importance.head().to_string(index=False))
    
    # Evaluate strategies
    equity = evaluate_strategies(df, labels, predictions, threshold=0.6)
    
    # Optional: Plot results
    plot_results(equity, importance)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return df, features, labels, model, predictions, equity


if __name__ == "__main__":
    df, features, labels, model, predictions, equity = main()
