"""
Meta-Labeling Pipeline with Real BIST100 Data
==============================================

This script fetches 10 years of BIST100 data from Yahoo Finance
and applies the complete Meta-Labeling framework.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA FETCHING FROM YAHOO FINANCE
# =============================================================================

def fetch_bist100_data(years: int = 10, csv_path: str = None) -> pd.DataFrame:
    """
    Fetch BIST100 index data from Yahoo Finance or load from CSV.
    
    Ticker: XU100.IS (BIST 100 Index on Istanbul Stock Exchange)
    
    If Yahoo Finance fails (rate limiting), generates synthetic data
    that mimics BIST100 characteristics.
    """
    
    # Option 1: Load from CSV if provided
    if csv_path is not None:
        print(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        print(f"  Loaded {len(df)} bars from CSV")
        return df
    
    # Option 2: Try Yahoo Finance
    print(f"Fetching {years} years of BIST100 data from Yahoo Finance...")
    
    try:
        import time
        time.sleep(2)  # Small delay to avoid rate limiting
        
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        
        ticker = "XU100.IS"
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(df) == 0:
            raise ValueError("Empty dataframe received from Yahoo Finance")
        
        # Handle multi-index columns (yfinance >= 0.2.0 returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        print(f"  Downloaded {len(df)} bars")
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Price range: {df['close'].min():,.0f} - {df['close'].max():,.0f}")
        
        return df
        
    except Exception as e:
        print(f"  [WARNING] Yahoo Finance failed: {e}")
        print(f"  [INFO] Generating synthetic BIST100-like data for demonstration...")
        return generate_synthetic_bist100(years=years)


def generate_synthetic_bist100(years: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that mimics BIST100 characteristics.
    
    BIST100 characteristics (2014-2024):
    - High volatility (25-40% annually)
    - Significant trends with regime changes
    - Price range roughly 60,000 to 10,000,000 (before/after denomination)
    """
    np.random.seed(seed)
    
    n_bars = years * 252  # Trading days per year
    
    # BIST100 characteristics
    initial_price = 80000.0  # Starting price circa 2014
    annual_drift = 0.15  # ~15% annual return (includes inflation effects)
    annual_vol = 0.30  # ~30% annual volatility
    
    # Generate returns with regime switching
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    # Regime probabilities (60% normal, 30% high vol, 10% crisis)
    regimes = np.random.choice([0, 1, 2], size=n_bars, p=[0.60, 0.30, 0.10])
    
    returns = np.zeros(n_bars)
    for i in range(n_bars):
        if regimes[i] == 0:  # Normal regime
            returns[i] = np.random.normal(daily_drift, daily_vol)
        elif regimes[i] == 1:  # High volatility
            returns[i] = np.random.normal(daily_drift * 0.5, daily_vol * 1.8)
        else:  # Crisis regime
            returns[i] = np.random.normal(-daily_drift * 2, daily_vol * 2.5)
    
    # Generate close prices
    close = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLC
    daily_range = 0.015  # ~1.5% average daily range
    high = close * (1 + np.abs(np.random.normal(0, daily_range, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, daily_range, n_bars)))
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n_bars)
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    # Generate volume
    base_volume = 500_000_000  # ~500M TRY daily volume
    volume = base_volume * (1 + np.abs(returns) * 20) * np.random.uniform(0.5, 1.5, n_bars)
    
    # Create datetime index (trading days only)
    end_date = pd.Timestamp.now()
    dates = pd.bdate_range(end=end_date, periods=n_bars)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int)
    }, index=dates)
    
    print(f"  Generated {len(df)} synthetic bars")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: {df['close'].min():,.0f} - {df['close'].max():,.0f}")
    
    return df


# =============================================================================
# SECTION 2: FEATURE ENGINEERING (Same as before, volatility-normalized)
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    high = df['high']
    low = df['low']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = compute_atr(df, period)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    
    return adx


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    
    return pd.DataFrame({
        'bb_upper': sma + std_dev * std,
        'bb_middle': sma,
        'bb_lower': sma - std_dev * std,
        'bb_width': (sma + std_dev * std - (sma - std_dev * std)) / sma,
        'bb_pct': (series - (sma - std_dev * std)) / (2 * std_dev * std + 1e-10)
    })


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-normalized features for BIST100.
    """
    features = pd.DataFrame(index=df.index)
    
    # Core volatility measure
    atr = compute_atr(df, period=14)
    
    # Volatility-Normalized Returns
    returns = df['close'].pct_change()
    features['norm_return_1'] = returns / (atr / df['close'])
    features['norm_return_5'] = df['close'].pct_change(5) / (atr / df['close'])
    features['norm_return_10'] = df['close'].pct_change(10) / (atr / df['close'])
    features['norm_return_20'] = df['close'].pct_change(20) / (atr / df['close'])
    
    # Distance from MAs Normalized by ATR
    ema_10 = df['close'].ewm(span=10).mean()
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    ema_200 = df['close'].ewm(span=200).mean()
    
    features['dist_ema10_norm'] = (df['close'] - ema_10) / atr
    features['dist_ema20_norm'] = (df['close'] - ema_20) / atr
    features['dist_ema50_norm'] = (df['close'] - ema_50) / atr
    features['dist_ema200_norm'] = (df['close'] - ema_200) / atr
    
    # Trend direction (MA crossovers)
    features['ema_10_20_cross'] = (ema_10 > ema_20).astype(int)
    features['ema_20_50_cross'] = (ema_20 > ema_50).astype(int)
    features['ema_50_200_cross'] = (ema_50 > ema_200).astype(int)
    
    # Volume Z-Score
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    features['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)
    
    # RSI
    features['rsi'] = compute_rsi(df['close'], period=14)
    features['rsi_5'] = compute_rsi(df['close'], period=5)
    
    # ADX (Trend Strength)
    features['adx'] = compute_adx(df, period=14)
    
    # ATR Ratio (Volatility Expansion/Contraction)
    atr_fast = compute_atr(df, period=7)
    atr_slow = compute_atr(df, period=21)
    features['atr_ratio'] = atr_fast / (atr_slow + 1e-10)
    
    # MACD features
    macd_df = compute_macd(df['close'])
    features['macd_norm'] = macd_df['histogram'] / atr
    features['macd_signal'] = (macd_df['macd'] > macd_df['signal']).astype(int)
    
    # Bollinger Bands
    bb_df = compute_bollinger_bands(df['close'])
    features['bb_pct'] = bb_df['bb_pct']
    features['bb_width'] = bb_df['bb_width']
    
    # Higher Highs / Lower Lows
    hh = (df['high'] > df['high'].shift(1)).astype(int)
    ll = (df['low'] < df['low'].shift(1)).astype(int)
    features['hh_count_5'] = hh.rolling(5).sum()
    features['ll_count_5'] = ll.rolling(5).sum()
    features['hh_count_10'] = hh.rolling(10).sum()
    features['ll_count_10'] = ll.rolling(10).sum()
    
    # Realized volatility
    features['realized_vol_10'] = returns.rolling(10).std() * np.sqrt(252)
    features['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
    
    # Store ATR for barrier calculations
    features['atr'] = atr
    
    return features


# =============================================================================
# SECTION 3: BASE STRATEGY
# =============================================================================

def generate_base_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Generate primary trading signals using multiple conditions.
    
    Buy signals:
    - RSI < 30 AND ADX > 20 (Oversold in trend)
    - OR MACD cross up AND RSI < 50 (Momentum shift)
    
    Sell signals:
    - RSI > 70 AND ADX > 20 (Overbought in trend)
    - OR MACD cross down AND RSI > 50 (Momentum shift)
    """
    signals = pd.Series(0, index=df.index)
    
    rsi = features['rsi']
    adx = features['adx']
    macd_signal = features['macd_signal']
    macd_prev = macd_signal.shift(1)
    
    # Buy conditions
    buy_rsi = (rsi < 30) & (adx > 20)
    buy_macd = (macd_signal == 1) & (macd_prev == 0) & (rsi < 50)
    buy_condition = buy_rsi | buy_macd
    
    # Sell conditions
    sell_rsi = (rsi > 70) & (adx > 20)
    sell_macd = (macd_signal == 0) & (macd_prev == 1) & (rsi > 50)
    sell_condition = sell_rsi | sell_macd
    
    signals[buy_condition] = 1
    signals[sell_condition] = -1
    
    # Remove consecutive signals
    signals = signals.where(signals != signals.shift(1), 0)
    
    return signals


# =============================================================================
# SECTION 4: TRIPLE BARRIER LABELING
# =============================================================================

def apply_triple_barrier(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 10
) -> pd.DataFrame:
    """Apply Triple Barrier labeling method."""
    
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
            
        direction = signal_values[i]
        entry_price = close[i]
        current_atr = atr[i]
        
        if np.isnan(current_atr) or current_atr == 0:
            continue
        
        take_profit = entry_price + direction * pt_mult * current_atr
        stop_loss = entry_price - direction * sl_mult * current_atr
        
        for j in range(i + 1, min(i + max_holding + 1, n)):
            current_close = close[j]
            
            if direction == 1:  # Long
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
            else:  # Short
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
# SECTION 5: XGBOOST META-MODEL
# =============================================================================

def train_meta_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int = 5
) -> tuple:
    """Train XGBoost meta-model using Walk-Forward Validation."""
    
    signal_mask = labels['direction'] != 0
    X = features[signal_mask].drop(columns=['atr'], errors='ignore')
    y = labels.loc[signal_mask, 'label']
    
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n{'='*60}")
    print(f"META-MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Total signal events: {len(X)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
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
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_predictions = pd.Series(index=X.index, dtype=float)
    fold_metrics = []
    
    y = y.astype(int)
    
    print(f"\nWalk-Forward Validation ({n_splits} splits):")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold+1}: Skipped (single class in training set)")
            continue
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        all_predictions.iloc[val_idx] = y_pred_proba
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        
        fold_metrics.append({'precision': precision, 'recall': recall, 'auc': auc})
        print(f"Fold {fold+1}: Precision={precision:.3f}, Recall={recall:.3f}, AUC={auc:.3f}")
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if fold_metrics:
        print("\n" + "-" * 50)
        print("Average Metrics Across Folds:")
        avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        for metric, value in avg_metrics.items():
            print(f"  {metric.upper()}: {value:.3f}")
    
    return final_model, all_predictions, importance


# =============================================================================
# SECTION 6: EVALUATION
# =============================================================================

def evaluate_strategies(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: pd.Series,
    threshold: float = 0.55
) -> pd.DataFrame:
    """Compare naive vs ML-filtered strategies."""
    
    print(f"\n{'='*60}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    signal_mask = labels['direction'] != 0
    signal_labels = labels[signal_mask].copy()
    signal_labels['pred_proba'] = predictions
    signal_labels = signal_labels.dropna(subset=['pred_proba'])
    
    # Naive strategy
    naive_returns = signal_labels['trade_return']
    naive_cumret = (1 + naive_returns).cumprod()
    
    # ML-Filtered strategy
    ml_mask = signal_labels['pred_proba'] > threshold
    ml_returns = signal_labels.loc[ml_mask, 'trade_return']
    
    ml_cumret = pd.Series(1.0, index=signal_labels.index)
    if ml_mask.sum() > 0:
        ml_cumret[ml_mask] = (1 + ml_returns).cumprod()
    ml_cumret = ml_cumret.ffill()
    
    print(f"\nNaive Strategy (all signals):")
    print(f"  Total trades: {len(naive_returns)}")
    print(f"  Win rate: {(naive_returns > 0).mean():.1%}")
    print(f"  Total return: {(naive_cumret.iloc[-1] - 1)*100:.2f}%")
    print(f"  Avg return per trade: {naive_returns.mean()*100:.2f}%")
    
    print(f"\nML-Filtered Strategy (P > {threshold}):")
    print(f"  Total trades: {ml_mask.sum()}")
    print(f"  Trades filtered out: {(~ml_mask).sum()} ({(~ml_mask).mean():.1%})")
    if ml_mask.sum() > 0:
        print(f"  Win rate: {(ml_returns > 0).mean():.1%}")
        print(f"  Total return: {(ml_cumret.iloc[-1] - 1)*100:.2f}%")
        print(f"  Avg return per trade: {ml_returns.mean()*100:.2f}%")
    
    # Final metrics
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
    
    equity = pd.DataFrame({
        'naive_cumret': naive_cumret,
        'ml_cumret': ml_cumret
    })
    
    return equity, signal_labels


def plot_results(df: pd.DataFrame, equity: pd.DataFrame, importance: pd.DataFrame, signal_labels: pd.DataFrame):
    """Plot comprehensive results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. BIST100 Price with signals
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['close'], 'b-', alpha=0.7, label='BIST100')
    
    # Mark signals
    buy_signals = signal_labels[signal_labels['direction'] == 1]
    sell_signals = signal_labels[signal_labels['direction'] == -1]
    
    ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'close'], 
                marker='^', c='green', s=50, label='Buy Signal', alpha=0.7)
    ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'close'], 
                marker='v', c='red', s=50, label='Sell Signal', alpha=0.7)
    
    ax1.set_title('BIST100 with Trading Signals', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity Curves
    ax2 = axes[0, 1]
    equity['naive_cumret'].plot(ax=ax2, label='Naive (All Signals)', linewidth=2)
    equity['ml_cumret'].plot(ax=ax2, label='ML-Filtered (P > 0.55)', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Equity Curves: Naive vs ML-Filtered', fontsize=14)
    ax2.set_xlabel('Trade Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax3 = axes[1, 0]
    top_n = min(15, len(importance))
    importance.head(top_n).plot(
        kind='barh', 
        x='feature', 
        y='importance', 
        ax=ax3,
        legend=False,
        color='steelblue'
    )
    ax3.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    ax3.set_xlabel('Importance')
    ax3.invert_yaxis()
    
    # 4. Prediction Distribution
    ax4 = axes[1, 1]
    valid_preds = signal_labels.dropna(subset=['pred_proba'])
    
    # Separate by actual outcome
    winners = valid_preds[valid_preds['label'] == 1]['pred_proba']
    losers = valid_preds[valid_preds['label'] == 0]['pred_proba']
    
    ax4.hist(losers, bins=20, alpha=0.6, label='Losing Trades', color='red')
    ax4.hist(winners, bins=20, alpha=0.6, label='Winning Trades', color='green')
    ax4.axvline(x=0.55, color='black', linestyle='--', label='Threshold (0.55)')
    ax4.set_title('Prediction Probability Distribution', fontsize=14)
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bist100_meta_labeling_results.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] Results saved to 'bist100_meta_labeling_results.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline for BIST100 Meta-Labeling."""
    
    print("=" * 60)
    print("META-LABELING PIPELINE FOR BIST100")
    print("=" * 60)
    
    # Step 1: Fetch Real Data
    print("\n[1/5] Fetching BIST100 data from Yahoo Finance...")
    df = fetch_bist100_data(years=10)
    
    # Step 2: Compute Features
    print("\n[2/5] Computing volatility-normalized features...")
    features = compute_features(df)
    print(f"  Total features: {len(features.columns) - 1}")
    
    # Step 3: Generate Base Signals
    print("\n[3/5] Generating base strategy signals...")
    signals = generate_base_signals(df, features)
    n_buy = (signals == 1).sum()
    n_sell = (signals == -1).sum()
    print(f"  Buy signals: {n_buy}")
    print(f"  Sell signals: {n_sell}")
    print(f"  Total events: {n_buy + n_sell}")
    
    # Step 4: Triple Barrier Labeling
    print("\n[4/5] Applying Triple Barrier labeling...")
    labels = apply_triple_barrier(
        df, features, signals,
        pt_mult=2.0,
        sl_mult=1.0,
        max_holding=10
    )
    
    signal_labels = labels[labels['direction'] != 0]
    barrier_counts = signal_labels['barrier_hit'].value_counts()
    print(f"  Barrier hit distribution:")
    for barrier, count in barrier_counts.items():
        if barrier != 'none':
            pct = count / len(signal_labels) * 100
            print(f"    {barrier}: {count} ({pct:.1f}%)")
    
    # Step 5: Train Meta-Model
    print("\n[5/5] Training XGBoost Meta-Model...")
    model, predictions, importance = train_meta_model(features, labels, n_splits=5)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Evaluate strategies
    equity, signal_labels_with_pred = evaluate_strategies(df, labels, predictions, threshold=0.55)
    
    # Plot results
    plot_results(df, equity, importance, signal_labels_with_pred)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return df, features, labels, model, predictions, equity


if __name__ == "__main__":
    df, features, labels, model, predictions, equity = main()
