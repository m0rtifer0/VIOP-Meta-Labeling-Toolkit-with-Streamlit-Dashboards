"""
BIST100 Meta-Labeling Dashboard with Stochastic Simulations
============================================================

Interactive Streamlit dashboard for:
1. BIST100 Meta-Labeling ML Pipeline
2. Strategy Monte Carlo Simulations
3. Walk-Forward Model Analysis
4. Risk Metrics and VaR

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="BIST100 Meta-Labeling Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e63946 0%, #1d3557 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA GENERATION
# =============================================================================

@st.cache_data
def generate_bist100_data(years: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic BIST100-like data."""
    np.random.seed(seed)
    
    n_bars = years * 252
    initial_price = 80000.0
    annual_drift = 0.15
    annual_vol = 0.30
    
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    regimes = np.random.choice([0, 1, 2], size=n_bars, p=[0.60, 0.30, 0.10])
    
    returns = np.zeros(n_bars)
    for i in range(n_bars):
        if regimes[i] == 0:
            returns[i] = np.random.normal(daily_drift, daily_vol)
        elif regimes[i] == 1:
            returns[i] = np.random.normal(daily_drift * 0.5, daily_vol * 1.8)
        else:
            returns[i] = np.random.normal(-daily_drift * 2, daily_vol * 2.5)
    
    close = initial_price * np.cumprod(1 + returns)
    
    daily_range = 0.015
    high = close * (1 + np.abs(np.random.normal(0, daily_range, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, daily_range, n_bars)))
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n_bars)
    
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    base_volume = 500_000_000
    volume = base_volume * (1 + np.abs(returns) * 20) * np.random.uniform(0.5, 1.5, n_bars)
    
    end_date = pd.Timestamp.now()
    dates = pd.bdate_range(end=end_date, periods=n_bars)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int)
    }, index=dates)
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df['high'], df['low']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1/period, min_periods=period).mean()


@st.cache_data
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for meta-labeling."""
    features = pd.DataFrame(index=df.index)
    atr = compute_atr(df, period=14)
    
    # Normalized returns
    returns = df['close'].pct_change()
    features['norm_return_1'] = returns / (atr / df['close'])
    features['norm_return_5'] = df['close'].pct_change(5) / (atr / df['close'])
    features['norm_return_10'] = df['close'].pct_change(10) / (atr / df['close'])
    
    # Distance from MAs
    for span in [10, 20, 50]:
        ema = df['close'].ewm(span=span).mean()
        features[f'dist_ema{span}_norm'] = (df['close'] - ema) / atr
    
    # Volume
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    features['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)
    
    # Indicators
    features['rsi'] = compute_rsi(df['close'], period=14)
    features['adx'] = compute_adx(df, period=14)
    
    # ATR ratio
    atr_fast = compute_atr(df, period=7)
    atr_slow = compute_atr(df, period=21)
    features['atr_ratio'] = atr_fast / (atr_slow + 1e-10)
    
    # MACD
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9).mean()
    features['macd_norm'] = (macd - signal) / atr
    features['macd_signal'] = (macd > signal).astype(int)
    
    # Bollinger
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    features['bb_pct'] = (df['close'] - (sma - 2*std)) / (4*std + 1e-10)
    
    features['atr'] = atr
    
    return features


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df: pd.DataFrame, features: pd.DataFrame, 
                     rsi_oversold: int = 30, rsi_overbought: int = 70,
                     adx_threshold: int = 20) -> pd.Series:
    """Generate trading signals based on RSI and ADX."""
    signals = pd.Series(0, index=df.index)
    
    rsi = features['rsi']
    adx = features['adx']
    macd_signal = features['macd_signal']
    macd_prev = macd_signal.shift(1)
    
    buy_rsi = (rsi < rsi_oversold) & (adx > adx_threshold)
    buy_macd = (macd_signal == 1) & (macd_prev == 0) & (rsi < 50)
    
    sell_rsi = (rsi > rsi_overbought) & (adx > adx_threshold)
    sell_macd = (macd_signal == 0) & (macd_prev == 1) & (rsi > 50)
    
    signals[buy_rsi | buy_macd] = 1
    signals[sell_rsi | sell_macd] = -1
    signals = signals.where(signals != signals.shift(1), 0)
    
    return signals


# =============================================================================
# TRIPLE BARRIER LABELING
# =============================================================================

def apply_triple_barrier(df: pd.DataFrame, features: pd.DataFrame, signals: pd.Series,
                         pt_mult: float = 2.0, sl_mult: float = 1.0, 
                         max_holding: int = 10) -> pd.DataFrame:
    """Apply triple barrier labeling."""
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
            
            if direction == 1:
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
            else:
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
    
    return pd.DataFrame({
        'direction': signals,
        'barrier_hit': barrier_hit,
        'label': labels,
        'trade_return': trade_returns,
        'bars_held': bars_held
    }, index=df.index)


# =============================================================================
# META-MODEL TRAINING
# =============================================================================

def train_meta_model(features: pd.DataFrame, labels: pd.DataFrame, 
                     n_splits: int = 5, max_depth: int = 4,
                     learning_rate: float = 0.05) -> tuple:
    """Train XGBoost meta-model."""
    signal_mask = labels['direction'] != 0
    X = features[signal_mask].drop(columns=['atr'], errors='ignore')
    y = labels.loc[signal_mask, 'label']
    
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    if len(X) < 20:
        return None, None, None, []
    
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
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
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_train)) < 2:
            continue
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        all_predictions.iloc[val_idx] = y_pred_proba
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        
        fold_metrics.append({
            'fold': fold + 1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        })
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return final_model, all_predictions, importance, fold_metrics


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def monte_carlo_strategy(trade_returns: np.ndarray, n_trades: int, 
                         n_simulations: int = 1000) -> np.ndarray:
    """Simulate strategy equity curves using bootstrap."""
    equity_curves = np.zeros((n_simulations, n_trades))
    
    for i in range(n_simulations):
        sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
        equity_curves[i] = np.cumsum(sampled_returns)
    
    return equity_curves


def calculate_var_cvar(returns: np.ndarray, confidence: float = 0.95) -> tuple:
    """Calculate VaR and CVaR."""
    alpha = 1 - confidence
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
    return var, cvar


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_price_with_signals(df: pd.DataFrame, labels: pd.DataFrame) -> go.Figure:
    """Plot price chart with trading signals."""
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BIST100'
    ))
    
    buy_signals = labels[labels['direction'] == 1]
    sell_signals = labels[labels['direction'] == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=df.loc[buy_signals.index, 'low'] * 0.98,
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='lime'),
        name='Buy Signal'
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=df.loc[sell_signals.index, 'high'] * 1.02,
        mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        name='Sell Signal'
    ))
    
    fig.update_layout(
        title='BIST100 Price with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_equity_curves(naive_returns: pd.Series, ml_returns: pd.Series, 
                       threshold: float) -> go.Figure:
    """Plot equity curves comparison."""
    naive_cumret = (1 + naive_returns).cumprod()
    ml_cumret = (1 + ml_returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=naive_cumret.index,
        y=naive_cumret.values,
        mode='lines',
        name='Naive Strategy',
        line=dict(color='#636EFA', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=ml_cumret.index,
        y=ml_cumret.values,
        mode='lines',
        name=f'ML-Filtered (P > {threshold})',
        line=dict(color='#00CC96', width=2)
    ))
    
    fig.add_hline(y=1, line_dash='dash', line_color='gray', opacity=0.5)
    
    fig.update_layout(
        title='Equity Curves: Naive vs ML-Filtered',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_monte_carlo(equity_curves: np.ndarray) -> go.Figure:
    """Plot Monte Carlo simulation results."""
    n_sims, n_trades = equity_curves.shape
    trade_axis = np.arange(1, n_trades + 1)
    
    fig = go.Figure()
    
    # Sample paths
    for i in range(min(50, n_sims)):
        fig.add_trace(go.Scatter(
            x=trade_axis,
            y=equity_curves[i] * 100,
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 149, 237, 0.2)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Percentiles
    p5 = np.percentile(equity_curves, 5, axis=0) * 100
    p25 = np.percentile(equity_curves, 25, axis=0) * 100
    p50 = np.percentile(equity_curves, 50, axis=0) * 100
    p75 = np.percentile(equity_curves, 75, axis=0) * 100
    p95 = np.percentile(equity_curves, 95, axis=0) * 100
    
    fig.add_trace(go.Scatter(x=trade_axis, y=p95, mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=trade_axis, y=p5, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255, 99, 71, 0.2)', name='90% CI'))
    
    fig.add_trace(go.Scatter(x=trade_axis, y=p75, mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=trade_axis, y=p25, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255, 215, 0, 0.3)', name='50% CI'))
    
    fig.add_trace(go.Scatter(x=trade_axis, y=p50, mode='lines',
                             line=dict(width=3, color='#00FF7F'), name='Median'))
    
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    
    fig.update_layout(
        title='Monte Carlo Strategy Simulation',
        xaxis_title='Trade Number',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=450
    )
    
    return fig


def plot_feature_importance(importance: pd.DataFrame) -> go.Figure:
    """Plot feature importance."""
    top_n = min(12, len(importance))
    data = importance.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=data['importance'],
        y=data['feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='Top Feature Importance',
        xaxis_title='Importance',
        yaxis_title='',
        template='plotly_dark',
        height=400,
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_prediction_distribution(labels_with_pred: pd.DataFrame) -> go.Figure:
    """Plot prediction probability distribution."""
    valid = labels_with_pred.dropna(subset=['pred_proba'])
    
    winners = valid[valid['label'] == 1]['pred_proba']
    losers = valid[valid['label'] == 0]['pred_proba']
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(x=losers, name='Losing Trades', 
                               marker_color='rgba(255, 99, 71, 0.7)', nbinsx=20))
    fig.add_trace(go.Histogram(x=winners, name='Winning Trades',
                               marker_color='rgba(0, 255, 127, 0.7)', nbinsx=20))
    
    fig.update_layout(
        title='Prediction Probability Distribution',
        xaxis_title='Predicted Probability',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_dark',
        height=350
    )
    
    return fig


# =============================================================================
# STREAMLIT PAGES
# =============================================================================

def page_meta_labeling():
    """Meta-Labeling Pipeline Page."""
    st.header("Meta-Labeling Pipeline")
    st.markdown("Train XGBoost to filter trading signals based on win probability.")
    
    # Sidebar parameters
    st.sidebar.subheader("Data Parameters")
    years = st.sidebar.slider("Years of Data", 3, 15, 10)
    seed = st.sidebar.number_input("Random Seed", 1, 100, 42)
    
    st.sidebar.subheader("Signal Parameters")
    rsi_oversold = st.sidebar.slider("RSI Oversold", 15, 40, 30)
    rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 85, 70)
    adx_threshold = st.sidebar.slider("ADX Threshold", 10, 35, 20)
    
    st.sidebar.subheader("Barrier Parameters")
    pt_mult = st.sidebar.slider("Take Profit (ATR)", 1.0, 4.0, 2.0, 0.5)
    sl_mult = st.sidebar.slider("Stop Loss (ATR)", 0.5, 2.0, 1.0, 0.25)
    max_holding = st.sidebar.slider("Max Holding (days)", 5, 20, 10)
    
    st.sidebar.subheader("Model Parameters")
    max_depth = st.sidebar.slider("Max Depth", 2, 8, 4)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.05, 0.01)
    threshold = st.sidebar.slider("Filter Threshold", 0.4, 0.7, 0.55, 0.05)
    
    run_pipeline = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)
    
    if run_pipeline:
        with st.spinner("Running Meta-Labeling Pipeline..."):
            # Generate data
            df = generate_bist100_data(years=years, seed=seed)
            features = compute_features(df)
            
            # Generate signals
            signals = generate_signals(df, features, rsi_oversold, rsi_overbought, adx_threshold)
            n_signals = (signals != 0).sum()
            
            # Apply labeling
            labels = apply_triple_barrier(df, features, signals, pt_mult, sl_mult, max_holding)
            
            # Train model
            model, predictions, importance, fold_metrics = train_meta_model(
                features, labels, max_depth=max_depth, learning_rate=learning_rate
            )
            
            if model is None:
                st.error("Not enough signals to train model. Try adjusting parameters.")
                return
            
            # Prepare results
            signal_labels = labels[labels['direction'] != 0].copy()
            signal_labels['pred_proba'] = predictions
            signal_labels = signal_labels.dropna(subset=['pred_proba'])
            
            naive_returns = signal_labels['trade_return']
            ml_mask = signal_labels['pred_proba'] > threshold
            ml_returns = signal_labels.loc[ml_mask, 'trade_return']
        
        # Display results
        st.subheader("Pipeline Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", n_signals)
        with col2:
            st.metric("Naive Win Rate", f"{(naive_returns > 0).mean():.1%}")
        with col3:
            st.metric("ML-Filtered Trades", ml_mask.sum())
        with col4:
            if len(ml_returns) > 0:
                st.metric("ML Win Rate", f"{(ml_returns > 0).mean():.1%}")
        
        # Cross-validation results
        if fold_metrics:
            st.subheader("Walk-Forward Validation")
            metrics_df = pd.DataFrame(fold_metrics)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Precision", f"{metrics_df['precision'].mean():.3f}")
            with col2:
                st.metric("Avg Recall", f"{metrics_df['recall'].mean():.3f}")
            with col3:
                st.metric("Avg AUC", f"{metrics_df['auc'].mean():.3f}")
            
            st.dataframe(metrics_df.style.format({
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'auc': '{:.3f}'
            }), use_container_width=True)
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Equity Curves", "Feature Importance", "Predictions"])
        
        with tab1:
            st.plotly_chart(plot_price_with_signals(df.tail(500), labels.tail(500)), use_container_width=True)
        
        with tab2:
            st.plotly_chart(plot_equity_curves(naive_returns, ml_returns, threshold), use_container_width=True)
        
        with tab3:
            st.plotly_chart(plot_feature_importance(importance), use_container_width=True)
        
        with tab4:
            st.plotly_chart(plot_prediction_distribution(signal_labels), use_container_width=True)
        
        # Store in session for simulation
        st.session_state['ml_returns'] = ml_returns.values
        st.session_state['naive_returns'] = naive_returns.values


def page_monte_carlo():
    """Monte Carlo Simulation Page."""
    st.header("Strategy Monte Carlo Simulation")
    st.markdown("Simulate future performance using historical trade returns.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        use_ml = st.checkbox("Use ML-Filtered Returns", value=True)
        n_trades = st.slider("Number of Trades", 20, 200, 100)
        n_simulations = st.slider("Simulations", 500, 5000, 1000, 100)
        confidence = st.slider("VaR Confidence (%)", 90, 99, 95)
        
        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
    
    with col2:
        if 'ml_returns' not in st.session_state:
            st.warning("Please run the Meta-Labeling Pipeline first to generate trade returns.")
            return
        
        if run_sim:
            returns = st.session_state['ml_returns'] if use_ml else st.session_state['naive_returns']
            
            if len(returns) < 10:
                st.error("Not enough trades for simulation.")
                return
            
            with st.spinner("Running Monte Carlo..."):
                equity_curves = monte_carlo_strategy(returns, n_trades, n_simulations)
                final_returns = equity_curves[:, -1]
                
                var, cvar = calculate_var_cvar(final_returns, confidence / 100)
            
            # Metrics
            st.subheader("Simulation Results")
            
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Expected Return", f"{final_returns.mean()*100:.1f}%")
            with metric_cols[1]:
                st.metric("Median Return", f"{np.median(final_returns)*100:.1f}%")
            with metric_cols[2]:
                st.metric("Std Dev", f"{final_returns.std()*100:.1f}%")
            with metric_cols[3]:
                st.metric(f"VaR ({confidence}%)", f"{var*100:.1f}%")
            with metric_cols[4]:
                st.metric(f"CVaR ({confidence}%)", f"{cvar*100:.1f}%")
            
            prob_profit = (final_returns > 0).mean() * 100
            st.info(f"Probability of Profit after {n_trades} trades: **{prob_profit:.1f}%**")
            
            # Chart
            st.plotly_chart(plot_monte_carlo(equity_curves), use_container_width=True)
            
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            max_dd_per_sim = np.zeros(n_simulations)
            for i in range(n_simulations):
                cummax = np.maximum.accumulate(equity_curves[i])
                drawdown = equity_curves[i] - cummax
                max_dd_per_sim[i] = drawdown.min()
            
            dd_cols = st.columns(3)
            with dd_cols[0]:
                st.metric("Avg Max Drawdown", f"{max_dd_per_sim.mean()*100:.1f}%")
            with dd_cols[1]:
                st.metric("Worst Max Drawdown", f"{max_dd_per_sim.min()*100:.1f}%")
            with dd_cols[2]:
                st.metric("P(DD > 20%)", f"{(max_dd_per_sim < -0.20).mean()*100:.1f}%")


def page_parameter_sensitivity():
    """Parameter Sensitivity Analysis Page."""
    st.header("Parameter Sensitivity Analysis")
    st.markdown("Analyze how different thresholds affect strategy performance.")
    
    if 'naive_returns' not in st.session_state:
        st.warning("Please run the Meta-Labeling Pipeline first.")
        return
    
    # This would require storing more data in session state
    st.info("Run multiple threshold values to see sensitivity analysis.")
    
    # Simple threshold sensitivity
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Threshold Range")
        min_thresh = st.slider("Min Threshold", 0.3, 0.5, 0.4, 0.05)
        max_thresh = st.slider("Max Threshold", 0.5, 0.8, 0.7, 0.05)
        n_points = st.slider("Number of Points", 5, 20, 10)
        
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    
    with col2:
        if run_analysis:
            st.info("This feature requires storing prediction probabilities. Please extend the implementation.")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">BIST100 Meta-Labeling Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered signal filtering with Monte Carlo simulations</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["Meta-Labeling Pipeline", "Monte Carlo Simulation", "Parameter Sensitivity"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    This dashboard combines:
    - Meta-Labeling for signal filtering
    - Triple Barrier labeling
    - XGBoost classification
    - Monte Carlo simulations
    - Risk analysis (VaR/CVaR)
    """)
    
    if page == "Meta-Labeling Pipeline":
        page_meta_labeling()
    elif page == "Monte Carlo Simulation":
        page_monte_carlo()
    else:
        page_parameter_sensitivity()


if __name__ == "__main__":
    main()
