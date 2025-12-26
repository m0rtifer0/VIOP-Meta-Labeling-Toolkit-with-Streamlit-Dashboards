"""
Stochastic Simulation Dashboard for BIST VIOP
==============================================

Interactive Streamlit dashboard for running multiple stochastic simulations:
1. Monte Carlo Option Pricing (Black-Scholes, GBM paths)
2. Strategy Backtesting with Confidence Intervals
3. Portfolio Risk Analysis (VaR, CVaR)
4. Meta-Label Probability Distributions

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="VIOP Stochastic Simulator",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    .stMetric {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 10px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

@st.cache_data
def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None
) -> np.ndarray:
    """
    Geometric Brownian Motion (GBM) price path simulation.
    
    dS = μ*S*dt + σ*S*dW
    
    Parameters:
    -----------
    S0: Initial price
    mu: Drift (annualized)
    sigma: Volatility (annualized)
    T: Time horizon (years)
    n_steps: Number of time steps
    n_paths: Number of simulation paths
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Generate random increments
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # GBM formula: S(t+dt) = S(t) * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Cumulative sum for log returns
    log_returns = drift + diffusion
    log_paths = np.cumsum(log_returns, axis=1)
    
    # Convert to price paths
    paths = S0 * np.exp(np.column_stack([np.zeros(n_paths), log_paths]))
    
    return paths


def monte_carlo_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    option_type: str = 'call'
) -> Tuple[float, float, np.ndarray]:
    """
    Monte Carlo option pricing using risk-neutral valuation.
    
    Returns: (price, std_error, payoffs)
    """
    # Simulate terminal prices under risk-neutral measure
    Z = np.random.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    
    return price, std_error, ST


def calculate_var_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
    """
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


def simulate_strategy_returns(
    n_trades: int,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    n_simulations: int
) -> np.ndarray:
    """
    Simulate strategy equity curves using Bernoulli process for trade outcomes.
    """
    outcomes = np.random.binomial(1, win_rate, (n_simulations, n_trades))
    
    # Apply win/loss amounts
    returns = np.where(outcomes == 1, avg_win, -avg_loss)
    
    # Cumulative returns
    equity_curves = np.cumsum(returns, axis=1)
    
    return equity_curves


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_price_paths(paths: np.ndarray, title: str = "GBM Price Paths") -> go.Figure:
    """Create interactive plot of simulated price paths."""
    n_paths, n_steps = paths.shape
    time_axis = np.linspace(0, 1, n_steps)
    
    fig = go.Figure()
    
    # Plot sample paths (max 100 for performance)
    n_display = min(n_paths, 100)
    for i in range(n_display):
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=paths[i],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add mean path
    mean_path = paths.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=mean_path,
        mode='lines',
        line=dict(width=3, color='#FFD700'),
        name='Mean Path'
    ))
    
    # Add percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=p95,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=p5,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 215, 0, 0.2)',
        name='90% CI'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Time (Years)",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_terminal_distribution(ST: np.ndarray, K: float = None) -> go.Figure:
    """Plot histogram of terminal prices with statistics."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ST,
        nbinsx=50,
        marker_color='rgba(100, 149, 237, 0.7)',
        name='Terminal Prices'
    ))
    
    # Add vertical lines for statistics
    mean_ST = np.mean(ST)
    median_ST = np.median(ST)
    
    fig.add_vline(x=mean_ST, line_dash="dash", line_color="#FFD700",
                  annotation_text=f"Mean: {mean_ST:.2f}")
    fig.add_vline(x=median_ST, line_dash="dot", line_color="#00FF7F",
                  annotation_text=f"Median: {median_ST:.2f}")
    
    if K is not None:
        fig.add_vline(x=K, line_dash="solid", line_color="#FF6347",
                      annotation_text=f"Strike: {K:.2f}")
    
    fig.update_layout(
        title="Terminal Price Distribution",
        xaxis_title="Price",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_equity_curves(equity_curves: np.ndarray) -> go.Figure:
    """Plot strategy equity curve simulations with confidence bands."""
    n_sims, n_trades = equity_curves.shape
    trade_axis = np.arange(1, n_trades + 1)
    
    fig = go.Figure()
    
    # Plot sample paths
    n_display = min(n_sims, 50)
    for i in range(n_display):
        fig.add_trace(go.Scatter(
            x=trade_axis,
            y=equity_curves[i],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add statistics
    mean_curve = equity_curves.mean(axis=0)
    p5 = np.percentile(equity_curves, 5, axis=0)
    p25 = np.percentile(equity_curves, 25, axis=0)
    p75 = np.percentile(equity_curves, 75, axis=0)
    p95 = np.percentile(equity_curves, 95, axis=0)
    
    # 90% CI
    fig.add_trace(go.Scatter(
        x=trade_axis, y=p95, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=trade_axis, y=p5, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 99, 71, 0.2)',
        name='90% CI'
    ))
    
    # 50% CI
    fig.add_trace(go.Scatter(
        x=trade_axis, y=p75, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=trade_axis, y=p25, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 215, 0, 0.3)',
        name='50% CI'
    ))
    
    # Mean
    fig.add_trace(go.Scatter(
        x=trade_axis, y=mean_curve, mode='lines',
        line=dict(width=3, color='#00FF7F'),
        name='Expected Value'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title="Strategy Equity Curve Simulations",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L (%)",
        template="plotly_dark",
        height=500
    )
    
    return fig


def plot_var_distribution(returns: np.ndarray, var: float, cvar: float) -> go.Figure:
    """Plot return distribution with VaR and CVaR markers."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,  # Convert to percentage
        nbinsx=50,
        marker_color='rgba(100, 149, 237, 0.7)',
        name='Returns'
    ))
    
    # VaR line
    fig.add_vline(
        x=var * 100,
        line_dash="solid",
        line_color="#FF6347",
        annotation_text=f"VaR: {var*100:.2f}%"
    )
    
    # CVaR line
    fig.add_vline(
        x=cvar * 100,
        line_dash="dash",
        line_color="#FF1493",
        annotation_text=f"CVaR: {cvar*100:.2f}%"
    )
    
    fig.update_layout(
        title="Portfolio Return Distribution with Risk Metrics",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400
    )
    
    return fig


# =============================================================================
# STREAMLIT APP PAGES
# =============================================================================

def page_gbm_simulation():
    """GBM Price Path Simulation Page."""
    st.header("Geometric Brownian Motion Simulation")
    st.markdown("Simulate future price paths using stochastic differential equations.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        S0 = st.number_input("Initial Price (S₀)", value=5000.0, min_value=1.0, step=100.0)
        mu = st.slider("Drift (μ) - Annual", min_value=-0.5, max_value=0.5, value=0.10, step=0.01)
        sigma = st.slider("Volatility (σ) - Annual", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        T = st.slider("Time Horizon (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        n_steps = st.slider("Time Steps", min_value=50, max_value=500, value=252)
        n_paths = st.slider("Number of Paths", min_value=100, max_value=10000, value=1000, step=100)
        seed = st.number_input("Random Seed (0 = random)", value=42, min_value=0)
        
        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
    
    with col2:
        if run_sim or 'gbm_paths' in st.session_state:
            with st.spinner("Simulating paths..."):
                paths = simulate_gbm_paths(
                    S0, mu, sigma, T, n_steps, n_paths,
                    seed if seed > 0 else None
                )
                st.session_state['gbm_paths'] = paths
            
            # Statistics
            final_prices = paths[:, -1]
            
            st.subheader("Simulation Results")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Mean Final Price", f"{final_prices.mean():,.2f}")
            with metric_cols[1]:
                st.metric("Std Dev", f"{final_prices.std():,.2f}")
            with metric_cols[2]:
                expected = S0 * np.exp(mu * T)
                st.metric("Theoretical E[S_T]", f"{expected:,.2f}")
            with metric_cols[3]:
                prob_profit = (final_prices > S0).mean() * 100
                st.metric("P(Profit)", f"{prob_profit:.1f}%")
            
            # Plots
            st.plotly_chart(plot_price_paths(paths), use_container_width=True)
            st.plotly_chart(plot_terminal_distribution(final_prices), use_container_width=True)


def page_option_pricing():
    """Monte Carlo Option Pricing Page."""
    st.header("Monte Carlo Option Pricing")
    st.markdown("Price European options using Monte Carlo simulation under risk-neutral measure.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Option Parameters")
        S0 = st.number_input("Spot Price (S₀)", value=5000.0, min_value=1.0, step=100.0)
        K = st.number_input("Strike Price (K)", value=5000.0, min_value=1.0, step=100.0)
        r = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        sigma = st.slider("Implied Volatility (σ)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        T = st.slider("Time to Expiry (Years)", min_value=0.01, max_value=2.0, value=0.25, step=0.01)
        option_type = st.radio("Option Type", ["Call", "Put"])
        n_paths = st.slider("Simulation Paths", min_value=1000, max_value=100000, value=10000, step=1000)
        
        run_pricing = st.button("Price Option", type="primary", use_container_width=True)
    
    with col2:
        if run_pricing:
            with st.spinner("Running Monte Carlo..."):
                price, std_error, ST = monte_carlo_option_price(
                    S0, K, r, sigma, T, n_paths, option_type.lower()
                )
            
            # Black-Scholes for comparison
            from scipy.stats import norm
            d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == "Call":
                bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
            
            st.subheader("Pricing Results")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("MC Price", f"₺{price:,.2f}")
            with metric_cols[1]:
                st.metric("Std Error", f"±₺{std_error:,.2f}")
            with metric_cols[2]:
                st.metric("BS Price", f"₺{bs_price:,.2f}")
            with metric_cols[3]:
                error_pct = abs(price - bs_price) / bs_price * 100
                st.metric("MC Error %", f"{error_pct:.3f}%")
            
            # 95% Confidence Interval
            ci_low = price - 1.96 * std_error
            ci_high = price + 1.96 * std_error
            st.info(f"95% Confidence Interval: ₺{ci_low:,.2f} - ₺{ci_high:,.2f}")
            
            # Greeks approximation
            st.subheader("Approximate Greeks (Bump & Revalue)")
            delta_bump = 0.01 * S0
            price_up, _, _ = monte_carlo_option_price(S0 + delta_bump, K, r, sigma, T, n_paths, option_type.lower())
            price_down, _, _ = monte_carlo_option_price(S0 - delta_bump, K, r, sigma, T, n_paths, option_type.lower())
            delta = (price_up - price_down) / (2 * delta_bump)
            gamma = (price_up - 2*price + price_down) / (delta_bump**2)
            
            greek_cols = st.columns(2)
            with greek_cols[0]:
                st.metric("Delta (Δ)", f"{delta:.4f}")
            with greek_cols[1]:
                st.metric("Gamma (Γ)", f"{gamma:.6f}")
            
            # Distribution plot
            st.plotly_chart(plot_terminal_distribution(ST, K), use_container_width=True)


def page_strategy_simulation():
    """Strategy Backtesting with Monte Carlo."""
    st.header("Strategy Monte Carlo Simulation")
    st.markdown("Simulate strategy performance under varying market conditions.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Strategy Parameters")
        n_trades = st.slider("Number of Trades", min_value=10, max_value=500, value=100)
        win_rate = st.slider("Win Rate (%)", min_value=20, max_value=80, value=55) / 100
        avg_win = st.slider("Avg Win (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        avg_loss = st.slider("Avg Loss (%)", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
        n_simulations = st.slider("Simulations", min_value=100, max_value=5000, value=1000, step=100)
        
        run_strat = st.button("Simulate Strategy", type="primary", use_container_width=True)
    
    with col2:
        if run_strat:
            with st.spinner("Running simulations..."):
                equity_curves = simulate_strategy_returns(
                    n_trades, win_rate, avg_win, avg_loss, n_simulations
                )
            
            final_returns = equity_curves[:, -1]
            
            st.subheader("Simulation Results")
            
            # Key metrics
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Expected Return", f"{final_returns.mean():.1f}%")
            with metric_cols[1]:
                st.metric("Median Return", f"{np.median(final_returns):.1f}%")
            with metric_cols[2]:
                st.metric("Std Dev", f"{final_returns.std():.1f}%")
            with metric_cols[3]:
                prob_profit = (final_returns > 0).mean() * 100
                st.metric("P(Profitable)", f"{prob_profit:.1f}%")
            with metric_cols[4]:
                sharpe = final_returns.mean() / final_returns.std() if final_returns.std() > 0 else 0
                st.metric("Sharpe-like Ratio", f"{sharpe:.2f}")
            
            # Expectancy calculation
            expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
            st.info(f"**Trade Expectancy:** {expectancy:.2f}% per trade | "
                   f"**Expected after {n_trades} trades:** {expectancy * n_trades:.1f}%")
            
            # Risk metrics
            var, cvar = calculate_var_cvar(final_returns / 100, 0.95)
            risk_cols = st.columns(2)
            with risk_cols[0]:
                st.metric("VaR (95%)", f"{var*100:.1f}%")
            with risk_cols[1]:
                st.metric("CVaR (95%)", f"{cvar*100:.1f}%")
            
            # Plots
            st.plotly_chart(plot_equity_curves(equity_curves), use_container_width=True)
            
            # Drawdown analysis
            max_dd_per_sim = np.zeros(n_simulations)
            for i in range(n_simulations):
                cummax = np.maximum.accumulate(equity_curves[i])
                drawdown = equity_curves[i] - cummax
                max_dd_per_sim[i] = drawdown.min()
            
            st.subheader("Maximum Drawdown Distribution")
            dd_cols = st.columns(3)
            with dd_cols[0]:
                st.metric("Avg Max DD", f"{max_dd_per_sim.mean():.1f}%")
            with dd_cols[1]:
                st.metric("Worst Max DD", f"{max_dd_per_sim.min():.1f}%")
            with dd_cols[2]:
                st.metric("P(DD > 20%)", f"{(max_dd_per_sim < -20).mean()*100:.1f}%")


def page_var_analysis():
    """Value at Risk Analysis Page."""
    st.header("Portfolio Risk Analysis (VaR/CVaR)")
    st.markdown("Analyze portfolio risk using Monte Carlo Value at Risk.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Portfolio Parameters")
        portfolio_value = st.number_input("Portfolio Value (₺)", value=1000000.0, min_value=1000.0, step=10000.0)
        daily_mean = st.slider("Daily Expected Return (%)", min_value=-1.0, max_value=1.0, value=0.05, step=0.01)
        daily_vol = st.slider("Daily Volatility (%)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        time_horizon = st.slider("Time Horizon (Days)", min_value=1, max_value=30, value=10)
        confidence = st.slider("Confidence Level (%)", min_value=90, max_value=99, value=95)
        n_simulations = st.slider("Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
        
        run_var = st.button("Calculate VaR", type="primary", use_container_width=True)
    
    with col2:
        if run_var:
            with st.spinner("Running risk analysis..."):
                # Simulate portfolio returns
                daily_returns = np.random.normal(
                    daily_mean / 100,
                    daily_vol / 100,
                    (n_simulations, time_horizon)
                )
                
                # Compound for multi-day
                cumulative_returns = np.prod(1 + daily_returns, axis=1) - 1
                
                # Calculate VaR and CVaR
                var, cvar = calculate_var_cvar(cumulative_returns, confidence / 100)
                
                var_amount = var * portfolio_value
                cvar_amount = cvar * portfolio_value
            
            st.subheader(f"Risk Metrics ({time_horizon}-Day Horizon)")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(f"VaR ({confidence}%)", f"₺{abs(var_amount):,.0f}")
            with metric_cols[1]:
                st.metric(f"CVaR ({confidence}%)", f"₺{abs(cvar_amount):,.0f}")
            with metric_cols[2]:
                st.metric("VaR %", f"{var*100:.2f}%")
            with metric_cols[3]:
                st.metric("CVaR %", f"{cvar*100:.2f}%")
            
            # Interpretation
            st.warning(f"""
            **Interpretation:**
            - There is a **{100-confidence}%** chance of losing more than **₺{abs(var_amount):,.0f}** over {time_horizon} days.
            - If losses exceed VaR, the expected loss is **₺{abs(cvar_amount):,.0f}** (CVaR).
            """)
            
            # Distribution plot
            st.plotly_chart(plot_var_distribution(cumulative_returns, var, cvar), use_container_width=True)
            
            # Scenario analysis table
            st.subheader("Scenario Analysis")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            scenario_data = []
            for p in percentiles:
                ret = np.percentile(cumulative_returns, p)
                scenario_data.append({
                    'Percentile': f"{p}%",
                    'Return (%)': f"{ret*100:.2f}%",
                    'P&L (₺)': f"₺{ret * portfolio_value:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">VIOP Stochastic Simulator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Simulation",
        ["GBM Price Paths", "Option Pricing", "Strategy Backtest", "VaR Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    This dashboard provides Monte Carlo simulation tools for:
    - Asset price modeling (GBM)
    - Option pricing
    - Strategy backtesting
    - Risk analysis (VaR/CVaR)
    
    All simulations use stochastic methods for robust uncertainty quantification.
    """)
    
    # Route to selected page
    if "GBM" in page:
        page_gbm_simulation()
    elif "Option" in page:
        page_option_pricing()
    elif "Strategy" in page:
        page_strategy_simulation()
    elif "VaR" in page:
        page_var_analysis()


if __name__ == "__main__":
    main()
