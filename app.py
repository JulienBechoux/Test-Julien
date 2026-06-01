"""
Freight Cost Simulator with Monte Carlo - SIMPLIFIED & STABLE
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Freight Cost Simulator", page_icon="📦", layout="wide")

st.markdown("<h1 style='color: #1f77b4;'>📦 Freight Cost Simulator</h1>", unsafe_allow_html=True)
st.markdown("*Analyze Year-To-Date freight costs with Monte Carlo forecasting*")
st.divider()

# ==================== SETUP ====================

@st.cache_data
def generate_ytd_data(current_day):
    """Generate YTD freight costs"""
    np.random.seed(42)
    
    start_date = datetime(2026, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(current_day)]
    
    # Simple cost model
    days = np.arange(current_day)
    daily_costs = 5000 + (days * 15) + 2000 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 800, current_day)
    daily_costs = np.maximum(daily_costs, 1000)
    
    df = pd.DataFrame({
        'Date': dates,
        'Daily_Cost': daily_costs,
        'Cumulative_Cost': np.cumsum(daily_costs)
    })
    return df

# ==================== SIDEBAR ====================

st.sidebar.header("⚙️ Settings")

current_day = st.sidebar.slider("Day of Year", 1, 365, value=datetime.now().timetuple().tm_yday)

num_sims = st.sidebar.slider("Simulations", 100, 2000, 1000, step=100)
inflation = st.sidebar.slider("Inflation (%)", -10.0, 20.0, 3.5, step=0.5)
volatility = st.sidebar.slider("Volatility", 0.5, 3.0, 1.0, step=0.1)

# ==================== MAIN ====================

# Load data
df = generate_ytd_data(current_day)
total_ytd = df['Daily_Cost'].sum()
avg_daily = df['Daily_Cost'].mean()
std_daily = df['Daily_Cost'].std()
days_remaining = 365 - len(df)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total YTD", f"${total_ytd:,.0f}")
col2.metric("Avg Daily", f"${avg_daily:,.0f}")
col3.metric("Days Done", f"{len(df)}/365")
col4.metric("Days Left", days_remaining)

# Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Daily_Cost'], mode='lines', 
                              line=dict(color='#1f77b4', width=2), 
                              fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'))
    fig1.update_layout(title="Daily Costs", xaxis_title="Date", yaxis_title="Cost ($)", 
                       height=350, template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Cost'], mode='lines',
                              line=dict(color='#ff7f0e', width=2),
                              fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.2)'))
    fig2.update_layout(title="Cumulative Costs", xaxis_title="Date", yaxis_title="Cost ($)",
                       height=350, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ==================== MONTE CARLO ====================

st.header("🎲 Monte Carlo Simulation")

if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    
    with st.spinner("Running simulation..."):
        # Parameters
        drift = (inflation / 100) / 365
        sigma = (std_daily / avg_daily) * volatility
        
        # Vectorized MC
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, days_remaining))
        
        # GBM: S(t+1) = S(t) * exp[(mu - sigma^2/2) + sigma*Z]
        exponents = (drift - 0.5 * sigma**2) + sigma * Z
        price_mult = np.exp(exponents)
        
        # Daily costs for each path
        daily_results = np.cumprod(price_mult, axis=1) * avg_daily
        daily_results = np.clip(daily_results, avg_daily * 0.5, avg_daily * 3)
        
        # Total costs for each path
        total_future = np.sum(daily_results, axis=1)
        total_annual = total_ytd + total_future
    
    st.success("✅ Done!")
    
    # Results
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Total", f"${np.mean(total_annual):,.0f}")
    col2.metric("Best Case (5%)", f"${np.percentile(total_annual, 5):,.0f}")
    col3.metric("Median (50%)", f"${np.percentile(total_annual, 50):,.0f}")
    col4.metric("Worst Case (95%)", f"${np.percentile(total_annual, 95):,.0f}")
    
    # 95% CI
    lower_ci = np.percentile(total_annual, 2.5)
    upper_ci = np.percentile(total_annual, 97.5)
    st.info(f"**95% Confidence Interval:** ${lower_ci:,.0f} — ${upper_ci:,.0f}")
    
    # Distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=total_annual, nbinsx=50, marker_color='#2ca02c', opacity=0.7))
    
    percentiles = [5, 25, 50, 75, 95]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#ff7f0e', '#d62728']
    
    for pct, color in zip(percentiles, colors):
        val = np.percentile(total_annual, pct)
        fig_dist.add_vline(x=val, line_dash="dash", line_color=color, 
                          annotation_text=f"{pct}%ile", annotation_position="top")
    
    fig_dist.update_layout(title="Distribution of Forecasted Annual Costs",
                          xaxis_title="Annual Cost ($)", yaxis_title="Frequency",
                          height=450, template='plotly_white')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Paths
    st.subheader("📉 Simulation Paths")
    
    fig_paths = go.Figure()
    
    # Sample 50 paths
    for i in np.random.choice(num_sims, min(50, num_sims), replace=False):
        future_dates = [df['Date'].iloc[-1] + timedelta(days=j) for j in range(1, days_remaining + 1)]
        cum_path = total_ytd + np.cumsum(daily_results[i])
        fig_paths.add_trace(go.Scatter(x=future_dates, y=cum_path, mode='lines',
                                       line=dict(width=0.5, color='rgba(31, 119, 180, 0.1)'),
                                       hoverinfo='skip', showlegend=False))
    
    # Median
    median_path = total_ytd + np.cumsum(np.median(daily_results, axis=0))
    future_dates = [df['Date'].iloc[-1] + timedelta(days=j) for j in range(1, days_remaining + 1)]
    fig_paths.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines',
                                   name='Median', line=dict(color='red', width=3)))
    
    fig_paths.update_layout(title="Forecasted Cost Paths",
                           xaxis_title="Date", yaxis_title="Cumulative Cost ($)",
                           height=450, template='plotly_white')
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # Percentile table
    st.subheader("📋 Percentile Breakdown")
    
    pct_data = {
        'Percentile': ['5th', '25th', '50th', '75th', '95th'],
        'Additional Cost': [f"${np.percentile(total_future, p):,.0f}" for p in [5, 25, 50, 75, 95]],
        'Total Annual': [f"${np.percentile(total_annual, p):,.0f}" for p in [5, 25, 50, 75, 95]]
    }
    st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

st.divider()
st.caption("🔬 Algorithm: Geometric Brownian Motion (GBM) | Formula: S(t+1) = S(t)·exp[(μ-σ²/2)+σ·Z]")
