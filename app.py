"""
Freight Cost Simulator with Monte Carlo Forecasting
A Streamlit application for analyzing Year-To-Date freight costs 
and performing Monte Carlo simulations for cost forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Freight Cost Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

@st.cache_data
def generate_ytd_freight_costs(days_in_year=365, current_day=None):
    """
    Generate realistic YTD freight costs with seasonal and trend components.
    
    Parameters:
    - days_in_year: Total days in year (365 or 366)
    - current_day: Current day of year (auto-calculated if None)
    
    Returns:
    - DataFrame with daily freight costs
    """
    if current_day is None:
        current_day = datetime.now().timetuple().tm_yday
    
    # Initialize date range
    start_date = datetime(2026, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(current_day)]
    
    # Generate base cost with trend
    days = np.arange(current_day)
    base_trend = 5000 + (days * 15)  # Linear trend increasing over time
    
    # Add seasonal pattern (higher costs mid-year)
    seasonal = 2000 * np.sin(2 * np.pi * days / 365)
    
    # Add random noise (realistic daily variations)
    noise = np.random.normal(0, 800, current_day)
    
    # Combine components
    daily_costs = base_trend + seasonal + noise
    daily_costs = np.maximum(daily_costs, 1000)  # Ensure minimum cost
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Daily_Cost': daily_costs,
        'Cumulative_Cost': np.cumsum(daily_costs)
    })
    
    return df

def calculate_statistics(df):
    """Calculate key statistics from cost data."""
    return {
        'total_ytd': df['Daily_Cost'].sum(),
        'avg_daily': df['Daily_Cost'].mean(),
        'std_daily': df['Daily_Cost'].std(),
        'min_daily': df['Daily_Cost'].min(),
        'max_daily': df['Daily_Cost'].max(),
        'days_elapsed': len(df)
    }

def monte_carlo_simulation(stats, days_remaining, num_simulations, 
                           cost_inflation, volatility_factor):
    """
    Perform Monte Carlo simulation for future freight costs.
    
    Algorithm: Geometric Brownian Motion (GBM)
    - Used for modeling financial time series
    - Incorporates drift (mean trend) and diffusion (volatility)
    
    Mathematical Formula:
    S(t+1) = S(t) * exp[(μ - σ²/2)*Δt + σ*√Δt*Z]
    
    Where:
    - S = Current cost
    - μ = Drift (mean trend from inflation)
    - σ = Volatility (standard deviation adjusted by multiplier)
    - Δt = Time increment (1/365 for daily)
    - Z = Standard normal random variable N(0,1)
    
    Parameters:
    - stats: Dictionary with statistical measures
    - days_remaining: Days until end of year
    - num_simulations: Number of simulation paths
    - cost_inflation: Annual inflation rate (%)
    - volatility_factor: Volatility multiplier for uncertainty
    
    Returns:
    - Tuple of (simulated paths, cumulative future costs)
    """
    
    # Extract statistics
    avg_cost = stats['avg_daily']
    std_cost = stats['std_daily']
    
    # Calculate daily drift from annual inflation
    drift = (cost_inflation / 100) / 365
    
    # Adjusted volatility (normalized by mean, scaled by factor)
    volatility = (std_cost / avg_cost) * volatility_factor
    
    # Initialize arrays to store simulations
    simulations = np.zeros((num_simulations, days_remaining))
    cumulative_costs = np.zeros(num_simulations)
    
    # Set random seed for reproducibility if enabled
    if st.session_state.get('random_seed', None) is not None:
        np.random.seed(42)
    
    # Run Monte Carlo simulation
    for i in range(num_simulations):
        current_cost = avg_cost
        
        # Simulate each remaining day
        for j in range(days_remaining):
            # Generate random shock (Wiener process increment)
            dW = np.random.standard_normal()
            
            # Apply GBM formula: S(t+1) = S(t) * exp[drift + volatility * Z]
            exponent = (drift - 0.5 * volatility**2) + volatility * dW
            current_cost = current_cost * np.exp(exponent)
            
            # Bound costs to realistic range (±50% of average)
            current_cost = np.clip(current_cost, avg_cost * 0.5, avg_cost * 3)
            
            # Store daily cost and accumulate
            simulations[i, j] = current_cost
            cumulative_costs[i] += current_cost
    
    return simulations, cumulative_costs

def calculate_forecast_metrics(total_ytd, cumulative_future_costs, days_elapsed):
    """
    Calculate comprehensive forecast metrics and confidence intervals.
    
    Returns:
    - Dictionary with percentiles, totals, and confidence intervals
    """
    
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(cumulative_future_costs, percentiles)
    
    # Add YTD to future costs to get total annual projection
    forecasted_totals = total_ytd + cumulative_future_costs
    
    metrics = {
        'percentiles': percentiles,
        'percentile_values': percentile_values,
        'forecasted_totals': forecasted_totals,
        'expected_total': np.mean(forecasted_totals),
        'min_forecast': np.min(forecasted_totals),
        'max_forecast': np.max(forecasted_totals),
        'std_forecast': np.std(forecasted_totals),
        'lower_ci': np.percentile(forecasted_totals, 2.5),
        'upper_ci': np.percentile(forecasted_totals, 97.5),
    }
    
    return metrics

# ==================== MAIN APPLICATION ====================

def main():
    # Title and description
    st.markdown("<div class='header-title'>📦 Freight Cost Simulator</div>", 
                unsafe_allow_html=True)
    st.markdown("*Analyze Year-To-Date freight costs and forecast with Monte Carlo simulations*")
    st.divider()
    
    # ==================== SIDEBAR CONFIGURATION ====================
    st.sidebar.header("⚙️ Configuration")
    
    # Data generation settings
    with st.sidebar.expander("📊 Data Settings", expanded=True):
        current_day = st.slider(
            "Current day of year",
            min_value=1,
            max_value=365,
            value=datetime.now().timetuple().tm_yday,
            help="Simulate as if we're at this day of the year"
        )
        
        if st.button("🔄 Regenerate Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Simulation parameters
    with st.sidebar.expander("🎲 Simulation Parameters", expanded=True):
        num_simulations = st.slider(
            "Number of simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Higher values provide more accurate results but take longer"
        )
        
        cost_inflation = st.slider(
            "Expected annual inflation (%)",
            min_value=-10.0,
            max_value=20.0,
            value=3.5,
            step=0.5,
            help="Expected inflation rate for freight costs"
        )
        
        volatility_factor = st.slider(
            "Volatility multiplier",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Adjust uncertainty: 1.0 = historical volatility, >1.0 = more uncertainty"
        )
        
        random_seed = st.checkbox("Use fixed random seed for reproducibility")
        if random_seed:
            st.session_state['random_seed'] = 42
        else:
            st.session_state['random_seed'] = None
    
    # ==================== GENERATE AND DISPLAY YTD DATA ====================
    st.header("📈 Year-To-Date Freight Costs")
    
    # Generate YTD data
    df_ytd = generate_ytd_freight_costs(current_day=current_day)
    stats_data = calculate_statistics(df_ytd)
    days_remaining = 365 - stats_data['days_elapsed']
    
    # Display key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total YTD Cost",
            value=f"${stats_data['total_ytd']:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Average Daily Cost",
            value=f"${stats_data['avg_daily']:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Std Dev (Daily)",
            value=f"${stats_data['std_daily']:,.0f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Days Elapsed",
            value=f"{stats_data['days_elapsed']}",
            delta=f"{days_remaining} remaining"
        )
    
    with col5:
        st.metric(
            label="Days Elapsed %",
            value=f"{stats_data['days_elapsed']/365*100:.1f}%",
            delta=None
        )
    
    # Plot daily costs
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df_ytd['Date'],
        y=df_ytd['Daily_Cost'],
        mode='lines',
        name='Daily Cost',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig_daily.update_layout(
        title="Daily Freight Costs",
        xaxis_title="Date",
        yaxis_title="Daily Cost ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Plot cumulative costs
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=df_ytd['Date'],
        y=df_ytd['Cumulative_Cost'],
        mode='lines',
        name='Cumulative Cost',
        line=dict(color='#ff7f0e', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    fig_cumulative.update_layout(
        title="Cumulative Freight Costs (YTD)",
        xaxis_title="Date",
        yaxis_title="Cumulative Cost ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    st.divider()
    
    # ==================== MONTE CARLO SIMULATION ====================
    st.header("🎲 Monte Carlo Simulation & Forecast")
    
    # Run simulation button
    if st.button("▶️ Run Monte Carlo Simulation", type="primary", use_container_width=True):
        
        with st.spinner("Running simulations... Please wait"):
            # Execute Monte Carlo simulation
            simulations, cumulative_future = monte_carlo_simulation(
                stats=stats_data,
                days_remaining=days_remaining,
                num_simulations=num_simulations,
                cost_inflation=cost_inflation,
                volatility_factor=volatility_factor
            )
            
            # Calculate forecast metrics
            metrics = calculate_forecast_metrics(
                stats_data['total_ytd'],
                cumulative_future,
                stats_data['days_elapsed']
            )
        
        st.success("✅ Simulation completed!")
        
        # ==================== FORECAST RESULTS ====================
        st.subheader("📊 Forecast Results")
        
        # Key forecast metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Expected Annual Total",
                value=f"${metrics['expected_total']:,.0f}",
                delta=f"${metrics['expected_total'] - stats_data['total_ytd']:,.0f} remaining"
            )
        
        with col2:
            st.metric(
                label="Best Case (5th percentile)",
                value=f"${metrics['min_forecast']:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Most Likely (50th percentile)",
                value=f"${np.percentile(metrics['forecasted_totals'], 50):,.0f}"
            )
        
        with col4:
            st.metric(
                label="Worst Case (95th percentile)",
                value=f"${metrics['max_forecast']:,.0f}"
            )
        
        # Confidence interval
        st.info(f"""
        **95% Confidence Interval**
        
        Annual freight costs are expected to fall between **${metrics['lower_ci']:,.0f}** 
        and **${metrics['upper_ci']:,.0f}** with 95% confidence.
        """)
        
        # Distribution of forecasted totals
        fig_distribution = go.Figure()
        
        fig_distribution.add_trace(go.Histogram(
            x=metrics['forecasted_totals'],
            nbinsx=50,
            name='Forecasted Total Cost',
            marker_color='#2ca02c',
            opacity=0.7
        ))
        
        # Add percentile reference lines
        percentile_colors = ['#d62728', '#ff7f0e', '#1f77b4', '#ff7f0e', '#d62728']
        percentile_names = ['5th %ile', '25th %ile', 'Median', '75th %ile', '95th %ile']
        
        for pct, val, color, name in zip(
            metrics['percentiles'],
            metrics['percentile_values'],
            percentile_colors,
            percentile_names
        ):
            fig_distribution.add_vline(
                x=val + stats_data['total_ytd'],
                line_dash="dash",
                line_color=color,
                annotation_text=f"{name}<br>${val + stats_data['total_ytd']:,.0f}",
                annotation_position="top"
            )
        
        fig_distribution.update_layout(
            title="Distribution of Forecasted Annual Costs",
            xaxis_title="Annual Cost ($)",
            yaxis_title="Frequency",
            hovermode='x',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_distribution, use_container_width=True)
        
        # Simulation paths visualization
        st.subheader("📉 Simulation Paths (Sample)")
        
        fig_paths = go.Figure()
        
        # Add sample of all simulation paths
        for i in range(min(500, num_simulations)):
            future_dates = [df_ytd['Date'].iloc[-1] + timedelta(days=j) 
                          for j in range(1, days_remaining + 1)]
            cumulative_path = stats_data['total_ytd'] + np.cumsum(simulations[i])
            
            fig_paths.add_trace(go.Scatter(
                x=future_dates,
                y=cumulative_path,
                mode='lines',
                line=dict(width=0.5, color='rgba(31, 119, 180, 0.1)'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add median path
        median_path = stats_data['total_ytd'] + np.cumsum(np.median(simulations, axis=0))
        future_dates = [df_ytd['Date'].iloc[-1] + timedelta(days=j) 
                       for j in range(1, days_remaining + 1)]
        
        fig_paths.add_trace(go.Scatter(
            x=future_dates,
            y=median_path,
            mode='lines',
            name='Median Path',
            line=dict(color='red', width=3)
        ))
        
        # Add confidence interval shading
        lower_ci_path = stats_data['total_ytd'] + np.cumsum(np.percentile(simulations, 2.5, axis=0))
        upper_ci_path = stats_data['total_ytd'] + np.cumsum(np.percentile(simulations, 97.5, axis=0))
        
        fig_paths.add_trace(go.Scatter(
            x=future_dates,
            y=upper_ci_path,
            mode='lines',
            name='95% CI Upper',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True
        ))
        
        fig_paths.add_trace(go.Scatter(
            x=future_dates,
            y=lower_ci_path,
            mode='lines',
            name='95% CI Lower',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(0,100,200,0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig_paths.update_layout(
            title="Forecasted Cost Paths (Future Period)",
            xaxis_title="Date",
            yaxis_title="Cumulative Cost ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_paths, use_container_width=True)
        
        # Percentile breakdown table
        st.subheader("📋 Percentile Breakdown")
        
        percentile_table = pd.DataFrame({
            'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
            'Additional Cost': [f"${v:,.0f}" for v in metrics['percentile_values']],
            'Total Annual Cost': [f"${v + stats_data['total_ytd']:,.0f}" 
                                 for v in metrics['percentile_values']]
        })
        
        st.dataframe(percentile_table, use_container_width=True, hide_index=True)
        
        # Sensitivity Analysis
        st.subheader("🎯 Sensitivity Analysis")
        
        sensitivity_col1, sensitivity_col2 = st.columns(2)
        
        with sensitivity_col1:
            st.write("**Impact of Inflation on Expected Cost**")
            inflation_range = np.arange(-5, 21, 2.5)
            inflation_impacts = []
            
            for inf in inflation_range:
                _, cumulative = monte_carlo_simulation(
                    stats=stats_data,
                    days_remaining=days_remaining,
                    num_simulations=500,
                    cost_inflation=inf,
                    volatility_factor=volatility_factor
                )
                inflation_impacts.append(np.mean(cumulative) + stats_data['total_ytd'])
            
            fig_inflation = go.Figure()
            fig_inflation.add_trace(go.Scatter(
                x=inflation_range,
                y=inflation_impacts,
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            fig_inflation.update_layout(
                title="Impact of Inflation on Annual Cost",
                xaxis_title="Inflation Rate (%)",
                yaxis_title="Expected Annual Cost ($)",
                hovermode='x',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_inflation, use_container_width=True)
        
        with sensitivity_col2:
            st.write("**Impact of Volatility on Expected Cost**")
            volatility_range = np.arange(0.5, 3.1, 0.3)
            volatility_impacts = []
            
            for vol in volatility_range:
                _, cumulative = monte_carlo_simulation(
                    stats=stats_data,
                    days_remaining=days_remaining,
                    num_simulations=500,
                    cost_inflation=cost_inflation,
                    volatility_factor=vol
                )
                volatility_impacts.append(np.mean(cumulative) + stats_data['total_ytd'])
            
            fig_volatility = go.Figure()
            fig_volatility.add_trace(go.Scatter(
                x=volatility_range,
                y=volatility_impacts,
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))
            
            fig_volatility.update_layout(
                title="Impact of Volatility on Annual Cost",
                xaxis_title="Volatility Multiplier",
                yaxis_title="Expected Annual Cost ($)",
                hovermode='x',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_volatility, use_container_width=True)

if __name__ == "__main__":
    main()
