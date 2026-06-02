"""
Freight Cost Analysis and Monte Carlo Simulation Application

This application provides:
- Real-time freight cost analysis (current year vs. last year)
- Interactive Monte Carlo simulation for cost forecasting
- Parameter tuning for various cost scenarios
- Statistical analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CostDistribution(Enum):
    """Supported probability distributions for Monte Carlo simulation."""
    NORMAL = "Normal (Bell Curve)"
    LOGNORMAL = "Log-Normal (Right-Skewed)"
    UNIFORM = "Uniform (Equal Probability)"
    TRIANGULAR = "Triangular (Mode-based)"


@dataclass
class FreightCostMetrics:
    """Container for freight cost statistics."""
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    median: float
    q25: float
    q75: float
    total: float


class FreightCostGenerator:
    """Generate realistic freight cost data."""

    @staticmethod
    def generate_historical_costs(year: int, num_months: int = 12) -> pd.DataFrame:
        """
        Generate realistic historical freight costs for a given year.
        
        Args:
            year: The year to generate costs for
            num_months: Number of months of data
            
        Returns:
            DataFrame with monthly freight costs and metadata
            
        Raises:
            ValueError: If num_months is not between 1 and 12
        """
        if num_months < 1 or num_months > 12:
            raise ValueError("num_months must be between 1 and 12")

        np.random.seed(year)  # Consistent data per year for reproducibility
        
        # Generate base seasonal pattern
        months = pd.date_range(start=f"{year}-01-01", periods=num_months, freq="M")
        seasonal_factor = 1 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, num_months))
        
        # Base monthly cost: $50,000 to $120,000
        base_cost = 85000
        variation = np.random.normal(0, 8000, num_months)
        noise = np.random.normal(1, 0.05, num_months)
        
        costs = (base_cost + variation) * seasonal_factor * noise
        costs = np.maximum(costs, 30000)  # Ensure positive costs
        
        data = {
            'Date': months,
            'Month': months.strftime('%B'),
            'Cost': costs,
            'Year': year,
            'Shipments': np.random.randint(80, 200, num_months),
            'Distance_km': np.random.randint(500, 2000, num_months),
            'Weight_tons': np.random.uniform(20, 100, num_months)
        }
        
        logger.info(f"Generated {num_months} months of freight costs for year {year}")
        return pd.DataFrame(data)

    @staticmethod
    def get_metrics(costs: pd.Series) -> FreightCostMetrics:
        """
        Calculate statistical metrics from cost data.
        
        Args:
            costs: Series of cost values
            
        Returns:
            FreightCostMetrics object with calculated statistics
        """
        return FreightCostMetrics(
            mean=float(costs.mean()),
            std_dev=float(costs.std()),
            min_val=float(costs.min()),
            max_val=float(costs.max()),
            median=float(costs.median()),
            q25=float(costs.quantile(0.25)),
            q75=float(costs.quantile(0.75)),
            total=float(costs.sum())
        )


class MonteCarloSimulator:
    """Perform Monte Carlo simulations for freight cost forecasting."""

    def __init__(self, historical_mean: float, historical_std: float):
        """
        Initialize simulator with historical data statistics.
        
        Args:
            historical_mean: Mean of historical costs
            historical_std: Standard deviation of historical costs
        """
        self.historical_mean = historical_mean
        self.historical_std = historical_std
        logger.info(f"Initialized simulator with mean=${historical_mean:.2f}, std=${historical_std:.2f}")

    def run_simulation(
        self,
        num_simulations: int,
        num_months: int,
        distribution: CostDistribution,
        mean_adjustment: float = 1.0,
        volatility_multiplier: float = 1.0,
        correlation_strength: float = 0.3
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for freight costs.
        
        Args:
            num_simulations: Number of simulation paths to generate
            num_months: Number of months to forecast
            distribution: Probability distribution to use
            mean_adjustment: Multiplier for mean (e.g., 1.1 = 10% increase)
            volatility_multiplier: Multiplier for volatility (std dev)
            correlation_strength: Autocorrelation strength (0-1)
            
        Returns:
            Array of shape (num_simulations, num_months) with simulated costs
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_simulations < 1 or num_months < 1:
            raise ValueError("num_simulations and num_months must be >= 1")
        if not 0 <= correlation_strength <= 1:
            raise ValueError("correlation_strength must be between 0 and 1")

        adjusted_mean = self.historical_mean * mean_adjustment
        adjusted_std = self.historical_std * volatility_multiplier
        
        simulations = np.zeros((num_simulations, num_months))
        
        for sim in range(num_simulations):
            current_value = adjusted_mean
            
            for month in range(num_months):
                # Generate correlated random variable
                if month == 0:
                    shock = self._generate_shock(adjusted_mean, adjusted_std, distribution)
                else:
                    prev_shock = simulations[sim, month - 1]
                    random_component = self._generate_shock(adjusted_mean, adjusted_std, distribution)
                    shock = (correlation_strength * prev_shock + 
                            (1 - correlation_strength) * random_component)
                
                simulations[sim, month] = max(shock, 5000)  # Floor at $5,000
        
        logger.info(f"Completed {num_simulations} simulations for {num_months} months")
        return simulations

    def _generate_shock(
        self,
        mean: float,
        std: float,
        distribution: CostDistribution
    ) -> float:
        """
        Generate a single random shock value from specified distribution.
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
            distribution: Type of distribution to use
            
        Returns:
            Generated shock value
        """
        if distribution == CostDistribution.NORMAL:
            return np.random.normal(mean, std)
        elif distribution == CostDistribution.LOGNORMAL:
            # Convert parameters for lognormal
            mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
            sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
            return np.random.lognormal(mu, sigma)
        elif distribution == CostDistribution.UNIFORM:
            lower = mean - 2 * std
            upper = mean + 2 * std
            return np.random.uniform(lower, upper)
        elif distribution == CostDistribution.TRIANGULAR:
            lower = mean - 2 * std
            upper = mean + 2 * std
            return np.random.triangular(lower, mean, upper)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    @staticmethod
    def calculate_percentiles(
        simulations: np.ndarray,
        percentiles: list
    ) -> dict:
        """
        Calculate percentiles from simulation results.
        
        Args:
            simulations: Array of simulation results
            percentiles: List of percentile values (0-100)
            
        Returns:
            Dictionary mapping percentiles to values
        """
        result = {}
        for p in percentiles:
            result[p] = np.percentile(simulations, p, axis=0)
        return result


def create_historical_comparison_chart(df_current: pd.DataFrame, df_previous: pd.DataFrame) -> go.Figure:
    """
    Create comparison chart for current vs. previous year costs.
    
    Args:
        df_current: Current year data
        df_previous: Previous year data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_current['Month'],
        y=df_current['Cost'],
        name='Current Year',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_previous['Month'],
        y=df_previous['Cost'],
        name='Previous Year',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Monthly Freight Costs: Current Year vs. Previous Year',
        xaxis_title='Month',
        yaxis_title='Cost ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        yaxis=dict(tickformat='$,.0f')
    )
    
    return fig


def create_simulation_chart(simulations: np.ndarray, percentiles: dict) -> go.Figure:
    """
    Create visualization of Monte Carlo simulation results.
    
    Args:
        simulations: Simulation results array
        percentiles: Percentile values for bands
        
    Returns:
        Plotly figure object
    """
    months = np.arange(1, simulations.shape[1] + 1)
    median = np.percentile(simulations, 50, axis=0)
    p5 = percentiles[5]
    p95 = percentiles[95]
    p25 = percentiles[25]
    p75 = percentiles[75]
    
    fig = go.Figure()
    
    # 90% confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='90% Confidence Band'
    ))
    
    # 50% confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 200, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='50% Confidence Band'
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=months,
        y=median,
        name='Median Forecast',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Monte Carlo Simulation: Freight Cost Forecast',
        xaxis_title='Month',
        yaxis_title='Cost ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        yaxis=dict(tickformat='$,.0f'),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_distribution_chart(simulations: np.ndarray, month_index: int = -1) -> go.Figure:
    """
    Create histogram of simulated values for a specific month.
    
    Args:
        simulations: Simulation results array
        month_index: Month to visualize (default: last month)
        
    Returns:
        Plotly figure object
    """
    values = simulations[:, month_index]
    
    fig = px.histogram(
        x=values,
        nbins=50,
        title=f'Distribution of Simulated Costs (Month {month_index + 1})',
        labels={'x': 'Cost ($)', 'count': 'Frequency'},
        template='plotly_white'
    )
    
    fig.add_vline(
        x=np.mean(values),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=np.median(values),
        line_dash="dash",
        line_color="green",
        annotation_text="Median",
        annotation_position="top left"
    )
    
    fig.update_layout(
        height=450,
        xaxis=dict(tickformat='$,.0f'),
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Freight Cost Monte Carlo Simulator",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📦 Freight Cost Analysis & Monte Carlo Simulator")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        current_year = datetime.now().year
        previous_year = current_year - 1
        
        st.subheader("Data Generation")
        data_seed = st.number_input(
            "Random Seed (for reproducibility)",
            value=42,
            min_value=0,
            help="Use same seed to generate identical data"
        )
        np.random.seed(data_seed)
        
        st.subheader("Simulation Parameters")
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations = more accurate but slower"
        )
        
        forecast_months = st.slider(
            "Forecast Horizon (Months)",
            min_value=1,
            max_value=24,
            value=12,
            help="Number of months to forecast"
        )
        
        distribution = st.selectbox(
            "Cost Distribution",
            [d for d in CostDistribution],
            format_func=lambda x: x.value,
            help="Probability distribution for cost variations"
        )
        
        st.subheader("Scenario Adjustments")
        mean_adjustment = st.slider(
            "Expected Cost Change",
            min_value=0.8,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Multiplier for expected costs (1.0 = no change)"
        )
        
        volatility_multiplier = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Multiplier for cost variability (1.0 = historical)"
        )
        
        correlation_strength = st.slider(
            "Month-to-Month Correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="How much one month depends on previous month"
        )
        
        st.subheader("Percentiles to Display")
        percentiles = st.multiselect(
            "Select percentiles for analysis",
            [5, 10, 25, 50, 75, 90, 95],
            default=[5, 25, 50, 75, 95]
        )
    
    try:
        # Generate historical data
        generator = FreightCostGenerator()
        df_current = generator.generate_historical_costs(current_year)
        df_previous = generator.generate_historical_costs(previous_year)
        
        # Calculate metrics
        current_metrics = FreightCostGenerator.get_metrics(df_current['Cost'])
        previous_metrics = FreightCostGenerator.get_metrics(df_previous['Cost'])
        
        # Display historical analysis
        st.header("📊 Historical Cost Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Year Total",
                f"${current_metrics.total:,.0f}",
                f"{((current_metrics.total / previous_metrics.total - 1) * 100):+.1f}%"
            )
        with col2:
            st.metric(
                "Current Year Average",
                f"${current_metrics.mean:,.0f}",
                f"{((current_metrics.mean / previous_metrics.mean - 1) * 100):+.1f}%"
            )
        with col3:
            st.metric(
                "Previous Year Total",
                f"${previous_metrics.total:,.0f}"
            )
        with col4:
            st.metric(
                "Previous Year Average",
                f"${previous_metrics.mean:,.0f}"
            )
        
        st.plotly_chart(
            create_historical_comparison_chart(df_current, df_previous),
            use_container_width=True
        )
        
        # Display detailed metrics in expandable section
        with st.expander("📈 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Current Year ({current_year})")
                stats_current = {
                    'Mean': f"${current_metrics.mean:,.2f}",
                    'Median': f"${current_metrics.median:,.2f}",
                    'Std Dev': f"${current_metrics.std_dev:,.2f}",
                    'Min': f"${current_metrics.min_val:,.2f}",
                    'Max': f"${current_metrics.max_val:,.2f}",
                    'Q1 (25%)': f"${current_metrics.q25:,.2f}",
                    'Q3 (75%)': f"${current_metrics.q75:,.2f}",
                    'Total': f"${current_metrics.total:,.2f}"
                }
                for label, value in stats_current.items():
                    st.text(f"{label}: {value}")
            
            with col2:
                st.subheader(f"Previous Year ({previous_year})")
                stats_previous = {
                    'Mean': f"${previous_metrics.mean:,.2f}",
                    'Median': f"${previous_metrics.median:,.2f}",
                    'Std Dev': f"${previous_metrics.std_dev:,.2f}",
                    'Min': f"${previous_metrics.min_val:,.2f}",
                    'Max': f"${previous_metrics.max_val:,.2f}",
                    'Q1 (25%)': f"${previous_metrics.q25:,.2f}",
                    'Q3 (75%)': f"${previous_metrics.q75:,.2f}",
                    'Total': f"${previous_metrics.total:,.2f}"
                }
                for label, value in stats_previous.items():
                    st.text(f"{label}: {value}")
        
        # Monte Carlo Simulation
        st.markdown("---")
        st.header("🎲 Monte Carlo Simulation")
        
        st.info(
            f"Running {num_simulations:,} simulations for {forecast_months} months "
            f"using {distribution.value} distribution with {mean_adjustment:.0%} "
            f"cost adjustment and {volatility_multiplier:.1f}x volatility"
        )
        
        # Run simulation
        simulator = MonteCarloSimulator(
            current_metrics.mean,
            current_metrics.std_dev
        )
        
        simulations = simulator.run_simulation(
            num_simulations=num_simulations,
            num_months=forecast_months,
            distribution=distribution,
            mean_adjustment=mean_adjustment,
            volatility_multiplier=volatility_multiplier,
            correlation_strength=correlation_strength
        )
        
        # Calculate percentiles
        all_percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = simulator.calculate_percentiles(simulations, all_percentiles)
        
        # Display simulation results
        col1, col2, col3, col4 = st.columns(4)
        
        final_month_values = simulations[:, -1]
        with col1:
            st.metric(
                "Expected Total (Forecast)",
                f"${np.mean(simulations.sum(axis=1)):,.0f}",
                f"±${np.std(simulations.sum(axis=1)):,.0f}"
            )
        with col2:
            st.metric(
                "Median (Month 12)",
                f"${np.median(final_month_values):,.0f}"
            )
        with col3:
            st.metric(
                "Best Case (5th percentile)",
                f"${np.percentile(final_month_values, 5):,.0f}"
            )
        with col4:
            st.metric(
                "Worst Case (95th percentile)",
                f"${np.percentile(final_month_values, 95):,.0f}"
            )
        
        # Simulation visualization
        st.plotly_chart(
            create_simulation_chart(simulations, percentile_values),
            use_container_width=True
        )
        
        # Distribution visualization
        st.plotly_chart(
            create_distribution_chart(simulations, -1),
            use_container_width=True
        )
        
        # Detailed simulation statistics
        with st.expander("📋 Simulation Statistics by Month"):
            sim_stats = []
            for month in range(forecast_months):
                month_values = simulations[:, month]
                sim_stats.append({
                    'Month': month + 1,
                    'Mean': f"${np.mean(month_values):,.2f}",
                    'Median': f"${np.median(month_values):,.2f}",
                    'Std Dev': f"${np.std(month_values):,.2f}",
                    'Min': f"${np.min(month_values):,.2f}",
                    'Max': f"${np.max(month_values):,.2f}",
                    'P5': f"${np.percentile(month_values, 5):,.2f}",
                    'P95': f"${np.percentile(month_values, 95):,.2f}"
                })
            
            st.dataframe(pd.DataFrame(sim_stats), use_container_width=True)
        
        # Percentile analysis
        with st.expander("📊 Percentile Analysis"):
            st.write("Values by month and percentile:")
            percentile_data = []
            for month in range(forecast_months):
                row = {'Month': month + 1}
                for p in all_percentiles:
                    row[f'P{p}'] = f"${percentile_values[p][month]:,.2f}"
                percentile_data.append(row)
            
            st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)
        
        # Scenario comparison
        with st.expander("🔍 Scenario Comparison"):
            st.write("""
            **Current Scenario Parameters:**
            - Cost Adjustment: {:.0%}
            - Volatility Multiplier: {:.1f}x
            - Month-to-Month Correlation: {:.0%}
            - Distribution: {}
            - Simulations: {:,}
            """.format(
                mean_adjustment,
                volatility_multiplier,
                correlation_strength,
                distribution.value,
                num_simulations
            ))
            
            st.write("**Total Cost Forecast (12 months):**")
            total_costs = simulations.sum(axis=1)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"${np.mean(total_costs):,.0f}")
            with col2:
                st.metric("Median", f"${np.median(total_costs):,.0f}")
            with col3:
                st.metric("Std Dev", f"${np.std(total_costs):,.0f}")
        
        # Download data
        with st.expander("💾 Download Results"):
            # Prepare data for download
            export_data = []
            for month in range(forecast_months):
                for percentile in [5, 25, 50, 75, 95]:
                    export_data.append({
                        'Month': month + 1,
                        'Percentile': percentile,
                        'Value': percentile_values[percentile][month]
                    })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download Simulation Results as CSV",
                data=csv,
                file_name="monte_carlo_results.csv",
                mime="text/csv"
            )
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check the application logs for more details.")


if __name__ == "__main__":
    main()
