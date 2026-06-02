# 📦 Freight Cost Analysis & Monte Carlo Simulator

A comprehensive **Streamlit-based web application** for analyzing freight costs and performing Monte Carlo simulations to forecast cost scenarios. This tool helps logistics and supply chain professionals make data-driven decisions about freight expenses.

---

## 🎯 Features

### Historical Analysis
- **Year-over-Year Comparison**: Visualize freight costs for current and previous years
- **Statistical Metrics**: Mean, median, standard deviation, quartiles, min/max values
- **Monthly Trends**: Identify seasonal patterns and anomalies
- **Detailed Breakdown**: Shipment counts, distances, and weight analysis

### Monte Carlo Simulation
- **Multiple Distributions**: Normal, Log-Normal, Uniform, and Triangular distributions
- **Customizable Parameters**:
  - Number of simulations (100–10,000)
  - Forecast horizon (1–24 months)
  - Cost adjustments (80%–150% of baseline)
  - Volatility multipliers (0.5x–3.0x)
  - Month-to-month correlation (0–100%)
  
### Interactive Visualizations
- Confidence bands (50% and 90%)
- Percentile analysis and distribution plots
- Dynamic charts with hover details
- Monthly statistics breakdown

### Data Export
- Download simulation results as CSV
- Full statistical output for further analysis

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd freight-cost-simulator
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access in browser**
   - Open your browser and navigate to `http://localhost:8501`
   - Streamlit will automatically reload on code changes

---

## 📊 How to Use

### Step 1: Review Historical Data

The app automatically generates realistic freight cost data for the current and previous years:

| Section | What You See |
|---------|-------------|
| **Key Metrics** | Total/average costs with year-over-year changes |
| **Comparison Chart** | Line graph showing monthly trends |
| **Detailed Statistics** | Expandable section with full statistical breakdown |

**Interpret the data:**
- Look for seasonal patterns (peaks/troughs)
- Identify month-to-month volatility
- Compare trends between years

---

### Step 2: Configure Simulation Parameters

Use the **sidebar controls** to customize your simulation:

#### Data Generation
- **Random Seed**: Set this to 42 (or any number) for reproducible results. Change the seed to generate different data.

#### Simulation Parameters
- **Number of Simulations**: 
  - 100–500: Fast, less accurate
  - 1,000 (default): Good balance
  - 5,000–10,000: More accurate, slower
  
- **Forecast Horizon**: How many months ahead to predict (1–24 months)

- **Cost Distribution**:
  - **Normal**: Bell curve distribution (most common)
  - **Log-Normal**: Right-skewed (asymmetric, common for costs)
  - **Uniform**: All values equally likely
  - **Triangular**: Mode-based distribution (symmetric around mode)

#### Scenario Adjustments
- **Expected Cost Change**:
  - 0.8 = 20% decrease
  - 1.0 = No change (default)
  - 1.2 = 20% increase
  
- **Volatility Multiplier**:
  - 0.5 = Half the historical variability
  - 1.0 = Same as historical (default)
  - 2.0 = Double the historical variability
  
- **Month-to-Month Correlation**:
  - 0.0 = Completely independent months
  - 0.3 = Weak correlation (default, realistic)
  - 0.8 = Strong correlation (costs tend to follow previous month)

#### Percentiles
Select which percentile bands to display (5th, 25th, 50th, 75th, 95th).

---

### Step 3: Interpret Simulation Results

#### Summary Metrics
| Metric | Meaning |
|--------|---------|
| **Expected Total** | Average sum over forecast period ± std dev |
| **Median (Month 12)** | Middle value for final month's cost |
| **Best Case (5th %)** | Only 5% of outcomes are lower |
| **Worst Case (95th %)** | Only 5% of outcomes are higher |

#### Visualization Tabs

1. **Main Forecast Chart**
   - Shaded bands show confidence intervals
   - Median line shows expected path
   - Hover for exact values

2. **Distribution Histogram**
   - Shows spread of possible outcomes
   - Mean (red) vs. Median (green) lines
   - Helps understand risk profile

3. **Simulation Statistics by Month**
   - Expandable table with detailed stats per month
   - Mean, median, std dev, min/max, P5/P95

4. **Percentile Analysis**
   - Full matrix of values by month and percentile
   - Use for detailed risk assessment

---

## 🔧 Technical Architecture

### Core Classes

#### `CostDistribution` (Enum)
Defines available probability distributions for simulations.

```python
NORMAL = "Normal (Bell Curve)"
LOGNORMAL = "Log-Normal (Right-Skewed)"
UNIFORM = "Uniform (Equal Probability)"
TRIANGULAR = "Triangular (Mode-based)"
```

#### `FreightCostMetrics` (Dataclass)
Container for statistical measures:
- `mean`, `std_dev`, `min_val`, `max_val`
- `median`, `q25`, `q75`, `total`

#### `FreightCostGenerator`
Generates realistic historical data:
- Seasonal patterns (higher in summer, lower in winter)
- Random variation around base cost
- Includes shipments, distance, and weight metadata

**Key method**: `generate_historical_costs(year, num_months)`
- Returns DataFrame with monthly costs
- Seed-based for reproducibility

#### `MonteCarloSimulator`
Performs the core simulation:
- Takes historical mean/std dev as input
- Generates correlated random shocks
- Supports multiple distributions
- Returns (num_simulations, num_months) array

**Key method**: `run_simulation(...)`
- Parameters: simulations, months, distribution, adjustments
- Returns: 2D array of simulated costs

---

## 📈 Understanding Monte Carlo Simulation

### What is it?
Monte Carlo simulation runs thousands of "what-if" scenarios to show the range of possible outcomes.

### Why use it for freight costs?
- **Captures uncertainty**: Real costs aren't predictable
- **Risk assessment**: Shows best/worst-case scenarios
- **Planning**: Helps set budgets with confidence levels
- **Scenario testing**: Quickly explore "what-if" questions

### How it works (simplified)
1. Start with historical cost statistics (mean, variability)
2. Generate 1,000+ random cost paths using a probability distribution
3. Calculate percentiles (5th, 25th, 50th, 75th, 95th)
4. Visualize the range of outcomes

---

## 🎛️ Example Scenarios

### Scenario 1: Stable Growth
- **Expected Cost Change**: 1.05 (5% increase)
- **Volatility Multiplier**: 1.0 (same as historical)
- **Correlation**: 0.3 (normal)
- **Use case**: Modest inflation scenario

### Scenario 2: High Uncertainty
- **Expected Cost Change**: 1.1 (10% increase)
- **Volatility Multiplier**: 2.0 (double uncertainty)
- **Correlation**: 0.5 (stronger month dependencies)
- **Use case**: Volatile fuel prices, supply chain disruption

### Scenario 3: Cost Reduction
- **Expected Cost Change**: 0.9 (10% decrease)
- **Volatility Multiplier**: 0.7 (lower variability)
- **Correlation**: 0.2 (more independent)
- **Use case**: Efficiency improvements, vendor negotiations

---

## 📊 Interpreting Results

### Best Practice 1: Set Budget Targets
Use the **75th percentile** of the forecast as your budget:
- Covers 75% of likely outcomes
- Provides reasonable safety margin
- Avoids excessive contingency

### Best Practice 2: Risk Zones
- **Green zone (P25–P50)**: Likely outcomes
- **Yellow zone (P50–P75)**: Possible outcomes
- **Red zone (P75–P95)**: Adverse scenarios

### Best Practice 3: Compare Scenarios
Run simulations with different parameters to:
- Evaluate mitigation strategies
- Test sensitivity to key assumptions
- Build a decision matrix

---

## 🔒 Security & Data Privacy

- **No external data storage**: All data generated locally
- **No API calls**: Completely offline operation
- **No credentials needed**: Standalone application
- **Reproducibility**: Set random seed for consistent results

---

## 🛠️ Customization

### Modify Data Generation
Edit the `FreightCostGenerator.generate_historical_costs()` method to:
- Change base cost range ($50K–$120K)
- Adjust seasonal patterns
- Add custom cost drivers

### Add New Distributions
Extend the `CostDistribution` enum and implement in `_generate_shock()`:
```python
elif distribution == CostDistribution.CUSTOM:
    return np.random.custom_distribution(...)
```

### Change Visualization Colors
Update Plotly `line` and `fillcolor` parameters in chart functions.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| **streamlit** | Web framework for interactive UI |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing and random generation |
| **plotly** | Interactive visualizations |
| **scipy** | Advanced statistical functions |

---

## ⚠️ Limitations & Considerations

1. **Data is simulated**: Not real historical costs. Replace with actual data for production use.
2. **Assumptions matter**: Results only as good as input parameters.
3. **Tail risk**: Extreme scenarios (>95th percentile) are less reliable.
4. **Correlation model**: Simple first-order correlation; doesn't capture complex dependencies.
5. **Performance**: Very large simulations (>50,000) may be slow.

---

## 🚀 Production Deployment

### Option 1: Streamlit Cloud
```bash
# Create GitHub repo, push code, connect to Streamlit Cloud
# Automatic deployments on every push
```

### Option 2: Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["streamlit", "run", "app.py"]
```

### Option 3: Self-Hosted
Deploy on AWS, Google Cloud, or your own server:
```bash
streamlit run app.py --server.port=8501 --logger.level=info
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"ModuleNotFoundError"** | Run `pip install -r requirements.txt` |
| **Simulation very slow** | Reduce num_simulations or forecast_months |
| **Charts not showing** | Clear browser cache, restart Streamlit |
| **Random results vary** | Set same random seed in sidebar |
| **Out of memory** | Reduce num_simulations significantly |

---

## 📝 License

MIT License - Feel free to use and modify for your needs.

---

## 🤝 Contributing

To improve this application:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Support

For questions or issues:
- Check the troubleshooting section above
- Review the code comments and docstrings
- Run with `--logger.level=debug` for more details

---

## 🔮 Future Enhancements

Potential features for future versions:
- [ ] Import real historical data from CSV/database
- [ ] Multiple cost categories (fuel, labor, equipment)
- [ ] Sensitivity analysis (tornado charts)
- [ ] Rolling forecast updates
- [ ] Scenario comparison dashboard
- [ ] VaR (Value at Risk) calculations
- [ ] Correlation matrix visualization
- [ ] Export to PowerPoint/PDF reports

---

## 📚 References

### Monte Carlo Simulation
- Palisade Corporation Monte Carlo Tutorial
- "Risk Analysis in Project Management" - David T. Hulett
- Investopedia: Monte Carlo Simulation

### Freight Cost Analysis
- American Trucking Association (ATA) Cost Indices
- GEODIS Supply Chain Insights
- Gartner Supply Chain Council

---

**Built with ❤️ using Streamlit, Pandas, and NumPy**
