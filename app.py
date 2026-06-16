# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Freight Costs Dashboard", layout="wide")

# -------------------------
# Configuration / Constants
# -------------------------
CURRENT_YEAR = datetime.now().year
LAST_YEAR = CURRENT_YEAR - 1
SOURCES = ["Manual accruals", "SAP ERP", "SAP TM"]
CARRIERS = ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]
REGIONS = ["EMEA", "AMER", "APAC"]
MODES = ["Air", "Ocean", "Road", "Rail"]

# -------------------------
# Data generation utilities
# -------------------------
@st.cache_data
def generate_random_costs(seed: int = 42, months: int = 24):
    np.random.seed(seed)
    rows = []
    start_year = LAST_YEAR
    start_month = 1
    # generate month sequence for last 24 months ending current month
    dates = pd.date_range(
        start=f"{start_year}-01-01", periods=months, freq="MS"
    ).to_pydatetime().tolist()
    for src in SOURCES:
        for dt in dates:
            year = dt.year
            month = dt.month
            # base cost depends on source and mode randomness
            for mode in MODES:
                for carrier in np.random.choice(CARRIERS, size=2, replace=False):
                    region = np.random.choice(REGIONS)
                    # create a plausible cost distribution
                    base = {
                        "Manual accruals": 20000,
                        "SAP ERP": 35000,
                        "SAP TM": 30000,
                    }[src]
                    mode_factor = {"Air": 1.8, "Ocean": 1.0, "Road": 0.9, "Rail": 0.7}[mode]
                    seasonal = 1 + 0.15 * np.sin((month / 12) * 2 * np.pi)
                    noise = np.random.normal(loc=0.0, scale=0.12)
                    amount = max(0, base * mode_factor * seasonal * (1 + noise) * np.random.uniform(0.6, 1.4))
                    rows.append(
                        {
                            "date": pd.Timestamp(year=year, month=month, day=1),
                            "year": year,
                            "month": month,
                            "source": src,
                            "mode": mode,
                            "carrier": carrier,
                            "region": region,
                            "amount": round(amount, 2),
                        }
                    )
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Data loading (simulate + optional uploads)
# -------------------------
st.title("Freight Costs — Year over Year Insights")

st.markdown("**Data sources:** Manual accruals, SAP ERP, SAP TM. Data below is simulated by the app; you can upload your own files to replace it.")

# Generate simulated dataset
df_sim = generate_random_costs(seed=123, months=24)

# Allow user to upload files to replace simulated sources
st.sidebar.header("Data input")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 3 files (CSV or Excel). If provided, the app will try to read and map columns: date, source, mode, carrier, region, amount.",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
)

def try_read_file(f):
    try:
        if f.name.lower().endswith(".csv"):
            return pd.read_csv(f)
        else:
            return pd.read_excel(f)
    except Exception as e:
        st.sidebar.error(f"Failed to read {f.name}: {e}")
        return None

def normalize_uploaded(df):
    # Try to map common column names
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    mapping_candidates = {
        "date": ["date", "shipment_date", "invoice_date", "posting_date"],
        "source": ["source", "file", "system"],
        "mode": ["mode", "transport_mode", "shipment_mode"],
        "carrier": ["carrier", "vendor", "shipper"],
        "region": ["region", "area", "zone"],
        "amount": ["amount", "cost", "freight_cost", "value", "total"],
    }
    for target, candidates in mapping_candidates.items():
        for cand in candidates:
            if cand in lower_cols:
                col_map[lower_cols[cand]] = target
                break
    df = df.rename(columns=col_map)
    # Ensure required columns exist
    required = ["date", "source", "mode", "carrier", "region", "amount"]
    if not all(c in df.columns for c in required):
        return None
    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df

if uploaded_files:
    uploaded_dfs = []
    for f in uploaded_files[:3]:
        raw = try_read_file(f)
        if raw is None:
            continue
        norm = normalize_uploaded(raw)
        if norm is None:
            st.sidebar.warning(f"Could not map columns for {f.name}. Expected columns like date, amount, source, mode, carrier, region.")
            continue
        uploaded_dfs.append(norm)
    if uploaded_dfs:
        df = pd.concat(uploaded_dfs, ignore_index=True)
        st.sidebar.success("Uploaded files loaded and normalized.")
    else:
        df = df_sim.copy()
        st.sidebar.info("Using simulated data because uploads could not be normalized.")
else:
    df = df_sim.copy()

# -------------------------
# Filters
# -------------------------
st.sidebar.header("Filters")
years_available = sorted(df["year"].unique(), reverse=True)
selected_years = st.sidebar.multiselect("Year", options=years_available, default=[CURRENT_YEAR, LAST_YEAR] if CURRENT_YEAR in years_available else years_available[:2])
sources_available = sorted(df["source"].unique())
selected_sources = st.sidebar.multiselect("Source", options=sources_available, default=sources_available)
modes_available = sorted(df["mode"].unique())
selected_modes = st.sidebar.multiselect("Mode", options=modes_available, default=modes_available)
regions_available = sorted(df["region"].unique())
selected_regions = st.sidebar.multiselect("Region", options=regions_available, default=regions_available)
carriers_available = sorted(df["carrier"].unique())
selected_carriers = st.sidebar.multiselect("Carrier", options=carriers_available, default=carriers_available)

df_filtered = df[
    (df["year"].isin(selected_years)) &
    (df["source"].isin(selected_sources)) &
    (df["mode"].isin(selected_modes)) &
    (df["region"].isin(selected_regions)) &
    (df["carrier"].isin(selected_carriers))
].copy()

# -------------------------
# KPIs
# -------------------------
st.header("Key metrics")
col1, col2, col3, col4 = st.columns(4)

total_by_year = df_filtered.groupby("year")["amount"].sum().reindex(selected_years, fill_value=0)
kpi_total = df_filtered["amount"].sum()
kpi_avg_month = df_filtered.groupby(["year", "month"])["amount"].sum().mean()
kpi_top_carrier = df_filtered.groupby("carrier")["amount"].sum().idxmax() if not df_filtered.empty else "N/A"

col1.metric("Total freight spend (filtered)", f"{kpi_total:,.2f}")
col2.metric("Average monthly spend", f"{kpi_avg_month:,.2f}")
col3.metric("Top carrier (by spend)", kpi_top_carrier)
col4.metric("Years shown", ", ".join(map(str, selected_years)))

# -------------------------
# Time series and comparisons
# -------------------------
st.markdown("---")
st.subheader("Spend over time")

# Aggregate monthly
monthly = df_filtered.groupby(["date", "year", "month"]).agg(total_amount=("amount", "sum")).reset_index()
monthly = monthly.sort_values("date")

if monthly.empty:
    st.info("No data for the selected filters. Adjust filters to see charts and tables.")
else:
    fig_ts = px.line(
        monthly,
        x="date",
        y="total_amount",
        color="year",
        markers=True,
        labels={"total_amount": "Total freight spend", "date": "Month", "year": "Year"},
        title="Monthly freight spend by year"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Breakdown by source
    st.subheader("Breakdown by source")
    by_source = df_filtered.groupby(["source", "year"])["amount"].sum().reset_index()
    fig_src = px.bar(
        by_source,
        x="source",
        y="amount",
        color="year",
        barmode="group",
        labels={"amount": "Total spend", "source": "Source"},
        title="Total spend by source and year"
    )
    st.plotly_chart(fig_src, use_container_width=True)

    # Mode share pie for selected year
    st.subheader("Mode share")
    year_for_pie = st.selectbox("Select year for mode share", options=selected_years, index=0)
    mode_share = df_filtered[df_filtered["year"] == year_for_pie].groupby("mode")["amount"].sum().reset_index()
    fig_pie = px.pie(mode_share, names="mode", values="amount", title=f"Mode share — {year_for_pie}")
    st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------
# Table and download
# -------------------------
st.markdown("---")
st.subheader("Detailed transactions")

# Show a sample table with pagination
table_cols = ["date", "year", "month", "source", "mode", "carrier", "region", "amount"]
table_df = df_filtered[table_cols].sort_values(["date", "amount"], ascending=[False, False]).reset_index(drop=True)
st.dataframe(table_df, use_container_width=True, height=300)

# Download filtered data as CSV
csv_buffer = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data as CSV",
    data=csv_buffer,
    file_name=f"freight_costs_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

# -------------------------
# Insights panel
# -------------------------
st.markdown("---")
st.subheader("Automated insights")
insights = []
if not df_filtered.empty:
    # YoY comparison for total spend
    totals = df_filtered.groupby("year")["amount"].sum()
    if CURRENT_YEAR in totals.index and LAST_YEAR in totals.index:
        cur = totals.loc[CURRENT_YEAR]
        prev = totals.loc[LAST_YEAR]
        pct = ((cur - prev) / prev * 100) if prev != 0 else np.nan
        insights.append(f"Total freight spend changed by {pct:.1f}% from {LAST_YEAR} to {CURRENT_YEAR}.")
    # Top region
    top_region = df_filtered.groupby("region")["amount"].sum().idxmax()
    insights.append(f"Top region by spend (filtered): {top_region}.")
    # Top mode
    top_mode = df_filtered.groupby("mode")["amount"].sum().idxmax()
    insights.append(f"Top transport mode by spend (filtered): {top_mode}.")
else:
    insights.append("No insights available for the current filter selection.")

for i, ins in enumerate(insights, 1):
    st.write(f"**Insight {i}:** {ins}")

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.caption("This dashboard simulates freight cost data for demonstration. Upload your own files to analyze real costs. Columns expected: date, source, mode, carrier, region, amount.")
