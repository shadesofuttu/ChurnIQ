"""Overview Page - Main Dashboard"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.analysis.churn_analytics import ChurnAnalytics
from src.visualization.plotting import ChurnVisualizer
from app.components.metrics import display_kpi_row

st.set_page_config(page_title="Overview - ChurnIQ", page_icon="📊", layout="wide")

st.title("📊 Overview - Customer Churn Analytics")
st.markdown("### Comprehensive view of customer retention metrics")
st.markdown("---")

# Load data from session state
if 'data' not in st.session_state:
    st.error("Please load data from the main page first.")
    st.stop()

df = st.session_state['data']
filtered_df = st.session_state.get('filtered_data', df)

# Initialize analytics
analytics = ChurnAnalytics(filtered_df)
kpis = analytics.calculate_overall_kpis()

# Display KPIs
st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value=f"{kpis['total_customers']:,}",
        help="Total number of customers in the dataset"
    )

with col2:
    st.metric(
        label="Churn Rate",
        value=f"{kpis['overall_churn_rate']:.2f}%",
        delta=f"-{100 - kpis['overall_churn_rate']:.1f}% retention",
        delta_color="inverse",
        help="Percentage of customers who have churned"
    )

with col3:
    st.metric(
        label="Churned Customers",
        value=f"{kpis['churned_customers']:,}",
        help="Number of customers who have exited"
    )

with col4:
    engagement_drop = analytics.calculate_engagement_drop_indicator()
    st.metric(
        label="Engagement Index",
        value=f"{100 - engagement_drop:.1f}",
        help="Customer engagement score (higher is better)"
    )

st.markdown("---")

# Quick insights
st.header("📈 Quick Insights")

insights = analytics.identify_churn_patterns()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Findings")
    for i, insight in enumerate(insights[:3], 1):
        st.info(f"**{i}.** {insight}")

with col2:
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Metric': ['Avg Age', 'Avg Tenure', 'Avg Balance', 'Avg Products'],
        'Value': [
            f"{kpis['avg_customer_age']:.1f} years",
            f"{kpis['avg_tenure']:.1f} years",
            f"€{kpis['avg_balance']:,.0f}",
            f"{kpis['avg_products']:.2f}"
        ]
    })
    st.dataframe(summary_stats, hide_index=True, use_container_width=True)

st.markdown("---")

# Churn distribution
st.header("Churn Distribution")

visualizer = ChurnVisualizer(filtered_df)
fig = visualizer.plot_churn_distribution(save=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Geographic overview
st.header("Geographic Overview")

geo_analysis = analytics.analyze_churn_by_geography()
fig = visualizer.plot_churn_by_geography(save=False)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    geo_analysis[['Total_Customers', 'Churn_Rate', 'Churned_Count', 'Risk_Index']],
    use_container_width=True
)
