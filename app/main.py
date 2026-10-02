"""Streamlit Dashboard for Customer Churn Analytics"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.settings import DATA_PROCESSED, APP_TITLE, APP_LAYOUT
from src.data.loader import load_processed_data
from src.analysis.churn_analytics import ChurnAnalytics
from src.visualization.plotting import ChurnVisualizer

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        df = load_processed_data()
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure data is processed first.")
        return None

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🏦 Customer Segmentation & Churn Analytics</div>', unsafe_allow_html=True)
    st.markdown("### European Banking - Customer Retention Intelligence Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Initialize analytics
    analytics = ChurnAnalytics(df)
    kpis = analytics.calculate_overall_kpis()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Geography filter
    geographies = ['All'] + sorted(df['Geography'].unique().tolist())
    selected_geo = st.sidebar.selectbox("Geography", geographies)
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Balance filter
    balance_threshold = st.sidebar.number_input(
        "Minimum Balance",
        min_value=0,
        max_value=int(df['Balance'].max()),
        value=0,
        step=10000
    )
    
    # Active member filter
    activity_filter = st.sidebar.multiselect(
        "Member Activity",
        ["Active", "Inactive"],
        default=["Active", "Inactive"]
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_geo != 'All':
        filtered_df = filtered_df[filtered_df['Geography'] == selected_geo]
    
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) &
        (filtered_df['Age'] <= age_range[1])
    ]
    
    filtered_df = filtered_df[filtered_df['Balance'] >= balance_threshold]
    
    if "Active" in activity_filter and "Inactive" not in activity_filter:
        filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
    elif "Inactive" in activity_filter and "Active" not in activity_filter:
        filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]
    
    # Update analytics with filtered data
    filtered_analytics = ChurnAnalytics(filtered_df)
    filtered_kpis = filtered_analytics.calculate_overall_kpis()
    
    # Key Performance Indicators
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{filtered_kpis['total_customers']:,}",
            delta=f"{filtered_kpis['total_customers'] - kpis['total_customers']}" if selected_geo != 'All' else None
        )
    
    with col2:
        st.metric(
            label="Overall Churn Rate",
            value=f"{filtered_kpis['overall_churn_rate']:.2f}%",
            delta=f"{filtered_kpis['overall_churn_rate'] - kpis['overall_churn_rate']:.2f}%" if selected_geo != 'All' else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Churned Customers",
            value=f"{filtered_kpis['churned_customers']:,}",
            delta=f"{filtered_kpis['churned_customers'] - kpis['churned_customers']}" if selected_geo != 'All' else None,
            delta_color="inverse"
        )
    
    with col4:
        engagement_drop = filtered_analytics.calculate_engagement_drop_indicator()
        st.metric(
            label="Engagement Drop Index",
            value=f"{engagement_drop:.1f}",
            help="Score 0-100, higher indicates more concerning engagement levels"
        )
    
    st.markdown("---")
    
    # Geographic Analysis
    st.header("🌍 Geographic Churn Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        visualizer = ChurnVisualizer(filtered_df)
        geo_fig = visualizer.plot_churn_by_geography(save=False)
        st.plotly_chart(geo_fig, use_container_width=True)
    
    with col2:
        st.subheader("Geographic Breakdown")
        geo_analysis = filtered_analytics.analyze_churn_by_geography()
        st.dataframe(
            geo_analysis[['Total_Customers', 'Churn_Rate', 'Risk_Index']],
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Demographic Analysis
    st.header("👥 Demographic Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Age Groups", "Gender", "Tenure"])
    
    demo_analysis = filtered_analytics.analyze_churn_by_demographics()
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            demo_fig = visualizer.plot_churn_by_demographics(save=False)
            st.plotly_chart(demo_fig, use_container_width=True)
        with col2:
            st.dataframe(demo_analysis['age'], use_container_width=True)
    
    with tab2:
        st.dataframe(demo_analysis['gender'], use_container_width=True)
        
        # Gender comparison
        gender_churn = filtered_df.groupby('Gender')['Exited'].mean() * 100
        if len(gender_churn) > 1:
            diff = abs(gender_churn.iloc[0] - gender_churn.iloc[1])
            st.info(f"Gender churn rate difference: {diff:.2f}%")
    
    with tab3:
        tenure_fig = visualizer.plot_churn_by_tenure(save=False)
        st.plotly_chart(tenure_fig, use_container_width=True)
        st.dataframe(demo_analysis['tenure'], use_container_width=True)
    
    st.markdown("---")
    
    # High-Value Customer Analysis
    st.header("💎 High-Value Customer Churn Analysis")
    
    high_value_threshold = st.slider(
        "Define High-Value Customer (Balance Threshold)",
        min_value=50000,
        max_value=200000,
        value=100000,
        step=10000
    )
    
    high_value_analysis = filtered_analytics.analyze_high_value_churn(high_value_threshold)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(high_value_analysis, use_container_width=True)
    
    with col2:
        # Calculate high-value churn impact
        high_value_customers = filtered_df[filtered_df['Balance'] >= high_value_threshold]
        hv_churn_rate = high_value_customers['Exited'].mean() * 100
        hv_churned = high_value_customers['Exited'].sum()
        
        st.markdown(f"""
        **Key Insights:**
        - High-value customers (Balance ≥ €{high_value_threshold:,}): **{len(high_value_customers):,}**
        - High-value churn rate: **{hv_churn_rate:.2f}%**
        - High-value customers lost: **{hv_churned}**
        - Average balance of churned high-value customers: **€{high_value_customers[high_value_customers['Exited']==1]['Balance'].mean():,.2f}**
        """)
        
        if hv_churn_rate > filtered_kpis['overall_churn_rate']:
            st.warning(f"⚠️ High-value customers are churning at a higher rate ({hv_churn_rate:.2f}%) than the overall average ({filtered_kpis['overall_churn_rate']:.2f}%)")
    
    st.markdown("---")
    
    # Product & Engagement Analysis
    st.header("🔗 Product & Engagement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        products_fig = visualizer.plot_products_vs_churn(save=False)
        st.plotly_chart(products_fig, use_container_width=True)
        
        # Product insights
        single_product_churn = filtered_df[filtered_df['NumOfProducts'] == 1]['Exited'].mean() * 100
        multi_product_churn = filtered_df[filtered_df['NumOfProducts'] >= 2]['Exited'].mean() * 100
        
        st.info(f"📊 Single-product customers have {single_product_churn:.1f}% churn rate vs {multi_product_churn:.1f}% for multi-product customers")
    
    with col2:
        engagement_fig = visualizer.plot_engagement_analysis(save=False)
        st.plotly_chart(engagement_fig, use_container_width=True)
        
        # Engagement insights
        inactive_churn = filtered_df[filtered_df['IsActiveMember'] == 0]['Exited'].mean() * 100
        active_churn = filtered_df[filtered_df['IsActiveMember'] == 1]['Exited'].mean() * 100
        
        st.warning(f"⚠️ Inactive members have {inactive_churn:.1f}% churn rate vs {active_churn:.1f}% for active members")
    
    st.markdown("---")
    
    # Segment Analysis (if available)
    if 'Segment_Name' in filtered_df.columns:
        st.header("🎯 Customer Segment Analysis")
        
        segment_churn = filtered_analytics.calculate_segment_churn_rates()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            segment_fig = visualizer.plot_segment_distribution(save=False)
            st.plotly_chart(segment_fig, use_container_width=True)
        
        with col2:
            st.subheader("Segment Churn Rates")
            st.dataframe(
                segment_churn[['Total_Customers', 'Churn_Rate', 'Avg_Balance']],
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Key Insights & Recommendations
    st.header("💡 Key Insights & Recommendations")
    
    insights = filtered_analytics.identify_churn_patterns()
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
            <strong>{i}.</strong> {insight}
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison table
    st.subheader("📈 Churned vs Retained Customer Comparison")
    comparison = filtered_analytics.compare_churned_vs_retained()
    st.dataframe(comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Export options
    st.header("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Analysis Report"):
            with st.spinner("Generating reports..."):
                report_path = filtered_analytics.export_analysis_report()
                st.success(f"Reports exported to: {report_path}")
    
    with col2:
        if st.button("📈 Export Visualizations"):
            with st.spinner("Generating visualizations..."):
                visualizer.create_all_eda_plots()
                st.success("Visualizations saved to figures directory")
    
    with col3:
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Filtered Data",
            data=csv,
            file_name="filtered_churn_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Customer Segmentation & Churn Pattern Analytics in European Banking</p>
        <p>Dashboard for data-driven customer retention strategies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
