"""Reusable Metric Components for Streamlit Dashboard"""

import streamlit as st
import pandas as pd
from typing import Optional, Union

def display_metric_card(title: str, value: Union[str, int, float], 
                       delta: Optional[Union[str, int, float]] = None,
                       delta_color: str = "normal",
                       help_text: Optional[str] = None):
    """
    Display a metric card with optional delta.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional change value
        delta_color: 'normal', 'inverse', or 'off'
        help_text: Optional help tooltip
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def display_kpi_row(kpis: dict):
    """
    Display a row of KPI metrics.
    
    Args:
        kpis: Dictionary with keys: title, value, delta (optional)
    """
    cols = st.columns(len(kpis))
    
    for i, (key, kpi_data) in enumerate(kpis.items()):
        with cols[i]:
            if isinstance(kpi_data, dict):
                st.metric(
                    label=kpi_data.get('title', key),
                    value=kpi_data.get('value'),
                    delta=kpi_data.get('delta'),
                    delta_color=kpi_data.get('delta_color', 'normal')
                )
            else:
                st.metric(label=key, value=kpi_data)

def display_comparison_metrics(metric_name: str, current: float, 
                              previous: float, format_str: str = "{:.2f}"):
    """
    Display comparison between current and previous values.
    
    Args:
        metric_name: Name of the metric
        current: Current value
        previous: Previous value
        format_str: Format string for values
    """
    delta = current - previous
    delta_pct = (delta / previous * 100) if previous != 0 else 0
    
    st.metric(
        label=metric_name,
        value=format_str.format(current),
        delta=f"{delta_pct:+.1f}%"
    )

def display_summary_table(df: pd.DataFrame, title: str = "Summary"):
    """
    Display a formatted summary table.
    
    Args:
        df: DataFrame to display
        title: Table title
    """
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

def display_insight_box(insights: list, title: str = "💡 Key Insights"):
    """
    Display insights in an info box.
    
    Args:
        insights: List of insight strings
        title: Box title
    """
    st.subheader(title)
    for i, insight in enumerate(insights, 1):
        st.info(f"**{i}.** {insight}")

def display_warning_box(message: str, title: str = "⚠️ Warning"):
    """
    Display a warning box.
    
    Args:
        message: Warning message
        title: Box title
    """
    st.warning(f"**{title}**\n\n{message}")

def display_success_box(message: str, title: str = "✅ Success"):
    """
    Display a success box.
    
    Args:
        message: Success message
        title: Box title
    """
    st.success(f"**{title}**\n\n{message}")
