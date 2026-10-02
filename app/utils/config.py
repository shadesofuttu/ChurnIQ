"""Configuration for Streamlit Dashboard"""

import streamlit as st

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'ChurnIQ Dashboard',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db',
    'churned': '#e74c3c',
    'retained': '#2ecc71',
    'high_risk': '#e74c3c',
    'medium_risk': '#f39c12',
    'low_risk': '#2ecc71'
}

# KPI thresholds
KPI_THRESHOLDS = {
    'churn_rate': {
        'excellent': 10,
        'good': 15,
        'warning': 20,
        'critical': 25
    },
    'engagement_score': {
        'excellent': 70,
        'good': 50,
        'warning': 30,
        'critical': 20
    }
}

# Chart configuration
CHART_CONFIG = {
    'height': 400,
    'template': 'plotly_white',
    'font_family': 'Arial, sans-serif'
}

# Export configuration
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'excel_engine': 'openpyxl',
    'date_format': '%Y-%m-%d'
}

def apply_custom_css():
    """
    Apply custom CSS styling to the dashboard.
    """
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
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
        }
        .danger-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

def get_color_by_threshold(value: float, thresholds: dict) -> str:
    """
    Get color based on value and thresholds.
    
    Args:
        value: Value to check
        thresholds: Dictionary with threshold levels
    
    Returns:
        Color hex code
    """
    if value <= thresholds['excellent']:
        return COLORS['success']
    elif value <= thresholds['good']:
        return COLORS['info']
    elif value <= thresholds['warning']:
        return COLORS['warning']
    else:
        return COLORS['danger']
