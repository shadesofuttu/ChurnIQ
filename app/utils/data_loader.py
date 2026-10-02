"""Data Loading Utilities for Streamlit App"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_processed_data
from config.settings import DATA_PROCESSED

@st.cache_data
def load_dashboard_data():
    """
    Load and cache data for dashboard.
    
    Returns:
        DataFrame with processed customer data
    """
    try:
        df = load_processed_data('cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error(
            "⚠️ Data file not found. Please run the training pipeline first:\n\n"
            "`python scripts/train_models.py`"
        )
        return None

@st.cache_data
def load_segment_data():
    """
    Load segmentation data if available.
    
    Returns:
        DataFrame with segment information or None
    """
    try:
        segment_file = DATA_PROCESSED / 'segmented_data.csv'
        if segment_file.exists():
            return pd.read_csv(segment_file)
        return None
    except Exception as e:
        st.warning(f"Could not load segment data: {e}")
        return None

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the data.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_records': len(df),
        'total_churned': df['Exited'].sum(),
        'churn_rate': df['Exited'].mean() * 100,
        'unique_geographies': df['Geography'].nunique(),
        'date_range': 'Historical data',
        'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
    }
