"""Reusable Filter Components for Streamlit Dashboard"""

import streamlit as st
import pandas as pd
from typing import List, Optional, Tuple

def create_geography_filter(df: pd.DataFrame, key: str = "geo_filter") -> str:
    """
    Create geography selection filter.
    
    Args:
        df: DataFrame with 'Geography' column
        key: Unique key for widget
    
    Returns:
        Selected geography or 'All'
    """
    geographies = ['All'] + sorted(df['Geography'].unique().tolist())
    return st.selectbox("🌍 Geography", geographies, key=key)

def create_age_range_filter(df: pd.DataFrame, key: str = "age_filter") -> Tuple[int, int]:
    """
    Create age range slider.
    
    Args:
        df: DataFrame with 'Age' column
        key: Unique key for widget
    
    Returns:
        Tuple of (min_age, max_age)
    """
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    return st.slider(
        "👤 Age Range",
        min_age,
        max_age,
        (min_age, max_age),
        key=key
    )

def create_balance_filter(df: pd.DataFrame, key: str = "balance_filter") -> float:
    """
    Create minimum balance filter.
    
    Args:
        df: DataFrame with 'Balance' column
        key: Unique key for widget
    
    Returns:
        Minimum balance threshold
    """
    return st.number_input(
        "💰 Minimum Balance (€)",
        min_value=0,
        max_value=int(df['Balance'].max()),
        value=0,
        step=10000,
        key=key
    )

def create_activity_filter(key: str = "activity_filter") -> List[str]:
    """
    Create member activity filter.
    
    Args:
        key: Unique key for widget
    
    Returns:
        List of selected activity statuses
    """
    return st.multiselect(
        "🔔 Member Activity",
        ["Active", "Inactive"],
        default=["Active", "Inactive"],
        key=key
    )

def create_product_filter(df: pd.DataFrame, key: str = "product_filter") -> List[int]:
    """
    Create number of products filter.
    
    Args:
        df: DataFrame with 'NumOfProducts' column
        key: Unique key for widget
    
    Returns:
        List of selected product counts
    """
    products = sorted(df['NumOfProducts'].unique().tolist())
    return st.multiselect(
        "📦 Number of Products",
        products,
        default=products,
        key=key
    )

def create_tenure_filter(df: pd.DataFrame, key: str = "tenure_filter") -> Tuple[int, int]:
    """
    Create tenure range filter.
    
    Args:
        df: DataFrame with 'Tenure' column
        key: Unique key for widget
    
    Returns:
        Tuple of (min_tenure, max_tenure)
    """
    min_tenure = int(df['Tenure'].min())
    max_tenure = int(df['Tenure'].max())
    return st.slider(
        "📅 Tenure (Years)",
        min_tenure,
        max_tenure,
        (min_tenure, max_tenure),
        key=key
    )

def create_gender_filter(key: str = "gender_filter") -> List[str]:
    """
    Create gender filter.
    
    Args:
        key: Unique key for widget
    
    Returns:
        List of selected genders
    """
    return st.multiselect(
        "⚧ Gender",
        ["Male", "Female"],
        default=["Male", "Female"],
        key=key
    )

def create_credit_card_filter(key: str = "card_filter") -> Optional[bool]:
    """
    Create credit card ownership filter.
    
    Args:
        key: Unique key for widget
    
    Returns:
        Selected credit card status or None for all
    """
    options = ["All", "Has Card", "No Card"]
    selection = st.radio(
        "💳 Credit Card",
        options,
        key=key,
        horizontal=True
    )
    
    if selection == "Has Card":
        return True
    elif selection == "No Card":
        return False
    else:
        return None

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply multiple filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter criteria
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Geography filter
    if 'geography' in filters and filters['geography'] != 'All':
        filtered_df = filtered_df[filtered_df['Geography'] == filters['geography']]
    
    # Age range filter
    if 'age_range' in filters:
        min_age, max_age = filters['age_range']
        filtered_df = filtered_df[
            (filtered_df['Age'] >= min_age) & 
            (filtered_df['Age'] <= max_age)
        ]
    
    # Balance filter
    if 'min_balance' in filters:
        filtered_df = filtered_df[filtered_df['Balance'] >= filters['min_balance']]
    
    # Activity filter
    if 'activity' in filters:
        activity_filter = filters['activity']
        if "Active" in activity_filter and "Inactive" not in activity_filter:
            filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
        elif "Inactive" in activity_filter and "Active" not in activity_filter:
            filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]
    
    # Product filter
    if 'products' in filters and filters['products']:
        filtered_df = filtered_df[filtered_df['NumOfProducts'].isin(filters['products'])]
    
    # Tenure filter
    if 'tenure_range' in filters:
        min_tenure, max_tenure = filters['tenure_range']
        filtered_df = filtered_df[
            (filtered_df['Tenure'] >= min_tenure) & 
            (filtered_df['Tenure'] <= max_tenure)
        ]
    
    # Gender filter
    if 'gender' in filters and filters['gender']:
        filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
    
    # Credit card filter
    if 'has_credit_card' in filters and filters['has_credit_card'] is not None:
        filtered_df = filtered_df[
            filtered_df['HasCrCard'] == (1 if filters['has_credit_card'] else 0)
        ]
    
    return filtered_df

def display_filter_summary(original_count: int, filtered_count: int):
    """
    Display summary of applied filters.
    
    Args:
        original_count: Original number of records
        filtered_count: Filtered number of records
    """
    pct = (filtered_count / original_count * 100) if original_count > 0 else 0
    
    st.info(
        f"📊 Showing **{filtered_count:,}** of **{original_count:,}** customers "
        f"({pct:.1f}%)"
    )
