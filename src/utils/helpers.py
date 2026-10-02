"""Utility Helper Functions"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage string.
    
    Args:
        value: Decimal value (e.g., 0.25)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string (e.g., '25.00%')
    """
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = '€', decimals: int = 2) -> str:
    """
    Format a value as currency string.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.{decimals}f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def create_bins(data: pd.Series, n_bins: int = 5, labels: List[str] = None) -> pd.Series:
    """
    Create quantile-based bins for continuous data.
    
    Args:
        data: Pandas Series to bin
        n_bins: Number of bins
        labels: Optional labels for bins
    
    Returns:
        Binned Series
    """
    return pd.qcut(data, q=n_bins, labels=labels, duplicates='drop')

def get_top_n_values(series: pd.Series, n: int = 10) -> pd.Series:
    """
    Get top N values from a Series.
    
    Args:
        series: Pandas Series
        n: Number of top values
    
    Returns:
        Series with top N values
    """
    return series.value_counts().head(n)

def normalize_column(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a pandas Series.
    
    Args:
        series: Series to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized Series
    """
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_summary_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Create summary statistics grouped by a column.
    
    Args:
        df: Input DataFrame
        group_col: Column to group by
        value_col: Column to aggregate
    
    Returns:
        DataFrame with summary statistics
    """
    return df.groupby(group_col)[value_col].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        series: Pandas Series
        multiplier: IQR multiplier (typically 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

def create_age_bins(ages: pd.Series) -> pd.Series:
    """
    Create standard age bins.
    
    Args:
        ages: Series of ages
    
    Returns:
        Series with age groups
    """
    bins = [0, 18, 30, 40, 50, 60, 100]
    labels = ['<18', '18-30', '31-40', '41-50', '51-60', '60+']
    return pd.cut(ages, bins=bins, labels=labels, right=False)

def calculate_churn_rate(df: pd.DataFrame, group_col: str = None) -> float:
    """
    Calculate churn rate overall or by group.
    
    Args:
        df: DataFrame with 'Exited' column
        group_col: Optional column to group by
    
    Returns:
        Churn rate(s)
    """
    if group_col:
        return df.groupby(group_col)['Exited'].mean() * 100
    else:
        return df['Exited'].mean() * 100

def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of DataFrame.
    
    Args:
        df: DataFrame
    
    Returns:
        Memory usage string
    """
    bytes_used = df.memory_usage(deep=True).sum()
    mb_used = bytes_used / (1024 ** 2)
    return f"{mb_used:.2f} MB"

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Memory: {get_memory_usage(df)}")
    print(f"\nColumn Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("None")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"{'='*60}\n")
