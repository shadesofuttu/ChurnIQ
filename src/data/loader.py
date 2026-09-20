"""Data Loading Utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_RAW, DATA_PROCESSED


def load_raw_data(filename: str = "bank_churn.csv") -> pd.DataFrame:
    """
    Load raw banking data from CSV file.
    
    Args:
        filename: Name of the CSV file in data/raw/
    
    Returns:
        DataFrame with raw data
    """
    filepath = DATA_RAW / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filename}")
    return df


def load_processed_data(filename: str = "cleaned_data.csv") -> pd.DataFrame:
    """
    Load processed/cleaned data.
    
    Args:
        filename: Name of the processed CSV file
    
    Returns:
        DataFrame with processed data
    """
    filepath = DATA_PROCESSED / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "cleaned_data.csv") -> None:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    filepath = DATA_PROCESSED / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} records to {filepath}")


def get_train_test_split(df: pd.DataFrame, 
                          target_col: str = 'Exited',
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[target_col] if target_col in df.columns else None
    )
    
    print(f"Train set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df