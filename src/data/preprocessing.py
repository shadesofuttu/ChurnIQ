"""Data Preprocessing and Cleaning"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class DataPreprocessor:
    """
    Handles data cleaning, validation, and preprocessing for banking churn data.
    """
    
    def __init__(self):
        self.missing_stats = {}
        self.outlier_stats = {}
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with missing value statistics
        """
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        self.missing_stats = missing.to_dict('records')
        return missing
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values based on strategy.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'mean', 'median', 'mode'
        
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mode':
            for col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Missing values handled using '{strategy}' strategy")
        print(f"Rows after handling: {len(df_clean)}")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check
            method: 'iqr' or 'zscore'
        
        Returns:
            Dictionary with outlier statistics
        """
        outlier_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > 3]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': round(len(outliers) / len(df) * 100, 2)
            }
        
        self.outlier_stats = outlier_info
        return outlier_info
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers by capping or removing.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            method: 'cap' (winsorize) or 'remove'
        
        Returns:
            DataFrame with handled outliers
        """
        df_clean = df.copy()
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Outliers handled using '{method}' method for {len(columns)} columns")
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            method: 'onehot' or 'label'
        
        Returns:
            DataFrame with encoded variables
        """
        df_encoded = df.copy()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                df_encoded[col] = le.fit_transform(df_encoded[col])
        
        print(f"Encoded {len(columns)} categorical columns using {method} encoding")
        return df_encoded
    
    def get_preprocessing_report(self) -> str:
        """
        Generate a summary report of preprocessing steps.
        
        Returns:
            String report
        """
        report = "=== Data Preprocessing Report ===\n\n"
        
        if self.missing_stats:
            report += "Missing Values Handled:\n"
            for stat in self.missing_stats:
                report += f"  - {stat['Column']}: {stat['Missing_Count']} ({stat['Missing_Percentage']}%)\n"
            report += "\n"
        
        if self.outlier_stats:
            report += "Outliers Detected:\n"
            for col, stats in self.outlier_stats.items():
                report += f"  - {col}: {stats['count']} ({stats['percentage']}%)\n"
        
        return report