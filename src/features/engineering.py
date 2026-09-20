"""Feature Engineering for Churn Analysis"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """
    Create new features for churn prediction model.
    """
    
    def __init__(self):
        self.feature_list = []
    
    def create_age_groups(self, df: pd.DataFrame, age_col: str = 'Age') -> pd.DataFrame:
        """
        Create age group categories.
        
        Args:
            df: Input DataFrame
            age_col: Name of age column
        
        Returns:
            DataFrame with age group feature
        """
        df = df.copy()
        df['Age_Group'] = pd.cut(
            df[age_col],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['18-30', '31-40', '41-50', '51-60', '60+']
        )
        self.feature_list.append('Age_Group')
        return df
    
    def create_balance_categories(self, df: pd.DataFrame, balance_col: str = 'Balance') -> pd.DataFrame:
        """
        Categorize balance into groups.
        
        Args:
            df: Input DataFrame
            balance_col: Name of balance column
        
        Returns:
            DataFrame with balance category feature
        """
        df = df.copy()
        df['Balance_Category'] = pd.cut(
            df[balance_col],
            bins=[-np.inf, 0, 50000, 100000, 150000, np.inf],
            labels=['Zero', 'Low', 'Medium', 'High', 'Very_High']
        )
        self.feature_list.append('Balance_Category')
        return df
    
    def create_tenure_groups(self, df: pd.DataFrame, tenure_col: str = 'Tenure') -> pd.DataFrame:
        """
        Create tenure group categories.
        
        Args:
            df: Input DataFrame
            tenure_col: Name of tenure column
        
        Returns:
            DataFrame with tenure group feature
        """
        df = df.copy()
        df['Tenure_Group'] = pd.cut(
            df[tenure_col],
            bins=[0, 2, 5, 7, 10],
            labels=['New', 'Short', 'Medium', 'Long'],
            include_lowest=True
        )
        self.feature_list.append('Tenure_Group')
        return df
    
    def create_credit_score_groups(self, df: pd.DataFrame, credit_col: str = 'CreditScore') -> pd.DataFrame:
        """
        Categorize credit scores.
        
        Args:
            df: Input DataFrame
            credit_col: Name of credit score column
        
        Returns:
            DataFrame with credit score category
        """
        df = df.copy()
        df['Credit_Category'] = pd.cut(
            df[credit_col],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent']
        )
        self.feature_list.append('Credit_Category')
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Balance per product
        if 'Balance' in df.columns and 'NumOfProducts' in df.columns:
            df['Balance_Per_Product'] = df['Balance'] / (df['NumOfProducts'] + 1)
            self.feature_list.append('Balance_Per_Product')
        
        # Tenure-Age ratio
        if 'Tenure' in df.columns and 'Age' in df.columns:
            df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
            self.feature_list.append('Tenure_Age_Ratio')
        
        # Active member with credit card
        if 'IsActiveMember' in df.columns and 'HasCrCard' in df.columns:
            df['Active_With_Card'] = df['IsActiveMember'] * df['HasCrCard']
            self.feature_list.append('Active_With_Card')
        
        # Credit to salary ratio
        if 'CreditScore' in df.columns and 'EstimatedSalary' in df.columns:
            df['Credit_Salary_Ratio'] = df['CreditScore'] / (df['EstimatedSalary'] / 1000)
            self.feature_list.append('Credit_Salary_Ratio')
        
        return df
    
    def create_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer engagement score based on multiple factors.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engagement score
        """
        df = df.copy()
        
        engagement_components = []
        
        if 'IsActiveMember' in df.columns:
            engagement_components.append(df['IsActiveMember'] * 30)
        
        if 'NumOfProducts' in df.columns:
            engagement_components.append((df['NumOfProducts'] / 4) * 30)
        
        if 'HasCrCard' in df.columns:
            engagement_components.append(df['HasCrCard'] * 20)
        
        if 'Balance' in df.columns:
            balance_normalized = (df['Balance'] - df['Balance'].min()) / (df['Balance'].max() - df['Balance'].min())
            engagement_components.append(balance_normalized * 20)
        
        df['Engagement_Score'] = sum(engagement_components)
        self.feature_list.append('Engagement_Score')
        
        return df
    
    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer risk score for churn.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with risk score
        """
        df = df.copy()
        
        risk = 0
        
        # Low balance increases risk
        if 'Balance' in df.columns:
            risk += (df['Balance'] == 0).astype(int) * 25
        
        # Not active member increases risk
        if 'IsActiveMember' in df.columns:
            risk += (1 - df['IsActiveMember']) * 25
        
        # Low number of products increases risk
        if 'NumOfProducts' in df.columns:
            risk += (df['NumOfProducts'] == 1).astype(int) * 20
        
        # Short tenure increases risk
        if 'Tenure' in df.columns:
            risk += (df['Tenure'] <= 2).astype(int) * 15
        
        # Low credit score increases risk
        if 'CreditScore' in df.columns:
            risk += (df['CreditScore'] < 600).astype(int) * 15
        
        df['Churn_Risk_Score'] = risk
        self.feature_list.append('Churn_Risk_Score')
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features at once.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all new features
        """
        df = self.create_age_groups(df)
        df = self.create_balance_categories(df)
        df = self.create_tenure_groups(df)
        df = self.create_credit_score_groups(df)
        df = self.create_interaction_features(df)
        df = self.create_engagement_score(df)
        df = self.create_risk_score(df)
        
        print(f"Created {len(self.feature_list)} new features")
        print(f"New features: {', '.join(self.feature_list)}")
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame, target: str = 'Exited') -> pd.DataFrame:
        """
        Calculate basic feature importance using correlation.
        
        Args:
            df: Input DataFrame
            target: Target variable name
        
        Returns:
            DataFrame with feature correlations
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values
        })
        
        return importance_df[importance_df['Feature'] != target]