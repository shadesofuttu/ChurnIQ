"""Comprehensive Churn Analysis for European Banking"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_PROCESSED, FIGURES_DIR

class ChurnAnalytics:
    """
    Comprehensive analytics for customer churn patterns.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize churn analytics.
        
        Args:
            df: DataFrame with customer data including churn labels
        """
        self.df = df.copy()
        self.kpis = {}
        self.insights = []
    
    def calculate_overall_kpis(self) -> Dict[str, float]:
        """
        Calculate key performance indicators.
        
        Returns:
            Dictionary of KPIs
        """
        total_customers = len(self.df)
        churned_customers = self.df['Exited'].sum()
        
        self.kpis = {
            'total_customers': total_customers,
            'churned_customers': int(churned_customers),
            'retained_customers': int(total_customers - churned_customers),
            'overall_churn_rate': round(churned_customers / total_customers * 100, 2),
            'retention_rate': round((1 - churned_customers / total_customers) * 100, 2),
            'avg_customer_age': round(self.df['Age'].mean(), 1),
            'avg_tenure': round(self.df['Tenure'].mean(), 1),
            'avg_balance': round(self.df['Balance'].mean(), 2),
            'avg_products': round(self.df['NumOfProducts'].mean(), 2),
        }
        
        return self.kpis
    
    def analyze_churn_by_geography(self) -> pd.DataFrame:
        """
        Analyze churn patterns across different geographies.
        
        Returns:
            DataFrame with geographic churn analysis
        """
        geo_analysis = self.df.groupby('Geography').agg({
            'Exited': ['sum', 'mean', 'count'],
            'Balance': 'mean',
            'NumOfProducts': 'mean',
            'IsActiveMember': 'mean'
        }).round(3)
        
        geo_analysis.columns = [
            'Churned_Count', 'Churn_Rate', 'Total_Customers',
            'Avg_Balance', 'Avg_Products', 'Active_Rate'
        ]
        
        geo_analysis['Churn_Rate'] = (geo_analysis['Churn_Rate'] * 100).round(2)
        geo_analysis['Active_Rate'] = (geo_analysis['Active_Rate'] * 100).round(2)
        geo_analysis = geo_analysis.sort_values('Churn_Rate', ascending=False)
        
        # Calculate geographic risk index
        geo_analysis['Risk_Index'] = (
            geo_analysis['Churn_Rate'] * 0.5 +
            (100 - geo_analysis['Active_Rate']) * 0.3 +
            (geo_analysis['Avg_Products'] < 2).astype(int) * 20
        ).round(2)
        
        return geo_analysis
    
    def analyze_churn_by_demographics(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze churn by demographic factors.
        
        Returns:
            Dictionary of demographic analysis DataFrames
        """
        results = {}
        
        # Age groups
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        self.df['Age_Group'] = pd.cut(self.df['Age'], bins=age_bins, labels=age_labels)
        
        results['age'] = self.df.groupby('Age_Group', observed=False).agg({
            'Exited': ['sum', 'mean', 'count']
        }).round(4)
        results['age'].columns = ['Churned', 'Churn_Rate', 'Total']
        results['age']['Churn_Rate'] = (results['age']['Churn_Rate'] * 100).round(2)
        
        # Gender
        results['gender'] = self.df.groupby('Gender').agg({
            'Exited': ['sum', 'mean', 'count']
        }).round(4)
        results['gender'].columns = ['Churned', 'Churn_Rate', 'Total']
        results['gender']['Churn_Rate'] = (results['gender']['Churn_Rate'] * 100).round(2)
        
        # Tenure groups
        tenure_bins = [0, 2, 5, 7, 11]
        tenure_labels = ['0-2 yrs', '3-5 yrs', '6-7 yrs', '8+ yrs']
        self.df['Tenure_Group'] = pd.cut(self.df['Tenure'], bins=tenure_bins, labels=tenure_labels)
        
        results['tenure'] = self.df.groupby('Tenure_Group', observed=False).agg({
            'Exited': ['sum', 'mean', 'count']
        }).round(4)
        results['tenure'].columns = ['Churned', 'Churn_Rate', 'Total']
        results['tenure']['Churn_Rate'] = (results['tenure']['Churn_Rate'] * 100).round(2)
        
        return results
    
    def analyze_high_value_churn(self, balance_threshold: float = 100000) -> pd.DataFrame:
        """
        Analyze churn among high-value customers.
        
        Args:
            balance_threshold: Balance threshold for high-value customers
        
        Returns:
            DataFrame with high-value customer analysis
        """
        high_value = self.df[self.df['Balance'] >= balance_threshold]
        regular_value = self.df[self.df['Balance'] < balance_threshold]
        
        analysis = pd.DataFrame({
            'Customer_Type': ['High-Value', 'Regular-Value'],
            'Total_Customers': [len(high_value), len(regular_value)],
            'Churned': [high_value['Exited'].sum(), regular_value['Exited'].sum()],
            'Churn_Rate': [
                round(high_value['Exited'].mean() * 100, 2),
                round(regular_value['Exited'].mean() * 100, 2)
            ],
            'Avg_Balance': [
                round(high_value['Balance'].mean(), 2),
                round(regular_value['Balance'].mean(), 2)
            ],
            'Avg_Products': [
                round(high_value['NumOfProducts'].mean(), 2),
                round(regular_value['NumOfProducts'].mean(), 2)
            ]
        })
        
        # Calculate revenue at risk (assuming avg revenue per customer)
        avg_annual_revenue = 500  # Placeholder
        analysis['Revenue_at_Risk'] = (
            analysis['Churned'] * avg_annual_revenue
        ).round(2)
        
        return analysis
    
    def analyze_churn_by_engagement(self) -> pd.DataFrame:
        """
        Analyze churn by customer engagement indicators.
        
        Returns:
            DataFrame with engagement analysis
        """
        engagement_analysis = []
        
        # Active vs Inactive members
        for active_status in [0, 1]:
            subset = self.df[self.df['IsActiveMember'] == active_status]
            engagement_analysis.append({
                'Engagement_Type': 'Active' if active_status else 'Inactive',
                'Total': len(subset),
                'Churned': subset['Exited'].sum(),
                'Churn_Rate': round(subset['Exited'].mean() * 100, 2),
                'Avg_Products': round(subset['NumOfProducts'].mean(), 2)
            })
        
        # Credit card ownership
        for card_status in [0, 1]:
            subset = self.df[self.df['HasCrCard'] == card_status]
            engagement_analysis.append({
                'Engagement_Type': 'Has Credit Card' if card_status else 'No Credit Card',
                'Total': len(subset),
                'Churned': subset['Exited'].sum(),
                'Churn_Rate': round(subset['Exited'].mean() * 100, 2),
                'Avg_Products': round(subset['NumOfProducts'].mean(), 2)
            })
        
        # Number of products
        for num_products in sorted(self.df['NumOfProducts'].unique()):
            subset = self.df[self.df['NumOfProducts'] == num_products]
            engagement_analysis.append({
                'Engagement_Type': f'{int(num_products)} Product(s)',
                'Total': len(subset),
                'Churned': subset['Exited'].sum(),
                'Churn_Rate': round(subset['Exited'].mean() * 100, 2),
                'Avg_Products': num_products
            })
        
        return pd.DataFrame(engagement_analysis)
    
    def calculate_segment_churn_rates(self, segment_col: str = 'Segment_Name') -> pd.DataFrame:
        """
        Calculate churn rates by customer segment.
        
        Args:
            segment_col: Name of segment column
        
        Returns:
            DataFrame with segment churn analysis
        """
        if segment_col not in self.df.columns:
            print(f"Warning: '{segment_col}' column not found")
            return None
        
        segment_analysis = self.df.groupby(segment_col).agg({
            'Exited': ['sum', 'mean', 'count'],
            'Balance': 'mean',
            'CreditScore': 'mean',
            'Age': 'mean'
        }).round(2)
        
        segment_analysis.columns = [
            'Churned_Count', 'Churn_Rate', 'Total_Customers',
            'Avg_Balance', 'Avg_Credit_Score', 'Avg_Age'
        ]
        
        segment_analysis['Churn_Rate'] = (segment_analysis['Churn_Rate'] * 100).round(2)
        segment_analysis = segment_analysis.sort_values('Churn_Rate', ascending=False)
        
        return segment_analysis
    
    def identify_churn_patterns(self) -> List[str]:
        """
        Identify key patterns and insights from churn analysis.
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # Geographic insights
        geo_analysis = self.analyze_churn_by_geography()
        highest_churn_geo = geo_analysis.index[0]
        highest_churn_rate = geo_analysis.loc[highest_churn_geo, 'Churn_Rate']
        
        insights.append(
            f"Geographic Risk: {highest_churn_geo} has the highest churn rate at {highest_churn_rate}%"
        )
        
        # Age insights
        demo_analysis = self.analyze_churn_by_demographics()
        age_churn = demo_analysis['age'].sort_values('Churn_Rate', ascending=False)
        highest_age_group = age_churn.index[0]
        
        insights.append(
            f"Age Pattern: Customers aged {highest_age_group} show highest churn tendency"
        )
        
        # Engagement insights
        active_churn = self.df[self.df['IsActiveMember'] == 0]['Exited'].mean()
        inactive_churn = self.df[self.df['IsActiveMember'] == 1]['Exited'].mean()
        
        if active_churn > inactive_churn * 1.5:
            insights.append(
                f"Engagement Critical: Inactive members are {round(active_churn/inactive_churn, 1)}x more likely to churn"
            )
        
        # Product insights
        single_product_churn = self.df[self.df['NumOfProducts'] == 1]['Exited'].mean()
        multi_product_churn = self.df[self.df['NumOfProducts'] >= 2]['Exited'].mean()
        
        if single_product_churn > multi_product_churn * 1.2:
            insights.append(
                f"Product Cross-Sell Opportunity: Single-product customers have {round(single_product_churn * 100, 1)}% churn vs {round(multi_product_churn * 100, 1)}% for multi-product"
            )
        
        # Balance insights
        zero_balance_churn = self.df[self.df['Balance'] == 0]['Exited'].mean()
        if zero_balance_churn > 0.5:
            insights.append(
                f"Zero Balance Risk: {round(zero_balance_churn * 100, 1)}% of customers with zero balance churned"
            )
        
        self.insights = insights
        return insights
    
    def generate_kpi_summary(self) -> Dict[str, any]:
        """
        Generate comprehensive KPI summary for dashboard.
        
        Returns:
            Dictionary with all KPIs and metrics
        """
        if not self.kpis:
            self.calculate_overall_kpis()
        
        geo_analysis = self.analyze_churn_by_geography()
        high_value = self.analyze_high_value_churn()
        
        summary = {
            'overall': self.kpis,
            'geographic': geo_analysis.to_dict('index'),
            'high_value': high_value.to_dict('records'),
            'insights': self.identify_churn_patterns()
        }
        
        return summary
    
    def compare_churned_vs_retained(self) -> pd.DataFrame:
        """
        Compare characteristics of churned vs retained customers.
        
        Returns:
            DataFrame with comparison
        """
        comparison_features = [
            'CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        churned = self.df[self.df['Exited'] == 1][comparison_features].mean()
        retained = self.df[self.df['Exited'] == 0][comparison_features].mean()
        
        comparison = pd.DataFrame({
            'Churned_Avg': churned,
            'Retained_Avg': retained,
            'Difference': churned - retained,
            'Difference_Pct': ((churned - retained) / retained * 100).round(2)
        }).round(2)
        
        return comparison
    
    def calculate_engagement_drop_indicator(self) -> float:
        """
        Calculate engagement drop indicator based on inactivity vs churn.
        
        Returns:
            Engagement drop score (0-100, higher = more concerning)
        """
        inactive_customers = len(self.df[self.df['IsActiveMember'] == 0])
        inactive_churn_rate = self.df[self.df['IsActiveMember'] == 0]['Exited'].mean()
        low_product_rate = len(self.df[self.df['NumOfProducts'] == 1]) / len(self.df)
        
        # Weighted score
        engagement_drop = (
            (inactive_customers / len(self.df)) * 40 +
            inactive_churn_rate * 40 +
            low_product_rate * 20
        ) * 100
        
        return round(engagement_drop, 2)
    
    def export_analysis_report(self, output_dir: Path = None) -> str:
        """
        Export comprehensive analysis report to CSV files.
        
        Args:
            output_dir: Directory to save reports
        
        Returns:
            Path to reports directory
        """
        if output_dir is None:
            output_dir = DATA_PROCESSED.parent / 'reports' / 'analysis'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export different analyses
        geo_analysis = self.analyze_churn_by_geography()
        geo_analysis.to_csv(output_dir / 'geographic_churn_analysis.csv')
        
        demo_analysis = self.analyze_churn_by_demographics()
        for key, df in demo_analysis.items():
            df.to_csv(output_dir / f'{key}_churn_analysis.csv')
        
        high_value = self.analyze_high_value_churn()
        high_value.to_csv(output_dir / 'high_value_churn_analysis.csv', index=False)
        
        engagement = self.analyze_churn_by_engagement()
        engagement.to_csv(output_dir / 'engagement_churn_analysis.csv', index=False)
        
        comparison = self.compare_churned_vs_retained()
        comparison.to_csv(output_dir / 'churned_vs_retained_comparison.csv')
        
        print(f"Analysis reports exported to {output_dir}")
        return str(output_dir)
