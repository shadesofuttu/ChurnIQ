"""Visualization Functions for Churn Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import FIGURES_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ChurnVisualizer:
    """
    Create visualizations for churn analysis.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: Path = None):
        """
        Initialize visualizer.
        
        Args:
            df: DataFrame with customer data
            output_dir: Directory to save figures
        """
        self.df = df
        self.output_dir = output_dir or FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_churn_distribution(self, save: bool = True) -> go.Figure:
        """
        Plot overall churn distribution.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        churn_counts = self.df['Exited'].value_counts()
        churn_pct = (churn_counts / len(self.df) * 100).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Retained', 'Churned'],
                y=churn_counts.values[::-1],
                text=[f'{pct}%' for pct in churn_pct.values[::-1]],
                textposition='auto',
                marker_color=['#2ecc71', '#e74c3c']
            )
        ])
        
        fig.update_layout(
            title='Customer Churn Distribution',
            xaxis_title='Customer Status',
            yaxis_title='Number of Customers',
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'churn_distribution.html')
        
        return fig
    
    def plot_churn_by_geography(self, save: bool = True) -> go.Figure:
        """
        Plot churn rate by geography.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        geo_churn = self.df.groupby('Geography')['Exited'].agg(['sum', 'mean', 'count'])
        geo_churn['churn_rate'] = (geo_churn['mean'] * 100).round(2)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Churn Rate by Country', 'Customer Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart - Churn rate
        fig.add_trace(
            go.Bar(
                x=geo_churn.index,
                y=geo_churn['churn_rate'],
                text=geo_churn['churn_rate'].apply(lambda x: f'{x}%'),
                textposition='auto',
                marker_color='#3498db',
                name='Churn Rate'
            ),
            row=1, col=1
        )
        
        # Pie chart - Customer distribution
        fig.add_trace(
            go.Pie(
                labels=geo_churn.index,
                values=geo_churn['count'],
                hole=0.3,
                name='Customers'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Geographic Churn Analysis',
            showlegend=True,
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'churn_by_geography.html')
        
        return fig
    
    def plot_churn_by_demographics(self, save: bool = True) -> go.Figure:
        """
        Plot churn by age and gender.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        # Create age groups
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        self.df['Age_Group'] = pd.cut(self.df['Age'], bins=age_bins, labels=age_labels)
        
        # Calculate churn rates
        age_churn = self.df.groupby('Age_Group', observed=False)['Exited'].mean() * 100
        gender_churn = self.df.groupby('Gender')['Exited'].mean() * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Churn Rate by Age Group', 'Churn Rate by Gender')
        )
        
        # Age groups
        fig.add_trace(
            go.Bar(
                x=age_churn.index.astype(str),
                y=age_churn.values,
                text=age_churn.values.round(2),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color='#9b59b6',
                name='Age'
            ),
            row=1, col=1
        )
        
        # Gender
        fig.add_trace(
            go.Bar(
                x=gender_churn.index,
                y=gender_churn.values,
                text=gender_churn.values.round(2),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color='#e67e22',
                name='Gender'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Demographic Churn Analysis',
            showlegend=False,
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'churn_by_demographics.html')
        
        return fig
    
    def plot_churn_by_tenure(self, save: bool = True) -> go.Figure:
        """
        Plot churn rate by customer tenure.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        tenure_churn = self.df.groupby('Tenure').agg({
            'Exited': ['mean', 'count']
        })
        tenure_churn.columns = ['churn_rate', 'count']
        tenure_churn['churn_rate'] = tenure_churn['churn_rate'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=tenure_churn.index,
            y=tenure_churn['churn_rate'],
            mode='lines+markers',
            name='Churn Rate',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Churn Rate by Customer Tenure',
            xaxis_title='Tenure (Years)',
            yaxis_title='Churn Rate (%)',
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'churn_by_tenure.html')
        
        return fig
    
    def plot_balance_vs_churn(self, save: bool = True) -> go.Figure:
        """
        Plot balance distribution for churned vs retained customers.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        churned = self.df[self.df['Exited'] == 1]['Balance']
        retained = self.df[self.df['Exited'] == 0]['Balance']
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=churned,
            name='Churned',
            opacity=0.7,
            marker_color='#e74c3c',
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=retained,
            name='Retained',
            opacity=0.7,
            marker_color='#2ecc71',
            nbinsx=50
        ))
        
        fig.update_layout(
            title='Account Balance Distribution: Churned vs Retained',
            xaxis_title='Account Balance',
            yaxis_title='Number of Customers',
            barmode='overlay',
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'balance_vs_churn.html')
        
        return fig
    
    def plot_products_vs_churn(self, save: bool = True) -> go.Figure:
        """
        Plot churn rate by number of products.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        products_churn = self.df.groupby('NumOfProducts').agg({
            'Exited': ['mean', 'count']
        })
        products_churn.columns = ['churn_rate', 'count']
        products_churn['churn_rate'] = products_churn['churn_rate'] * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=products_churn.index,
            y=products_churn['churn_rate'],
            text=products_churn['churn_rate'].round(2),
            texttemplate='%{text}%',
            textposition='auto',
            marker_color='#16a085',
            name='Churn Rate'
        ))
        
        fig.update_layout(
            title='Churn Rate by Number of Products',
            xaxis_title='Number of Products',
            yaxis_title='Churn Rate (%)',
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'products_vs_churn.html')
        
        return fig
    
    def plot_engagement_analysis(self, save: bool = True) -> go.Figure:
        """
        Plot churn by engagement indicators.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        # Active member analysis
        active_churn = self.df.groupby('IsActiveMember')['Exited'].mean() * 100
        
        # Credit card analysis
        card_churn = self.df.groupby('HasCrCard')['Exited'].mean() * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Active Member Status', 'Credit Card Ownership')
        )
        
        # Active member
        fig.add_trace(
            go.Bar(
                x=['Inactive', 'Active'],
                y=active_churn.values,
                text=active_churn.values.round(2),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color=['#e74c3c', '#2ecc71'],
                name='Active Status'
            ),
            row=1, col=1
        )
        
        # Credit card
        fig.add_trace(
            go.Bar(
                x=['No Card', 'Has Card'],
                y=card_churn.values,
                text=card_churn.values.round(2),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color=['#e67e22', '#3498db'],
                name='Card Status'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Customer Engagement vs Churn',
            showlegend=False,
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'engagement_analysis.html')
        
        return fig
    
    def plot_correlation_heatmap(self, save: bool = True) -> go.Figure:
        """
        Plot correlation heatmap of features with churn.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_with_churn = self.df[numeric_cols].corrwith(self.df['Exited']).sort_values(ascending=False)
        
        fig = go.Figure(data=go.Bar(
            x=corr_with_churn.values,
            y=corr_with_churn.index,
            orientation='h',
            marker=dict(
                color=corr_with_churn.values,
                colorscale='RdYlGn_r',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title='Feature Correlation with Churn',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Features',
            template='plotly_white',
            height=600
        )
        
        if save:
            fig.write_html(self.output_dir / 'eda' / 'correlation_heatmap.html')
        
        return fig
    
    def plot_segment_distribution(self, segment_col: str = 'Segment_Name', save: bool = True) -> go.Figure:
        """
        Plot customer segment distribution.
        
        Args:
            segment_col: Name of segment column
            save: Whether to save the figure
        
        Returns:
            Plotly figure
        """
        if segment_col not in self.df.columns:
            print(f"Warning: '{segment_col}' column not found")
            return None
        
        segment_counts = self.df[segment_col].value_counts()
        segment_churn = self.df.groupby(segment_col)['Exited'].mean() * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Segment Distribution', 'Churn Rate by Segment'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=segment_counts.index,
                values=segment_counts.values,
                hole=0.3
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=segment_churn.index,
                y=segment_churn.values,
                text=segment_churn.values.round(2),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color='#8e44ad'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Customer Segmentation Analysis',
            showlegend=True,
            template='plotly_white',
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'segmentation' / 'segment_distribution.html')
        
        return fig
    
    def create_all_eda_plots(self) -> None:
        """Create and save all EDA visualizations."""
        print("Generating EDA visualizations...")
        
        (self.output_dir / 'eda').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'segmentation').mkdir(parents=True, exist_ok=True)
        
        self.plot_churn_distribution()
        self.plot_churn_by_geography()
        self.plot_churn_by_demographics()
        self.plot_churn_by_tenure()
        self.plot_balance_vs_churn()
        self.plot_products_vs_churn()
        self.plot_engagement_analysis()
        self.plot_correlation_heatmap()
        
        print(f"All visualizations saved to {self.output_dir}")
