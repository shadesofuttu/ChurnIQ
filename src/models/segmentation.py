"""Customer Segmentation Analysis"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, RANDOM_STATE, N_CLUSTERS

class CustomerSegmentation:
    """
    Perform customer segmentation using clustering techniques.
    """
    
    def __init__(self, n_clusters: int = N_CLUSTERS):
        """
        Initialize customer segmentation.
        
        Args:
            n_clusters: Number of segments to create
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.scaler = StandardScaler()
        self.pca = None
        self.segment_profiles = None
        self.feature_columns = None
    
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM-style features for banking data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with RFM features
        """
        df_rfm = df.copy()
        
        # Recency: Tenure (inverse - newer customers have lower tenure)
        df_rfm['Recency'] = df_rfm['Tenure'].max() - df_rfm['Tenure']
        
        # Frequency: Number of products
        df_rfm['Frequency'] = df_rfm['NumOfProducts']
        
        # Monetary: Balance + Estimated Salary proxy
        df_rfm['Monetary'] = df_rfm['Balance'] + (df_rfm['EstimatedSalary'] * 0.1)
        
        return df_rfm
    
    def prepare_segmentation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for segmentation.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with segmentation features
        """
        feature_cols = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
            'EstimatedSalary'
        ]
        
        # Add geography if available
        if 'Geography' in df.columns:
            df_encoded = pd.get_dummies(df[['Geography']], drop_first=True)
            feature_cols.extend(df_encoded.columns.tolist())
            df = pd.concat([df, df_encoded], axis=1)
        
        # Store feature columns
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        return df[self.feature_columns]
    
    def fit_segments(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Fit segmentation model and assign segments.
        
        Args:
            df: Input DataFrame with all features
            method: 'standard' or 'rfm'
        
        Returns:
            DataFrame with segment assignments
        """
        if method == 'rfm':
            df_with_rfm = self.create_rfm_features(df)
            X = df_with_rfm[['Recency', 'Frequency', 'Monetary']]
            self.feature_columns = ['Recency', 'Frequency', 'Monetary']
        else:
            X = self.prepare_segmentation_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        print(f"Fitting {self.n_clusters} segments using {method} method...")
        self.kmeans.fit(X_scaled)
        
        # Assign segments
        df['Segment'] = self.kmeans.labels_
        
        # Create segment profiles
        self.segment_profiles = self._create_segment_profiles(df)
        
        # Name segments based on characteristics
        segment_names = self._name_segments(self.segment_profiles)
        df['Segment_Name'] = df['Segment'].map(segment_names)
        
        print(f"Segmentation complete. Distribution:")
        print(df['Segment_Name'].value_counts())
        
        return df
    
    def _create_segment_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create profiles for each segment.
        
        Args:
            df: DataFrame with segment assignments
        
        Returns:
            DataFrame with segment profiles
        """
        profile_features = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'IsActiveMember', 'HasCrCard',
            'EstimatedSalary', 'Exited'
        ]
        
        available_features = [f for f in profile_features if f in df.columns]
        
        profiles = df.groupby('Segment')[available_features].agg([
            'mean', 'median', 'std'
        ]).round(2)
        
        # Add segment size
        profiles['Size'] = df.groupby('Segment').size()
        
        return profiles
    
    def _name_segments(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """
        Assign meaningful names to segments based on characteristics.
        
        Args:
            profiles: Segment profiles DataFrame
        
        Returns:
            Dictionary mapping segment ID to name
        """
        segment_names = {}
        
        # Extract key metrics
        avg_balance = profiles[('Balance', 'mean')]
        avg_products = profiles[('NumOfProducts', 'mean')]
        avg_active = profiles[('IsActiveMember', 'mean')]
        churn_rate = profiles[('Exited', 'mean')] if ('Exited', 'mean') in profiles.columns else None
        
        for segment in profiles.index:
            balance = avg_balance[segment]
            products = avg_products[segment]
            active = avg_active[segment]
            churn = churn_rate[segment] if churn_rate is not None else 0
            
            # Naming logic
            if balance > avg_balance.median() and products >= 2 and active > 0.7:
                name = "High-Value Engaged"
            elif balance > avg_balance.median() and active < 0.5:
                name = "High-Value At-Risk"
            elif balance <= avg_balance.quantile(0.25) and products == 1:
                name = "Low-Engagement"
            elif active > 0.7 and products >= 2:
                name = "Loyal Multi-Product"
            else:
                name = f"Standard Segment {segment}"
            
            segment_names[segment] = name
        
        return segment_names
    
    def predict_segment(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict segment for new customers.
        
        Args:
            df: DataFrame with customer features
        
        Returns:
            Array of segment predictions
        """
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def analyze_segment_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze churn rate by segment.
        
        Args:
            df: DataFrame with segments and churn labels
        
        Returns:
            DataFrame with churn analysis by segment
        """
        if 'Exited' not in df.columns:
            print("Warning: 'Exited' column not found for churn analysis")
            return None
        
        churn_analysis = df.groupby(['Segment_Name']).agg({
            'Exited': ['sum', 'mean', 'count']
        }).round(4)
        
        churn_analysis.columns = ['Churned_Count', 'Churn_Rate', 'Total_Customers']
        churn_analysis = churn_analysis.sort_values('Churn_Rate', ascending=False)
        
        print("\n=== Churn Rate by Segment ===")
        print(churn_analysis)
        
        return churn_analysis
    
    def analyze_segment_by_geography(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze segment distribution by geography.
        
        Args:
            df: DataFrame with segments and geography
        
        Returns:
            Cross-tabulation of segments and geography
        """
        if 'Geography' not in df.columns:
            print("Warning: 'Geography' column not found")
            return None
        
        geo_segment = pd.crosstab(
            df['Geography'], 
            df['Segment_Name'],
            normalize='index'
        ).round(3) * 100
        
        print("\n=== Segment Distribution by Geography (%) ===")
        print(geo_segment)
        
        return geo_segment
    
    def get_segment_recommendations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate actionable recommendations for each segment.
        
        Args:
            df: DataFrame with segments
        
        Returns:
            Dictionary of recommendations by segment
        """
        recommendations = {}
        
        for segment_name in df['Segment_Name'].unique():
            segment_data = df[df['Segment_Name'] == segment_name]
            
            recs = []
            
            # Analyze characteristics
            avg_balance = segment_data['Balance'].mean()
            avg_products = segment_data['NumOfProducts'].mean()
            active_pct = segment_data['IsActiveMember'].mean()
            churn_rate = segment_data['Exited'].mean() if 'Exited' in segment_data.columns else 0
            
            # Generate recommendations
            if churn_rate > 0.25:
                recs.append("High churn risk - implement retention campaigns")
            
            if avg_products < 1.5:
                recs.append("Low product adoption - cross-sell opportunities")
            
            if active_pct < 0.5:
                recs.append("Low engagement - re-engagement campaigns needed")
            
            if avg_balance > 100000:
                recs.append("High-value segment - premium service offerings")
            
            if not recs:
                recs.append("Stable segment - maintain current service level")
            
            recommendations[segment_name] = recs
        
        return recommendations
    
    def visualize_segments_pca(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce dimensions using PCA for visualization.
        
        Args:
            df: DataFrame with segmentation features
        
        Returns:
            Tuple of (PCA components, explained variance)
        """
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca, self.pca.explained_variance_ratio_
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save segmentation model.
        
        Args:
            filepath: Path to save model
        """
        if filepath is None:
            filepath = MODELS_DIR / "segmentation_model.pkl"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'feature_columns': self.feature_columns,
            'segment_profiles': self.segment_profiles
        }
        
        joblib.dump(model_data, filepath)
        print(f"Segmentation model saved to {filepath}")
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load segmentation model.
        
        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = MODELS_DIR / "segmentation_model.pkl"
        
        model_data = joblib.load(filepath)
        
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.n_clusters = model_data['n_clusters']
        self.feature_columns = model_data['feature_columns']
        self.segment_profiles = model_data['segment_profiles']
        
        print(f"Segmentation model loaded from {filepath}")

def perform_optimal_clustering(df: pd.DataFrame, max_clusters: int = 10) -> pd.DataFrame:
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        df: Input DataFrame
        max_clusters: Maximum number of clusters to test
    
    Returns:
        DataFrame with inertia scores
    """
    segmenter = CustomerSegmentation(n_clusters=2)
    X = segmenter.prepare_segmentation_features(df)
    X_scaled = segmenter.scaler.fit_transform(X)
    
    inertias = []
    cluster_range = range(2, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    results = pd.DataFrame({
        'n_clusters': list(cluster_range),
        'inertia': inertias
    })
    
    print("\n=== Elbow Method Results ===")
    print(results)
    
    return results
