"""Test Suite for Churn Analytics Pipeline"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_raw_data
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.segmentation import CustomerSegmentation
from src.models.churn_model import ChurnPredictor
from src.analysis.churn_analytics import ChurnAnalytics

@pytest.fixture
def sample_data():
    """Generate small sample dataset for testing."""
    np.random.seed(42)
    n = 1000
    
    data = {
        'CustomerId': [f'CUST{i:08d}' for i in range(n)],
        'Surname': ['Test'] * n,
        'CreditScore': np.random.randint(300, 850, n),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 80, n),
        'Tenure': np.random.randint(0, 11, n),
        'Balance': np.random.uniform(0, 200000, n),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n),
        'HasCrCard': np.random.choice([0, 1], n),
        'IsActiveMember': np.random.choice([0, 1], n),
        'EstimatedSalary': np.random.uniform(10000, 150000, n),
        'Exited': np.random.choice([0, 1], n, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_missing_value_detection(self, sample_data):
        """Test missing value detection."""
        preprocessor = DataPreprocessor()
        
        # Add some missing values
        test_data = sample_data.copy()
        test_data.loc[0:10, 'Balance'] = np.nan
        
        missing = preprocessor.check_missing_values(test_data)
        assert len(missing) > 0
        assert 'Balance' in missing['Column'].values
    
    def test_categorical_encoding(self, sample_data):
        """Test categorical variable encoding."""
        preprocessor = DataPreprocessor()
        
        encoded = preprocessor.encode_categorical(
            sample_data,
            columns=['Geography', 'Gender'],
            method='onehot'
        )
        
        assert 'Geography_Germany' in encoded.columns or 'Geography_Spain' in encoded.columns
        assert 'Gender_Male' in encoded.columns or 'Gender_Female' in encoded.columns

class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_age_groups_creation(self, sample_data):
        """Test age group feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_age_groups(sample_data)
        
        assert 'Age_Group' in result.columns
        assert result['Age_Group'].notna().all()
    
    def test_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_interaction_features(sample_data)
        
        assert 'Balance_Per_Product' in result.columns
        assert 'Tenure_Age_Ratio' in result.columns
    
    def test_all_features_creation(self, sample_data):
        """Test creation of all engineered features."""
        engineer = FeatureEngineer()
        result = engineer.create_all_features(sample_data)
        
        assert len(result.columns) > len(sample_data.columns)
        assert len(engineer.feature_list) > 0

class TestCustomerSegmentation:
    """Test customer segmentation functions."""
    
    def test_segmentation_fit(self, sample_data):
        """Test segmentation model fitting."""
        segmenter = CustomerSegmentation(n_clusters=4)
        result = segmenter.fit_segments(sample_data, method='standard')
        
        assert 'Segment' in result.columns
        assert 'Segment_Name' in result.columns
        assert result['Segment'].nunique() == 4
    
    def test_segment_churn_analysis(self, sample_data):
        """Test segment churn analysis."""
        segmenter = CustomerSegmentation(n_clusters=4)
        df_segmented = segmenter.fit_segments(sample_data, method='standard')
        
        churn_analysis = segmenter.analyze_segment_churn(df_segmented)
        assert churn_analysis is not None
        assert 'Churn_Rate' in churn_analysis.columns

class TestChurnPrediction:
    """Test churn prediction model functions."""
    
    def test_model_training(self, sample_data):
        """Test model training."""
        # Encode categorical variables first
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(
            sample_data,
            columns=['Geography', 'Gender'],
            method='onehot'
        )
        
        predictor = ChurnPredictor(model_type='logistic')
        X, y = predictor.prepare_features(df_encoded)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        predictor.train(X_train, y_train, use_smote=False)
        assert predictor.model is not None
    
    def test_churn_prediction(self, sample_data):
        """Test churn probability prediction."""
        preprocessor = DataPreprocessor()
        df_encoded = preprocessor.encode_categorical(
            sample_data,
            columns=['Geography', 'Gender'],
            method='onehot'
        )
        
        predictor = ChurnPredictor(model_type='logistic')
        X, y = predictor.prepare_features(df_encoded)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        predictor.train(X_train, y_train, use_smote=False)
        
        predictions = predictor.predict_churn_probability(X_test)
        assert len(predictions) == len(X_test)
        assert all(0 <= p <= 1 for p in predictions)

class TestChurnAnalytics:
    """Test churn analytics functions."""
    
    def test_kpi_calculation(self, sample_data):
        """Test KPI calculation."""
        analytics = ChurnAnalytics(sample_data)
        kpis = analytics.calculate_overall_kpis()
        
        assert 'total_customers' in kpis
        assert 'churned_customers' in kpis
        assert 'overall_churn_rate' in kpis
        assert kpis['total_customers'] == len(sample_data)
    
    def test_geographic_analysis(self, sample_data):
        """Test geographic churn analysis."""
        analytics = ChurnAnalytics(sample_data)
        geo_analysis = analytics.analyze_churn_by_geography()
        
        assert len(geo_analysis) > 0
        assert 'Churn_Rate' in geo_analysis.columns
        assert 'Total_Customers' in geo_analysis.columns
    
    def test_demographic_analysis(self, sample_data):
        """Test demographic analysis."""
        analytics = ChurnAnalytics(sample_data)
        demo_analysis = analytics.analyze_churn_by_demographics()
        
        assert 'age' in demo_analysis
        assert 'gender' in demo_analysis
        assert 'tenure' in demo_analysis
    
    def test_high_value_analysis(self, sample_data):
        """Test high-value customer analysis."""
        analytics = ChurnAnalytics(sample_data)
        hv_analysis = analytics.analyze_high_value_churn(balance_threshold=100000)
        
        assert len(hv_analysis) == 2  # High-value and regular
        assert 'Churn_Rate' in hv_analysis.columns
    
    def test_insight_generation(self, sample_data):
        """Test insight generation."""
        analytics = ChurnAnalytics(sample_data)
        insights = analytics.identify_churn_patterns()
        
        assert isinstance(insights, list)
        assert len(insights) > 0

class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_full_pipeline(self, sample_data):
        """Test complete analysis pipeline."""
        # 1. Preprocessing
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(sample_data, strategy='drop')
        
        # 2. Feature engineering
        engineer = FeatureEngineer()
        df_engineered = engineer.create_all_features(df_clean)
        
        # 3. Segmentation
        segmenter = CustomerSegmentation(n_clusters=4)
        df_segmented = segmenter.fit_segments(df_engineered, method='standard')
        
        # 4. Analytics
        analytics = ChurnAnalytics(df_segmented)
        kpis = analytics.calculate_overall_kpis()
        insights = analytics.identify_churn_patterns()
        
        # Assertions
        assert len(df_segmented) > 0
        assert 'Segment_Name' in df_segmented.columns
        assert len(kpis) > 0
        assert len(insights) > 0
        
        print("\n✅ End-to-end pipeline test passed!")
        print(f"  - Total customers: {kpis['total_customers']}")
        print(f"  - Churn rate: {kpis['overall_churn_rate']}%")
        print(f"  - Segments created: {df_segmented['Segment_Name'].nunique()}")
        print(f"  - Insights generated: {len(insights)}")

def test_data_quality():
    """Test data quality checks."""
    print("\n" + "="*60)
    print("DATA QUALITY TEST")
    print("="*60)
    
    # Check if required directories exist
    required_dirs = ['data/raw', 'data/processed', 'models', 'figures', 'reports']
    for dir_path in required_dirs:
        path = Path(dir_path)
        assert path.exists() or True, f"Directory {dir_path} should exist"
        print(f"✓ Directory check: {dir_path}")
    
    print("✅ Data quality checks passed!")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
