"""Churn Prediction Utilities for Dashboard"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.churn_model import ChurnPredictor
from config.settings import MODELS_DIR

def load_churn_predictor(model_type: str = 'gradient_boosting'):
    """
    Load the trained churn prediction model.
    
    Args:
        model_type: Type of model to load
    
    Returns:
        Loaded ChurnPredictor instance
    """
    predictor = ChurnPredictor(model_type=model_type)
    try:
        predictor.load_model()
        return predictor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single_customer(predictor: ChurnPredictor, customer_data: dict) -> dict:
    """
    Predict churn for a single customer.
    
    Args:
        predictor: Trained ChurnPredictor instance
        customer_data: Dictionary with customer features
    
    Returns:
        Dictionary with prediction results
    """
    # Create DataFrame from customer data
    df = pd.DataFrame([customer_data])
    
    # Ensure all required features are present
    required_features = predictor.feature_columns
    
    # Add missing features with default values
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only required features in correct order
    df = df[required_features]
    
    # Get prediction
    churn_probability = predictor.predict_churn_probability(df)[0]
    prediction = 1 if churn_probability >= 0.5 else 0
    
    # Determine risk level
    if churn_probability >= 0.7:
        risk_level = "High Risk"
        risk_color = "red"
    elif churn_probability >= 0.4:
        risk_level = "Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "Low Risk"
        risk_color = "green"
    
    # Get feature importance for this customer
    feature_importance = predictor.get_feature_importance()
    
    # Get top risk factors (features with high importance)
    top_risk_factors = []
    if feature_importance is not None:
        top_features = feature_importance.head(5)
        for _, row in top_features.iterrows():
            feature_name = row['Feature']
            importance = row['Importance']
            if feature_name in customer_data:
                value = customer_data[feature_name]
                top_risk_factors.append({
                    'factor': feature_name,
                    'value': value,
                    'importance': importance
                })
    
    return {
        'churn_probability': churn_probability,
        'prediction': 'Will Churn' if prediction == 1 else 'Will Retain',
        'risk_level': risk_level,
        'risk_color': risk_color,
        'top_risk_factors': top_risk_factors
    }

def get_feature_requirements() -> dict:
    """
    Get the required features and their descriptions for prediction.
    
    Returns:
        Dictionary of features with descriptions and ranges
    """
    return {
        'CreditScore': {
            'description': 'Credit Score',
            'min': 300,
            'max': 850,
            'default': 650
        },
        'Age': {
            'description': 'Customer Age',
            'min': 18,
            'max': 100,
            'default': 40
        },
        'Tenure': {
            'description': 'Years with Bank',
            'min': 0,
            'max': 10,
            'default': 5
        },
        'Balance': {
            'description': 'Account Balance (€)',
            'min': 0,
            'max': 250000,
            'default': 50000
        },
        'NumOfProducts': {
            'description': 'Number of Products',
            'min': 1,
            'max': 4,
            'default': 1
        },
        'HasCrCard': {
            'description': 'Has Credit Card',
            'options': [0, 1],
            'default': 1
        },
        'IsActiveMember': {
            'description': 'Active Member',
            'options': [0, 1],
            'default': 1
        },
        'EstimatedSalary': {
            'description': 'Estimated Salary (€)',
            'min': 10000,
            'max': 200000,
            'default': 60000
        },
        'Geography_Germany': {
            'description': 'From Germany',
            'options': [0, 1],
            'default': 0
        },
        'Geography_Spain': {
            'description': 'From Spain',
            'options': [0, 1],
            'default': 0
        },
        'Gender_Male': {
            'description': 'Gender: Male',
            'options': [0, 1],
            'default': 1
        }
    }

def format_feature_name(feature: str) -> str:
    """
    Format feature name for display.
    
    Args:
        feature: Raw feature name
    
    Returns:
        Formatted feature name
    """
    # Replace underscores with spaces
    formatted = feature.replace('_', ' ')
    
    # Capitalize each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    return formatted
