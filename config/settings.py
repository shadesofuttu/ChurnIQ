"""Project Configuration Settings"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_SEGMENTS = BASE_DIR / "data" / "segments"

# Model paths
MODELS_DIR = BASE_DIR / "models"
CHURN_MODEL = MODELS_DIR / "churn_model.pkl"
SCALER = MODELS_DIR / "scaler.pkl"
SEGMENTATION_MODEL = MODELS_DIR / "segmentation_model.pkl"
FEATURE_COLUMNS = MODELS_DIR / "feature_columns.json"

# Figure paths
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_EDA = FIGURES_DIR / "eda"
FIGURES_SEGMENTATION = FIGURES_DIR / "segmentation"
FIGURES_MODEL = FIGURES_DIR / "model"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Segmentation parameters
N_CLUSTERS = 4
RFM_WEIGHTS = {
    'recency': 0.3,
    'frequency': 0.3,
    'monetary': 0.4
}

# Feature groups
DEMOGRAPHIC_FEATURES = ['Geography', 'Gender', 'Age']
ACCOUNT_FEATURES = ['Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
FINANCIAL_FEATURES = ['CreditScore', 'EstimatedSalary']

# App settings
APP_TITLE = "ChurnIQ Dashboard"
APP_LAYOUT = "wide"
APP_THEME = "light"