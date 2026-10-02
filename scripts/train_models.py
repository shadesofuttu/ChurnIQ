"""Training Script for Churn Models and Segmentation"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.data.loader import load_raw_data, save_processed_data
from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.segmentation import CustomerSegmentation
from src.models.churn_model import ChurnPredictor, compare_models
from src.analysis.churn_analytics import ChurnAnalytics
from src.visualization.plotting import ChurnVisualizer
from config.settings import *

def main():
    """Main training pipeline."""
    
    print("="*60)
    print("CUSTOMER CHURN ANALYTICS - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_raw_data()
    print(f"Loaded {len(df)} records")
    
    preprocessor = DataPreprocessor()
    
    # Check and handle missing values
    missing = preprocessor.check_missing_values(df)
    if len(missing) > 0:
        print(f"Found {len(missing)} columns with missing values")
        df = preprocessor.handle_missing_values(df, strategy='drop')
    
    print(f"Final dataset size: {len(df)} records")
    
    # 2. Feature engineering
    print("\n[2/6] Creating engineered features...")
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.create_all_features(df)
    print(f"Created {len(feature_engineer.feature_list)} new features")
    
    # 3. Customer segmentation
    print("\n[3/6] Performing customer segmentation...")
    segmenter = CustomerSegmentation(n_clusters=N_CLUSTERS)
    df_segmented = segmenter.fit_segments(df_engineered, method='standard')
    
    # Analyze segments
    segment_churn = segmenter.analyze_segment_churn(df_segmented)
    print("\nSegment Churn Rates:")
    print(segment_churn)
    
    # Save segmentation model
    segmenter.save_model()
    print("Segmentation model saved")
    
    # 4. Prepare data for churn prediction
    print("\n[4/6] Preparing data for churn prediction...")
    df_model = df_segmented.copy()
    
    # Encode categorical variables
    df_model = preprocessor.encode_categorical(
        df_model,
        columns=['Geography', 'Gender'],
        method='onehot'
    )
    
    # Drop non-numeric categorical columns
    categorical_to_drop = ['Age_Group', 'Balance_Category', 'Tenure_Group', 
                          'Credit_Category', 'Segment_Name']
    df_model = df_model.drop(
        columns=[col for col in categorical_to_drop if col in df_model.columns],
        errors='ignore'
    )
    
    # 5. Train churn prediction models
    print("\n[5/6] Training churn prediction models...")
    
    # Prepare features
    predictor = ChurnPredictor(model_type='random_forest')
    X, y = predictor.prepare_features(df_model)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Compare models
    print("\nComparing model performance...")
    comparison = compare_models(X_train, y_train, X_test, y_test)
    
    # Save comparison results
    comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    print("Model comparison saved")
    
    # Train best model
    best_model_type = comparison.loc[comparison['ROC AUC'].idxmax(), 'Model']
    print(f"\nBest model: {best_model_type}")
    
    best_model = ChurnPredictor(model_type=best_model_type)
    best_model.train(X_train, y_train, use_smote=True, tune_hyperparameters=False)
    metrics = best_model.evaluate(X_test, y_test)
    
    # Save best model
    best_model.save_model()
    print("Churn prediction model saved")
    
    # Feature importance
    feature_importance = best_model.get_feature_importance()
    if feature_importance is not None:
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
    
    # 6. Generate analytics and visualizations
    print("\n[6/6] Generating analytics and visualizations...")
    
    # Save processed data
    save_processed_data(df_segmented, 'cleaned_data.csv')
    
    # Generate analytics
    analytics = ChurnAnalytics(df_segmented)
    kpis = analytics.calculate_overall_kpis()
    insights = analytics.identify_churn_patterns()
    
    print("\nKey Performance Indicators:")
    for key, value in kpis.items():
        print(f"  {key}: {value}")
    
    print("\nKey Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    # Export reports
    report_path = analytics.export_analysis_report()
    print(f"\nAnalysis reports exported to: {report_path}")
    
    # Generate visualizations
    visualizer = ChurnVisualizer(df_segmented)
    visualizer.create_all_eda_plots()
    print("All visualizations generated")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the analysis reports in the 'reports' directory")
    print("2. Check visualizations in the 'figures' directory")
    print("3. Launch the Streamlit dashboard: streamlit run app/main.py")
    print("="*60)

if __name__ == "__main__":
    main()
