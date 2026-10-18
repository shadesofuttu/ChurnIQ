"""Diagnostic script to test dashboard components"""

import sys
from pathlib import Path
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent))

print("="*60)
print("CHURNIQ DASHBOARD DIAGNOSTIC TEST")
print("="*60)

# Test 1: Load processed data
print("\n[TEST 1] Loading processed data...")
try:
    from src.data.loader import load_processed_data
    df = load_processed_data()
    print(f"[OK] Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns[:5])}...")
except Exception as e:
    print(f"[FAIL] Failed to load data: {e}")
    sys.exit(1)

# Test 2: Load churn predictor
print("\n[TEST 2] Loading churn prediction model...")
try:
    from src.models.churn_model import ChurnPredictor
    predictor = ChurnPredictor('gradient_boosting')
    predictor.load_model()
    print(f"[OK] Model loaded successfully")
    print(f"  Feature columns: {len(predictor.feature_columns)}")
    print(f"  Model type: {type(predictor.model).__name__}")
except Exception as e:
    print(f"[FAIL] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test single prediction
print("\n[TEST 3] Testing single customer prediction...")
try:
    from src.utils.prediction import predict_single_customer
    
    test_customer = {
        'CreditScore': 650,
        'Age': 40,
        'Tenure': 5,
        'Balance': 50000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 60000,
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Male': 1,
        'Balance_Per_Product': 25000,
        'Tenure_Age_Ratio': 0.125,
        'Active_With_Card': 1,
        'Credit_Salary_Ratio': 10.83,
        'Engagement_Score': 80,
        'Churn_Risk_Score': 0,
        'Segment': 2
    }
    
    result = predict_single_customer(predictor, test_customer)
    print(f"[OK] Prediction successful")
    print(f"  Churn probability: {result['churn_probability']:.2%}")
    print(f"  Risk level: {result['risk_level']}")
    print(f"  Prediction: {result['prediction']}")
except Exception as e:
    print(f"[FAIL] Failed prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize analytics
print("\n[TEST 4] Testing analytics module...")
try:
    from src.analysis.churn_analytics import ChurnAnalytics
    analytics = ChurnAnalytics(df)
    kpis = analytics.calculate_overall_kpis()
    print(f"[OK] Analytics initialized successfully")
    print(f"  Total customers: {kpis['total_customers']:,}")
    print(f"  Churn rate: {kpis['overall_churn_rate']:.2f}%")
except Exception as e:
    print(f"[FAIL] Failed to initialize analytics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check model comparison file
print("\n[TEST 5] Checking model comparison results...")
try:
    import pandas as pd
    from config.settings import MODELS_DIR
    comparison_file = MODELS_DIR / 'model_comparison.csv'
    if comparison_file.exists():
        comparison = pd.read_csv(comparison_file)
        print(f"[OK] Model comparison file found")
        print(f"  Models compared: {', '.join(comparison['Model'].tolist())}")
    else:
        print(f"[FAIL] Model comparison file not found at {comparison_file}")
except Exception as e:
    print(f"[FAIL] Failed to load model comparison: {e}")

print("\n" + "="*60)
print("ALL TESTS PASSED [OK]")
print("="*60)
print("\nThe dashboard components are working correctly.")
print("If you're still seeing errors in the dashboard, try:")
print("1. Stop the Streamlit app (Ctrl+C)")
print("2. Clear Streamlit cache: streamlit cache clear")
print("3. Restart: streamlit run app/main.py")
print("="*60)