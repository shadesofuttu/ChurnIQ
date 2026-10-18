"""Check if all required files exist for deployment"""

import sys
from pathlib import Path

print("=" * 60)
print("DEPLOYMENT ENVIRONMENT CHECK")
print("=" * 60)

# Check current directory
print(f"\nCurrent directory: {Path.cwd()}")

# Check if models exist
models_dir = Path("models")
print(f"\nModels directory exists: {models_dir.exists()}")

if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    print(f"Model files found: {len(model_files)}")
    for f in model_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")

# Check if data exists
data_dir = Path("data/processed")
print(f"\nProcessed data directory exists: {data_dir.exists()}")

if data_dir.exists():
    data_files = list(data_dir.glob("*.csv"))
    print(f"Data files found: {len(data_files)}")
    for f in data_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")

# Check if feature columns exist
feature_file = models_dir / "feature_columns.txt"
print(f"\nFeature columns file exists: {feature_file.exists()}")

# Try loading model
print("\nAttempting to load model...")
try:
    sys.path.append(str(Path.cwd()))
    from src.models.churn_model import ChurnPredictor
    
    predictor = ChurnPredictor('gradient_boosting')
    predictor.load_model()
    print("[SUCCESS] Model loaded!")
    print(f"Feature columns: {len(predictor.feature_columns)}")
except Exception as e:
    print(f"[FAILED] {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
"""