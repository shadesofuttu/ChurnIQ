# Quick Start Guide

## Customer Segmentation & Churn Analytics - European Banking

### 🚀 Get Started in 5 Minutes

---

## Step 1: Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd customer-churn-analytics

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Prepare Your Data

### Place your data file

Put your banking customer CSV file in `data/raw/` directory:

```bash
data/raw/bank_churn.csv
```

### Required columns:
- CustomerId
- Surname
- CreditScore
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (0 or 1)

### Sample data format:
```csv
CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
15634602,Hargrave,619,France,Female,42,2,0.00,1,1,1,101348.88,1
15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
```

---

## Step 3: Train Models and Generate Analysis

### Option A: Run the complete pipeline

```bash
python scripts/train_models.py
```

This script will:
- ✅ Load and preprocess data
- ✅ Create engineered features
- ✅ Perform customer segmentation
- ✅ Train churn prediction models
- ✅ Generate analysis reports
- ✅ Create visualizations

**Expected runtime:** 5-10 minutes

### Option B: Use the Jupyter notebook

```bash
jupyter notebook notebooks/01_comprehensive_churn_analysis.ipynb
```

Step through the analysis interactively.

---

## Step 4: Launch the Dashboard

```bash
streamlit run app/main.py
```

**The dashboard will open automatically at:** http://localhost:8501

### Dashboard Features:
- 📊 Interactive KPI metrics
- 🌍 Geographic churn analysis
- 👥 Demographic breakdowns
- 💎 High-value customer analysis
- 🎯 Customer segmentation
- 📥 Export capabilities

---

## Step 5: Review Outputs

### Generated Files:

1. **Processed Data:**
   - `data/processed/cleaned_data.csv` - Clean dataset with segments

2. **Models:**
   - `models/churn_model_*.pkl` - Trained prediction model
   - `models/segmentation_model.pkl` - Customer segmentation
   - `models/scaler.pkl` - Feature scaler

3. **Reports:**
   - `reports/research_paper.md` - Comprehensive analysis
   - `reports/executive_summary.md` - Executive summary
   - `reports/analysis/*.csv` - Detailed analytics

4. **Visualizations:**
   - `figures/eda/*.html` - Exploratory plots
   - `figures/segmentation/*.html` - Segmentation charts

---

## Common Use Cases

### Use Case 1: Analyze Churn Patterns

```python
from src.data.loader import load_processed_data
from src.analysis.churn_analytics import ChurnAnalytics

# Load data
df = load_processed_data()

# Create analytics instance
analytics = ChurnAnalytics(df)

# Get KPIs
kpis = analytics.calculate_overall_kpis()
print(kpis)

# Analyze by geography
geo_analysis = analytics.analyze_churn_by_geography()
print(geo_analysis)

# Get insights
insights = analytics.identify_churn_patterns()
for insight in insights:
    print(insight)
```

### Use Case 2: Predict Churn for New Customers

```python
from src.models.churn_model import ChurnPredictor
import pandas as pd

# Load trained model
predictor = ChurnPredictor(model_type='gradient_boosting')
predictor.load_model()

# Prepare new customer data
new_customers = pd.DataFrame({
    'CreditScore': [650, 720],
    'Age': [35, 45],
    'Tenure': [2, 5],
    'Balance': [50000, 120000],
    'NumOfProducts': [1, 2],
    'HasCrCard': [1, 1],
    'IsActiveMember': [0, 1],
    'EstimatedSalary': [60000, 95000],
    'Geography_Germany': [1, 0],
    'Geography_Spain': [0, 1],
    'Gender_Male': [1, 0]
})

# Predict churn probability
churn_probs = predictor.predict_churn_probability(new_customers)
print(f"Churn probabilities: {churn_probs}")
```

### Use Case 3: Segment Customers

```python
from src.models.segmentation import CustomerSegmentation

# Create segmentation instance
segmenter = CustomerSegmentation(n_clusters=4)

# Fit and assign segments
df_segmented = segmenter.fit_segments(df, method='standard')

# Analyze segments
segment_churn = segmenter.analyze_segment_churn(df_segmented)
print(segment_churn)

# Get recommendations
recommendations = segmenter.get_segment_recommendations(df_segmented)
for segment, recs in recommendations.items():
    print(f"\n{segment}:")
    for rec in recs:
        print(f"  - {rec}")
```

---

## Troubleshooting

### Issue: "Data file not found"

**Solution:** Ensure your CSV file is in `data/raw/bank_churn.csv`

```bash
# Check if file exists
ls data/raw/
```

### Issue: "Module not found"

**Solution:** Reinstall requirements

```bash
pip install -r requirements.txt --upgrade
```

### Issue: Dashboard not loading

**Solution:** Check if models are trained

```bash
# Train models first
python scripts/train_models.py

# Then launch dashboard
streamlit run app/main.py
```

### Issue: "No processed data found"

**Solution:** Run the training pipeline or notebook first to generate processed data

---

## Next Steps

### For Analysts:
1. 📊 Explore the Jupyter notebook for detailed analysis
2. 📈 Customize visualizations in `src/visualization/plotting.py`
3. 🔍 Add custom analytics in `src/analysis/`

### For Data Scientists:
1. 🤖 Experiment with different models in `src/models/churn_model.py`
2. 🎯 Refine segmentation in `src/models/segmentation.py`
3. ⚙️ Tune hyperparameters in config/settings.py

### For Business Users:
1. 💼 Use the Streamlit dashboard for interactive analysis
2. 📄 Review the executive summary for strategic insights
3. 📊 Export reports for presentations

---

## Configuration

### Key settings in `config/settings.py`:

```python
# Model parameters
TEST_SIZE = 0.2          # Train/test split
RANDOM_STATE = 42        # Reproducibility seed
CV_FOLDS = 5             # Cross-validation folds

# Segmentation
N_CLUSTERS = 4           # Number of customer segments

# Paths (auto-configured)
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "figures"
```

---

## Best Practices

### ✅ Do:
- Keep data in the designated folders
- Run the training pipeline before using the dashboard
- Review the research paper for methodology details
- Use version control for code changes
- Document your analysis and findings

### ❌ Don't:
- Commit sensitive data files to version control
- Modify config settings without understanding impact
- Skip data validation steps
- Deploy models without proper testing

---

## Resources

### Documentation:
- **README.md** - Full project documentation
- **reports/research_paper.md** - Detailed methodology and findings
- **reports/executive_summary.md** - Strategic overview

### Support:
- Check existing issues in the repository
- Review code comments for implementation details
- Consult the research paper for analytical questions

---

## Example Workflow

### Complete Analysis Workflow:

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Place data file
cp /path/to/your/data.csv data/raw/bank_churn.csv

# 3. Run analysis pipeline
python scripts/train_models.py

# 4. Launch dashboard
streamlit run app/main.py

# 5. Open browser to http://localhost:8501
# 6. Explore insights and export reports
```

**Total time:** 10-15 minutes from start to insights!

---

## Success Checklist

- ✅ Environment set up and dependencies installed
- ✅ Data file placed in `data/raw/`
- ✅ Training pipeline completed successfully
- ✅ Models saved in `models/` directory
- ✅ Dashboard launches without errors
- ✅ Visualizations generated in `figures/`
- ✅ Reports available in `reports/`

**Congratulations! You're ready to analyze customer churn! 🎉**

---

**Need help?** Open an issue in the repository or contact the data science team.

**Last Updated:** October 2024
