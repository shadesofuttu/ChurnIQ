# Customer Segmentation & Churn Pattern Analytics in European Banking

## 📊 Project Overview

This project provides a comprehensive, data-driven analysis of customer churn patterns in European banking. By combining advanced segmentation techniques with predictive analytics, we deliver actionable insights to reduce customer attrition and improve retention strategies.

### 🎯 Project Objectives

**Primary Objectives:**
- Measure overall churn rate across European markets
- Identify churn distribution across customer segments
- Compare churn behavior across European regions

**Secondary Objectives:**
- Understand churn among high-value customers
- Evaluate engagement and tenure patterns
- Support strategic planning and marketing decisions

### 🔑 Key Features

- **Advanced Analytics**: Comprehensive churn pattern analysis across geography, demographics, and behavior
- **Customer Segmentation**: RFM-based and behavioral segmentation with 4+ distinct customer groups
- **Predictive Modeling**: Machine learning models (Random Forest, Gradient Boosting, Logistic Regression)
- **Interactive Dashboard**: Real-time Streamlit dashboard with dynamic filters and drill-down capabilities
- **Automated Reporting**: Export-ready analysis reports and visualizations

## 📁 Project Structure

```
├── app/                          # Streamlit dashboard application
│   ├── main.py                   # Main dashboard application
│   ├── components/               # Reusable UI components
│   ├── pages/                    # Multi-page dashboard
│   └── utils/                    # Dashboard utilities
├── config/                       # Configuration files
│   └── settings.py               # Project settings and parameters
├── data/
│   ├── raw/                      # Raw data files
│   ├── processed/                # Cleaned and processed data
│   └── segments/                 # Segmentation results
├── figures/                      # Generated visualizations
│   ├── eda/                      # Exploratory data analysis plots
│   ├── model/                    # Model performance plots
│   └── segmentation/             # Segmentation visualizations
├── models/                       # Trained models
│   ├── churn_model.pkl           # Churn prediction model
│   ├── segmentation_model.pkl    # Customer segmentation model
│   └── scaler.pkl                # Feature scaler
├── notebooks/                    # Jupyter notebooks
│   └── 01_comprehensive_churn_analysis.ipynb
├── reports/                      # Generated reports
│   ├── research_paper.md         # Detailed research paper
│   └── executive_summary.md      # Executive summary
├── src/                          # Source code
│   ├── analysis/                 # Analytics modules
│   │   └── churn_analytics.py    # Comprehensive churn analysis
│   ├── data/                     # Data processing
│   │   ├── loader.py             # Data loading utilities
│   │   └── preprocessing.py      # Data cleaning and preprocessing
│   ├── features/                 # Feature engineering
│   │   └── engineering.py        # Feature creation and transformation
│   ├── models/                   # Machine learning models
│   │   ├── churn_model.py        # Churn prediction models
│   │   └── segmentation.py       # Customer segmentation
│   ├── utils/                    # Utility functions
│   └── visualization/            # Visualization functions
│       └── plotting.py           # Plotting utilities
├── scripts/                      # Executable scripts
│   └── train_models.py           # Model training pipeline
├── tests/                        # Unit tests
├── .gitignore
├── README.md
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd customer-churn-analytics
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Data Setup

Place your banking customer data CSV file in the `data/raw/` directory with the following columns:

**Required Columns:**
- `CustomerId`: Unique customer identifier
- `Surname`: Customer surname
- `CreditScore`: Credit score
- `Geography`: Country (France, Spain, Germany)
- `Gender`: Male/Female
- `Age`: Customer age
- `Tenure`: Years with the bank
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products
- `HasCrCard`: Credit card ownership (0/1)
- `IsActiveMember`: Active member status (0/1)
- `EstimatedSalary`: Estimated annual salary
- `Exited`: Churn indicator (0=Retained, 1=Churned)

## 📊 Usage

### 1. Train Models and Generate Analysis

```bash
python scripts/train_models.py
```

This script will:
- Load and preprocess the data
- Create engineered features
- Perform customer segmentation
- Train churn prediction models
- Generate analysis reports and visualizations

### 2. Run Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/01_comprehensive_churn_analysis.ipynb
```

### 3. Launch Interactive Dashboard

```bash
streamlit run app/main.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📈 Key Performance Indicators (KPIs)

### Overall Metrics
- **Overall Churn Rate**: Percentage of customers who exited
- **Segment Churn Rate**: Churn rate by customer segment
- **High-Value Churn Ratio**: Churn among premium customers
- **Geographic Risk Index**: Regional churn exposure
- **Engagement Drop Indicator**: Inactivity vs churn correlation

### Analytical Dimensions

**Churn Distribution Analysis:**
- Overall churn rate
- Segment-wise churn rates
- Churn contribution by segment size
- Comparison of churned vs retained profiles

**Comparative Demographic Analysis:**
- Gender-based churn differences
- Geography-age interaction analysis
- Financial stability vs churn comparison

**High-Value Customer Churn Analysis:**
- Identify high-balance churners
- Compare salary vs balance churn patterns
- Quantify revenue risk from churn

## 🎯 Analytical Methodology

### Data Processing Pipeline

1. **Data Ingestion & Validation**
   - Load dataset
   - Validate engagement and product fields
   - Ensure binary variables consistency
   - Confirm churn labeling accuracy

2. **Data Cleaning & Preparation**
   - Remove non-analytical fields (surname, customer ID)
   - Convert categorical variables for grouping
   - Create derived segmentation fields

3. **Customer Segmentation Design**
   - **Geographic Segmentation**: France, Spain, Germany
   - **Age Segmentation**: <30, 30-45, 46-60, 60+
   - **Credit Score Bands**: Low, Medium, High
   - **Tenure Groups**: <3 yrs, 3-7 yrs, 8+ yrs

4. **Feature Engineering**
   - Age groups and tenure categories
   - Balance categories (Zero, Low, Medium, High, Very High)
   - Interaction features (Balance per product, Tenure-age ratio)
   - Engagement score and risk score

5. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest (ensemble)
   - Gradient Boosting (advanced)
   - SMOTE for class imbalance handling
   - Cross-validation and hyperparameter tuning

## 📱 Streamlit Dashboard Features

### Core Modules

1. **Overall Churn Summary**
   - Total customers and churn rate
   - Geography-wise churn visualization
   - Segment-wise churn comparison

2. **Demographic Analysis**
   - Age & tenure churn comparison
   - Gender-based analysis
   - Credit score impact

3. **Geographic Analysis**
   - Country-level churn rates
   - Regional risk indices
   - Cross-country comparisons

4. **High-Value Customer Churn Explorer**
   - Balance threshold customization
   - Premium customer analysis
   - Revenue at risk calculation

5. **Segment Analysis**
   - Segment distribution
   - Churn rates by segment
   - Segment recommendations

### User Capabilities

- **Segment Filters**: Filter by geography, age, balance, activity
- **Dynamic KPI Updates**: Real-time metric recalculation
- **Drill-Down Views**: Click-through to detailed analysis
- **Export Options**: Download filtered data and reports

## 📄 Deliverables

### 1. Research Paper
- Comprehensive EDA with insights and recommendations
- Methodology documentation
- Statistical analysis and findings
- Location: `reports/research_paper.md`

### 2. Streamlit Dashboard
- Live analytics with interactive filters
- Real-time KPI monitoring
- Customizable views and exports

### 3. Executive Summary
- High-level findings for stakeholders
- Strategic recommendations
- Action items and priorities
- Location: `reports/executive_summary.md`

## 🔍 Key Findings Preview

**Geographic Patterns:**
- Churn rates vary significantly by country
- Germany shows highest churn concentration
- Regional strategies needed

**Demographic Insights:**
- Age 40-60 shows elevated churn risk
- Tenure inversely correlates with churn
- Gender differences are minimal

**Engagement Critical:**
- Inactive members 2-3x more likely to churn
- Single-product customers at higher risk
- Credit card ownership correlates with retention

**High-Value Risk:**
- Premium customers require targeted attention
- Zero-balance accounts signal imminent churn
- Multi-product strategy improves retention

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning models
- **XGBoost & LightGBM**: Advanced ML algorithms
- **Streamlit**: Interactive dashboard
- **Plotly & Seaborn**: Data visualization
- **Matplotlib**: Statistical plots
- **Jupyter**: Exploratory analysis

## 📊 Model Performance

Typical performance metrics:

- **Accuracy**: 85-88%
- **F1 Score**: 0.75-0.80
- **ROC AUC**: 0.85-0.90
- **Cross-validation AUC**: 0.83-0.87

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Data Science Team
- European Banking Analytics Division

## 🙏 Acknowledgments

- European Central Bank for data standards guidance
- Banking analytics community for best practices
- Open-source contributors

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: October 2026
**Version**: 1.0.0
**Status**: Production Ready