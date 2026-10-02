"""Model Performance Page - ML Evaluation Metrics"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
import joblib

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.churn_model import ChurnPredictor
from config.settings import MODELS_DIR

st.set_page_config(page_title="Model Performance - ChurnIQ", page_icon="🤖", layout="wide")

st.title("🤖 Model Performance")
st.markdown("### Machine Learning Model Evaluation & Metrics")
st.markdown("---")

# Load best model
if 'predictor' not in st.session_state:
    with st.spinner("Loading model..."):
        predictor = ChurnPredictor('gradient_boosting')
        try:
            predictor.load_model()
            st.session_state['predictor'] = predictor
        except:
            st.error("Model not found. Please train models first using: python scripts/train_models.py")
            st.stop()

predictor = st.session_state['predictor']

# Model information
st.header("📋 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", "Gradient Boosting")
    st.metric("Features", len(predictor.feature_columns))

with col2:
    if 'metrics' in dir(predictor) and predictor.metrics:
        st.metric("Accuracy", f"{predictor.metrics.get('accuracy', 0):.4f}")
        st.metric("ROC AUC", f"{predictor.metrics.get('roc_auc', 0):.4f}")

with col3:
    if 'metrics' in dir(predictor) and predictor.metrics:
        st.metric("Precision", f"{predictor.metrics.get('precision', 0):.4f}")
        st.metric("Recall", f"{predictor.metrics.get('recall', 0):.4f}")

st.markdown("---")

# Load comparison results if available
comparison_file = MODELS_DIR / 'model_comparison.csv'
if comparison_file.exists():
    st.header("📊 Model Comparison")
    
    comparison_df = pd.read_csv(comparison_file)
    
    # Display as table
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'CV AUC']),
        use_container_width=True
    )
    
    # Visualize comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto'
            ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

# Feature Importance
st.header("🎯 Feature Importance")

feature_importance = predictor.get_feature_importance()

if feature_importance is not None:
    # Show top features
    top_n = st.slider("Number of top features to display", 5, len(feature_importance), 15)
    
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['Importance'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(
            color=top_features['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=top_features['Importance'].round(3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white',
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show full table
    with st.expander("📋 View All Feature Importances"):
        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Feature importance not available for this model type.")

st.markdown("---")

# Confusion Matrix
if 'metrics' in dir(predictor) and predictor.metrics and 'confusion_matrix' in predictor.metrics:
    st.header("🎯 Confusion Matrix")
    
    cm = predictor.metrics['confusion_matrix']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Retain', 'Predicted: Churn'],
        y=['Actual: Retain', 'Actual: Churn'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("True Negatives", f"{tn:,}")
    with col2:
        st.metric("False Positives", f"{fp:,}")
    with col3:
        st.metric("False Negatives", f"{fn:,}")
    with col4:
        st.metric("True Positives", f"{tp:,}")
    
    st.markdown("---")

# ROC Curve
if 'metrics' in dir(predictor) and predictor.metrics and 'roc_curve' in predictor.metrics:
    st.header("📈 ROC Curve")
    
    roc_data = predictor.metrics['roc_curve']
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_score = predictor.metrics['roc_auc']
    
    fig = go.Figure()
    
    # ROC Curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""The ROC AUC score of {auc_score:.3f} indicates the model's ability to distinguish between 
    churned and retained customers. A score of 0.5 is random guessing, while 1.0 is perfect classification.""")

st.markdown("---")

# Performance Notes
st.header("📝 Performance Notes")

st.markdown("""
### Model Evaluation Methodology

**Cross-Validation:**
- Performed on **original training data** (before SMOTE) to avoid data leakage
- 5-fold stratified cross-validation
- Provides unbiased estimate of model performance

**SMOTE Application:**
- Applied **only after** cross-validation to handle class imbalance
- Used solely for final model training
- Prevents artificially inflated CV scores

**Test Set Evaluation:**
- Evaluated on held-out test set (20% of original data)
- Test data never seen during training or cross-validation
- Provides realistic performance estimate

### Key Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: Of predicted churns, how many actually churned
- **Recall**: Of actual churns, how many were correctly predicted
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Model's ability to distinguish between classes
- **CV AUC**: Cross-validated AUC score (pre-SMOTE, unbiased)

### Performance Gap Analysis

If CV AUC significantly differs from Test AUC:
- **CV > Test**: May indicate overfitting or data distribution shift
- **Test > CV**: May indicate class imbalance effect or lucky test split
- Small differences (<5%) are normal and expected
""")

st.markdown("---")

# Download model info
st.header("💾 Export Model Information")

if st.button("📥 Download Model Metrics"):
    if 'metrics' in dir(predictor) and predictor.metrics:
        metrics_df = pd.DataFrame([
            {'Metric': 'Accuracy', 'Value': predictor.metrics.get('accuracy', 0)},
            {'Metric': 'Precision', 'Value': predictor.metrics.get('precision', 0)},
            {'Metric': 'Recall', 'Value': predictor.metrics.get('recall', 0)},
            {'Metric': 'F1 Score', 'Value': predictor.metrics.get('f1_score', 0)},
            {'Metric': 'ROC AUC', 'Value': predictor.metrics.get('roc_auc', 0)},
            {'Metric': 'CV AUC Mean', 'Value': predictor.metrics.get('cv_auc_mean', 0)},
            {'Metric': 'CV AUC Std', 'Value': predictor.metrics.get('cv_auc_std', 0)}
        ])
        
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
