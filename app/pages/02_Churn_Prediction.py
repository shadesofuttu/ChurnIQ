"""Churn Prediction Page - Individual Customer Risk Assessment"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.prediction import load_churn_predictor, predict_single_customer, get_feature_requirements, format_feature_name

st.set_page_config(page_title="Churn Prediction - ChurnIQ", page_icon="🎯", layout="wide")

st.title("🎯 Customer Churn Prediction")
st.markdown("### Predict individual customer churn risk")
st.markdown("---")

# Load predictor
if 'predictor' not in st.session_state:
    with st.spinner("Loading prediction model..."):
        predictor = load_churn_predictor('gradient_boosting')
        if predictor is None:
            st.error("Failed to load prediction model. Please ensure models are trained.")
            st.stop()
        st.session_state['predictor'] = predictor

predictor = st.session_state['predictor']

# Input form
st.header("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Information")
    age = st.slider("Age", 18, 100, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

with col2:
    st.subheader("Financial Information")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    balance = st.number_input("Account Balance (€)", 0, 250000, 50000, step=1000)
    estimated_salary = st.number_input("Estimated Salary (€)", 10000, 200000, 60000, step=1000)

with col3:
    st.subheader("Banking Relationship")
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active = st.selectbox("Active Member", ["Yes", "No"])

st.markdown("---")

# Prepare customer data
customer_data = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': 1 if has_credit_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active == "Yes" else 0,
    'EstimatedSalary': estimated_salary,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0
}

# Add any engineered features that might be required
if 'Balance_Per_Product' in predictor.feature_columns:
    customer_data['Balance_Per_Product'] = balance / (num_products + 1)
if 'Tenure_Age_Ratio' in predictor.feature_columns:
    customer_data['Tenure_Age_Ratio'] = tenure / age
if 'Active_With_Card' in predictor.feature_columns:
    customer_data['Active_With_Card'] = customer_data['IsActiveMember'] * customer_data['HasCrCard']
if 'Credit_Salary_Ratio' in predictor.feature_columns:
    customer_data['Credit_Salary_Ratio'] = credit_score / (estimated_salary / 1000)
if 'Engagement_Score' in predictor.feature_columns:
    engagement = customer_data['IsActiveMember'] * 30 + (num_products / 4) * 30 + customer_data['HasCrCard'] * 20
    customer_data['Engagement_Score'] = engagement
if 'Churn_Risk_Score' in predictor.feature_columns:
    risk = 0
    risk += (balance == 0) * 25
    risk += (1 - customer_data['IsActiveMember']) * 25
    risk += (num_products == 1) * 20
    risk += (tenure <= 2) * 15
    risk += (credit_score < 600) * 15
    customer_data['Churn_Risk_Score'] = risk
if 'Segment' in predictor.feature_columns:
    customer_data['Segment'] = 0  # Default segment

# Predict button
if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer risk..."):
        result = predict_single_customer(predictor, customer_data)
    
    st.markdown("---")
    st.header("Prediction Results")
    
    # Display results in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{result['churn_probability']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Risk Level",
            value=result['risk_level']
        )
    
    with col3:
        st.metric(
            label="Prediction",
            value=result['prediction']
        )
    
    with col4:
        # Color-coded risk indicator
        if result['risk_level'] == "High Risk":
            st.error("⚠️ High Churn Risk")
        elif result['risk_level'] == "Medium Risk":
            st.warning("⚡ Medium Churn Risk")
        else:
            st.success("✅ Low Churn Risk")
    
    st.markdown("---")
    
    # Risk gauge visualization
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['churn_probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': result['risk_color']},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top risk factors
    if result['top_risk_factors']:
        st.subheader("🔍 Top Risk Factors")
        st.markdown("These features have the highest impact on churn prediction:")
        
        risk_factors_df = pd.DataFrame(result['top_risk_factors'])
        risk_factors_df['factor'] = risk_factors_df['factor'].apply(format_feature_name)
        risk_factors_df['importance'] = (risk_factors_df['importance'] * 100).round(2)
        
        st.dataframe(
            risk_factors_df.rename(columns={
                'factor': 'Risk Factor',
                'value': 'Current Value',
                'importance': 'Importance (%)'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if result['churn_probability'] >= 0.7:
        recommendations.append("**Urgent Action Required**: Immediate personal outreach by relationship manager")
        recommendations.append("Offer retention incentives or special promotions")
    elif result['churn_probability'] >= 0.4:
        recommendations.append("**Monitor Closely**: Include in retention campaign")
        recommendations.append("Conduct satisfaction survey to identify pain points")
    
    if customer_data['IsActiveMember'] == 0:
        recommendations.append("Customer is inactive - launch re-engagement campaign")
    
    if num_products == 1:
        recommendations.append("Single-product customer - cross-sell additional products")
    
    if balance == 0:
        recommendations.append("Zero balance detected - high churn signal, immediate intervention needed")
    
    if tenure <= 2:
        recommendations.append("New customer (<2 years) - enhance onboarding and early engagement")
    
    if not recommendations:
        recommendations.append("Customer appears stable - maintain current service level")
    
    for i, rec in enumerate(recommendations, 1):
        st.info(f"{i}. {rec}")

st.markdown("---")

# Batch prediction option
st.header("📊 Batch Prediction")
st.markdown("Upload a CSV file with customer data to predict churn for multiple customers at once.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(batch_df)} customers")
    
    if st.button("Run Batch Prediction"):
        with st.spinner("Processing batch predictions..."):
            # Add predictions to dataframe
            predictions = []
            for _, row in batch_df.iterrows():
                row_dict = row.to_dict()
                result = predict_single_customer(predictor, row_dict)
                predictions.append({
                    'Churn_Probability': result['churn_probability'],
                    'Prediction': result['prediction'],
                    'Risk_Level': result['risk_level']
                })
            
            pred_df = pd.DataFrame(predictions)
            result_df = pd.concat([batch_df, pred_df], axis=1)
            
            st.success("Batch prediction completed!")
            st.dataframe(result_df, use_container_width=True)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
