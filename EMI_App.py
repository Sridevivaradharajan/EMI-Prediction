"""
EMI Eligibility Prediction System - Cloud Ready Version
Professional Streamlit Application with Google Drive Integration
Author: Sridevi V
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import gdown
import tempfile
import shutil
from pathlib import Path

# ============================================================================
# COLOR SCHEME CONFIGURATION
# ============================================================================

PRIMARY = "#EF7722"
SECONDARY = "#FAA533"
BACKGROUND = "#EBEBEB"
ACCENT = "#0BA6DF"
WHITE = "#FFFFFF"
TEXT = "#2C3E50"
SUCCESS = "#27AE60"
DANGER = "#E74C3C"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EMI Eligibility Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    st.markdown(f"""
        <style>
        /* Main Background */
        .stApp {{
            background-color: {BACKGROUND};
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {TEXT};
            font-family: 'Arial', sans-serif;
        }}
        
        h1 {{
            color: {PRIMARY};
            border-bottom: 3px solid {SECONDARY};
            padding-bottom: 10px;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {WHITE};
            border-right: 2px solid {SECONDARY};
        }}
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            color: {PRIMARY};
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {PRIMARY};
            color: {WHITE};
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {SECONDARY};
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Info Boxes */
        .info-box {{
            background-color: {WHITE};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {ACCENT};
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .success-box {{
            background-color: {WHITE};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {SUCCESS};
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .warning-box {{
            background-color: {WHITE};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {SECONDARY};
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .danger-box {{
            background-color: {WHITE};
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {DANGER};
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {{
            color: {PRIMARY};
            font-size: 28px;
            font-weight: bold;
        }}
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {{
            border: 2px solid {ACCENT};
            border-radius: 5px;
            padding: 10px;
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {SUCCESS};
        }}
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ============================================================================
# GOOGLE DRIVE CONFIGURATION
# ============================================================================

GDRIVE_FOLDER_ID = '1cUbTU0LwyNLhzRjqepxayBFpHtGYnhP0'

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def setup_model_directory():
    """Creates a temporary directory to store downloaded models."""
    temp_dir = tempfile.mkdtemp(prefix="emi_models_")
    return temp_dir

def download_folder_from_gdrive(folder_id, dest_path):
    """Downloads all files from a public Google Drive folder using gdown."""
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=url, output=dest_path, quiet=False, use_cookies=False)
        return True
    except Exception as e:
        print("Download failed:", e)
        return False

@st.cache_resource
def load_models_from_gdrive_folder():
    """Load all models from a shared Google Drive folder."""
    try:
        with st.spinner('Loading models from cloud storage...'):
            # Create temporary directory
            models_dir = setup_model_directory()

            # Download GDrive folder
            if not download_folder_from_gdrive(GDRIVE_FOLDER_ID, models_dir):
                st.error("Could not download Google Drive folder")
                return None

            # Detect saved_models folder
            saved_models_path = None
            for root, dirs, files in os.walk(models_dir):
                if 'saved_models' in dirs:
                    saved_models_path = os.path.join(root, 'saved_models')
                    break

            if not saved_models_path:
                saved_models_path = models_dir

            classification_model_name = "best_classification_model_Stacking_Ensemble.pkl"
            regression_model_name = "best_regression_model_XGBoost.pkl"
            
            # Load models
            clf_model = joblib.load(os.path.join(saved_models_path, classification_model_name))
            reg_model = joblib.load(os.path.join(saved_models_path, regression_model_name))
            
            # Label encoder & scalers
            label_encoder = joblib.load(os.path.join(saved_models_path, 'GLOBAL_LABEL_ENCODER.pkl'))
            clf_scaler = joblib.load(os.path.join(saved_models_path, 'classification_scaler.pkl'))
            reg_scaler = joblib.load(os.path.join(saved_models_path, 'regression_scaler.pkl'))

            # Metadata + features
            metadata = json.load(open(os.path.join(saved_models_path, 'model_metadata.json'), 'r'))
            clf_features = json.load(open(os.path.join(saved_models_path, 'classification_features.json'), 'r'))
            reg_features = json.load(open(os.path.join(saved_models_path, 'regression_features.json'), 'r'))

            st.success("Models loaded successfully!")

            # Return all components
            return {
                'classification_model': clf_model,
                'regression_model': reg_model,
                'label_encoder': label_encoder,
                'classification_scaler': clf_scaler,
                'regression_scaler': reg_scaler,
                'metadata': metadata,
                'classification_features': clf_features,
                'regression_features': reg_features,
                'temp_dir': models_dir
            }

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def engineer_features(input_data):
    """Apply feature engineering to input data"""
    df = input_data.copy()
    
    # 1. Debt-to-Income Ratio
    df['debt_to_income_ratio'] = (
        (df['current_emi_amount'] / df['monthly_salary'].replace(0, 1)) * 100
    ).clip(0, 200)
    
    # 2. Total Monthly Expenses
    expense_cols = ['school_fees', 'college_fees', 'travel_expenses',
                   'groceries_utilities', 'other_monthly_expenses', 'monthly_rent']
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
    
    # 3. Expense-to-Income Ratio
    df['expense_to_income_ratio'] = (
        (df['total_monthly_expenses'] / df['monthly_salary'].replace(0, 1)) * 100
    ).clip(0, 150)
    
    # 4. Disposable Income
    df['disposable_income'] = (
        df['monthly_salary'] - 
        df['total_monthly_expenses'] - 
        df['current_emi_amount']
    ).clip(lower=0)
    
    # 5. Savings Rate
    df['savings_rate'] = (
        ((df['monthly_salary'] - df['total_monthly_expenses']) / 
         df['monthly_salary'].replace(0, 1)) * 100
    ).clip(0, 100)
    
    # 6. Credit Utilization
    df['credit_utilization'] = (
        (df['requested_amount'] / (df['monthly_salary'] * 12).replace(0, 1)) * 100
    ).clip(0, 200)
    
    # 7. Financial Stability Score
    df['financial_stability_score'] = (
        (df['credit_score'] / 850 * 0.35) +
        (df['years_of_employment'] / 40 * 0.25) +
        ((100 - df['debt_to_income_ratio']) / 100 * 0.25) +
        (df['savings_rate'] / 100 * 0.15)
    ) * 100
    df['financial_stability_score'] = df['financial_stability_score'].clip(0, 100)
    
    # 8. Dependent Ratio
    df['dependent_ratio'] = (
        (df['dependents'] / df['family_size'].replace(0, 1))
    ).fillna(0) * 100
    
    # 9. Loan Burden Score
    df['loan_burden_score'] = (
        df['requested_amount'] / df['disposable_income'].replace(0, 1)
    ).clip(0, 100)
    
    # 10. Emergency Buffer
    df['emergency_buffer_months'] = (
        df['bank_balance'] / df['total_monthly_expenses'].replace(0, 1)
    ).clip(0, 50)
    
    return df

def prepare_classification_input(input_data, feature_names, preprocessing_type):
    """Prepare input data for classification model"""
    categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        df_encoded = pd.get_dummies(input_data, columns=categorical_cols, 
                                    drop_first=True, dtype=int)
    else:
        df_encoded = input_data.copy()
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[feature_names]
    
    if preprocessing_type == 'linear':
        skewed_features = ['monthly_salary', 'bank_balance', 'requested_amount', 
                          'emergency_fund', 'disposable_income', 'total_monthly_expenses']
        
        for col in skewed_features:
            if col in df_encoded.columns:
                log_col_name = f'{col}_log'
                df_encoded[log_col_name] = np.log1p(df_encoded[col])
    
    return df_encoded

def prepare_regression_input(input_data, feature_names, preprocessing_type):
    """Prepare input data for regression model"""
    categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        df_encoded = pd.get_dummies(input_data, columns=categorical_cols, 
                                    drop_first=True, dtype=int)
    else:
        df_encoded = input_data.copy()
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[feature_names]
    
    if preprocessing_type == 'linear':
        skewed_features = ['monthly_salary', 'bank_balance', 'requested_amount', 
                          'emergency_fund', 'disposable_income', 'total_monthly_expenses']
        
        for col in skewed_features:
            if col in df_encoded.columns:
                log_col_name = f'{col}_log'
                df_encoded[log_col_name] = np.log1p(df_encoded[col])
    
    return df_encoded

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_eligibility(input_data, models):
    """Predict EMI eligibility"""
    try:
        input_with_features = engineer_features(input_data)
        preprocessing_type = models['metadata']['classification']['preprocessing_type']
        
        X_clf = prepare_classification_input(
            input_with_features, 
            models['classification_features'],
            preprocessing_type
        )
        
        if preprocessing_type == 'linear':
            X_clf_scaled = models['classification_scaler'].transform(X_clf)
            X_clf_final = pd.DataFrame(
                X_clf_scaled, 
                columns=X_clf.columns, 
                index=X_clf.index
            )
        else:
            X_clf_final = X_clf
        
        prediction_encoded = models['classification_model'].predict(X_clf_final)
        prediction_proba = models['classification_model'].predict_proba(X_clf_final)
        
        prediction = models['label_encoder'].inverse_transform(prediction_encoded)[0]
        
        class_probs = {}
        for idx, class_name in enumerate(models['label_encoder'].classes_):
            class_probs[class_name] = prediction_proba[0][idx]
        
        return {
            'prediction': prediction,
            'probabilities': class_probs,
            'confidence': max(prediction_proba[0])
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def predict_emi_amount(input_data, models):
    """Predict maximum EMI amount"""
    try:
        input_with_features = engineer_features(input_data)
        preprocessing_type = models['metadata']['regression']['preprocessing_type']
        
        X_reg = prepare_regression_input(
            input_with_features, 
            models['regression_features'],
            preprocessing_type
        )
        
        if preprocessing_type == 'linear':
            X_reg_scaled = models['regression_scaler'].transform(X_reg)
            X_reg_final = pd.DataFrame(
                X_reg_scaled, 
                columns=X_reg.columns, 
                index=X_reg.index
            )
        else:
            X_reg_final = X_reg
        
        emi_prediction = models['regression_model'].predict(X_reg_final)[0]
        
        return max(0, emi_prediction)
    except Exception as e:
        st.error(f"EMI prediction error: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_probability_chart(probabilities):
    """Create probability distribution chart"""
    colors = [PRIMARY, SECONDARY, ACCENT]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=colors[:len(probabilities)],
            text=[f"{v*100:.1f}%" for v in probabilities.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Eligibility Probability Distribution",
        xaxis_title="Category",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=TEXT),
        height=400
    )
    
    return fig

def create_financial_metrics_chart(input_data):
    """Create financial metrics visualization"""
    metrics = {
        'DTI Ratio': input_data['debt_to_income_ratio'].values[0],
        'Expense Ratio': input_data['expense_to_income_ratio'].values[0],
        'Savings Rate': input_data['savings_rate'].values[0],
        'Credit Score': (input_data['credit_score'].values[0] / 850) * 100
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=[PRIMARY, SECONDARY, ACCENT, SUCCESS],
            text=[f"{v:.1f}%" for v in metrics.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Key Financial Metrics",
        xaxis_title="Metric",
        yaxis_title="Percentage",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=TEXT),
        height=400
    )
    
    return fig

def create_gauge_chart(value, title, max_value=100):
    """Create gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': TEXT}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': TEXT},
            'bar': {'color': PRIMARY},
            'steps': [
                {'range': [0, max_value*0.33], 'color': DANGER},
                {'range': [max_value*0.33, max_value*0.66], 'color': SECONDARY},
                {'range': [max_value*0.66, max_value], 'color': SUCCESS}
            ],
            'threshold': {
                'line': {'color': TEXT, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor=WHITE,
        font={'color': TEXT},
        height=300
    )
    
    return fig

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def single_prediction_page(models):
    """Single prediction interface"""
    st.header("Single Applicant Prediction")
    
    st.markdown(f"""
        <div class='info-box'>
            <p>Enter the applicant's information below to get instant EMI eligibility prediction and recommended EMI amount.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        family_size = st.number_input("Family Size", min_value=1, max_value=15, value=1)
        
    with col2:
        st.subheader("Financial Information")
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, value=100000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        requested_amount = st.number_input("Requested Loan Amount (INR)", min_value=0, value=500000)
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, value=0)
        
    with col3:
        st.subheader("Employment & Expenses")
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        occupation = st.selectbox("Occupation", [
            "Salaried", "Self-Employed", "Business", "Professional", "Other"
        ])
        school_fees = st.number_input("School Fees (Monthly, INR)", min_value=0, value=0)
        college_fees = st.number_input("College Fees (Monthly, INR)", min_value=0, value=0)
        travel_expenses = st.number_input("Travel Expenses (Monthly, INR)", min_value=0, value=0)
        groceries_utilities = st.number_input("Groceries & Utilities (Monthly, INR)", min_value=0, value=10000)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (INR)", min_value=0, value=0)
        monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0, value=0)
        emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, value=50000)
    
    if st.button("Predict Eligibility", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'marital_status': [marital_status],
            'dependents': [dependents],
            'family_size': [family_size],
            'monthly_salary': [monthly_salary],
            'bank_balance': [bank_balance],
            'credit_score': [credit_score],
            'requested_amount': [requested_amount],
            'current_emi_amount': [current_emi_amount],
            'years_of_employment': [years_of_employment],
            'occupation': [occupation],
            'school_fees': [school_fees],
            'college_fees': [college_fees],
            'travel_expenses': [travel_expenses],
            'groceries_utilities': [groceries_utilities],
            'other_monthly_expenses': [other_monthly_expenses],
            'monthly_rent': [monthly_rent],
            'emergency_fund': [emergency_fund]
        })
        
        with st.spinner('Analyzing application...'):
            eligibility_result = predict_eligibility(input_data, models)
            emi_amount = predict_emi_amount(input_data, models)
            input_with_features = engineer_features(input_data)
        
        if eligibility_result and emi_amount is not None:
            st.markdown("---")
            st.header("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if eligibility_result['prediction'] == 'High':
                    st.markdown(f"""
                        <div class='success-box'>
                            <h3>Eligibility Status: HIGH</h3>
                            <p>Confidence: {eligibility_result['confidence']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif eligibility_result['prediction'] == 'Medium':
                    st.markdown(f"""
                        <div class='warning-box'>
                            <h3>Eligibility Status: MEDIUM</h3>
                            <p>Confidence: {eligibility_result['confidence']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='danger-box'>
                            <h3>Eligibility Status: LOW</h3>
                            <p>Confidence: {eligibility_result['confidence']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Maximum Recommended EMI", f"INR {emi_amount:,.2f}")
                st.metric("Requested Amount", f"INR {requested_amount:,.2f}")
            
            with col3:
                approval_percentage = (emi_amount / requested_amount * 100) if requested_amount > 0 else 0
                st.metric("Approval Percentage", f"{min(approval_percentage, 100):.1f}%")
                st.metric("Credit Score", credit_score)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_probability_chart(eligibility_result['probabilities']), 
                              use_container_width=True)
            
            with col2:
                st.plotly_chart(create_financial_metrics_chart(input_with_features), 
                              use_container_width=True)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(
                    create_gauge_chart(
                        input_with_features['debt_to_income_ratio'].values[0],
                        "Debt-to-Income Ratio",
                        100
                    ),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_gauge_chart(
                        input_with_features['savings_rate'].values[0],
                        "Savings Rate",
                        100
                    ),
                    use_container_width=True
                )
            
            with col3:
                st.plotly_chart(
                    create_gauge_chart(
                        input_with_features['financial_stability_score'].values[0],
                        "Financial Stability Score",
                        100
                    ),
                    use_container_width=True
                )

def batch_prediction_page(models):
    """Batch prediction interface"""
    st.header("Batch Prediction")
    
    st.markdown(f"""
        <div class='info-box'>
            <p>Upload a CSV file containing multiple applicant records for batch processing.</p>
            <p><strong>Required columns:</strong> age, gender, marital_status, dependents, family_size, 
            monthly_salary, bank_balance, credit_score, requested_amount, current_emi_amount, 
            years_of_employment, occupation, school_fees, college_fees, travel_expenses, 
            groceries_utilities, other_monthly_expenses, monthly_rent, emergency_fund</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Found {len(df)} records.")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Run Batch Prediction", use_container_width=True):
                with st.spinner('Processing batch predictions...'):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        input_data = pd.DataFrame([row])
                        
                        eligibility_result = predict_eligibility(input_data, models)
                        emi_amount = predict_emi_amount(input_data, models)
                        
                        results.append({
                            'Index': idx,
                            'Eligibility': eligibility_result['prediction'] if eligibility_result else 'Error',
                            'Confidence': f"{eligibility_result['confidence']*100:.1f}%" if eligibility_result else 'N/A',
                            'Recommended_EMI': f"{emi_amount:,.2f}" if emi_amount else 'N/A'
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success("Batch prediction completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_count = (results_df['Eligibility'] == 'High').sum()
                        st.metric("High Eligibility", high_count)
                    
                    with col2:
                        medium_count = (results_df['Eligibility'] == 'Medium').sum()
                        st.metric("Medium Eligibility", medium_count)
                    
                    with col3:
                        low_count = (results_df['Eligibility'] == 'Low').sum()
                        st.metric("Low Eligibility", low_count)
                    
                    st.subheader("Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"emi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    fig = px.pie(
                        results_df, 
                        names='Eligibility', 
                        title='Eligibility Distribution',
                        color_discrete_map={'High': SUCCESS, 'Medium': SECONDARY, 'Low': DANGER}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains all required columns.")

def model_info_page(models):
    """Display model information"""
    st.header("Model Information")
    
    st.markdown(f"""
        <div class='info-box'>
            <h3>System Overview</h3>
            <p>This system uses advanced machine learning models to predict EMI eligibility 
            and recommend optimal EMI amounts based on comprehensive financial analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Model")
        st.markdown(f"""
            <div class='success-box'>
                <p><strong>Model Type:</strong> {models['metadata']['classification']['best_model']}</p>
                <p><strong>Accuracy:</strong> {models['metadata']['classification']['best_score']*100:.2f}%</p>
                <p><strong>Preprocessing:</strong> {models['metadata']['classification']['preprocessing_type']}</p>
                <p><strong>Purpose:</strong> Predicts eligibility category (High/Medium/Low)</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Classification Features")
        with st.expander("View Features"):
            features_df = pd.DataFrame({
                'Feature': models['classification_features']
            })
            st.dataframe(features_df, use_container_width=True)
    
    with col2:
        st.subheader("Regression Model")
        st.markdown(f"""
            <div class='success-box'>
                <p><strong>Model Type:</strong> {models['metadata']['regression']['best_model']}</p>
                <p><strong>R² Score:</strong> {models['metadata']['regression']['best_score']:.4f}</p>
                <p><strong>Preprocessing:</strong> {models['metadata']['regression']['preprocessing_type']}</p>
                <p><strong>Purpose:</strong> Predicts maximum recommended EMI amount</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Regression Features")
        with st.expander("View Features"):
            features_df = pd.DataFrame({
                'Feature': models['regression_features']
            })
            st.dataframe(features_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Feature Engineering Pipeline")
    
    features_info = {
        'Feature': [
            'Debt-to-Income Ratio',
            'Total Monthly Expenses',
            'Expense-to-Income Ratio',
            'Disposable Income',
            'Savings Rate',
            'Credit Utilization',
            'Financial Stability Score',
            'Dependent Ratio',
            'Loan Burden Score',
            'Emergency Buffer Months'
        ],
        'Description': [
            'Ratio of current EMI to monthly salary',
            'Sum of all monthly expenses',
            'Ratio of expenses to income',
            'Income remaining after expenses and EMI',
            'Percentage of income saved',
            'Requested amount vs annual income',
            'Composite score based on multiple factors',
            'Ratio of dependents to family size',
            'Requested amount vs disposable income',
            'Months of expenses covered by bank balance'
        ],
        'Impact': [
            'High',
            'Medium',
            'High',
            'High',
            'Medium',
            'High',
            'Very High',
            'Medium',
            'High',
            'Medium'
        ]
    }
    
    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='info-box'>
                <h4>Classification Metrics</h4>
                <p><strong>Accuracy:</strong> {models['metadata']['classification']['best_score']*100:.2f}%</p>
                <p><strong>Model:</strong> {models['metadata']['classification']['best_model']}</p>
                <p><strong>Training Date:</strong> {models['metadata']['training_date']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='info-box'>
                <h4>Regression Metrics</h4>
                <p><strong>R² Score:</strong> {models['metadata']['regression']['best_score']:.4f}</p>
                <p><strong>Model:</strong> {models['metadata']['regression']['best_model']}</p>
                <p><strong>Training Date:</strong> {models['metadata']['training_date']}</p>
            </div>
        """, unsafe_allow_html=True)

def about_page():
    """About page"""
    st.header("About EMI Eligibility Prediction System")
    
    st.markdown(f"""
        <div class='info-box'>
            <h3>System Overview</h3>
            <p>The EMI Eligibility Prediction System is a sophisticated machine learning application 
            designed to assess loan eligibility and recommend optimal EMI amounts based on comprehensive 
            financial analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='success-box'>
                <h4>Advanced ML Models</h4>
                <ul>
                    <li>Stacking Ensemble Classifier</li>
                    <li>XGBoost Regressor</li>
                    <li>Feature Engineering Pipeline</li>
                    <li>Multi-metric Evaluation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='success-box'>
                <h4>User-Friendly Interface</h4>
                <ul>
                    <li>Single Prediction Mode</li>
                    <li>Batch Processing</li>
                    <li>Interactive Visualizations</li>
                    <li>Downloadable Reports</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='success-box'>
                <h4>Comprehensive Analysis</h4>
                <ul>
                    <li>Financial Stability Assessment</li>
                    <li>Risk Factor Analysis</li>
                    <li>EMI Affordability Calculation</li>
                    <li>Credit Score Evaluation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='success-box'>
                <h4>Cloud-Ready Architecture</h4>
                <ul>
                    <li>Google Drive Integration</li>
                    <li>Scalable Design</li>
                    <li>Real-time Predictions</li>
                    <li>Secure Data Processing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("How It Works")
    
    st.markdown("""
        ### Step 1: Data Collection
        The system collects comprehensive financial information including:
        - Personal demographics (age, gender, marital status)
        - Employment details (years of employment, occupation)
        - Financial metrics (salary, bank balance, credit score)
        - Expense breakdown (rent, fees, utilities, etc.)
        - Current financial obligations (existing EMIs)
        
        ### Step 2: Feature Engineering
        Advanced feature engineering creates derived metrics:
        - Debt-to-Income Ratio
        - Financial Stability Score
        - Savings Rate
        - Credit Utilization
        - Emergency Buffer Assessment
        
        ### Step 3: ML Prediction
        Two specialized models work together:
        - **Classification Model**: Determines eligibility category (High/Medium/Low)
        - **Regression Model**: Calculates maximum recommended EMI amount
        
        ### Step 4: Results & Visualization
        The system provides:
        - Eligibility prediction with confidence scores
        - Recommended EMI amount
        - Interactive charts and gauges
        - Detailed financial analysis
        - Downloadable reports (for batch mode)
    """)
    
    st.markdown("---")
    
    st.subheader("Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='info-box'>
                <h4>Machine Learning</h4>
                <ul>
                    <li>Scikit-learn</li>
                    <li>XGBoost</li>
                    <li>Pandas</li>
                    <li>NumPy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='info-box'>
                <h4>Visualization</h4>
                <ul>
                    <li>Plotly</li>
                    <li>Streamlit</li>
                    <li>Custom CSS</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='info-box'>
                <h4>Cloud Integration</h4>
                <ul>
                    <li>Google Drive</li>
                    <li>gdown</li>
                    <li>joblib</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Developer Information")
    
    st.markdown(f"""
        <div class='info-box'>
            <p><strong>Author:</strong> Sridevi V</p>
            <p><strong>Version:</strong> 2.0 (Cloud-Ready)</p>
            <p><strong>Last Updated:</strong> {datetime.now().strftime('%B %Y')}</p>
            <p><strong>Purpose:</strong> Educational and Professional Use</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Disclaimer")
    
    st.markdown(f"""
        <div class='warning-box'>
            <p><strong>Important Notice:</strong></p>
            <p>This system is designed for informational and analytical purposes. 
            The predictions and recommendations should not be considered as final loan approval decisions. 
            Financial institutions should use this as a decision support tool alongside their existing 
            evaluation processes and regulatory requirements.</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown(f"""
        <div style='background-color: {PRIMARY}; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: {WHITE}; text-align: center; margin: 0;'>
                EMI Eligibility Prediction System
            </h1>
            <p style='color: {WHITE}; text-align: center; margin: 5px 0 0 0;'>
                Cloud-Ready Machine Learning Application
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models from Google Drive
    models = load_models_from_gdrive_folder() 
    
    if models is None:
        st.error("Failed to load models from Google Drive.")
        st.info("""
        **Setup Instructions:**
        1. Upload your model files to Google Drive
        2. Make the folder publicly accessible
        3. Extract the folder ID from the sharing link
        4. Update the GDRIVE_FOLDER_ID variable at the top of the code
        
        **Folder ID Format:**
        From: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
        Use: `FOLDER_ID_HERE`
        """)
        return
    
    st.sidebar.markdown(f"<h2 style='color: {PRIMARY};'>Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Select Page",
        ["Single Prediction", "Batch Prediction", "Model Information", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div class='info-box'>
            <h3 style='color: {PRIMARY};'>System Status</h3>
            <p style='color: {TEXT};'>
                <b>Classification Model:</b> {models['metadata']['classification']['best_model']}<br>
                <b>Regression Model:</b> {models['metadata']['regression']['best_model']}<br>
                <b>Status:</b> <span style='color: {SUCCESS};'>Cloud Active</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if page == "Single Prediction":
        single_prediction_page(models)
    elif page == "Batch Prediction":
        batch_prediction_page(models)
    elif page == "Model Information":
        model_info_page(models)
    else:
        about_page()

if __name__ == "__main__":
    main()
