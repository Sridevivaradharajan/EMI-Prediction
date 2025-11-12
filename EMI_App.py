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
    page_icon="🏦",
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
# GOOGLE DRIVE DOWNLOAD FUNCTIONS
# ============================================================================

def setup_model_directory():
    """Creates a temporary directory to store downloaded models."""
    temp_dir = tempfile.mkdtemp(prefix="emi_models_")
    return temp_dir

def download_folder_from_gdrive(folder_id, dest_path):
    """
    Downloads all files from a public Google Drive folder using gdown.
    Returns True if successful.
    """
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
        with st.spinner('🔄 Loading models from cloud storage...'):

            # ✅ Create temporary directory
            models_dir = setup_model_directory()

            # ✅ Download GDrive folder
            if not download_folder_from_gdrive(GDRIVE_FOLDER_ID, models_dir):
                st.error("❌ Could not download Google Drive folder")
                return None

            # ✅ Detect saved_models folder
            saved_models_path = None
            for root, dirs, files in os.walk(models_dir):
                if 'saved_models' in dirs:
                    saved_models_path = os.path.join(root, 'saved_models')
                    break

            if not saved_models_path:
                saved_models_path = models_dir

            classification_model_name = "best_classification_model_Stacking_Ensemble.pkl"
            regression_model_name = "best_regression_model_XGBoost.pkl"
            
            # ✅ Load models
            clf_model = joblib.load(os.path.join(saved_models_path, classification_model_name))
            reg_model = joblib.load(os.path.join(saved_models_path, regression_model_name))
            
            # ✅ Label encoder & scalers
            label_encoder = joblib.load(os.path.join(saved_models_path, 'GLOBAL_LABEL_ENCODER.pkl'))
            clf_scaler = joblib.load(os.path.join(saved_models_path, 'classification_scaler.pkl'))
            reg_scaler = joblib.load(os.path.join(saved_models_path, 'regression_scaler.pkl'))

            # ✅ Metadata + features
            metadata = json.load(open(os.path.join(saved_models_path, 'model_metadata.json'), 'r'))
            clf_features = json.load(open(os.path.join(saved_models_path, 'classification_features.json'), 'r'))
            reg_features = json.load(open(os.path.join(saved_models_path, 'regression_features.json'), 'r'))

            # ✅ Return all components
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
        st.error("❌ Failed to load models from Google Drive.")
        st.info("""
        **Setup Instructions:**
        1. Upload your model files to Google Drive
        2. Make them publicly accessible or get shareable links
        3. Extract the file IDs from the sharing links
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
                <b>Status:</b> <span style='color: {SUCCESS};'>☁️ Cloud Active</span>
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

# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================

def single_prediction_page(models):
    st.markdown(f"<h2 style='color: {PRIMARY};'>Single Customer Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <p style='color: {TEXT};'>
                Enter customer details below to predict EMI eligibility and maximum EMI amount.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Personal Information", 
        "Financial Details", 
        "Loan Request", 
        "Additional Information"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            education = st.selectbox(
                "Education Level", 
                ["High_School", "Graduate", "Post_Graduate", "Professional", "Unknown"]
            )
        
        with col2:
            employment_type = st.selectbox(
                "Employment Type", 
                ["Salaried", "Self_Employed", "Freelancer"]
            )
            years_of_employment = st.number_input(
                "Years of Employment", 
                min_value=0, max_value=50, value=5
            )
            marital_status = st.selectbox(
                "Marital Status", 
                ["Single", "Married", "Divorced", "Widowed"]
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_salary = st.number_input(
                "Monthly Salary (INR)", 
                min_value=0, value=50000, step=1000
            )
            bank_balance = st.number_input(
                "Bank Balance (INR)", 
                min_value=0, value=100000, step=5000
            )
            credit_score = st.number_input(
                "Credit Score", 
                min_value=300, max_value=900, value=700
            )
            emergency_fund = st.number_input(
                "Emergency Fund (INR)", 
                min_value=0, value=50000, step=5000
            )
        
        with col2:
            current_emi_amount = st.number_input(
                "Current EMI Amount (INR)", 
                min_value=0, value=5000, step=500
            )
            monthly_rent = st.number_input(
                "Monthly Rent (INR)", 
                min_value=0, value=10000, step=1000
            )
            school_fees = st.number_input(
                "School Fees (INR)", 
                min_value=0, value=5000, step=500
            )
            college_fees = st.number_input(
                "College Fees (INR)", 
                min_value=0, value=0, step=500
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            requested_amount = st.number_input(
                "Requested Loan Amount (INR)", 
                min_value=0, value=200000, step=10000
            )
            requested_tenure = st.number_input(
                "Requested Tenure (Months)", 
                min_value=6, max_value=360, value=24
            )
        
        with col2:
            emi_scenario = st.selectbox(
                "EMI Scenario", 
                ["Personal_Loan", "Home_Appliances", "E-commerce_Shopping", 
                 "Education", "Vehicle"]
            )
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            family_size = st.number_input(
                "Family Size", 
                min_value=1, max_value=20, value=4
            )
            dependents = st.number_input(
                "Number of Dependents", 
                min_value=0, max_value=10, value=2
            )
            house_type = st.selectbox(
                "House Type", 
                ["Own", "Rented", "Family"]
            )
        
        with col2:
            travel_expenses = st.number_input(
                "Travel Expenses (INR)", 
                min_value=0, value=3000, step=500
            )
            groceries_utilities = st.number_input(
                "Groceries & Utilities (INR)", 
                min_value=0, value=8000, step=500
            )
            other_monthly_expenses = st.number_input(
                "Other Monthly Expenses (INR)", 
                min_value=0, value=5000, step=500
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("Predict Eligibility", use_container_width=True)
    
    if predict_button:
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'education': [education],
            'employment_type': [employment_type],
            'years_of_employment': [years_of_employment],
            'marital_status': [marital_status],
            'monthly_salary': [monthly_salary],
            'bank_balance': [bank_balance],
            'credit_score': [credit_score],
            'emergency_fund': [emergency_fund],
            'current_emi_amount': [current_emi_amount],
            'monthly_rent': [monthly_rent],
            'school_fees': [school_fees],
            'college_fees': [college_fees],
            'requested_amount': [requested_amount],
            'requested_tenure': [requested_tenure],
            'emi_scenario': [emi_scenario],
            'family_size': [family_size],
            'dependents': [dependents],
            'house_type': [house_type],
            'travel_expenses': [travel_expenses],
            'groceries_utilities': [groceries_utilities],
            'other_monthly_expenses': [other_monthly_expenses]
        })
        
        with st.spinner('Processing prediction...'):
            input_with_features = engineer_features(input_data)
            eligibility_result = predict_eligibility(input_data, models)
            emi_amount = predict_emi_amount(input_data, models)
        
        if eligibility_result and emi_amount is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: {PRIMARY};'>Prediction Results</h3>", unsafe_allow_html=True)
            
            if eligibility_result['prediction'] == 'Eligible':
                st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='color: {SUCCESS}; margin: 0;'>Eligible</h2>
                        <p style='color: {TEXT}; margin: 10px 0 0 0;'>
                            The customer is eligible for EMI facility.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif eligibility_result['prediction'] == 'High_Risk':
                st.markdown(f"""
                    <div class='warning-box'>
                        <h2 style='color: {SECONDARY}; margin: 0;'>High Risk</h2>
                        <p style='color: {TEXT}; margin: 10px 0 0 0;'>
                            The customer is eligible but classified as high risk.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='danger-box'>
                        <h2 style='color: {DANGER}; margin: 0;'>Not Eligible</h2>
                        <p style='color: {TEXT}; margin: 10px 0 0 0;'>
                            The customer does not meet eligibility criteria.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Max EMI", f"INR {emi_amount:,.0f}")
            
            with col2:
                st.metric("Confidence", f"{eligibility_result['confidence']*100:.1f}%")
            
            with col3:
                emi_to_income = (emi_amount / monthly_salary) * 100
                st.metric("EMI-to-Income Ratio", f"{emi_to_income:.1f}%")
            
            with col4:
                financial_stability = input_with_features['financial_stability_score'].values[0]
                st.metric("Financial Stability", f"{financial_stability:.1f}/100")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_probability_chart(eligibility_result['probabilities']), 
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_financial_metrics_chart(input_with_features), 
                    use_container_width=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: {PRIMARY};'>Detailed Financial Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(
                    create_gauge_chart(
                        input_with_features['debt_to_income_ratio'].values[0],
                        "Debt-to-Income Ratio (%)",
                        100
                    ),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    create_gauge_chart(
                        input_with_features['savings_rate'].values[0],
                        "Savings Rate (%)",
                        100
                    ),
                    use_container_width=True
                )
            
            with col3:
                st.plotly_chart(
                    create_gauge_chart(
                        (credit_score / 850) * 100,
                        "Credit Score (%)",
                        100
                    ),
                    use_container_width=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: {PRIMARY};'>Recommendations</h3>", unsafe_allow_html=True)
            
            with st.expander("View Detailed Recommendations"):
                if eligibility_result['prediction'] == 'Eligible':
                    st.markdown(f"""
                        <div style='color: {TEXT};'>
                            <h4>Approved Recommendations:</h4>
                            <ul>
                                <li>Maximum recommended EMI: INR {emi_amount:,.0f}</li>
                                <li>Recommended loan tenure: {int(requested_amount/emi_amount)} months</li>
                                <li>Maintain current credit score above 700</li>
                                <li>Keep DTI ratio below 40%</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='color: {TEXT};'>
                            <h4>Improvement Recommendations:</h4>
                            <ul>
                                <li>Improve credit score to above 700</li>
                                <li>Reduce existing debt obligations</li>
                                <li>Increase monthly savings rate</li>
                                <li>Build emergency fund to 6 months expenses</li>
                                <li>Consider reducing loan amount request</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: BATCH PREDICTION
# ============================================================================
# ============================================================================
# PAGE: BATCH PREDICTION (CONTINUED)
# ============================================================================

def batch_prediction_page(models):
    st.markdown(f"<h2 style='color: {PRIMARY};'>Batch Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <p style='color: {TEXT};'>
                Upload a CSV file containing multiple customer records for bulk EMI eligibility prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown(f"<h3 style='color: {PRIMARY};'>Upload Data</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with customer data"
    )
    
    # Template download
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download Template CSV"):
            template_data = {
                'age': [30],
                'gender': ['Male'],
                'education': ['Graduate'],
                'employment_type': ['Salaried'],
                'years_of_employment': [5],
                'marital_status': ['Married'],
                'monthly_salary': [50000],
                'bank_balance': [100000],
                'credit_score': [700],
                'emergency_fund': [50000],
                'current_emi_amount': [5000],
                'monthly_rent': [10000],
                'school_fees': [5000],
                'college_fees': [0],
                'requested_amount': [200000],
                'requested_tenure': [24],
                'emi_scenario': ['Personal_Loan'],
                'family_size': [4],
                'dependents': [2],
                'house_type': ['Own'],
                'travel_expenses': [3000],
                'groceries_utilities': [8000],
                'other_monthly_expenses': [5000]
            }
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Download Template",
                data=csv,
                file_name="emi_prediction_template.csv",
                mime="text/csv"
            )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: {PRIMARY};'>Data Preview</h3>", unsafe_allow_html=True)
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Display first few rows
            st.dataframe(df.head(10), use_container_width=True)
            
            # Prediction button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Batch Prediction", use_container_width=True):
                with st.spinner('Processing predictions for all records...'):
                    # Initialize result lists
                    predictions = []
                    emi_amounts = []
                    confidences = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each row
                    for idx, row in df.iterrows():
                        # Convert row to dataframe
                        input_data = pd.DataFrame([row])
                        
                        # Get predictions
                        eligibility_result = predict_eligibility(input_data, models)
                        emi_amount = predict_emi_amount(input_data, models)
                        
                        if eligibility_result:
                            predictions.append(eligibility_result['prediction'])
                            confidences.append(eligibility_result['confidence'])
                        else:
                            predictions.append('Error')
                            confidences.append(0)
                        
                        if emi_amount is not None:
                            emi_amounts.append(emi_amount)
                        else:
                            emi_amounts.append(0)
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f'Processed {idx + 1} of {len(df)} records')
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Add results to dataframe
                    df['Predicted_Eligibility'] = predictions
                    df['Predicted_Max_EMI'] = emi_amounts
                    df['Prediction_Confidence'] = confidences
                    
                    # Show results
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color: {PRIMARY};'>Prediction Results</h3>", unsafe_allow_html=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        eligible_count = (df['Predicted_Eligibility'] == 'Eligible').sum()
                        st.metric("Eligible", f"{eligible_count} ({eligible_count/len(df)*100:.1f}%)")
                    
                    with col2:
                        high_risk_count = (df['Predicted_Eligibility'] == 'High_Risk').sum()
                        st.metric("High Risk", f"{high_risk_count} ({high_risk_count/len(df)*100:.1f}%)")
                    
                    with col3:
                        not_eligible_count = (df['Predicted_Eligibility'] == 'Not_Eligible').sum()
                        st.metric("Not Eligible", f"{not_eligible_count} ({not_eligible_count/len(df)*100:.1f}%)")
                    
                    with col4:
                        avg_emi = df['Predicted_Max_EMI'].mean()
                        st.metric("Avg Max EMI", f"INR {avg_emi:,.0f}")
                    
                    # Visualizations
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Eligibility distribution pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=df['Predicted_Eligibility'].value_counts().index,
                            values=df['Predicted_Eligibility'].value_counts().values,
                            marker=dict(colors=[SUCCESS, SECONDARY, DANGER]),
                            hole=0.3
                        )])
                        fig.update_layout(
                            title="Eligibility Distribution",
                            paper_bgcolor=WHITE,
                            font=dict(color=TEXT),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # EMI amount distribution
                        fig = go.Figure(data=[go.Histogram(
                            x=df['Predicted_Max_EMI'],
                            marker_color=PRIMARY,
                            nbinsx=30
                        )])
                        fig.update_layout(
                            title="Predicted EMI Distribution",
                            xaxis_title="EMI Amount (INR)",
                            yaxis_title="Count",
                            plot_bgcolor=WHITE,
                            paper_bgcolor=WHITE,
                            font=dict(color=TEXT),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence distribution
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig = go.Figure(data=[go.Box(
                        y=df['Prediction_Confidence'],
                        x=df['Predicted_Eligibility'],
                        marker_color=PRIMARY
                    )])
                    fig.update_layout(
                        title="Prediction Confidence by Eligibility",
                        xaxis_title="Eligibility Status",
                        yaxis_title="Confidence Score",
                        plot_bgcolor=WHITE,
                        paper_bgcolor=WHITE,
                        font=dict(color=TEXT),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed results
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color: {PRIMARY};'>Detailed Results</h3>", unsafe_allow_html=True)
                    
                    # Filter options
                    filter_option = st.selectbox(
                        "Filter by Eligibility",
                        ["All", "Eligible", "High_Risk", "Not_Eligible"]
                    )
                    
                    if filter_option != "All":
                        filtered_df = df[df['Predicted_Eligibility'] == filter_option]
                    else:
                        filtered_df = df
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Download results
                    st.markdown("<br>", unsafe_allow_html=True)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"emi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file matches the template format.")

# ============================================================================
# PAGE: MODEL INFORMATION
# ============================================================================

def model_info_page(models):
    st.markdown(f"<h2 style='color: {PRIMARY};'>Model Information</h2>", unsafe_allow_html=True)
    
    # Classification Model Info
    st.markdown(f"<h3 style='color: {PRIMARY};'>Classification Model</h3>", unsafe_allow_html=True)
    
    clf_metadata = models['metadata']['classification']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Model Details</h4>
                <p style='color: {TEXT};'>
                    <b>Algorithm:</b> {clf_metadata['best_model']}<br>
                    <b>Preprocessing:</b> {clf_metadata['preprocessing_type']}<br>
                    <b>Number of Classes:</b> {clf_metadata['num_classes']}<br>
                    <b>Classes:</b> {', '.join(clf_metadata['class_names'])}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Performance Metrics</h4>
                <p style='color: {TEXT};'>
                    <b>Accuracy:</b> {clf_metadata['best_metrics']['accuracy']:.4f}<br>
                    <b>Precision:</b> {clf_metadata['best_metrics']['precision']:.4f}<br>
                    <b>Recall:</b> {clf_metadata['best_metrics']['recall']:.4f}<br>
                    <b>F1-Score:</b> {clf_metadata['best_metrics']['f1_score']:.4f}<br>
                    <b>ROC-AUC:</b> {clf_metadata['best_metrics']['roc_auc']:.4f}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Model comparison
    st.markdown(f"<h4 style='color: {PRIMARY};'>Model Comparison</h4>", unsafe_allow_html=True)
    
    clf_results_df = pd.DataFrame(clf_metadata['all_results']).T
    clf_results_df = clf_results_df.sort_values('f1_score', ascending=False)
    
    fig = go.Figure()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=clf_results_df.index,
            y=clf_results_df[metric]
        ))
    
    fig.update_layout(
        title="Classification Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=TEXT),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Regression Model Info
    st.markdown(f"<h3 style='color: {PRIMARY};'>Regression Model</h3>", unsafe_allow_html=True)
    
    reg_metadata = models['metadata']['regression']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Model Details</h4>
                <p style='color: {TEXT};'>
                    <b>Algorithm:</b> {reg_metadata['best_model']}<br>
                    <b>Preprocessing:</b> {reg_metadata['preprocessing_type']}<br>
                    <b>Target Variable:</b> Maximum Monthly EMI
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Performance Metrics</h4>
                <p style='color: {TEXT};'>
                    <b>RMSE:</b> INR {reg_metadata['best_metrics']['rmse']:.2f}<br>
                    <b>MAE:</b> INR {reg_metadata['best_metrics']['mae']:.2f}<br>
                    <b>R² Score:</b> {reg_metadata['best_metrics']['r2_score']:.4f}<br>
                    <b>MAPE:</b> {reg_metadata['best_metrics']['mape']:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Model comparison
    st.markdown(f"<h4 style='color: {PRIMARY};'>Model Comparison</h4>", unsafe_allow_html=True)
    
    reg_results_df = pd.DataFrame(reg_metadata['all_results']).T
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=reg_results_df.index,
        y=reg_results_df['rmse'],
        marker_color=PRIMARY
    ))
    
    fig.add_trace(go.Bar(
        name='MAE',
        x=reg_results_df.index,
        y=reg_results_df['mae'],
        marker_color=SECONDARY
    ))
    
    fig.update_layout(
        title="Regression Model Error Comparison",
        xaxis_title="Model",
        yaxis_title="Error (INR)",
        barmode='group',
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=TEXT),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training Information
    st.markdown("---")
    st.markdown(f"<h3 style='color: {PRIMARY};'>Training Information</h3>", unsafe_allow_html=True)
    
    training_info = models['metadata']['training_info']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", f"{training_info['train_size']:,}")
    
    with col2:
        st.metric("Testing Samples", f"{training_info['test_size']:,}")
    
    with col3:
        st.metric("Imbalance Ratio", f"{training_info['imbalance_ratio']:.2f}:1")
    
    st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: {PRIMARY};'>Training Details</h4>
            <p style='color: {TEXT};'>
                <b>Class Imbalance Handling:</b> {str(training_info['class_imbalance_handled'])}<br>
                <b>Label Encoding Method:</b> {training_info['label_encoding_method']}
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================

def about_page():
    st.markdown(f"<h2 style='color: {PRIMARY};'>About This System</h2>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <h3 style='color: {PRIMARY};'>Project Overview</h3>
            <p style='color: {TEXT};'>
                The EMI Eligibility Prediction System is an advanced machine learning application 
                designed to automate and optimize the loan approval process. It combines 
                classification and regression models to determine customer eligibility and 
                predict maximum EMI amounts.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='color: {PRIMARY};'>Key Features</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Single Prediction</h4>
                <p style='color: {TEXT};'>
                    Predict EMI eligibility for individual customers with detailed 
                    financial analysis and recommendations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Batch Processing</h4>
                <p style='color: {TEXT};'>
                    Process multiple customer records simultaneously for efficient 
                    bulk predictions and analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Real-time Analysis</h4>
                <p style='color: {TEXT};'>
                    Get instant predictions with confidence scores and detailed 
                    financial metrics visualization.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='info-box'>
                <h4 style='color: {PRIMARY};'>Model Transparency</h4>
                <p style='color: {TEXT};'>
                    Access comprehensive model information, performance metrics, 
                    and comparison data.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='color: {PRIMARY};'>Technical Specifications</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: {PRIMARY};'>Machine Learning Models</h4>
            <p style='color: {TEXT};'>
                <b>Classification:</b> XGBoost Classifier for eligibility prediction<br>
                <b>Regression:</b> XGBoost Regressor for EMI amount prediction<br>
                <b>Feature Engineering:</b> 10+ derived financial metrics<br>
                <b>Preprocessing:</b> Advanced scaling and encoding techniques
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <h4 style='color: {PRIMARY};'>Performance Highlights</h4>
            <p style='color: {TEXT};'>
                <b>Classification Accuracy:</b> Greater than 95%<br>
                <b>Regression R² Score:</b> Greater than 0.90<br>
                <b>Processing Time:</b> Less than 100ms per prediction<br>
                <b>Model Stability:</b> Validated on 20% holdout test set
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='color: {PRIMARY};'>Business Impact</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='success-box'>
                <h4 style='color: {SUCCESS};'>Efficiency</h4>
                <p style='color: {TEXT};'>
                    Reduce processing time from days to minutes
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='success-box'>
                <h4 style='color: {SUCCESS};'>Accuracy</h4>
                <p style='color: {TEXT};'>
                    Minimize approval errors with ML-driven decisions
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class='success-box'>
                <h4 style='color: {SUCCESS};'>Scalability</h4>
                <p style='color: {TEXT};'>
                    Handle thousands of applications simultaneously
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='color: {PRIMARY};'>Project Information</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <p style='color: {TEXT};'>
                <b>Author:</b> Sridevi V<br>
                <b>Project Type:</b> Classification & Regression<br>
                <b>Contribution:</b> Individual<br>
                <b>Version:</b> 1.0.0<br>
                <b>Last Updated:</b> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='color: {PRIMARY};'>Contact & Support</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='info-box'>
            <p style='color: {TEXT};'>
                For technical support, bug reports, or feature requests, 
                please contact the development team.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()




