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
# GOOGLE DRIVE DOWNLOAD FUNCTIONS
# ============================================================================
# ================================================================
# ✅ IMPORTS
# ================================================================
import os
import json
import joblib
import streamlit as st
import tempfile
import gdown

# ================================================================
# ✅ GOOGLE DRIVE FOLDER ID (REQUIRED)
# ================================================================
GDRIVE_FOLDER_ID = '1cUbTU0LwyNLhzRjqepxayBFpHtGYnhP0?usp=drive_link'

# ================================================================
# ✅ 1. Create temporary directory for model downloads
# ================================================================
def setup_model_directory():
    """Creates a temporary directory to store downloaded models."""
    temp_dir = tempfile.mkdtemp(prefix="emi_models_")
    return temp_dir


# ================================================================
# ✅ 2. Function to download entire Google Drive folder
# ================================================================
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


# ================================================================
# ✅ 3. Load models from downloaded Google Drive folder
# ================================================================
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

            # ✅ Classification model
            clf_files = [f for f in os.listdir(saved_models_path)
                         if 'classification_model' in f and f.endswith('.pkl')]
            if len(clf_files) == 0:
                st.error("❌ Classification model file not found.")
                return None
            clf_model = joblib.load(os.path.join(saved_models_path, clf_files[0]))

            # ✅ Regression model
            reg_files = [f for f in os.listdir(saved_models_path)
                         if 'regression_model' in f and f.endswith('.pkl')]
            if len(reg_files) == 0:
                st.error("❌ Regression model file not found.")
                return None
            reg_model = joblib.load(os.path.join(saved_models_path, reg_files[0]))

            # ✅ Label encoder & scalers
            label_encoder = joblib.load(os.path.join(saved_models_path, 'GLOBAL_LABEL_ENCODER.pkl'))
            clf_scaler = joblib.load(os.path.join(saved_models_path, 'classification_scaler.pkl'))
            reg_scaler = joblib.load(os.path.join(saved_models_path, 'regression_scaler.pkl'))

            # ✅ Metadata + features
            metadata = json.load(open(os.path.join(saved_models_path, 'model_metadata.json'), 'r'))
            clf_features = json.load(open(os.path.join(saved_models_path, 'classification_features.json'), 'r'))
            reg_features = json.load(open(os.path.join(saved_models_path, 'regression_features.json'), 'r'))

            st.success("✅ Models loaded successfully!")

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
# MAIN APPLICATION (continued in next part)
# ============================================================================

def main():
    st.markdown(f"""
        <div style='background-color: {PRIMARY}; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: {WHITE}; text-align: center; margin: 0;'>
                ☁️ EMI Eligibility Prediction System
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
        4. Update the GDRIVE_CONFIG dictionary at the top of the code
        
        **File ID Format:**
        From: `https://drive.google.com/file/d/FILE_ID_HERE/view`
        Use: `FILE_ID_HERE`
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

# [Include all the page functions from the original code]
# single_prediction_page(), batch_prediction_page(), 
# model_info_page(), about_page() remain the same

if __name__ == "__main__":
    main()





