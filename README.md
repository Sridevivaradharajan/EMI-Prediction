# EMI Prediction & Eligibility Classification

This repository contains a machine learning project with two major components:

1. **EMI Amount Prediction** (Regression Model)  
   - Predicts the Equated Monthly Installment (EMI) using loan amount, interest rate, and tenure.

2. **EMI Eligibility Prediction** (Classification Model)  
   - Predicts whether a customer is eligible for a loan based on financial indicators.

---

## 🚀 Project Features

### ✅ 1. EMI Amount Prediction (Regression)
- Predicts monthly EMI using:
  - Loan Amount  
  - Annual Interest Rate  
  - Loan Tenure (in months)
- Machine Learning Models:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor
- Evaluated using RMSE, MAE, and R².

### ✅ 2. EMI Eligibility Prediction (Classification)
- Predicts loan approval status (Approved / Not Approved).
- Input features can include:
  - Income  
  - Credit Score  
  - Existing Loans  
  - Age, Employment Type, etc.
- Machine Learning Models:
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier
  - Voting Ensemble
  - Stacking Ensemble  
- Evaluated using Accuracy, F1-score, ROC-AUC.

---


## 🛠️ Installation

```bash
git clone https://github.com/Sridevivaradharajan/EMI-Prediction.git
cd EMI-Prediction
pip install -r requirements.txt
````

---


## 📜 License

This project is licensed under the
**Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license.
See the **LICENSE** file for full details.
