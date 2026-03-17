# Credit Risk Default Prediction using Machine Learning

## Overview
Financial institutions issue thousands of loans daily, and identifying borrowers who may default is critical to reducing financial losses.

This project builds a machine learning system to predict whether a borrower will **default on a loan or not** using demographic, financial, and credit history information.

The project includes complete **EDA, feature engineering, model training, and evaluation** using multiple machine learning algorithms.

---

## Business Problem
Banks and financial institutions must assess credit risk before approving loans.

Incorrect predictions can lead to:

- Financial losses
- Poor interest rate decisions
- Increased default rates  

The goal of this project is to develop a machine learning model that can **accurately identify high-risk borrowers.**

---

## Dataset

Dataset: Credit Risk Dataset (Kaggle)

Records: **32,581 loan applications**

Target Variable:

`loan_status`
- 0 → Non-Default
- 1 → Default

### Features

| Feature | Description |
|------|------|
| person_age | Age of borrower |
| person_income | Annual income |
| person_home_ownership | Rent / Mortgage / Own |
| person_emp_length | Employment duration |
| loan_intent | Purpose of loan |
| loan_grade | Credit grade |
| loan_amnt | Loan amount |
| loan_int_rate | Interest rate |
| loan_percent_income | Loan to income ratio |
| cb_person_default_on_file | Historical default |
| cb_person_cred_hist_length | Credit history length |

---

## Project Workflow

### 1 Data Cleaning
- Removed unrealistic values in `person_age` (>90)
- Removed extreme employment length outliers (>60)
- Handled missing values using **median imputation**
- Removed **165 duplicate records**

Final dataset: **32,408 rows**

---

### 2 Exploratory Data Analysis

EDA included:

- Correlation heatmaps
- Pair plots
- Histograms and boxplots
- Categorical feature analysis

Key findings:

- Most borrowers live in **rented homes or mortgages**
- Loan purposes are **fairly balanced across categories**
- Majority borrowers fall under **loan grades A and B**

---

### 3 Feature Engineering

Three new features were created:

- loan_to_income_ratio = loan_amnt / person_income
- loan_to_emp_length_ratio = person_emp_length / loan_amnt
- int_rate_to_loan_amt_ratio = loan_int_rate / loan_amnt

These ratios capture borrower **financial burden and repayment ability**.

---

### 4 Data Preprocessing

• Train-Test split → **75% / 25%**

Categorical features:
- One-Hot Encoding

Numerical features:
- StandardScaler

---

### 5 Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. CatBoost

Evaluation Metrics:

• Accuracy  
• Precision  
• Recall  
• F1 Score  
• ROC-AUC

Special focus was placed on **Recall and F1 Score**, since detecting defaulters is critical.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| Logistic Regression | 0.87 | 0.78 | 0.56 | 0.65 |
| Random Forest | 0.93 | 0.97 | 0.71 | 0.82 |
| XGBoost | 0.92 | 0.83 | 0.80 | 0.82 |
| CatBoost | 0.92 | 0.84 | 0.79 | 0.82 |

### Best Model: **XGBoost**

Performance:

Recall: **80%**  
F1 Score: **82%**  
ROC-AUC: **0.95**

---

## Feature Importance

Top predictors of loan default:

• person_income  
• loan_int_rate  
• loan_to_income_ratio  
• loan_amnt  
• person_age  
• credit history length

These features strongly influence borrower risk.

---

## Business Insights

Key insights from the model:

• Low income and high **loan-to-income ratio** significantly increase default risk  
• Higher interest rates correlate with higher default probability  
• Longer credit history reduces default likelihood  

Banks can use this model to:

• Screen risky applicants  
• Adjust interest rates  
• Improve loan approval decisions  
• Reduce financial losses

---

## Technologies Used

Python  
Pandas  
NumPy  
Matplotlib  
Seaborn  
Scikit-learn  
XGBoost  
CatBoost

---

## Repository Structure
```
credit-risk-default-prediction/

│
├── data/
│ └── credit_risk_dataset.csv
│
├── notebooks/
│ └── Credit_Risk_Prediction.ipynb
│
├── reports/
│ └── Detailed_Report_Analysis.pdf
│
├── images/
│ └── correlation_heatmap.png
│ └── loan_intent_distribution.png
│ └── model_comparison.png
│ └── roc_curve_xgboost.png
│ └── xgboost_feature_importance.png
│
├── models/
│ └── final_feature_order.pkl
│ └── onehot_columns.pkl
│ └── onehot_encoder.pkl
│ └── scaler_columns.kl
│ └── standard_scaler.pkl
│ └── xgb_credit_model.pkl
│
├── requirements.txt
│
├── model_comparison.csv
│
├── predict.py
│
└── README.md
```

----

## Author

Pahuldeep Singh Dhingra  
MS Data Science & Analytics  
Florida Atlantic University
