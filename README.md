# Credit Risk Default Prediction using Machine Learning

## Overview
Financial institutions issue thousands of loans daily, and identifying borrowers who may default is critical to reducing financial losses.

This project builds a machine learning system to predict whether a borrower will **default on a loan or not** using demographic, financial, and credit history information.

The project includes complete **EDA, feature engineering, model training, and evaluation** using multiple machine learning algorithms.

---

## Business Problem
Banks and financial institutions must assess credit risk before approving loans.

Incorrect predictions can lead to:

• Financial losses  
• Poor interest rate decisions  
• Increased default rates  

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
• Removed unrealistic values in `person_age` (>90)  
• Removed extreme employment length outliers (>60)  
• Handled missing values using **median imputation**  
• Removed **165 duplicate records**

Final dataset: **32,408 rows**

---

### 2 Exploratory Data Analysis

EDA included:

• Correlation heatmaps  
• Pair plots  
• Histograms and boxplots  
• Categorical feature analysis

Key findings:

• Most borrowers live in **rented homes or mortgages**  
• Loan purposes are **fairly balanced across categories**  
• Majority borrowers fall under **loan grades A and B**

---

### 3 Feature Engineering

Three new features were created:
