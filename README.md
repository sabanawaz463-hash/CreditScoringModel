# Credit Scoring Model Using Machine Learning

## Project Overview

This project implements a Credit Scoring System using Machine Learning to predict whether a customer is creditworthy based on their financial and credit history.

The system analyzes customer information such as income, loan amount, employment length, credit history, and other financial indicators to classify applicants as either low-risk or high-risk borrowers.

---

## Objectives

* Predict customer creditworthiness.
* Reduce lending risk.
* Compare multiple machine learning algorithms.
* Evaluate model performance using standard classification metrics.

---

## Dataset

Dataset: Credit Risk Dataset

### Features

* person_age
* person_income
* person_home_ownership
* person_emp_length
* loan_intent
* loan_grade
* loan_amnt
* loan_int_rate
* loan_percent_income
* cb_person_default_on_file
* cb_person_cred_hist_length

### Target Variable

* loan_status

Where:

* 0 = Creditworthy / Low Risk
* 1 = High Risk / Potential Default

---

## Project Workflow

### 1. Data Collection

Load the credit risk dataset.

### 2. Data Preprocessing

* Handle missing values
* Remove duplicates
* Encode categorical variables

### 3. Exploratory Data Analysis (EDA)

* Dataset overview
* Missing value analysis
* Target distribution
* Correlation analysis

### 4. Feature Engineering

Created:

* loan_income_ratio

### 5. Model Training

The following models were trained:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 6. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### 7. Model Selection

Random Forest was selected as the final model due to its superior performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook

---

## Project Structure

```text
CreditScoringModel/
│
├── dataset/
│   └── credit_risk_dataset.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── models/
│   ├── credit_scoring_model.pkl
│   └── scaler.pkl
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sabanawaz463-hash/CreditScoringModel.git
```

Navigate to the project:

```bash
cd CreditScoringModel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Train the model:

```bash
python train.py
```

Run predictions:

```bash
python predict.py
```

Run the web application:

```bash
streamlit run app.py
```

---

## Results

The trained model can predict whether a customer is likely to repay a loan based on financial information.

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Future Improvements

* Hyperparameter tuning
* XGBoost implementation
* Model explainability using SHAP
* Deployment using Hugging Face Spaces
* Deployment using Docker

---

## Author

Saba Nawaz

BS Computer Science

Machine Learning Project
