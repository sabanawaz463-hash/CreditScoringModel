# -*- coding: utf-8 -*-
"""CreditScoringModel-EDA.ipynb

# Credit Scoring Model - EDA.ipynb

Install **Libraries**
"""

pip install pandas

pip install numpy

pip install matplotlib

pip install seaborn

pip install scikit-learn

"""## 1. Import Libraries"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

"""## 2. Load Dataset"""

df = pd.read_csv("dataset/credit_risk_dataset.csv")

df.head()

"""## 3. Basic Dataset Information"""

print("Shape:", df.shape)

df.info()

df.describe()

"""## 4. Check Missing Values"""

df.isnull().sum()

"""Visualize missing values:"""

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

"""## 5. Handle Missing Values"""

df.dropna(inplace=True)

"""Verify:"""

df.isnull().sum()

"""## 6. Check Duplicate Records"""

print("Duplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

"""## 7. Target Variable Distribution"""

sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.show()

df['loan_status'].value_counts()

"""## 8. Numerical Feature Distributions"""

df.hist(figsize=(15,12))
plt.tight_layout()
plt.show()

"""## 9. Correlation Matrix"""

plt.figure(figsize=(12,8))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")
plt.show()

"""## 10. Encode Categorical Variables"""

categorical_cols = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

"""Check:"""

df.dtypes

"""No object columns should remain.

## 11. Feature Engineering

Create Loan-Income Ratio
"""

df['loan_income_ratio'] = (
    df['loan_amnt'] /
    df['person_income']
)

"""Preview:"""

df[['loan_amnt',
    'person_income',
    'loan_income_ratio']].head()

"""## 12. Define Features and Target"""

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print(X.shape)
print(y.shape)

"""## 13. Train-Test Split"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

"""## 14. Feature Scaling

Required for Logistic Regression
"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

"""## 15. Logistic Regression"""

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

"""## 16. Decision Tree"""

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

"""## 17. Random Forest"""

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

"""## 18. Evaluation Function"""

def evaluate_model(
    model_name,
    y_true,
    y_pred
):

    print(f"\n{model_name}")

    print(
        "Accuracy:",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Precision:",
        precision_score(y_true, y_pred)
    )

    print(
        "Recall:",
        recall_score(y_true, y_pred)
    )

    print(
        "F1:",
        f1_score(y_true, y_pred)
    )

"""## 19. Evaluate Models"""

evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred
)

evaluate_model(
    "Decision Tree",
    y_test,
    dt_pred
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

"""## 20. Classification Report"""

print(
    classification_report(
        y_test,
        rf_pred
    )
)

"""## 21. Confusion Matrix"""

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

"""## 22. ROC-AUC Score"""

rf_probs = rf.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(
    y_test,
    rf_probs
)

print("ROC-AUC:", roc_auc)

"""## 23. Feature Importance"""

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

importance

"""## 24. Plot Feature Importance"""

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.show()

"""## 25. Compare Models"""

results = pd.DataFrame({
    'Model':[
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy':[
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred)
    ]
})

results

"""## 26. Save Best Model"""

import joblib

joblib.dump(
    rf,
    "models/credit_scoring_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

"""## 27. Test New Customer"""

new_customer = pd.DataFrame([{
    'person_age':30,
    'person_income':60000,
    'person_home_ownership':1,
    'person_emp_length':5,
    'loan_intent':2,
    'loan_grade':1,
    'loan_amnt':10000,
    'loan_int_rate':10,
    'loan_percent_income':0.16,
    'cb_person_default_on_file':0,
    'cb_person_cred_hist_length':8,
    'loan_income_ratio':0.16
}])

prediction = rf.predict(new_customer)

print(prediction)