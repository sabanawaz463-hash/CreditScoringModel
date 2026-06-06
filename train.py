import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/credit_risk_dataset.csv")

df.dropna(inplace=True)

for col in [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df["loan_income_ratio"] = (
    df["loan_amnt"] /
    df["person_income"]
)

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/credit_scoring_model.pkl"
)

print("Model saved successfully")