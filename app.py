pip install streamlit

import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "models/credit_scoring_model.pkl"
)

st.title("Credit Scoring System")

age = st.number_input("Age")

income = st.number_input("Income")

loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):

    data = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": 1,
        "person_emp_length": 5,
        "loan_intent": 2,
        "loan_grade": 1,
        "loan_amnt": loan_amount,
        "loan_int_rate": 10,
        "loan_percent_income": loan_amount/income,
        "cb_person_default_on_file": 0,
        "cb_person_cred_hist_length": 8,
        "loan_income_ratio": loan_amount/income
    }])

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.success("Creditworthy")
    else:
        st.error("High Risk")