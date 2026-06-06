import pandas as pd
import joblib

model = joblib.load(
    "models/credit_scoring_model.pkl"
)

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

prediction = model.predict(
    new_customer
)

if prediction[0] == 0:
    print("Creditworthy")
else:
    print("High Risk")