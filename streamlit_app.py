import joblib
import pandas as pd

import streamlit as st

# Load trained model, scaler, and feature order
rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.title("💳 Bank Customer Churn Predictor")

# Collect raw user input
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0, format="%.2f")
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_crcard = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
est_salary = st.number_input(
    "Estimated Salary", min_value=0.0, value=60000.0, format="%.2f"
)

# Build dataframe
raw_df = pd.DataFrame(
    [
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_crcard,
            "IsActiveMember": is_active,
            "EstimatedSalary": est_salary,
        }
    ]
)

# Apply same encoding as training
df_encoded = pd.get_dummies(raw_df, drop_first=True)

# Align columns with training feature set
df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

# Scale numerics
X_scaled = scaler.transform(df_encoded)

# Predict
if st.button("Predict"):
    pred = rf.predict(X_scaled)[0]
    prob = rf.predict_proba(X_scaled)[0, 1]

    if pred == 1:
        st.error(f"⚠️ Customer likely to **leave** (probability {prob:.1%})")
    else:
        st.success(f"✅ Customer likely to **stay** (probability {prob:.1%})")
