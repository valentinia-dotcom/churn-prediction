import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

# Threshold selected during validation
THRESHOLD = 0.33

# -----------------------------
# App title
# -----------------------------
st.title("Customer Churn Prediction")

st.write(
    "Enter customer details below to estimate the probability "
    "of customer churn using the trained Random Forest model."
)

# -----------------------------
# User inputs
# -----------------------------
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.slider(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

num_products = st.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_credit_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

# -----------------------------
# Convert inputs
# -----------------------------
has_credit_card_value = 1 if has_credit_card == "Yes" else 0
is_active_member_value = 1 if is_active_member == "Yes" else 0

input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_credit_card_value],
        "IsActiveMember": [is_active_member_value],
        "EstimatedSalary": [estimated_salary]
    }
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn Risk"):

    probability = model.predict_proba(input_data)[0][1]

    prediction = 1 if probability >= THRESHOLD else 0

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("High Churn Risk")
        st.write(
            "The model predicts that this customer is likely to churn."
        )
    else:
        st.success("Low Churn Risk")
        st.write(
            "The model predicts that this customer is likely to stay."
        )

    st.caption(
        "Decision threshold: 0.33, selected during model validation."
    )

# -----------------------------
# Project information
# -----------------------------
st.divider()

st.subheader("About the Model")

st.write(
    """
    This application uses a Random Forest classifier trained on customer
    banking data. Class weighting was used to handle class imbalance,
    and the classification threshold was optimized using validation data.

    **Model:** Balanced Random Forest  
    **Decision Threshold:** 0.33  
    **Framework:** Scikit-learn  
    """
)
