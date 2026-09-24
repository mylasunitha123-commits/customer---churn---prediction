import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Prepare data
data = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data = pd.get_dummies(data, columns=["Geography", "Gender"], drop_first=True)

X = data.drop("Exited", axis=1)
y = data["Exited"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train, y_train)

st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 850, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
card = st.selectbox("Has Credit Card", ["Yes", "No"])
active = st.selectbox("Is Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

# Create input
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": 1 if card == "Yes" else 0,
    "IsActiveMember": 1 if active == "Yes" else 0,
    "EstimatedSalary": salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0
}])

# Ensure same column order
input_data = input_data.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")
