import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🚢 Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Encode categorical features
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

# Predict
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(input_data)[0]
    st.success("Survived" if prediction == 1 else "Did not survive")