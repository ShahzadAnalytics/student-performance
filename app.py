import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# UI
st.title("🎓 Student Performance Prediction")
st.write("Enter student details to predict performance")

# Inputs (modify according to your dataset)
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.number_input("Previous Exam Score", min_value=0, max_value=100, step=1)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=12.0, step=0.5)

# Predict button
if st.button("Predict Performance"):
    features = np.array([[study_hours, attendance, previous_score, sleep_hours]])
    prediction = model.predict(features)

    st.success(f"📊 Predicted Performance: {prediction[0]}")
