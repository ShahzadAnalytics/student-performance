import streamlit as st
import pickle
import os
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details to predict performance")

# -----------------------------
# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "student_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}")
        st.stop()
    return joblib.load(model_path)  # ✅ correct usage

model = load_model()

# -----------------------------
# Inputs (modify according to your dataset)
# -----------------------------
study_hours = st.number_input(
    "Study Hours per Day", min_value=0.0, max_value=24.0, step=0.5
)

attendance = st.slider(
    "Attendance (%)", min_value=0, max_value=100, value=75
)

previous_score = st.number_input(
    "Previous Exam Score", min_value=0, max_value=100, step=1
)

sleep_hours = st.number_input(
    "Sleep Hours per Day", min_value=0.0, max_value=12.0, step=0.5
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Performance"):
    try:
        # ⚠️ Feature order must match training
        features = np.array([[study_hours, attendance, previous_score, sleep_hours]])
        prediction = model.predict(features)
        st.success(f"📊 Predicted Performance: {prediction[0]}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.error(e)


