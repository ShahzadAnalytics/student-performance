import streamlit as st
import pickle
import os
import numpy as np

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# --------------------------------------------------
# Load the trained model (pickle)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎓 Student Performance Prediction App")
st.write("Fill in the student details to predict performance")

# Input fields (CHANGE according to your dataset)
study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

attendance = st.slider(
    "Attendance Percentage (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.number_input(
    "Previous Exam Score",
    min_value=0,
    max_value=100,
    step=1
)

sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=12.0,
    step=0.5
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Performance"):
    try:
        # ⚠️ Feature order MUST match training order
        features = np.array([[study_hours, attendance, previous_score, sleep_hours]])
        prediction = model.predict(features)

        st.success(f"📊 Predicted Performance: {prediction[0]}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.error(e)
