
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATH = Path("student_performance_model.pkl")

st.title("🎓 Student Performance Prediction")
st.write(
    "Enter student details to predict the performance "
    "category using a Random Forest model."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please upload the model again.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.header("Enter Student Details")

with st.form("student_form"):
    study_hours = st.number_input(
        "Study Hours per Day", min_value=0.0,
        max_value=24.0, value=6.0, step=0.5
    )
    attendance = st.number_input(
        "Attendance (%)", min_value=0.0,
        max_value=100.0, value=85.0
    )
    previous_marks = st.number_input(
        "Previous Marks", min_value=0.0,
        max_value=100.0, value=75.0
    )
    assignment_score = st.number_input(
        "Assignment Score", min_value=0.0,
        max_value=100.0, value=80.0
    )
    internal_score = st.number_input(
        "Internal Score", min_value=0.0,
        max_value=100.0, value=78.0
    )

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    student_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Assignment_Score": assignment_score,
        "Internal_Score": internal_score
    }])

    try:
        prediction = model.predict(student_data)[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Performance: {prediction}")

        explanations = {
            "Poor": (
                "The model classified this student in the Poor "
                "performance category. Review the student's study "
                "habits and scores to identify areas for improvement."
            ),
            "Average": (
                "The model classified this student in the Average "
                "performance category. Consistent study and practice "
                "may help improve performance."
            ),
            "Good": (
                "The model classified this student in the Good "
                "performance category. Maintaining consistent "
                "study habits may help sustain performance."
            )
        }

        st.info(explanations.get(
            str(prediction),
            "Review the predicted category and the student's scores."
        ))

        st.caption(
            "This is a machine-learning prediction, not a guarantee "
            "of actual academic performance."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
