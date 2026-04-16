import streamlit as st
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Load models
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")

st.title("🎓 Student Placement Predictor (AI Powered)")

st.write("Fill your details to check your placement chances")

# -------- INPUTS --------
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
attendance = st.slider("Attendance (%)", 0, 100, 75)
internships = st.number_input("Internships", 0, 10, 1)
projects = st.number_input("Projects", 0, 10, 2)

python_skill = st.slider("Python Skill (1-5)", 1, 5, 3)
ml_skill = st.slider("ML Skill (1-5)", 1, 5, 2)
communication = st.slider("Communication Skill (1-5)", 1, 5, 3)

branch = st.selectbox("Branch", ["CSE", "ECE", "EE", "ME"])
gender = st.selectbox("Gender", ["Male", "Female"])

# -------- PREDICT --------
if st.button("Predict Placement"):

    input_data = pd.DataFrame([{
        'cgpa': cgpa,
        'attendance': attendance,
        'internships': internships,
        'projects': projects,
        'python_skill': python_skill,
        'ml_skill': ml_skill,
        'communication_skill': communication,
        'branch': branch,
        'gender': gender
    }])

    # Predictions
    xgb_pred = xgb_model.predict(input_data)[0]
    lgb_pred = lgb_model.predict(input_data)[0]

    xgb_prob = xgb_model.predict_proba(input_data)[0][1]
    lgb_prob = lgb_model.predict_proba(input_data)[0][1]

    # Ensemble (average)
    final_prob = (xgb_prob + lgb_prob) / 2

    # -------- OUTPUT --------
    st.subheader("📊 Prediction Results")

    st.write(f"XGBoost Probability: {xgb_prob*100:.2f}%")
    st.write(f"LightGBM Probability: {lgb_prob*100:.2f}%")
    st.write(f"Final Placement Chance: {final_prob*100:.2f}%")

    if final_prob >= 0.6:
        st.success("🎉 High chance of getting placed!")
    else:
        st.error("⚠️ Not ready yet for placement")

        st.info("""
📚 Suggestions to improve:
- Increase technical skills (Python, ML)
- Build real-world projects
- Improve communication skills
- Gain internship experience

👉 Keep learning — you will be ready soon!
        """)
