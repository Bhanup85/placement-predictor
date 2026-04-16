import streamlit as st
import joblib
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")

st.title("🎓 Student Placement Predictor")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
attendance = st.slider("Attendance", 0, 100, 75)
internships = st.number_input("Internships", 0, 10, 1)

branch = st.selectbox("Branch", ["CSE", "ECE", "EE", "ME"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'cgpa': cgpa,
        'attendance': attendance,
        'internships': internships,
        'branch': branch,
        'gender': gender
    }])

    xgb_prob = xgb_model.predict_proba(input_data)[0][1]
    lgb_prob = lgb_model.predict_proba(input_data)[0][1]

    final_prob = (xgb_prob + lgb_prob) / 2

    st.write(f"Placement Chance: {final_prob*100:.2f}%")

    if final_prob > 0.6:
        st.success("🎉 High chance of placement!")
    else:
        st.error("⚠️ Not ready yet")
        st.info("👉 Improve skills, projects, and internships — you will be ready soon!")
