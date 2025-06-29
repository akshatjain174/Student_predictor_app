import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🎓 Student Performance Predictor")

# Input fields
hours = st.number_input("Hours Studied", 0.0, 10.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100)
assignments = st.selectbox("Assignments Completed", [0, 1])
marks = st.slider("Internal Marks (out of 15)", 0, 15)

# Predict button
if st.button("Predict Result"):
    input_data = np.array([[hours, attendance, assignments, marks]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        st.success("✅ Predicted Result: PASS 🎉")
    else:
        st.error("❌ Predicted Result: FAIL 😔")
