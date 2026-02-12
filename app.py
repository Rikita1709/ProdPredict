import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("productivity_model.pkl")

st.set_page_config(page_title="AI Productivity Analyzer", layout="wide")

st.title("🧠 AI-Powered Productivity & Wellness Analyzer")

st.markdown("Enter your daily stats to predict productivity score.")

# Input section
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    study = st.slider("Study Hours", 0.0, 12.0, 4.0)
    mood = st.slider("Mood Score (1-10)", 1, 10, 5)

with col2:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    energy = st.slider("Energy Level (1-10)", 1, 10, 6)

if st.button("Predict Productivity 🚀"):
    
    input_data = [[sleep, study, mood, stress, energy]]
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Productivity Score: {round(prediction,2)}")

    if prediction >= 8:
        st.info("🔥 Excellent productivity expected!")
    elif prediction >= 5:
        st.warning("🙂 Moderate productivity. Manage stress & energy.")
    else:
        st.error("⚠ Low productivity. Consider better sleep & stress control.")
