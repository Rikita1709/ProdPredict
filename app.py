import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("productivity_model.pkl")

st.set_page_config(page_title="AI Productivity Analyzer", layout="wide")

st.title("🧠 AI-Powered Productivity & Wellness Analyzer")
st.markdown("Enter your daily stats to predict productivity score.")

# ---------- AI Advice Function ----------
def generate_ai_advice(sleep, stress):
    advice = []
    if sleep < 6:
        advice.append("Try sleeping earlier to improve focus.")
    if stress > 6:
        advice.append("Engage in relaxation activities like deep breathing or a short walk.")
    if not advice:
        advice.append("Maintain the current routine — it's working well.")
    return " ".join(advice)

# Input section
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    study = st.slider("Study Hours", 0.0, 12.0, 4.0)
    mood = st.slider("Mood Score (1-10)", 1, 10, 5)

with col2:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Automatically calculate energy (same logic as dataset)
energy = max(1, min(10, int(sleep - (stress * 0.3))))

if st.button("Predict Productivity 🚀"):

    input_data = [[sleep, study, mood, stress, energy]]
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Productivity Score: {round(prediction,2)}")

    # Productivity Category
    if prediction >= 8:
        st.info("🔥 Excellent productivity expected!")
    elif prediction >= 5:
        st.warning("🙂 Moderate productivity. Manage stress & energy.")
    else:
        st.error("⚠ Low productivity. Consider better sleep & stress control.")

    # Show AI Advice
    advice = generate_ai_advice(sleep, stress)
    st.markdown("### 💡 Personalized AI Advice")
    st.write(advice)
