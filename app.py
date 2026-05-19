import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load trained model
model = joblib.load("productivity_model.pkl")

st.set_page_config(page_title="AI Productivity Analyzer", layout="wide")

st.title("🧠 AI-Based Burnout Detection & Productivity Optimization System")
st.markdown("Enter your daily stats to predict productivity and assess burnout risk.")

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

# ---------- Burnout Risk Function ----------
def burnout_risk(sleep, stress, productivity):
    risk_score = 0

    if sleep < 5:
        risk_score += 2
    if stress > 7:
        risk_score += 2
    if productivity < 4:
        risk_score += 2

    if risk_score >= 4:
        return "High Burnout Risk 🚨"
    elif risk_score >= 2:
        return "Moderate Burnout Risk ⚠"
    else:
        return "Low Burnout Risk ✅"

# ---------- Optimization Function ----------
def suggest_improvement(sleep, study, mood, stress):
    best_productivity = 0
    best_sleep = sleep

    for extra_sleep in range(int(sleep), 10):
        energy_sim = max(1, min(10, int(extra_sleep - (stress * 0.3))))
        pred = model.predict([[extra_sleep, study, mood, stress, energy_sim]])[0]

        if pred > best_productivity:
            best_productivity = pred
            best_sleep = extra_sleep

    return best_sleep, best_productivity

# ---------- Input Section ----------
col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    study = st.slider("Study Hours", 0.0, 12.0, 4.0)
    mood = st.slider("Mood Score (1-10)", 1, 10, 5)

with col2:
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Automatically calculate energy
energy = max(1, min(10, int(sleep - (stress * 0.3))))

# ---------- Prediction Button ----------
if st.button("Analyze My Productivity 🚀"):

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

    # Burnout Risk
    risk = burnout_risk(sleep, stress, prediction)
    st.markdown("### 🧯 Burnout Risk Assessment")
    st.write(risk)

    # Advice
    advice = generate_ai_advice(sleep, stress)
    st.markdown("### 💡 Personalized AI Advice")
    st.write(advice)

    # Optimization Suggestion
    optimal_sleep, improved_score = suggest_improvement(sleep, study, mood, stress)
    st.markdown("### 📈 Optimization Suggestion")
    st.write(f"If you increase sleep to {optimal_sleep} hours, predicted productivity becomes {round(improved_score,2)}")
    st.markdown("## 📊 Productivity Analytics Dashboard")

    # Load dataset
    df = pd.read_csv("final_productivity_dataset.csv")

    # Graph 1: Sleep vs Productivity
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.scatter(df["Sleep_Hours"], df["Productivity_Score"])
    ax1.set_xlabel("Sleep Hours")
    ax1.set_ylabel("Productivity Score")
    ax1.set_title("Sleep vs Productivity")

    st.pyplot(fig1)

    # Graph 2: Stress vs Productivity
    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.scatter(df["Stress_Level"], df["Productivity_Score"])
    ax2.set_xlabel("Stress Level")
    ax2.set_ylabel("Productivity Score")
    ax2.set_title("Stress vs Productivity")

    st.pyplot(fig2)
