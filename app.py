import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from groq import Groq


# Load trained model
model = joblib.load("productivity_model.pkl")

try:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception:
    client = None

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
# ---------- AI Wellness Coach ----------
def ai_wellness_coach(sleep, study, mood, stress, productivity, risk):

    if client is None:
        return "⚠️ AI Coach is unavailable. Please add your GROQ_API_KEY in Streamlit Secrets."

    prompt = f"""
You are an AI Wellness Coach.

Analyze the following student data:

Sleep Hours: {sleep}
Study Hours: {study}
Mood Score: {mood}/10
Stress Level: {stress}/10
Predicted Productivity: {round(productivity,2)}
Burnout Risk: {risk}

Explain:

1. Why the productivity score is this value.
2. Whether burnout is likely.
3. Give 5 personalized improvement suggestions.
4. Suggest an ideal routine for tomorrow.
5. End with one motivational sentence.

Keep the answer friendly and under 180 words.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert productivity and wellness coach."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

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

    # ---------- Dashboard Metrics ----------
st.markdown("## 📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Productivity",
        value=f"{round(prediction,2)}/10"
    )

with col2:
    st.metric(
        label="😴 Sleep",
        value=f"{sleep} hrs"
    )

with col3:
    st.metric(
        label="😟 Stress",
        value=f"{stress}/10"
    )

with col4:
    st.metric(
        label="⚡ Energy",
        value=f"{energy}/10"
    )

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
    # ---------- Burnout Meter ----------
st.markdown("### 📉 Burnout Level")

if "Low" in risk:
    st.progress(20)

elif "Moderate" in risk:
    st.progress(60)

else:
    st.progress(100)

    # Advice
    advice = generate_ai_advice(sleep, stress)
    st.markdown("### 💡 Personalized AI Advice")
    st.info(advice)

    # Optimization Suggestion
    optimal_sleep, improved_score = suggest_improvement(sleep, study, mood, stress)
    st.markdown("### 📈 Optimization Suggestion")
    st.markdown("### 🎯 Productivity Level")

    st.progress(int(prediction * 10))
    st.write(f"If you increase sleep to {optimal_sleep} hours, predicted productivity becomes {round(improved_score,2)}")
    st.markdown("## 📊 Productivity Analytics Dashboard")
    # ---------- AI Wellness Coach ----------
    st.markdown("## 🤖 AI Wellness Coach")

    with st.spinner("Analyzing your wellness using Llama 3..."):

        coach_response = ai_wellness_coach(
        sleep,
        study,
        mood,
        stress,
        prediction,
        risk
    )

    st.success(coach_response)

    # Load dataset
    df = pd.read_csv("final_productivity_dataset.csv")

    # Graph 1: Sleep vs Productivity
    fig1, ax1 = plt.subplots(figsize=(3,2))
    ax1.scatter(df["Sleep_Hours"], df["Productivity_Score"])
    ax1.set_xlabel("Sleep Hours")
    ax1.set_ylabel("Productivity Score")
    ax1.set_title("Sleep vs Productivity")

    

    # Graph 2: Stress vs Productivity
    fig2, ax2 = plt.subplots(figsize=(3,2))
    ax2.scatter(df["Stress_Level"], df["Productivity_Score"])
    ax2.set_xlabel("Stress Level")
    ax2.set_ylabel("Productivity Score")
    ax2.set_title("Stress vs Productivity")

    graph1, graph2 = st.columns(2)

with graph1:
    st.pyplot(fig1)

with graph2:
    st.pyplot(fig2)
    # ---------- AI Chat Assistant ----------
st.markdown("---")
st.markdown("## 💬 Chat with AI Wellness Coach")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Ask anything about your productivity, stress, studies or wellness...")

if user_question:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Create prompt using current prediction
    prompt = f"""
You are an AI Productivity and Wellness Coach.

Current User Data:
Sleep Hours: {sleep}
Study Hours: {study}
Mood Score: {mood}
Stress Level: {stress}
Predicted Productivity: {round(prediction,2)}
Burnout Risk: {risk}

User Question:
{user_question}

Answer in a friendly, practical and motivational way.
"""

    if client is None:
        answer = "⚠️ Please configure your GROQ_API_KEY."
    else:

        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI Wellness Coach."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("---")

st.caption(
    "Developed by Rikita Maiti | AI-Based Burnout Detection & Productivity Optimization System | Powered by Random Forest + Llama 3"
)
