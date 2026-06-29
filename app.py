import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from groq import Groq

# -------------------------------
# Load ML Model
# -------------------------------
model = joblib.load("productivity_model.pkl")

# -------------------------------
# Initialize Groq Client
# -------------------------------
try:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception:
    client = None

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Productivity Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Burnout Detection & Productivity Optimization System")
st.markdown(
    "Enter your daily lifestyle statistics to predict productivity, assess burnout risk, and receive AI-powered wellness guidance."
)

# ==========================================================
# AI Advice Function
# ==========================================================

def generate_ai_advice(sleep, stress):

    advice = []

    if sleep < 6:
        advice.append(
            "😴 Try increasing your sleep to improve concentration and energy."
        )

    if stress > 6:
        advice.append(
            "🧘 Practice meditation, breathing exercises, or take regular breaks."
        )

    if not advice:
        advice.append(
            "✅ Your routine looks healthy. Keep maintaining this balance."
        )

    return " ".join(advice)

# ==========================================================
# Burnout Risk Function
# ==========================================================

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
        return "Moderate Burnout Risk ⚠️"

    else:
        return "Low Burnout Risk ✅"

# ==========================================================
# Productivity Optimization
# ==========================================================

def suggest_improvement(sleep, study, mood, stress):

    best_productivity = 0
    best_sleep = sleep

    for extra_sleep in range(int(sleep), 10):

        energy_sim = max(
            1,
            min(10, int(extra_sleep - (stress * 0.3)))
        )

        pred = model.predict(
            [[extra_sleep, study, mood, stress, energy_sim]]
        )[0]

        if pred > best_productivity:
            best_productivity = pred
            best_sleep = extra_sleep

    return best_sleep, best_productivity

# ==========================================================
# AI Wellness Coach
# ==========================================================

def ai_wellness_coach(
    sleep,
    study,
    mood,
    stress,
    productivity,
    risk
):

    if client is None:
        return "⚠️ Groq API Key not found. Please configure Streamlit Secrets."

    prompt = f"""
You are an expert AI Wellness Coach.

Student Details

Sleep Hours : {sleep}
Study Hours : {study}
Mood Score : {mood}/10
Stress Level : {stress}/10
Predicted Productivity : {round(productivity,2)}
Burnout Risk : {risk}

Please answer in under 180 words.

Include:

1. Explain why the productivity score is this value.
2. Explain the burnout risk.
3. Give 5 personalized improvement tips.
4. Suggest an ideal routine for tomorrow.
5. End with one motivational sentence.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert productivity coach."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
# ==========================================================
# INPUT SECTION
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider(
        "😴 Sleep Hours",
        0.0,
        12.0,
        7.0
    )

    study = st.slider(
        "📚 Study Hours",
        0.0,
        12.0,
        4.0
    )

    mood = st.slider(
        "😊 Mood Score (1-10)",
        1,
        10,
        5
    )

with col2:

    stress = st.slider(
        "😟 Stress Level (1-10)",
        1,
        10,
        5
    )

# Automatically calculate Energy

energy = max(
    1,
    min(
        10,
        int(sleep - (stress * 0.3))
    )
)

# ==========================================================
# PREDICTION BUTTON
# ==========================================================

if st.button("🚀 Analyze My Productivity"):

    input_data = [[
        sleep,
        study,
        mood,
        stress,
        energy
    ]]

    prediction = model.predict(input_data)[0]

    # ======================================================
    # DASHBOARD
    # ======================================================

    st.markdown("---")
    st.subheader("📊 Productivity Dashboard")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "📈 Productivity",
            f"{round(prediction,2)}/10"
        )

    with m2:
        st.metric(
            "😴 Sleep",
            f"{sleep} hrs"
        )

    with m3:
        st.metric(
            "😟 Stress",
            f"{stress}/10"
        )

    with m4:
        st.metric(
            "⚡ Energy",
            f"{energy}/10"
        )

    st.markdown("---")

    # ======================================================
    # PRODUCTIVITY CATEGORY
    # ======================================================

    if prediction >= 8:

        st.success(
            "🔥 Excellent productivity expected!"
        )

    elif prediction >= 5:

        st.warning(
            "🙂 Moderate productivity. A few improvements can boost your performance."
        )

    else:

        st.error(
            "⚠️ Low productivity predicted. Focus on improving sleep and reducing stress."
        )

    # ======================================================
    # BURNOUT RISK
    # ======================================================

    risk = burnout_risk(
        sleep,
        stress,
        prediction
    )

    st.subheader("🧯 Burnout Risk")

    st.write(risk)

    if "Low" in risk:

        st.progress(20)

    elif "Moderate" in risk:

        st.progress(60)

    else:

        st.progress(100)

    # ======================================================
    # PERSONALIZED ADVICE
    # ======================================================

    advice = generate_ai_advice(
        sleep,
        stress
    )

    st.subheader("💡 Personalized AI Advice")

    st.info(advice)
    # ======================================================
# OPTIMIZATION SUGGESTION
# ======================================================

optimal_sleep, improved_score = suggest_improvement(
    sleep,
    study,
    mood,
    stress
)

st.subheader("📈 Productivity Optimization")

st.success(
    f"If you increase your sleep to **{optimal_sleep} hours**, "
    f"your predicted productivity could improve to **{round(improved_score,2)}/10**."
)

st.progress(int(improved_score * 10))

# ======================================================
# AI WELLNESS COACH
# ======================================================

st.markdown("---")
st.subheader("🤖 AI Wellness Coach (Powered by Llama 3)")

with st.spinner("🧠 AI is analyzing your lifestyle..."):

    coach_response = ai_wellness_coach(
        sleep,
        study,
        mood,
        stress,
        prediction,
        risk
    )

st.success(coach_response)

# ======================================================
# ANALYTICS DASHBOARD
# ======================================================

st.markdown("---")
st.subheader("📊 Productivity Analytics Dashboard")

df = pd.read_csv("final_productivity_dataset.csv")

graph1, graph2 = st.columns(2)

# ------------------------------------------------------
# Graph 1
# ------------------------------------------------------

with graph1:

    fig1, ax1 = plt.subplots(figsize=(5,4))

    ax1.scatter(
        df["Sleep_Hours"],
        df["Productivity_Score"],
        alpha=0.7
    )

    ax1.set_title("Sleep vs Productivity")
    ax1.set_xlabel("Sleep Hours")
    ax1.set_ylabel("Productivity")

    st.pyplot(fig1)

# ------------------------------------------------------
# Graph 2
# ------------------------------------------------------

with graph2:

    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax2.scatter(
        df["Stress_Level"],
        df["Productivity_Score"],
        alpha=0.7
    )

    ax2.set_title("Stress vs Productivity")
    ax2.set_xlabel("Stress Level")
    ax2.set_ylabel("Productivity")

    st.pyplot(fig2)
    # ======================================================
# AI CHAT ASSISTANT
# ======================================================

st.markdown("---")
st.subheader("💬 Chat with AI Wellness Coach")

# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input(
    "Ask anything about productivity, stress, studies or wellness..."
)

if user_question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    if client is None:

        answer = (
            "⚠️ Groq API Key not configured. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    else:

        prompt = f"""
You are an AI Productivity and Wellness Coach.

Current User Data

Sleep Hours: {sleep}
Study Hours: {study}
Mood Score: {mood}/10
Stress Level: {stress}/10
Energy Level: {energy}/10
Predicted Productivity: {round(prediction,2)}/10
Burnout Risk: {risk}

The user asked:

{user_question}

Give a practical, friendly and motivating answer.
Keep it under 200 words.
"""

        with st.spinner("🤖 Thinking..."):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI Wellness Coach "
                            "who helps students improve productivity "
                            "and avoid burnout."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "Developed by Rikita Maiti | AI-Based Burnout Detection & Productivity Optimization System | Powered by Random Forest + Llama 3 (Groq)"
)