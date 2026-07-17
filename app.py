import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from groq import Groq

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Productivity Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Load ML Model
# ----------------------------
model = joblib.load("productivity_model.pkl")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("final_productivity_dataset.csv")

# ----------------------------
# Connect to Groq API
# ----------------------------
try:
    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )
    st.success("Groq client initialized")
except Exception as e:
    st.error(f"Groq Error: {e}")
    client = None

# ----------------------------
# Session State
# ----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk" not in st.session_state:
    st.session_state.risk = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Title
# ----------------------------
st.title("🧠 AI-Based Burnout Detection & Productivity Optimization System")

st.markdown(
    "Predict productivity, detect burnout risk, receive AI-powered wellness advice, "
    "and chat with an intelligent wellness coach."
)
# ==========================================================
# RULE-BASED AI ADVICE
# ==========================================================

def generate_ai_advice(sleep, stress):

    advice = []

    if sleep < 6:
        advice.append("😴 Increase your sleep duration to improve concentration and recovery.")

    if stress > 6:
        advice.append("🧘 Practice meditation, deep breathing or take regular short breaks.")

    if sleep >= 7 and stress <= 5:
        advice.append("✅ Maintain your current lifestyle. Your routine looks balanced.")

    if len(advice) == 0:
        advice.append("Keep maintaining healthy daily habits.")

    return " ".join(advice)


# ==========================================================
# BURNOUT RISK
# ==========================================================

def burnout_risk(sleep, stress, productivity):

    score = 0

    if sleep < 5:
        score += 2

    if stress > 7:
        score += 2

    if productivity < 4:
        score += 2

    if score >= 4:
        return "High Burnout Risk 🚨"

    elif score >= 2:
        return "Moderate Burnout Risk ⚠️"

    else:
        return "Low Burnout Risk ✅"


# ==========================================================
# PRODUCTIVITY OPTIMIZATION
# ==========================================================

def suggest_improvement(sleep, study, mood, stress):

    best_sleep = sleep
    best_score = 0

    for hrs in range(int(sleep), 10):

        energy = max(1, min(10, int(hrs - (stress * 0.3))))

        pred = model.predict([[hrs, study, mood, stress, energy]])[0]

        if pred > best_score:
            best_score = pred
            best_sleep = hrs

    return best_sleep, best_score


# ==========================================================
# AI WELLNESS COACH
# ==========================================================

def ai_wellness_coach(sleep, study, mood, stress, productivity, risk):

    if client is None:
        return "⚠️ Groq API Key not found. Add GROQ_API_KEY inside Streamlit Secrets."

    prompt = f"""
You are an expert AI Wellness Coach.

Student Details

Sleep Hours : {sleep}
Study Hours : {study}
Mood Score : {mood}/10
Stress Level : {stress}/10
Predicted Productivity : {round(productivity,2)}
Burnout Risk : {risk}

Explain:

1. Why this productivity score occurred.
2. Explain burnout risk.
3. Give 5 personalized suggestions.
4. Suggest tomorrow's routine.
5. End with one motivational sentence.

Keep response under 180 words.
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


# ==========================================================
# AI CHAT
# ==========================================================

def ask_ai(question):

    if client is None:
        return "⚠️ Groq API Key not configured."

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content": "You are a helpful AI Productivity Coach. Answer student questions related to studies, productivity, burnout, stress management and wellness."
            },

            {
                "role": "user",
                "content": question
            }

        ]

    )

    return response.choices[0].message.content
# ==========================================================
# USER INPUT
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
        "😊 Mood Score",
        1,
        10,
        5
    )

with col2:

    stress = st.slider(
        "😟 Stress Level",
        1,
        10,
        5
    )

energy = max(
    1,
    min(
        10,
        int(sleep - (stress * 0.3))
    )
)

# ==========================================================
# ANALYZE BUTTON
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

    st.success(
        f"Predicted Productivity Score: {round(prediction,2)}/10"
    )

    if prediction >= 8:

        st.success("🔥 Excellent productivity expected!")

    elif prediction >= 5:

        st.warning("🙂 Moderate productivity. Manage stress and energy wisely.")

    else:

        st.error("⚠ Low productivity expected. Improve sleep and reduce stress.")

    risk = burnout_risk(
        sleep,
        stress,
        prediction
    )

    st.subheader("🧯 Burnout Risk Assessment")
    st.write(risk)

    advice = generate_ai_advice(
        sleep,
        stress
    )

    st.subheader("💡 Personalized AI Advice")
    st.write(advice)
        # ==========================================================
    # SAVE RESULTS
    # ==========================================================

    st.session_state.prediction = prediction
    st.session_state.risk = risk

    # ==========================================================
    # DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.subheader("📊 Productivity Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📈 Productivity",
            f"{round(prediction,2)}/10"
        )

    with c2:
        st.metric(
            "😴 Sleep",
            f"{sleep} hrs"
        )

    with c3:
        st.metric(
            "😟 Stress",
            f"{stress}/10"
        )

    with c4:
        st.metric(
            "⚡ Energy",
            f"{energy}/10"
        )

    # ==========================================================
    # OPTIMIZATION
    # ==========================================================

    optimal_sleep, improved_score = suggest_improvement(
        sleep,
        study,
        mood,
        stress
    )

    st.subheader("📈 Productivity Optimization")

    st.success(
        f"If you increase your sleep to **{optimal_sleep} hours**, "
        f"your predicted productivity can improve to **{round(improved_score,2)}/10**."
    )

    st.progress(min(int(improved_score * 10), 100))

    # ==========================================================
    # ANALYTICS DASHBOARD
    # ==========================================================

    st.markdown("---")
    st.subheader("📊 Productivity Analytics Dashboard")

    g1, g2 = st.columns(2)

    with g1:

        fig1, ax1 = plt.subplots(figsize=(5,4))

        ax1.scatter(
            df["Sleep_Hours"],
            df["Productivity_Score"],
            alpha=0.7
        )

        ax1.set_title("Sleep vs Productivity")
        ax1.set_xlabel("Sleep Hours")
        ax1.set_ylabel("Productivity Score")

        st.pyplot(fig1)

    with g2:

        fig2, ax2 = plt.subplots(figsize=(5,4))

        ax2.scatter(
            df["Stress_Level"],
            df["Productivity_Score"],
            alpha=0.7
        )

        ax2.set_title("Stress vs Productivity")
        ax2.set_xlabel("Stress Level")
        ax2.set_ylabel("Productivity Score")

        st.pyplot(fig2)
        # ==========================================================
# AI WELLNESS COACH
# ==========================================================

st.markdown("---")
st.subheader("🤖 AI Wellness Coach (Powered by Llama 3)")

if st.button("🧠 Generate AI Wellness Report"):

    if client is None:

        st.error(
            "Groq API Key not found. Please configure GROQ_API_KEY in Streamlit Secrets."
        )

    elif st.session_state.prediction is None:

        st.warning(
            "⚠ Please click 'Analyze My Productivity' first."
        )

    else:

        with st.spinner("Analyzing your lifestyle..."):

            coach_response = ai_wellness_coach(

                sleep,

                study,

                mood,

                stress,

                st.session_state.prediction,

                st.session_state.risk

            )

        st.success("AI Analysis Completed ✅")

        st.markdown(coach_response)
        # ==========================================================
# AI CHAT ASSISTANT
# ==========================================================

st.markdown("---")
st.subheader("💬 Chat with AI Wellness Coach")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input(
    "Ask anything about productivity, stress, studies or wellness..."
)

if user_question:

    # Save & display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Check Groq
    if client is None:

        answer = (
            "⚠️ Groq API Key not configured. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    else:

        # Use prediction if available
        productivity = (
            round(st.session_state.prediction, 2)
            if st.session_state.prediction is not None
            else "Not analyzed yet"
        )

        burnout = (
            st.session_state.risk
            if st.session_state.risk is not None
            else "Unknown"
        )

        prompt = f"""
You are an AI Productivity and Wellness Coach.

Current user information:

Sleep Hours: {sleep}
Study Hours: {study}
Mood Score: {mood}/10
Stress Level: {stress}/10
Energy Level: {energy}/10
Predicted Productivity: {productivity}
Burnout Risk: {burnout}

User Question:
{user_question}

Answer in a friendly, motivating and practical way.
Keep your answer under 200 words.
"""

        with st.spinner("🤖 AI is thinking..."):

            response = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[

                    {
                        "role": "system",
                        "content":
                        "You are an expert AI Wellness Coach helping students improve productivity and reduce burnout."
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

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Developed by Rikita Maiti | AI-Based Burnout Detection & Productivity Optimization System | Powered by Random Forest + Llama 3 (Groq)"
)
