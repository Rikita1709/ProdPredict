# 🧠 AI-Based Burnout Detection & Productivity Optimization System

An AI-powered machine learning web application that predicts productivity, detects burnout risk, and provides personalized optimization suggestions using behavioral and lifestyle factors.

---

## 📌 Project Overview

This project focuses on analyzing human behavioral patterns such as sleep, stress, mood, study habits, and energy levels to predict productivity and identify possible burnout risks.

Unlike traditional productivity tools that only manage tasks, this system integrates machine learning, behavioral analytics, and AI-generated recommendations to help users improve both productivity and mental well-being 

consists of an AI wellness coach and an interactive AI chat assistant powered by Llama 3 via Groq.

The application is deployed as an interactive web app using Streamlit.


---

# 🚀 Features

✅ Productivity Prediction using Machine Learning  
✅ Burnout Risk Detection  
✅ Personalized AI Advice  
✅ Productivity Optimization Suggestions  
✅ Interactive Web Interface  
✅ AI Wellness Coach (Llama 3 via Groq)
✅ Analytics Dashboard with Visualizations  
✅ Real-Time Predictions  
✅ Live Deployment using Streamlit Cloud  

---

# 🧠 Machine Learning Model

The system uses:

- Random Forest Regression
- Trained using Scikit-learn
- Predicts productivity score based on:
  - Sleep Hours
  - Study Hours
  - Mood Score
  - Stress Level
  - Energy Level

---

# 📊 Synthetic Dataset Generation

One of the major highlights of this project is that the complete dataset was created manually using a custom synthetic rule-based generation system developed in Python.

Instead of using prebuilt datasets, the dataset was generated to simulate realistic human productivity and behavioral patterns.

---

## 📁 Dataset Details

The generated dataset contains 900+ records with the following features:

| Feature | Description |
|---|---|
| Date | Daily record date |
| Sleep_Hours | Hours of sleep |
| Study_Hours | Daily study/work hours |
| Mood | Mood category |
| Mood_Score | Numerical mood rating |
| Stress_Level | Stress rating |
| Energy_Level | Calculated energy score |
| Productivity_Score | Predicted productivity |
| AI_Summary | AI-generated daily summary |
| AI_Advice | AI-generated recommendation |

---

# ⚙️ Synthetic Rule-Based Logic

The dataset was generated using custom behavioral rules such as:

- Higher sleep → higher energy
- Higher stress → lower energy
- Higher study hours + mood + energy → higher productivity
- Low sleep or high stress → burnout indicators

Example logic used:

```python
def generate_energy_level(sleep, stress):
    base = sleep - (stress * 0.3)
    return max(1, min(10, int(base + random.uniform(-1, 1))))
```

```python
def generate_productivity(hours, mood_score, energy):
    return max(1, min(10, int(
        (hours * 0.8) +
        (mood_score * 0.4) +
        (energy * 0.5)
    )))
```

This approach helped simulate realistic productivity trends and behavioral relationships.

---

# 🧯 Burnout Detection System

The project includes a burnout risk assessment module.

Burnout risk is detected using:
- Low sleep
- High stress
- Low predicted productivity

The system classifies users into:
- Low Burnout Risk
- Moderate Burnout Risk
- High Burnout Risk

---

# 📈 Productivity Optimization Engine

A unique optimization module was implemented to suggest improvements.

The system:
- Simulates different sleep values
- Predicts future productivity
- Suggests the optimal sleep duration for improved performance

---

# 📊 Analytics Dashboard

The application also contains visualization modules including:
- Sleep vs Productivity
- Stress vs Productivity
- Behavioral trend analysis

Implemented using:
- Matplotlib
- Streamlit

---

# 🖥️ Tech Stack

## Languages & Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Groq API
- Llama 3.3

---

# 🌐 Deployment

The project is deployed using:
- GitHub
- Streamlit Community Cloud

---

# 📂 Project Structure

```bash
├── app.py
├── train_model.py
├── final_productivity_dataset.csv
├── productivity_model.pkl
├── requirements.txt
└── README.md
```

---

# ▶️ How to Run Locally

## 1️⃣ Clone Repository

```bash
git clone <your-repository-link>
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Train Model

```bash
python train_model.py
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

Future enhancements may include:
- Real-world user data integration
- Deep learning models
- Mobile application support
- User authentication
- Cloud database integration
- Wearable device integration
- Real-time mental health monitoring

---

# 🎯 Project Outcome

This project demonstrates:
- End-to-end machine learning workflow
- Synthetic dataset engineering
- Behavioral analytics
- Burnout detection
- Real-time deployment
- AI-powered recommendation systems

The project successfully combines machine learning and behavioral analysis into an interactive productivity enhancement platform.

---

# 👩‍💻 Developer

Rikita Maiti  
B.Tech CSE Student  
Machine Learning & AI Enthusiast
