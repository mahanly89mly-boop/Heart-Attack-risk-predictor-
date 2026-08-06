"""
Heart Attack Risk Predictor — Streamlit web app.

This is a browser-based version of the original Tkinter desktop app (Src.py).
Same model, same preprocessing, same 24 input features — just accessible via
a shareable link instead of running locally.

Run locally:
    streamlit run app.py

Deploy for free:
    https://share.streamlit.io  (Streamlit Community Cloud) — recommended
    https://huggingface.co/spaces  (Hugging Face Spaces) — alternative
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

PRIMARY = "#B23A48"   # clinical red accent
INK = "#1F2937"
MUTED = "#6B7280"
BG_CARD = "#F7F5F2"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {INK}; }}
    .subtitle {{ color: {MUTED}; font-size: 1.02rem; margin-top: -0.6rem; }}
    .disclaimer {{
        background-color: #FFF7ED; border: 1px solid #FBBF77; border-radius: 8px;
        padding: 0.75rem 1rem; font-size: 0.88rem; color: #7C4A03; margin-bottom: 1.2rem;
    }}
    .result-card {{
        border-radius: 12px; padding: 1.4rem 1.6rem; margin-top: 1rem;
    }}
    .result-high {{ background-color: #FDECEC; border: 1px solid #F3B7B7; }}
    .result-low {{ background-color: #EAF7EE; border: 1px solid #A9DDB6; }}
    .result-title {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 0.2rem; }}
    .result-sub {{ color: {MUTED}; font-size: 0.92rem; }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("❤️ Heart Attack Risk Predictor")
st.markdown(
    '<div class="subtitle">Estimate heart-disease risk from clinical, lab, and lifestyle/wearable data.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠️ Educational demo only — not a certified medical device. '
    "Do not use this for real diagnosis or treatment decisions. Always consult a qualified "
    "healthcare provider.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model + encoder (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("Model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Model/OneHotEncoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    return model, ohe


try:
    model, ohe = load_artifacts()
    load_error = None
except Exception as exc:  # pragma: no cover
    model, ohe = None, None
    load_error = str(exc)

CATEGORICAL_COLS = ["smoker_status", "chest_pain_type", "sex"]
NUMERIC_COLS = [
    "age", "resting_bp_systolic", "resting_bp_diastolic", "cholesterol_total",
    "hdl", "ldl", "triglycerides", "fasting_blood_sugar", "hba1c", "bmi",
    "resting_heart_rate", "max_heart_rate_achieved", "exercise_induced_angina",
    "st_depression", "family_history", "alcohol_units_per_week",
    "exercise_minutes_per_week", "sleep_hours", "stress_score",
    "wearable_owner", "daily_steps", "diet_quality_score",
]

with st.sidebar:
    st.header("About")
    st.write(
        "A neural network (TensorFlow/Keras) trained on ~9,000 patient records, combining "
        "24 clinical, lab, and lifestyle/wearable features to classify heart-disease risk."
    )
    st.metric("Test accuracy", "90.5%")
    st.metric("Precision / Recall", "87.4% / 79.9%")
    st.markdown("---")
    st.markdown(
        "**Source code:** [GitHub repository]"
        "(https://github.com/mahanly89mly-boop/Heart-Attack-risk-predictor)"
    )
    st.markdown("Built by **Mahan Liaghatmand**")

if load_error:
    st.error(
        "Couldn't load the trained model artifacts. Make sure `Model/model.pkl` and "
        f"`Model/OneHotEncoder.pkl` are present in the deployment.\n\nDetails: {load_error}"
    )
    st.stop()

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("risk_form"):
    tab1, tab2, tab3 = st.tabs(["🩺 Vitals & History", "🧪 Lab Results", "🏃 Lifestyle & Wearable"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 90, 54)
            sex = st.selectbox("Sex", ["Female", "Male"])
            resting_bp_systolic = st.slider("Resting BP — systolic (mmHg)", 85, 181, 128)
            resting_bp_diastolic = st.slider("Resting BP — diastolic (mmHg)", 50, 126, 81)
            resting_heart_rate = st.slider("Resting heart rate (bpm)", 48, 111, 81)
            max_heart_rate_achieved = st.slider("Max heart rate achieved (bpm)", 93, 210, 166)
        with c2:
            chest_pain_type = st.selectbox(
                "Chest pain type",
                ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"],
            )
            exercise_induced_angina = st.checkbox("Exercise-induced angina")
            st_depression = st.slider("ST depression", 0.0, 6.5, 0.7, 0.1)
            family_history = st.checkbox("Family history of heart disease")
            smoker_status = st.selectbox("Smoker status", ["Never", "Former", "Current"])

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            cholesterol_total = st.slider("Total cholesterol (mg/dL)", 90, 314, 189)
            hdl = st.slider("HDL (mg/dL)", 18, 110, 55)
            ldl = st.slider("LDL (mg/dL)", 35, 207, 103)
        with c2:
            triglycerides = st.slider("Triglycerides (mg/dL)", 35, 390, 151)
            fasting_blood_sugar = st.slider("Fasting blood sugar (mg/dL)", 60, 204, 119)
            hba1c = st.slider("HbA1c (%)", 4.0, 8.6, 5.8, 0.1)
        bmi = st.slider("BMI", 15.0, 43.3, 25.3, 0.1)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            alcohol_units_per_week = st.slider("Alcohol (units/week)", 0.0, 45.9, 4.2, 0.1)
            exercise_minutes_per_week = st.slider("Exercise (min/week)", 0, 366, 139)
            sleep_hours = st.slider("Sleep (hours/night)", 3.1, 11.0, 7.0, 0.1)
        with c2:
            stress_score = st.slider("Stress score (0–100)", 0.0, 100.0, 48.1, 0.1)
            daily_steps = st.slider("Daily steps", 500, 13950, 6178, 50)
            diet_quality_score = st.slider("Diet quality score (0–100)", 4.8, 100.0, 59.6, 0.1)
        wearable_owner = st.checkbox("Owns a fitness wearable")

    submitted = st.form_submit_button("Assess risk", use_container_width=True)

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if submitted:
    row = {
        "age": age, "sex": sex,
        "resting_bp_systolic": resting_bp_systolic, "resting_bp_diastolic": resting_bp_diastolic,
        "cholesterol_total": cholesterol_total, "hdl": hdl, "ldl": ldl,
        "triglycerides": triglycerides, "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c": hba1c, "bmi": bmi, "resting_heart_rate": resting_heart_rate,
        "max_heart_rate_achieved": max_heart_rate_achieved, "chest_pain_type": chest_pain_type,
        "exercise_induced_angina": exercise_induced_angina, "st_depression": st_depression,
        "family_history": family_history, "smoker_status": smoker_status,
        "alcohol_units_per_week": alcohol_units_per_week,
        "exercise_minutes_per_week": exercise_minutes_per_week, "sleep_hours": sleep_hours,
        "stress_score": stress_score, "wearable_owner": wearable_owner,
        "daily_steps": daily_steps, "diet_quality_score": diet_quality_score,
    }
    df = pd.DataFrame([row])

    try:
        encoded = ohe.transform(df[CATEGORICAL_COLS])
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        encoded = np.asarray(encoded).reshape(1, -1)

        numeric = df[NUMERIC_COLS].astype(np.float32).values.reshape(1, -1)

        X = np.concatenate([numeric, encoded], axis=1)
        prediction = model.predict(X)
        probability = float(np.ravel(prediction)[0])
        is_high_risk = probability >= 0.5

        css_class = "result-high" if is_high_risk else "result-low"
        label = "High risk of heart disease" if is_high_risk else "Low risk of heart disease"
        icon = "⚠️" if is_high_risk else "✅"

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="result-title">{icon} {label}</div>
                <div class="result-sub">Estimated probability: <b>{probability:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))

    except Exception as exc:
        st.error(f"Something went wrong while scoring this input: {exc}")
