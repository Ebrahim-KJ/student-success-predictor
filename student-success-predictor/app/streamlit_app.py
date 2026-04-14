"""
streamlit_app.py
----------------
Live interactive dashboard for the Student Success Predictor.
Users can input a student profile and get:
  - A real-time risk score
  - Top factors driving the prediction (SHAP)
  - Causal intervention recommendations
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    return model, scaler, feature_names

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🎓 Student Success Predictor")
st.markdown("""
Predict whether a student is **at risk of academic failure** using machine learning.  
Built on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance).
""")

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — STUDENT PROFILE INPUT
# ─────────────────────────────────────────────

st.sidebar.header("📋 Student Profile")
st.sidebar.markdown("Fill in the student's details below:")

# Demographics
age = st.sidebar.slider("Age", 15, 22, 17)
sex = st.sidebar.selectbox("Sex", ["F", "M"])
address = st.sidebar.selectbox("Address Type", ["Urban", "Rural"])
famsize = st.sidebar.selectbox("Family Size", ["≤3", ">3"])
Pstatus = st.sidebar.selectbox("Parents' Cohabitation", ["Together", "Apart"])

# Parents
Medu = st.sidebar.slider("Mother's Education (0=None, 4=Higher)", 0, 4, 2)
Fedu = st.sidebar.slider("Father's Education (0=None, 4=Higher)", 0, 4, 2)
Mjob = st.sidebar.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
Fjob = st.sidebar.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])

# School
school = st.sidebar.selectbox("School", ["GP", "MS"])
reason = st.sidebar.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
guardian = st.sidebar.selectbox("Guardian", ["mother", "father", "other"])
traveltime = st.sidebar.slider("Travel Time to School (1=<15min, 4=>1hr)", 1, 4, 2)
studytime = st.sidebar.slider("Weekly Study Time (1=<2hrs, 4=>10hrs)", 1, 4, 2)
failures = st.sidebar.slider("Past Class Failures", 0, 3, 0)

# Support
schoolsup = st.sidebar.checkbox("School Extra Educational Support", value=False)
famsup = st.sidebar.checkbox("Family Educational Support", value=True)
paid = st.sidebar.checkbox("Extra Paid Classes", value=False)
activities = st.sidebar.checkbox("Extra-Curricular Activities", value=False)
nursery = st.sidebar.checkbox("Attended Nursery School", value=True)
higher = st.sidebar.checkbox("Wants to Pursue Higher Education", value=True)
internet = st.sidebar.checkbox("Internet Access at Home", value=True)
romantic = st.sidebar.checkbox("In a Romantic Relationship", value=False)

# Lifestyle
famrel = st.sidebar.slider("Family Relationship Quality (1=Very Bad, 5=Excellent)", 1, 5, 4)
freetime = st.sidebar.slider("Free Time After School (1=Very Low, 5=Very High)", 1, 5, 3)
goout = st.sidebar.slider("Going Out with Friends (1=Very Low, 5=Very High)", 1, 5, 3)
Dalc = st.sidebar.slider("Weekday Alcohol Consumption (1=Very Low, 5=Very High)", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol Consumption (1=Very Low, 5=Very High)", 1, 5, 2)
health = st.sidebar.slider("Current Health Status (1=Very Bad, 5=Very Good)", 1, 5, 4)
absences = st.sidebar.slider("Number of School Absences", 0, 50, 4)

# Grades
st.sidebar.markdown("**📝 Previous Grades (0–20 scale)**")
G1 = st.sidebar.slider("Period 1 Grade", 0, 20, 12)
G2 = st.sidebar.slider("Period 2 Grade", 0, 20, 12)

# ─────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────

def build_input_df():
    """Convert UI inputs to model-ready dataframe."""
    raw = {
        "school": 0 if school == "GP" else 1,
        "sex": 0 if sex == "F" else 1,
        "age": age,
        "address": 0 if address == "Urban" else 1,
        "famsize": 0 if famsize == "≤3" else 1,
        "Pstatus": 0 if Pstatus == "Together" else 1,
        "Medu": Medu, "Fedu": Fedu,
        "Mjob": ["at_home","health","other","services","teacher"].index(Mjob),
        "Fjob": ["at_home","health","other","services","teacher"].index(Fjob),
        "reason": ["course","home","other","reputation"].index(reason),
        "guardian": ["father","mother","other"].index(guardian),
        "traveltime": traveltime, "studytime": studytime,
        "failures": failures,
        "schoolsup": int(schoolsup), "famsup": int(famsup),
        "paid": int(paid), "activities": int(activities),
        "nursery": int(nursery), "higher": int(higher),
        "internet": int(internet), "romantic": int(romantic),
        "famrel": famrel, "freetime": freetime, "goout": goout,
        "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences,
        "G1": G1, "G2": G2,
    }

    df = pd.DataFrame([raw])

    # Engineer features (must match training pipeline)
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
    df["alcohol_total"] = df["Dalc"] + df["Walc"]
    df["high_alcohol"] = (df["alcohol_total"] >= 6).astype(int)
    df["social_risk_score"] = (
        (df["goout"] >= 4).astype(int) +
        (df["alcohol_total"] >= 5).astype(int) +
        df["romantic"]
    )
    df["support_score"] = df["famsup"] + df["schoolsup"] + df["paid"]
    df["grade_trajectory"] = df["G2"] - df["G1"]
    df["grade_declining"] = (df["grade_trajectory"] < 0).astype(int)
    df["avg_grade"] = (df["G1"] + df["G2"]) / 2
    df["study_efficiency"] = df["studytime"] / (df["freetime"] + 1)
    df["has_failed_before"] = (df["failures"] > 0).astype(int)

    return df


# ─────────────────────────────────────────────
# PREDICTION & DISPLAY
# ─────────────────────────────────────────────

try:
    model, scaler, feature_names = load_model()

    col1, col2, col3 = st.columns([1.2, 1, 1.2])

    input_df = build_input_df()

    # Align columns with training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    risk_prob = model.predict_proba(input_df)[0][1]
    risk_label = "🔴 At Risk" if risk_prob >= 0.5 else "🟢 Not At Risk"

    with col1:
        st.subheader("🔮 Prediction")
        st.markdown(f"### {risk_label}")

        # Risk gauge
        st.metric(
            label="Risk Probability",
            value=f"{risk_prob:.1%}",
            delta=f"{'High' if risk_prob >= 0.7 else 'Moderate' if risk_prob >= 0.4 else 'Low'} Risk"
        )

        st.progress(float(risk_prob))

        if risk_prob >= 0.7:
            st.error("⚠️ **High Risk** — Immediate intervention recommended.")
        elif risk_prob >= 0.4:
            st.warning("⚠️ **Moderate Risk** — Monitor closely and provide support.")
        else:
            st.success("✅ **Low Risk** — Student appears to be on track.")

    with col2:
        st.subheader("📊 Key Factors")

        # SHAP waterfall for this student
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Get top contributing features
        feature_impacts = list(zip(feature_names, shap_values[0]))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = feature_impacts[:8]

        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        colors = ["#d32f2f" if v > 0 else "#388e3c" for v in values]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(names[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Risk Score")
        ax.set_title("Top Factors")
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("💡 Recommendations")

        recs = []

        if G2 < G1:
            recs.append("📉 **Grade declining** — tutoring or academic support may help.")
        if failures > 0:
            recs.append("❌ **Prior failures detected** — consider early intervention programme.")
        if not schoolsup and risk_prob >= 0.5:
            recs.append("🏫 **School support** — enrolling in extra tutoring could reduce risk.")
        if Dalc >= 3 or Walc >= 4:
            recs.append("🍺 **High alcohol consumption** — may be impacting performance.")
        if absences > 10:
            recs.append(f"📅 **{absences} absences** — attendance is a major risk factor.")
        if not higher:
            recs.append("🎯 **No higher education goal** — motivational counselling recommended.")
        if goout >= 4 and studytime <= 2:
            recs.append("⏱️ **Low study time + high social activity** — time management support.")
        if not recs:
            recs.append("✅ No major risk factors identified. Keep it up!")

        for rec in recs:
            st.markdown(rec)

except FileNotFoundError:
    st.warning("⚠️ No trained models found. Please run `python src/models/train.py` first.")
    st.code("python src/models/train.py", language="bash")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
**Built by Ebrahim Kashif Jahangiri**  
MSc Data Science for Public Policy — Cornell University | MEng — Imperial College London  
[GitHub](https://github.com/YOUR_USERNAME/student-success-predictor) · [LinkedIn](https://linkedin.com/in/yourprofile)
""")
