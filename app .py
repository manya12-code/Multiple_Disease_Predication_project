import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve model files relative to THIS FILE, not the current working directory.
# This makes the app work no matter where `streamlit run` is launched from.
# Model files live in the SAME folder as app.py (repo root) — not a "models" subfolder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR


# --------------------------------------------------------------------------------------
# MODEL / SCALER LOADING (cached so files are read only once)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_pickle(path):
    if not os.path.exists(path):
        # Give a genuinely helpful error instead of a bare FileNotFoundError,
        # so it's obvious what's missing and where the app is looking.
        existing = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else None
        st.error(
            f"**Could not find model file:** `{path}`\n\n"
            f"Looking inside folder: `{MODEL_DIR}`\n\n"
            f"Files actually found there: {existing if existing is not None else '⚠️ this folder does not exist at all!'}\n\n"
            "👉 Make sure this exact file was uploaded to your GitHub repo, in the same "
            "folder as `app.py`, with the filename spelled exactly the same (case-sensitive)."
        )
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_blood_assets():
    scaler = load_pickle(f"{MODEL_DIR}/bloodt_scaler.pkl")
    model = load_pickle(f"{MODEL_DIR}/bloodtest_model.sav")
    return scaler, model


@st.cache_resource
def load_liver_assets():
    scaler = load_pickle(f"{MODEL_DIR}/liver_scaler.pkl")
    model = load_pickle(f"{MODEL_DIR}/logistic_model_liver.pkl")
    return scaler, model


@st.cache_resource
def load_heart_assets():
    scaler = load_pickle(f"{MODEL_DIR}/heart_scaler.pkl")
    model = load_pickle(f"{MODEL_DIR}/logistic_model (2).pkl")
    return scaler, model


# --------------------------------------------------------------------------------------
# SHARED HELPERS
# --------------------------------------------------------------------------------------
def show_probability_bars(labels, probs, highlight_idx):
    """Render a simple horizontal probability breakdown for each class."""
    order = np.argsort(probs)[::-1]
    for i in order:
        label = labels[i]
        pct = probs[i] * 100
        bar_color = "#2563eb" if i == highlight_idx else "#cbd5e1"
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:2px;">
                    <span>{label}</span><span><b>{pct:.1f}%</b></span>
                </div>
                <div style="background:#e2e8f0;border-radius:6px;height:10px;width:100%;">
                    <div style="background:{bar_color};width:{pct}%;height:10px;border-radius:6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def disclaimer():
    st.caption(
        "⚠️ This tool provides a statistical estimate based on a machine-learning model "
        "and is **not** a medical diagnosis. Please consult a qualified healthcare "
        "professional for any health concerns."
    )


# --------------------------------------------------------------------------------------
# PAGE: HOME
# --------------------------------------------------------------------------------------
def home_page():
    st.title("🏥 Multi-Disease Prediction System")
    st.markdown(
        """
        Welcome! This app uses trained machine-learning models to estimate the risk of
        three different conditions from simple health measurements.

        Choose a module from the sidebar to get started:
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🩸 Blood Test Analysis")
        st.write("Predicts overall blood-health status (Normal, Anemia Risk, "
                  "Diabetes Risk, or Cholesterol Risk) from basic blood-panel values.")
    with col2:
        st.subheader("🫀 Liver Disease")
        st.write("Predicts the likelihood of liver disease from liver-function "
                  "test (LFT) values.")
    with col3:
        st.subheader("❤️ Heart Failure")
        st.write("Predicts heart-failure risk from clinical records such as "
                  "ejection fraction and serum levels.")

    st.divider()
    disclaimer()


# --------------------------------------------------------------------------------------
# PAGE: BLOOD TEST
# --------------------------------------------------------------------------------------
def blood_test_page():
    st.title("🩸 Blood Test Health Analysis")
    st.write("Enter the patient's blood-panel values below to estimate their overall health status.")

    scaler, model = load_blood_assets()
    class_labels = {0: "Anemia Risk", 1: "Cholesterol Risk", 2: "Diabetes Risk", 3: "Normal"}

    with st.form("blood_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=25.0, value=13.5, step=0.1)
        with col2:
            glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=100.0, step=0.5)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80.0, max_value=400.0, value=190.0, step=0.5)
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70.0, max_value=220.0, value=120.0, step=0.5)

        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

    if submitted:
        gender_encoded = 1 if gender == "Male" else 0
        features = pd.DataFrame(
            [[age, gender_encoded, hemoglobin, glucose, cholesterol, systolic_bp]],
            columns=["Age", "Gender_Encoded", "Hemoglobin", "Glucose", "Cholesterol", "Systolic_BP"],
        )
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]
        result_label = class_labels[pred]

        st.divider()
        if result_label == "Normal":
            st.success(f"### ✅ Result: {result_label}")
        else:
            st.warning(f"### ⚠️ Result: {result_label}")

        labels_in_order = [class_labels[c] for c in model.classes_]
        highlight_idx = list(model.classes_).index(pred)
        st.subheader("Confidence Breakdown")
        show_probability_bars(labels_in_order, probs, highlight_idx)

        st.divider()
        disclaimer()


# --------------------------------------------------------------------------------------
# PAGE: LIVER DISEASE
# --------------------------------------------------------------------------------------
def liver_disease_page():
    st.title("🫀 Liver Disease Prediction")
    st.write("Enter the patient's liver-function test (LFT) results below.")

    scaler, model = load_liver_assets()

    with st.form("liver_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=80.0, value=1.0, step=0.1)
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.1, max_value=20.0, value=0.3, step=0.1)
            alk_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", min_value=50.0, max_value=3000.0, value=200.0, step=1.0)
        with col2:
            sgpt = st.number_input("SGPT / ALT (IU/L)", min_value=1.0, max_value=2500.0, value=30.0, step=1.0)
            sgot = st.number_input("SGOT / AST (IU/L)", min_value=1.0, max_value=5000.0, value=35.0, step=1.0)
            total_proteins = st.number_input("Total Proteins (g/dL)", min_value=1.0, max_value=10.0, value=6.5, step=0.1)
            albumin = st.number_input("Albumin (g/dL)", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.1, max_value=3.0, value=1.0, step=0.01)

        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

    if submitted:
        gender_encoded = 1 if gender == "Male" else 0
        features = pd.DataFrame(
            [[age, gender_encoded, total_bilirubin, direct_bilirubin, alk_phosphotase,
              sgpt, sgot, total_proteins, albumin, ag_ratio]],
            columns=[
                "Age of the patient", "Gender of the patient", "Total Bilirubin",
                "Direct Bilirubin", "Alkaline_Phosphotase", "Sgpt Alamine Aminotransferase",
                "Sgot Aspartate Aminotransferase", "Total Protiens", "ALB Albumin",
                "A/G Ratio Albumin and Globulin Ratio",
            ],
        )
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]

        # Model classes_: 1 = Liver Disease, 2 = No Liver Disease
        class_labels = {1: "Liver Disease Detected", 2: "No Liver Disease"}
        result_label = class_labels[pred]

        st.divider()
        if pred == 1:
            st.error(f"### ⚠️ Result: {result_label}")
        else:
            st.success(f"### ✅ Result: {result_label}")

        labels_in_order = [class_labels[c] for c in model.classes_]
        highlight_idx = list(model.classes_).index(pred)
        st.subheader("Confidence Breakdown")
        show_probability_bars(labels_in_order, probs, highlight_idx)

        st.divider()
        disclaimer()


# --------------------------------------------------------------------------------------
# PAGE: HEART FAILURE
# --------------------------------------------------------------------------------------
def heart_failure_page():
    st.title("❤️ Heart Failure Risk Prediction")
    st.write("Enter the patient's clinical record values below.")

    scaler, model = load_heart_assets()

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=110, value=60)
            anaemia = st.selectbox("Anaemia", ["No", "Yes"])
            cpk = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=10.0, max_value=8000.0, value=250.0, step=1.0)
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        with col2:
            ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=10, max_value=80, value=38)
            high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])
            platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=20000.0, max_value=900000.0, value=263000.0, step=1000.0)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.1, step=0.1)
        with col3:
            serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=137)
            sex = st.selectbox("Sex", ["Male", "Female"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            time_followup = st.number_input(
                "Follow-up Period (days)", min_value=1, max_value=300, value=130,
                help="Number of days the patient was followed up clinically after diagnosis.",
            )

        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

    if submitted:
        features = pd.DataFrame(
            [[
                age,
                1 if anaemia == "Yes" else 0,
                cpk,
                1 if diabetes == "Yes" else 0,
                ejection_fraction,
                1 if high_bp == "Yes" else 0,
                platelets,
                serum_creatinine,
                serum_sodium,
                1 if sex == "Male" else 0,
                1 if smoking == "Yes" else 0,
                time_followup,
            ]],
            columns=[
                "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
                "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
                "sex", "smoking", "time",
            ],
        )
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]

        class_labels = {0: "Low Risk (Survived)", 1: "High Risk (Heart Failure)"}
        result_label = class_labels[pred]

        st.divider()
        if pred == 1:
            st.error(f"### ⚠️ Result: {result_label}")
        else:
            st.success(f"### ✅ Result: {result_label}")

        labels_in_order = [class_labels[c] for c in model.classes_]
        highlight_idx = list(model.classes_).index(pred)
        st.subheader("Confidence Breakdown")
        show_probability_bars(labels_in_order, probs, highlight_idx)

        st.divider()
        disclaimer()


# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------------
def main():
    st.sidebar.title("🏥 Navigation")
    page = st.sidebar.radio(
        "Choose a prediction module:",
        ["🏠 Home", "🩸 Blood Test", "🫀 Liver Disease", "❤️ Heart Failure"],
    )

    st.sidebar.divider()
    st.sidebar.info(
        "**About**\n\nThis app demonstrates three machine-learning models "
        "(Random Forest / Logistic Regression) trained to flag potential "
        "health risks from routine test values."
    )

    if page == "🏠 Home":
        home_page()
    elif page == "🩸 Blood Test":
        blood_test_page()
    elif page == "🫀 Liver Disease":
        liver_disease_page()
    elif page == "❤️ Heart Failure":
        heart_failure_page()


if __name__ == "__main__":
    main()
