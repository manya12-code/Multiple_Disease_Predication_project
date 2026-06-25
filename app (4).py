import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    heart_model = pickle.load(open("logistic_model (2).pkl", "rb"))
    heart_scaler = pickle.load(open("heart_scaler.pkl", "rb"))

    liver_model = pickle.load(open("logistic_model_liver.pkl", "rb"))
    liver_scaler = pickle.load(open("liver_scaler.pkl", "rb"))

    blood_model = pickle.load(open("bloodtest_model.sav", "rb"))
    blood_scaler = pickle.load(open("bloodt_scaler.pkl", "rb"))

    return (
        heart_model,
        heart_scaler,
        liver_model,
        liver_scaler,
        blood_model,
        blood_scaler
    )

(
    heart_model,
    heart_scaler,
    liver_model,
    liver_scaler,
    blood_model,
    blood_scaler
) = load_models()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#0E7490;
    font-size:42px;
    font-weight:bold;
}
.result-box{
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='main-title'>🏥 Multi Disease Prediction System</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Disease Selection")

selected = st.sidebar.selectbox(
    "Choose Prediction",
    [
        "Heart Disease",
        "Liver Disease",
        "Blood Health"
    ]
)

# =====================================================
# HEART DISEASE
# =====================================================
if selected == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 45)
        anaemia = st.selectbox("Anaemia", [0, 1])
        creatinine_phosphokinase = st.number_input(
            "Creatinine Phosphokinase",
            value=250
        )
        diabetes = st.selectbox("Diabetes", [0, 1])
        ejection_fraction = st.number_input(
            "Ejection Fraction",
            value=38
        )
        high_blood_pressure = st.selectbox(
            "High Blood Pressure",
            [0, 1]
        )

    with col2:
        platelets = st.number_input(
            "Platelets",
            value=250000.0
        )
        serum_creatinine = st.number_input(
            "Serum Creatinine",
            value=1.0
        )
        serum_sodium = st.number_input(
            "Serum Sodium",
            value=137
        )
        sex = st.selectbox("Sex", [0, 1])
        smoking = st.selectbox("Smoking", [0, 1])
        time = st.number_input("Time", value=100)

    if st.button("Predict Heart Disease"):

        data = np.array([[
            age,
            anaemia,
            creatinine_phosphokinase,
            diabetes,
            ejection_fraction,
            high_blood_pressure,
            platelets,
            serum_creatinine,
            serum_sodium,
            sex,
            smoking,
            time
        ]])

        scaled = heart_scaler.transform(data)
        prediction = heart_model.predict(scaled)[0]

        if prediction == 1:
            st.error("🔴 High Heart Disease Risk")
        else:
            st.success("🟢 Low Heart Disease Risk")

# =====================================================
# LIVER DISEASE
# =====================================================
elif selected == "Liver Disease":

    st.header("🫀 Liver Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 40)
        gender = st.selectbox("Gender", [0, 1])
        total_bilirubin = st.number_input(
            "Total Bilirubin",
            value=1.0
        )
        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            value=0.2
        )
        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase",
            value=200
        )

    with col2:
        sgpt = st.number_input(
            "SGPT Alamine Aminotransferase",
            value=30
        )
        sgot = st.number_input(
            "SGOT Aspartate Aminotransferase",
            value=40
        )
        total_protiens = st.number_input(
            "Total Proteins",
            value=6.5
        )
        albumin = st.number_input(
            "Albumin",
            value=3.5
        )
        ag_ratio = st.number_input(
            "Albumin/Globulin Ratio",
            value=1.0
        )

    if st.button("Predict Liver Disease"):

        data = np.array([[
            age,
            gender,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            sgpt,
            sgot,
            total_protiens,
            albumin,
            ag_ratio
        ]])

        scaled = liver_scaler.transform(data)
        prediction = liver_model.predict(scaled)[0]

        if prediction == 1:
            st.error("🔴 Liver Disease Detected")
        else:
            st.success("🟢 No Liver Disease Detected")

# =====================================================
# BLOOD HEALTH
# =====================================================
elif selected == "Blood Health":

    st.header("🩸 Blood Health Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", [0, 1])
        hemoglobin = st.number_input(
            "Hemoglobin",
            value=13.0
        )

    with col2:
        glucose = st.number_input(
            "Glucose",
            value=90
        )
        cholesterol = st.number_input(
            "Cholesterol",
            value=180
        )
        systolic_bp = st.number_input(
            "Systolic BP",
            value=120
        )

    if st.button("Predict Blood Health"):

        data = np.array([[
            age,
            gender,
            hemoglobin,
            glucose,
            cholesterol,
            systolic_bp
        ]])

        scaled = blood_scaler.transform(data)
        prediction = blood_model.predict(scaled)[0]

        # Adjust mapping if needed
        if prediction == 0:
            st.success("🟢 Healthy")

        elif prediction == 1:
            st.warning("🟡 Anaemia Risk")

        elif prediction == 2:
            st.error("🔴 Diabetes Risk")

        else:
            st.info(f"Prediction Class: {prediction}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "Developed using Machine Learning, Scikit-Learn and Streamlit"
)
