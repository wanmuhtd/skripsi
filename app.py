import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ======================================================
# 1. KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. CUSTOM CSS (MINIMALIS & PROFESIONAL)
# ======================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Background */
.main {
    background-color: #fafafa;
    padding: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Headings */
h1, h2, h3 {
    color: #111827;
    font-weight: 600;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 44px;
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #1e40af;
}

/* Card */
.card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}

/* Metric */
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD MODEL
# ======================================================
@st.cache_resource
def load_resources():
    model_path = "models"
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    dae_model = load_model(os.path.join(model_path, "dae_model.h5"), compile=False)
    stacking_model = joblib.load(os.path.join(model_path, "stacking_model.pkl"))
    return scaler, dae_model, stacking_model

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# 4. SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("### Diabetes AI System")
    st.caption("Clinical Decision Support System")

    st.divider()

    st.markdown("""
    **Metodologi**
    - Median Imputation  
    - Denoising Autoencoder (DAE)  
    - Stacking Ensemble Classifier  
    """)

    st.divider()
    st.caption("Developed by Arwan Muhtada")
    st.caption("Teknik Informatika")

# ======================================================
# 5. HEADER
# ======================================================
st.title("Diabetes Risk Prediction System")
st.caption("Stacking Ensemble dengan Feature Compensation (DAE)")

# ======================================================
# 6. TABS
# ======================================================
tab1, tab2 = st.tabs(["Prediction", "Methodology"])

# ======================================================
# TAB 1: PREDIKSI
# ======================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Clinical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 117)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 72)

    with col2:
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 29)
        insulin = st.number_input("Insulin", 0, 900, 131)
        bmi = st.number_input("BMI", 0.0, 70.0, 32.05)

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
        age = st.number_input("Age", 1, 120, 25)

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # ==================================================
    # BUTTON PREDIKSI
    # ==================================================
    if st.button("Run Prediction"):
        with st.spinner("Processing data..."):
            feature_names = [
                "Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction"
            ]

            input_data = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure,
                  skin_thickness, insulin, bmi, dpf]],
                columns=feature_names
            )

            # ---------------------------
            # IMPUTASI NILAI 0
            # ---------------------------
            zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            medians = {
                "Glucose": 117.0,
                "BloodPressure": 72.0,
                "SkinThickness": 29.0,
                "Insulin": 131.0,
                "BMI": 32.05
            }

            imputed = False
            for col in zero_cols:
                if input_data[col][0] == 0:
                    input_data[col] = medians[col]
                    imputed = True

            if imputed:
                st.info("Nilai 0 terdeteksi dan telah digantikan menggunakan imputasi median.")

            # ---------------------------
            # SCALING & DAE
            # ---------------------------
            input_scaled = scaler.transform(input_data)

            if age > 30:
                features_dae_scaled = dae_model.predict(input_scaled, verbose=0)
                features_dae_original = scaler.inverse_transform(features_dae_scaled)

                st.subheader("Feature Compensation Result (DAE)")
                fig = go.Figure()
                fig.add_bar(name="Original", x=feature_names, y=input_data.values[0])
                fig.add_bar(name="After DAE", x=feature_names, y=features_dae_original[0])
                fig.update_layout(
                    barmode="group",
                    height=350,
                    template="simple_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                final_features = features_dae_scaled
            else:
                st.caption("DAE compensation is not applied (Age ≤ 30).")
                final_features = input_scaled

            # ---------------------------
            # PREDIKSI
            # ---------------------------
            prediction = stacking_model.predict(final_features)[0]
            prob = stacking_model.predict_proba(final_features)[0]

        st.divider()

        # ==================================================
        # HASIL
        # ==================================================
        col1, col2 = st.columns([1, 2])

        with col1:
            status = "Positive" if prediction == 1 else "Negative"
            st.markdown(f"### Prediction Result: **{status}**")
            st.metric("Confidence Level", f"{max(prob)*100:.2f}%")

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[1] * 100,
                title={"text": "Diabetes Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626" if prob[1] > 0.5 else "#16a34a"},
                    "bgcolor": "white"
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ======================================================
# TAB 2: METODOLOGI
# ======================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Research Methodology")

    st.markdown("""
    **1. Denoising Autoencoder (DAE)**  
    Digunakan untuk melakukan kompensasi fitur dengan merekonstruksi data
    pada kelompok usia di atas 30 tahun.

    **2. Stacking Ensemble Classifier**  
    Menggabungkan beberapa model pembelajaran mesin untuk meningkatkan
    performa klasifikasi.

    **3. Median Imputation**  
    Menangani nilai tidak valid (0) pada atribut medis yang tidak
    secara klinis bernilai nol.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
