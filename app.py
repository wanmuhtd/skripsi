import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ======================================================
# 1. PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ======================================================
# 2. GLOBAL CSS (STABLE & FLEXIBLE)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0d1117;
}

/* Card System */
.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 26px;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.section-header {
    color: #8b949e;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 12px;
    margin-bottom: 30px;
}
.stTabs [data-baseweb="tab"] {
    background: #21262d;
    border-radius: 20px;
    padding: 8px 30px;
    border: 1px solid #30363d;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background: #58a6ff !important;
    color: white !important;
}

/* Button */
.stButton>button {
    width: 100%;
    background: #238636;
    color: white;
    border-radius: 10px;
    padding: 14px;
    font-weight: 700;
    border: none;
    margin-top: 30px;
}
.stButton>button:hover {
    background: #2ea043;
}

/* Metric */
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD MODELS
# ======================================================
@st.cache_resource
def load_resources():
    path = "models"
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    dae = load_model(os.path.join(path, "dae_model.h5"), compile=False)
    stacking = joblib.load(os.path.join(path, "stacking_model.pkl"))
    return scaler, dae, stacking

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# 4. PREPROCESS FUNCTIONS
# ======================================================
def preprocess_input(preg, glu, bp, stk, ins, bmi, dpf):
    df = pd.DataFrame([[preg, glu, bp, stk, ins, bmi, dpf]],
        columns=["Pregnancies","Glucose","BloodPressure",
                 "SkinThickness","Insulin","BMI","DPF"])
    medians = {
        "Glucose":117,
        "BloodPressure":72,
        "SkinThickness":29,
        "Insulin":131,
        "BMI":32.05
    }
    for c,v in medians.items():
        df.loc[df[c] == 0, c] = v
    return scaler.transform(df.values)

def apply_dae(features, age):
    if age > 30:
        return dae_model.predict(features, verbose=0), True
    return features, False

# ======================================================
# 5. HEADER
# ======================================================
st.markdown("<h1 style='text-align:center;color:white;'>Diabetes Intelligence System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#8b949e;margin-bottom:35px;'>"
    "Denoising Autoencoder–Based Feature Compensation & Stacking Ensemble Classification"
    "</p>", unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔍 Prediction Tool", "📖 Methodology"])

# ======================================================
# 6. TAB PREDICTION
# ======================================================
with tab1:
    col_left, col_right = st.columns([1, 1.25], gap="large")

    # ---------------- LEFT : INPUT ----------------
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Patient Clinical Parameters</p>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 1)
            glu  = st.number_input("Glucose", 0, 300, 117)
            bp   = st.number_input("Blood Pressure", 0, 150, 72)
            ins  = st.number_input("Insulin", 0, 850, 131)
        with c2:
            stk = st.number_input("Skin Thickness", 0, 100, 29)
            bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
            dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.47)
            age = st.number_input("Patient Age", 1, 120, 25)

        st.markdown("<div style='flex-grow:1'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyze Diabetes Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RIGHT : OUTPUT ----------------
    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Analyzing patient risk profile..."):
                scaled = preprocess_input(preg, glu, bp, stk, ins, bmi, dpf)
                final_features, compensated = apply_dae(scaled, age)
                prob = stacking_model.predict_proba(final_features)[0]
                pred = stacking_model.predict(final_features)[0]

            # Feature Visualization
            st.markdown("<p class='section-header'>Feature Representation (Scaled)</p>", unsafe_allow_html=True)
            feature_names = ["Preg","Glu","BP","Skin","Ins","BMI","DPF"]

            fig = go.Figure()
            fig.add_bar(x=feature_names, y=scaled[0], name="Original", marker_color="#30363d")
            if compensated:
                fig.add_bar(x=feature_names, y=final_features[0], name="DAE Output", marker_color="#58a6ff")

            fig.update_layout(
                height=260,
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#8b949e",
                margin=dict(t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            # Result
            label = "Diabetic Risk Detected" if pred == 1 else "No Diabetic Risk"
            color = "#ff7b72" if pred == 1 else "#3fb950"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<p style='color:#8b949e;font-size:0.8rem;'>Prediction Result</p>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='color:{color};font-size:2.1rem;font-weight:800;margin-top:-5px;'>{label}</p>",
                    unsafe_allow_html=True
                )
                st.metric("Model Confidence", f"{max(prob)*100:.2f}%")

            with c2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1] * 100,
                    gauge={"axis": {"range": [0,100]}, "bar": {"color": color}},
                    title={"text": "Diabetes Probability (%)"}
                ))
                gauge.update_layout(height=190, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(gauge, use_container_width=True)

        else:
            st.markdown(
                "<div style='flex-grow:1;display:flex;align-items:center;justify-content:center;"
                "border:1px dashed #30363d;border-radius:12px;color:#484f58;'>"
                "Prediction output will appear here after analysis."
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 7. TAB METHODOLOGY
# ======================================================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Methodological Framework")
    st.write("""
    **1. Median Imputation**  
    Mengatasi nilai nol pada fitur klinis utama untuk menjaga konsistensi distribusi data.

    **2. Denoising Autoencoder (DAE)**  
    Digunakan sebagai mekanisme kompensasi fitur pada pasien usia dewasa (>30 tahun) untuk mereduksi noise dan meningkatkan representasi laten.

    **3. Stacking Ensemble Classifier**  
    Menggabungkan beberapa model dasar untuk menghasilkan keputusan klasifikasi akhir yang lebih robust dan akurat.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

