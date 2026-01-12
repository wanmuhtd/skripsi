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
    page_title="Diabetes Intelligence",
    page_icon="🩺",
    layout="wide",
)

# ======================================================
# 2. CSS OPTIMASI UKURAN & SIMETRIS
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

    /* Memaksa kontainer kolom memiliki tinggi yang sama */
    div[data-testid="column"] > div {
        height: 100%;
    }

    /* Styling Card Utama */
    div[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
        min-height: 600px !important; /* Tinggi dikunci agar simetris */
        display: flex;
        flex-direction: column;
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }

    /* Judul Seksi */
    .section-header {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 15px;
        text-transform: uppercase;
    }

    /* Button di dasar kolom kiri */
    .stButton {
        margin-top: auto !important;
        padding-top: 10px;
    }
    .stButton>button {
        width: 100%;
        background: #238636;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-weight: 700;
        border: none;
    }

    /* Menghilangkan margin berlebih pada metric */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD RESOURCES
# ======================================================
@st.cache_resource
def load_resources():
    path = "models"
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    dae_model = load_model(os.path.join(path, "dae_model.h5"), compile=False)
    stacking_model = joblib.load(os.path.join(path, "stacking_model.pkl"))
    return scaler, dae_model, stacking_model

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource: {e}")
    st.stop()

# ======================================================
# 4. MAIN INTERFACE
# ======================================================
st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 0;'>🩺 Diabetes Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>CDSS with DAE Compensation & Stacking Ensemble</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Analysis Tool", "Methodology"])

with tab1:
    col_left, col_right = st.columns([1, 1.2], gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-header">Patient Parameters</p>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                preg = st.number_input("Pregnancies", 0, 20, 1)
                glu = st.number_input("Glucose", 0, 300, 117)
                bp = st.number_input("Blood Pressure", 0, 150, 72)
                ins = st.number_input("Insulin", 0, 850, 131)
            with c2:
                stk = st.number_input("Skin Thickness", 0, 100, 29)
                bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
                dpf = st.number_input("Pedigree Func", 0.0, 3.0, 0.47)
                age = st.number_input("Patient Age", 1, 120, 25)
            
            predict_btn = st.button("RUN ANALYSIS")

    with col_right:
        if predict_btn:
            # --- LOGIC PROCESSING ---
            feature_names = ["Preg", "Glu", "BP", "Skin", "Ins", "BMI", "DPF"]
            df_raw = pd.DataFrame([[preg, glu, bp, stk, ins, bmi, dpf]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"])
            
            # Median Imputation
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            scaled_input = scaler.transform(df_raw.values)
            final_features = dae_model.predict(scaled_input, verbose=0) if age > 30 else scaled_input

            with st.container(border=True):
                st.markdown('<p class="section-header">Feature Analysis (Scaled)</p>', unsafe_allow_html=True)
                
                # Plotly Chart - UKURAN DIKONTROL
                fig = go.Figure()
                fig.add_bar(name="Input", x=feature_names, y=scaled_input[0], marker_color='#30363d')
                if age > 30:
                    fig.add_bar(name="DAE Comp.", x=feature_names, y=final_features[0], marker_color='#58a6ff')
                
                fig.update_layout(
                    height=200, # Perkecil tinggi chart
                    barmode='group', margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', legend=dict(orientation="h", y=1.2, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # --- RESULTS ---
                prob = stacking_model.predict_proba(final_features)[0]
                pred = stacking_model.predict(final_features)[0]
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    status = "POSITIVE" if pred == 1 else "NEGATIVE"
                    color = "#ff7b72" if pred == 1 else "#3fb950"
                    st.markdown(f"<p style='color:#8b949e; font-size:0.8rem; margin-bottom:0;'>Diagnosis</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color}; font-size:1.8rem; font-weight:800; margin-top:-5px;'>{status}</p>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{max(prob)*100:.1f}%")

                with res_col2:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob[1] * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                               'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                        title={'text': "Risk Prob %", 'font': {'size': 12}}
                    ))
                    fig_g.update_layout(height=140, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_g, use_container_width=True)
        else:
            with st.container(border=True):
                st.markdown(
                    "<div style='flex: 1; display: flex; align-items: center; justify-content: center; color: #484f58; text-align: center; border: 1px dashed #30363d; border-radius: 12px; height: 100%;'>"
                    "Analysis results will appear here."
                    "</div>", 
                    unsafe_allow_html=True
                )

with tab2:
    with st.container(border=True):
        st.markdown("### Metodologi")
        st.write("Penelitian ini menggabungkan **Denoising Autoencoder (DAE)** untuk meningkatkan kualitas fitur dan **Stacking Ensemble** untuk klasifikasi risiko diabetes mellitus.")
