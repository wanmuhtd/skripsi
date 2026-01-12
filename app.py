# !pip install streamlit pandas numpy scikit-learn plotly tensorflow joblib
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
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide",
)

# ======================================================
# 2. CUSTOM CSS (FIX TINGGI SIMETRIS)
# ======================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
    }

    /* Memaksa kontainer di dalam kolom untuk mengisi tinggi 100% */
    [data-testid="column"] > div > [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }

    /* Menjaga agar konten di dalam card tetap rapi */
    .section-header {
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }

    /* Menyelaraskan tombol di bagian bawah jika diperlukan */
    .stButton>button {
        width: 100%;
        background: #238636;
        color: white;
        border-radius: 12px;
        padding: 12px;
        font-weight: 700;
        border: none;
        margin-top: auto; /* Mendorong tombol ke bawah */
    }

    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD RESOURCES (Sesuai Konteks Skripsi)
# ======================================================
@st.cache_resource
def load_resources():
    path = "models"
    # Pastikan file-file ini tersedia di direktori 'models'
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    dae_model = load_model(os.path.join(path, "dae_model.h5"), compile=False)
    stacking_model = joblib.load(os.path.join(path, "stacking_model.pkl"))
    return scaler, dae_model, stacking_model

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# 4. MAIN LAYOUT
# ======================================================
st.markdown("<h1 style='text-align: center; color: white;'>Diabetes Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 30px;'>Kompensasi Fitur DAE & Stacking Ensemble</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analysis Tool", "📖 Methodology"])

with tab1:
    # Menggunakan kolom dengan proporsi yang seimbang
    col_left, col_right = st.columns([1, 1.2], gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-header">Patient Clinical Parameters</p>', unsafe_allow_html=True)
            
            # Grid Input 2 Kolom
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
            
            # Spacer untuk mendorong tombol ke bawah agar sejajar secara visual
            st.markdown("<div style='flex-grow: 1; min-height: 20px;'></div>", unsafe_allow_html=True)
            predict_btn = st.button("Analyze Diabetes Risk")

    with col_right:
        if predict_btn:
            # --- LOGIC PROCESSING ---
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
            df_raw = pd.DataFrame([[preg, glu, bp, stk, ins, bmi, dpf]], columns=feature_names)
            
            # Imputasi Median
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            # Scaling & DAE Logic
            scaled_input = scaler.transform(df_raw.values)
            final_features = dae_model.predict(scaled_input, verbose=0) if age > 30 else scaled_input

            with st.container(border=True):
                st.markdown('<p class="section-header">Feature Analysis (Scaled 0-1)</p>', unsafe_allow_html=True)
                
                # Grafik Perbandingan
                fig = go.Figure()
                fig.add_bar(name="Input", x=feature_names, y=scaled_input[0], marker_color='#30363d')
                if age > 30:
                    fig.add_bar(name="DAE Comp.", x=feature_names, y=final_features[0], marker_color='#58a6ff')
                
                fig.update_layout(
                    height=240, barmode='group', margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b949e', legend=dict(orientation="h", y=1.2, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # --- RESULTS ---
                prob = stacking_model.predict_proba(final_features)[0]
                pred = stacking_model.predict(final_features)[0]
                
                res1, res2 = st.columns(2)
                with res1:
                    status = "POSITIVE" if pred == 1 else "NEGATIVE"
                    color = "#ff7b72" if pred == 1 else "#3fb950"
                    st.markdown(f"### <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{max(prob)*100:.2f}%")

                with res2:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob[1] * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                               'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                        title={'text': "Risk Prob %", 'font': {'size': 12}}
                    ))
                    fig_g.update_layout(height=160, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_g, use_container_width=True)
        else:
            with st.container(border=True):
                st.markdown(
                    "<div style='height: 480px; display: flex; align-items: center; justify-content: center; color: #484f58; text-align: center;'>"
                    "Hasil analisis akan muncul di sini secara otomatis."
                    "</div>", 
                    unsafe_allow_html=True
                )

with tab2:
    with st.container(border=True):
        st.markdown("### Metodologi Penelitian")
        st.write("Aplikasi ini menggunakan alur kerja:")
        st.write("- **Imputasi Median**: Mengatasi data nol.")
        st.write("- **Min-Max Scaling**: Normalisasi fitur ke rentang $$[0, 1]$$.")
        st.write("- **DAE**: Perbaikan fitur untuk usia > 30 tahun.")
