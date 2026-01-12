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
# 2. CUSTOM CSS (FIX UNTUK KERAPIAN & TINGGI SEJAJAR)
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

    /* Target utama container agar card memiliki tinggi yang sama */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 16px !important;
        min-height: 600px !important; /* Tinggi dikunci agar simetris */
        display: flex;
        flex-direction: column;
    }

    /* Mengatur jarak antar baris input agar lebih rapi */
    .stNumberInput {
        margin-bottom: 12px !important;
    }

    /* Judul Seksi */
    .section-header {
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }

    /* Tab Centering */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 5px;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #21262d;
        border-radius: 20px;
        padding: 8px 30px;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background: #58a6ff !important;
        color: white !important;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: #238636;
        color: white;
        border-radius: 8px;
        padding: 15px;
        font-weight: 700;
        border: none;
        margin-top: 25px;
    }
    .stButton>button:hover {
        background: #2ea043;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD RESOURCES (DAE & STACKING)
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
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# 4. MAIN LAYOUT
# ======================================================
st.markdown("<h1 style='text-align: center; color: white;'>Diabetes Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 30px;'>Kompensasi Fitur Denoising Autoencoder & Klasifikasi Stacking Ensemble</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Prediction Tool", "📖 Methodology"])

with tab1:
    col_left, col_right = st.columns([1, 1.2], gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-header">Patient Clinical Parameters</p>', unsafe_allow_html=True)
            
            # Input Grid
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
            
            # Button diletakkan di bawah input
            st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
            predict_btn = st.button("Analyze Diabetes Risk")

    with col_right:
        if predict_btn:
            # --- LOGIC PROCESSING ---
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF"]
            df_raw = pd.DataFrame([[preg, glu, bp, stk, ins, bmi, dpf]], columns=feature_names)
            
            # Median Imputation (Simulation based on your context)
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            # Scaling (Gunakan .values untuk menghindari feature name error)
            scaled_input = scaler.transform(df_raw.values)
            
            # DAE Logic
            if age > 30:
                final_features = dae_model.predict(scaled_input, verbose=0)
                is_compensated = True
            else:
                final_features = scaled_input
                is_compensated = False

            # --- VISUALIZATION ---
            with st.container(border=True):
                st.markdown('<p class="section-header">Feature Analysis (Scaled 0-1)</p>', unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_bar(name="Input", x=feature_names, y=scaled_input[0], marker_color='#30363d')
                if is_compensated:
                    fig.add_bar(name="DAE Comp.", x=feature_names, y=final_features[0], marker_color='#58a6ff')
                
                fig.update_layout(
                    height=200, barmode='group', margin=dict(t=20, b=0, l=0, r=0),
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
                    label = "POSITIVE" if pred == 1 else "NEGATIVE"
                    color = "#ff7b72" if pred == 1 else "#3fb950"
                    st.markdown(f"<p style='color:#8b949e; font-size:0.8rem; margin-bottom:0;'>Result</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color}; font-size:2rem; font-weight:800; margin-top:-5px;'>{label}</p>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{max(prob)*100:.2f}%")

                with res_col2:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob[1] * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                               'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                        title={'text': "Risk Prob %", 'font': {'size': 12}}
                    ))
                    fig_g.update_layout(height=200, margin=dict(t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_g, use_container_height=True)
        else:
            # Tampilan awal agar tetap simetris
            with st.container(border=True):
                st.markdown('<p class="section-header">Feature Analysis</p>', unsafe_allow_html=True)
                st.markdown(
                    "<div style='height: 470px; display: flex; align-items: center; justify-content: center; color: #484f58; text-align: center; border: 1px dashed #30363d; border-radius: 10px; margin-bottom: 20px;'>"
                    "Hasil analisis akan muncul di sini setelah Anda memasukkan data pasien dan menekan tombol 'Analyze'."
                    "</div>", 
                    unsafe_allow_html=True
                )

with tab2:
    with st.container(border=True):
        st.markdown("### Landasan Teori & Metodologi [cite: 2025-12-18]")
        st.write("""
        1. **Imputasi Median**: Mengatasi data kosong pada fitur Glucose, Blood Pressure, dsb.
        2. **Denoising Autoencoder (DAE)**: Digunakan untuk mengkompensasi fitur (menghilangkan noise) khusus pada pasien dewasa (> 30 tahun).
        3. **Stacking Ensemble**: Menggabungkan beberapa model klasifikasi dasar untuk meningkatkan akurasi akhir prediksi risiko diabetes.
        """)





















