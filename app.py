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
# 2. MODERN DASHBOARD CSS (EQUAL HEIGHT FIX)
# ======================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* Gaya Card menggunakan Container asli Streamlit */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        /* Menyamakan tinggi minimal agar simetris */
        min-height: 540px !important; 
        display: flex;
        flex-direction: column;
    }

    .section-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 1.5rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #21262d;
        border-radius: 30px;
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
        border-radius: 12px;
        border: none;
        padding: 16px;
        font-weight: 800;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background: #2ea043;
        box-shadow: 0 0 15px rgba(35, 134, 54, 0.3);
    }

    .res-label { font-size: 0.9rem; color: #8b949e; margin-bottom: 0px; }
    .res-value { font-size: 2rem; font-weight: 800; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. RESOURCE LOADING
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
# 4. HEADER
# ======================================================
st.markdown("<h1 style='text-align: center; font-weight: 800;'>🩺 Diabetes Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>CDSS with DAE Compensation & Stacking Ensemble</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Analyze Patient", "Research Methodology"])

with tab1:
    # Membagi layout menjadi 2 kolom utama
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="section-title">PATIENT CLINICAL PARAMETERS</p>', unsafe_allow_html=True)
            
            # Grid Input
            in1, in2 = st.columns(2)
            with in1:
                preg = st.number_input("Pregnancies", 0, 20, 1)
                glu = st.number_input("Glucose", 0, 300, 117)
                bp = st.number_input("Blood Pressure", 0, 150, 72)
                ins = st.number_input("Insulin", 0, 850, 131)
            with in2:
                stk = st.number_input("Skin Thickness", 0, 100, 29)
                bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
                dpf = st.number_input("Pedigree Func", 0.0, 3.0, 0.47)
                age = st.number_input("Patient Age", 1, 120, 25)
            
            # Memberikan ruang agar tombol selalu di bawah
            st.write("") 
            predict_btn = st.button("RUN ANALYSIS")

    with col_right:
        if predict_btn:
            # --- LOGIC ---
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF"]
            raw_data = [preg, glu, bp, stk, ins, bmi, dpf]
            df_raw = pd.DataFrame([raw_data], columns=feature_names)
            
            # Median Imputation
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            scaled_input = scaler.transform(df_raw.values)
            final_features = dae_model.predict(scaled_input, verbose=0) if age > 30 else scaled_input

            with st.container(border=True):
                st.markdown('<p class="section-title">FEATURE ANALYSIS (SCALED)</p>', unsafe_allow_html=True)
                
                # Plotly Histogram
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
                
                # --- RESULT ---
                prob = stacking_model.predict_proba(final_features)[0]
                pred = stacking_model.predict(final_features)[0]
                
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    status = "POSITIVE" if pred == 1 else "NEGATIVE"
                    color = "#ff7b72" if pred == 1 else "#3fb950"
                    st.markdown(f"<p class='res-label'>Diagnosis Result</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='res-value' style='color:{color};'>{status}</p>", unsafe_allow_html=True)
                    st.metric("Confidence Score", f"{max(prob)*100:.2f}%")
                
                with res_c2:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob[1] * 100,
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                               'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                        title={'text': "Risk Prob %", 'font': {'size': 14}}
                    ))
                    fig_g.update_layout(height=170, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_g, use_container_width=True)
        else:
            # Placeholder agar tinggi kolom kanan sama dengan kiri saat awal dibuka
            with st.container(border=True):
                st.markdown("<div style='height: 480px; display: flex; align-items: center; justify-content: center; color: #484f58; text-align: center;'>Analysis results will be displayed here after you click the Run Analysis button.</div>", unsafe_allow_html=True)

with tab2:
    with st.container(border=True):
        st.markdown("### Methodology")
        st.write("Sistem ini menggunakan Denoising Autoencoder (DAE) untuk kompensasi fitur dan Stacking Ensemble Classifier.")
