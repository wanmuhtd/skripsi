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
# 2. MODERN DASHBOARD CSS
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

    /* Card Styling */
    .main-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem;
        height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Subheader Styling */
    .section-title {
        color: #8b949e;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1.5rem;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 1rem;
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
        border: none;
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: #238636;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 18px;
        font-weight: 800;
        font-size: 1rem;
        transition: 0.4s;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: #2ea043;
        box-shadow: 0 0 15px rgba(35, 134, 54, 0.4);
    }

    /* Result Typography */
    .res-label { font-size: 1rem; color: #8b949e; margin-bottom: 0px; }
    .res-value { font-size: 2.2rem; font-weight: 800; margin-top: -10px; }
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
    st.error(f"Error loading resources: {e}")
    st.stop()

# ======================================================
# 4. HEADER SECTION
# ======================================================
st.markdown("<h1 style='text-align: center; color: white; font-weight: 800; margin-bottom: 0px;'>🩺 Diabetes Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 1.1rem;'>Machine Learning-Based Risk Classification System</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Analyze Patient", "Research Methodology"])

with tab1:
    # Membagi layout menjadi 2 kolom utama dengan gap besar agar rapi
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">PATIENT CLINICAL PARAMETERS</p>', unsafe_allow_html=True)
        
        # Grid input 2 kolom di dalam card kiri
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
        
        predict_btn = st.button("RUN DIAGNOSTIC ANALYSIS")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if predict_btn:
            # --- PROCESSING ---
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF"]
            raw_data = [preg, glu, bp, stk, ins, bmi, dpf]
            df_raw = pd.DataFrame([raw_data], columns=feature_names)
            
            # Median Imputation
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            # Scaling & DAE
            scaled_input = scaler.transform(df_raw.values)
            if age > 30:
                final_features = dae_model.predict(scaled_input, verbose=0)
                is_comp = True
            else:
                final_features = scaled_input
                is_comp = False

            # --- VISUALIZATION CARD ---
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">FEATURE ANALYSIS (SCALED)</p>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_bar(name="Input", x=feature_names, y=scaled_input[0], marker_color='#30363d')
            if is_comp:
                fig.add_bar(name="DAE Comp.", x=feature_names, y=final_features[0], marker_color='#58a6ff')
            
            fig.update_layout(
                height=250, barmode='group', margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8b949e', legend=dict(orientation="h", y=1.2, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- RESULT CARD ---
            st.divider()
            prob = stacking_model.predict_proba(final_features)[0]
            pred = stacking_model.predict(final_features)[0]
            
            r_col1, r_col2 = st.columns([1, 1])
            with r_col1:
                label = "POSITIVE" if pred == 1 else "NEGATIVE"
                color = "#ff7b72" if pred == 1 else "#3fb950"
                st.markdown(f"<p class='res-label'>Diagnosis Result</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='res-value' style='color:{color};'>{label}</p>", unsafe_allow_html=True)
                st.metric("Model Confidence", f"{max(prob)*100:.2f}%")
            
            with r_col2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1] * 100,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                           'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                    title={'text': "Risk Probability", 'font': {'size': 14}}
                ))
                fig_g.update_layout(height=180, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_g, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder untuk menjaga layout tetap simetris sebelum dianalisis
            st.markdown('<div class="main-card" style="display: flex; align-items: center; justify-content: center; min-height: 400px; border-style: dashed;">', unsafe_allow_html=True)
            st.markdown("<p style='color: #484f58; font-size: 1.1rem;'>Patient analysis results will appear here.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 Scientific Methodology
    Sistem ini dibangun dengan alur kerja sebagai berikut:
    
    1.  **Data Acquisition**: Menggunakan data medis dengan 8 parameter klinis utama.
    2.  **Preprocessing**: Menangani *missing values* melalui median imputation untuk menjaga integritas data.
    3.  **Feature Normalization**: Menggunakan *Min-Max Scaling* untuk mentransformasi data ke rentang [0, 1], memastikan stabilitas komputasi pada model *Neural Network*.
    4.  **DAE Compensation**: Khusus untuk pasien usia > 30 tahun, data dilewatkan melalui *Denoising Autoencoder* untuk mengkompensasi variansi fitur.
    5.  **Ensemble Classification**: Menggunakan teknik *Stacking* (Random Forest, XGBoost, LR) untuk mencapai akurasi prediksi yang optimal.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
