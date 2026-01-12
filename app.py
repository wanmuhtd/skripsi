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
    page_title="Diabetes AI | Clinical DSS",
    page_icon="🩺",
    layout="wide",
)

# ======================================================
# 2. CUSTOM CSS (MODERN GLASSMORPHISM & ANIMATION)
# ======================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1d2129, #12141a);
        color: #e6edf3;
    }

    /* Custom Card Styling */
    .css-card {
        background: rgba(33, 37, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        margin-bottom: 20px;
    }

    /* Input Styling */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        transition: 0.3s;
    }
    .stNumberInput div[data-baseweb="input"]:focus-within {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.3) !important;
    }

    /* Centered Tabs (Pill Shape) */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #21262d;
        border-radius: 25px;
        padding: 0 30px;
        border: 1px solid #30363d;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d1117 !important;
        font-weight: 600;
    }

    /* Primary Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        color: white;
        border: none;
        padding: 16px;
        border-radius: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.4s all;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4);
    }

    /* Titles */
    .header-text {
        text-align: center;
        background: linear-gradient(to right, #ffffff, #8b949e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. HELPER FUNCTIONS & LOADING
# ======================================================
@st.cache_resource
def load_resources():
    # Simpan model di folder 'models/'
    path = "models"
    scaler = joblib.load(f"{path}/scaler.pkl")
    dae = load_model(f"{path}/dae_model.h5", compile=False)
    stacking = joblib.load(f"{path}/stacking_model.pkl")
    return scaler, dae, stacking

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error("Resources not found. Please check 'models/' directory.")
    st.stop()

# ======================================================
# 4. HEADER SECTION
# ======================================================
st.markdown('<h1 class="header-text">Diabetes Intelligence System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#8b949e; margin-bottom:40px;">Clinical Decision Support System with DAE & Stacking Ensemble</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🩺 Diagnosis Tool", "🧬 Methodology"])

with tab1:
    col_input, col_viz = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("📋 Patient Metrics")
        
        # UX: Mengelompokkan input agar lebih logis
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
            glu = st.number_input("Glucose (mg/dL)", 0, 300, 117, help="Plasma glucose concentration")
            bp = st.number_input("Blood Pressure (mm Hg)", 0, 150, 72)
            ins = st.number_input("Insulin (mu U/ml)", 0, 850, 131)
        with c2:
            stk = st.number_input("Skin Thickness (mm)", 0, 100, 29)
            bmi = st.number_input("BMI (Weight in kg/(m)^2)", 0.0, 70.0, 32.0)
            dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.47, help="Diabetes pedigree function score")
            age = st.number_input("Patient Age", 1, 120, 25)
        
        predict_btn = st.button("Run Clinical Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        if predict_btn:
            # --- LOGIC PROCESSING ---
            feats = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
            data = pd.DataFrame([[preg, glu, bp, stk, ins, bmi, dpf]], columns=feats)
            
            # Median Imputation (Simulation)
            for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                if data[col][0] == 0: data[col] = 100 # Replace with your real medians
            
            scaled_data = scaler.transform(data)
            
            # DAE Compensation Logic
            if age > 30:
                final_feats = dae_model.predict(scaled_data, verbose=0)
                recon_data = scaler.inverse_transform(final_feats)
                show_dae = True
            else:
                final_feats = scaled_data
                show_dae = False

            # --- VISUALIZATION ---
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            
            # Bar Chart Comparison
            fig = go.Figure()
            fig.add_bar(name="Input", x=feats, y=data.values[0], marker_color='#30363d')
            if show_dae:
                fig.add_bar(name="DAE Corrected", x=feats, y=recon_data[0], marker_color='#238636')
            
            fig.update_layout(
                height=250, barmode='group', margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8b949e', legend=dict(orientation="h", y=1.2)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prediction
            prob = stacking_model.predict_proba(final_feats)[0]
            pred = stacking_model.predict(final_feats)[0]
            
            st.divider()
            
            res1, res2 = st.columns(2)
            with res1:
                label = "POSITIVE" if pred == 1 else "NEGATIVE"
                color = "#ff7b72" if pred == 1 else "#3fb950"
                st.markdown(f"### Result:<br><span style='color:{color}; font-size:32px; font-weight:800;'>{label}</span>", unsafe_allow_html=True)
                st.metric("Confidence", f"{max(prob)*100:.1f}%")
            
            with res2:
                # Gauge Chart
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1] * 100,
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': color},
                           'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                    title={'text': "Risk Probability", 'font': {'size': 14}}
                ))
                fig_g.update_layout(height=180, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
                st.plotly_chart(fig_g, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Empty state UX
            st.info("👈 Fill in the patient data and click 'Run Clinical Analysis' to see the result.")

with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 Research Methodology
    Aplikasi ini mengimplementasikan alur kerja data science tingkat lanjut:
    1. **Data Cleaning**: Menggunakan *Median Imputation* untuk menangani nilai nol pada parameter medis.
    2. **Feature Enhancement**: *Denoising Autoencoder (DAE)* digunakan khusus untuk pasien di atas 30 tahun guna menangani variansi data.
    3. **Ensemble Learning**: Model *Stacking* menggabungkan kekuatan Random Forest, XGBoost, dan Logistic Regression.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
