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
# 2. CUSTOM CSS (PREMIUM DARK UI)
# ======================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0d1117; color: #e6edf3; }
    
    /* Card Container */
    .css-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { display: flex; justify-content: center; gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: #21262d; border-radius: 20px; padding: 5px 30px; color: #8b949e; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] { background: #f0f6fc !important; color: #0d1117 !important; font-weight: bold; }

    /* Button Styling */
    .stButton>button {
        width: 100%; background: #238636; color: white; border-radius: 8px; border: none;
        padding: 12px; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { background: #2ea043; transform: translateY(-2px); }

    /* Input Field */
    div[data-baseweb="input"] { background: #0d1117 !important; border: 1px solid #30363d !important; color: white !important; }
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
    st.error(f"Gagal memuat model/scaler: {e}")
    st.stop()

# ======================================================
# 4. HEADER
# ======================================================
st.markdown("<h1 style='text-align: center; color: white;'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Clinical Decision Support System with DAE Compensation & Stacking Ensemble</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analysis Tool", "📚 Methodology"])

with tab1:
    col_input, col_viz = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Patient Clinical Data")
        
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
            age = st.number_input("Age", 1, 120, 25)
        
        predict_btn = st.button("Analyze Diabetes Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        if predict_btn:
            # --- LOGIC PROCESSING ---
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF"]
            raw_data = [preg, glu, bp, stk, ins, bmi, dpf]
            df_raw = pd.DataFrame([raw_data], columns=feature_names)
            
            # 1. Median Imputation (Preprocessing)
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if df_raw[col][0] == 0: df_raw[col] = val

            # 2. SCALING (Penting: Transformasi data ke skala model)
            scaled_input = scaler.transform(df_raw)
            
            # 3. DAE Processing (Jika usia > 30)
            if age > 30:
                scaled_dae = dae_model.predict(scaled_input, verbose=0)
                final_features = scaled_dae
                is_compensated = True
            else:
                final_features = scaled_input
                is_compensated = False

            # --- VISUALIZATION: SCALED FEATURES HISTOGRAM ---
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.caption("📊 Feature Value Comparison (Scaled 0-1)")
            
            fig = go.Figure()
            fig.add_bar(name="Scaled Input", x=feature_names, y=scaled_input[0], marker_color='#444c56')
            
            if is_compensated:
                fig.add_bar(name="DAE Compensated", x=feature_names, y=scaled_dae[0], marker_color='#58a6ff')
            
            fig.update_layout(
                height=250, barmode='group', margin=dict(t=10, b=10, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8b949e', legend=dict(orientation="h", y=1.2)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- COMPARISON TABLE (RAW VS SCALED) ---
            st.caption("📋 Detailed Feature Values")
            comparison_df = pd.DataFrame({
                "Feature": feature_names,
                "Raw Value": raw_data,
                "Scaled (Model View)": np.round(final_features[0], 4)
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- PREDICTION RESULT ---
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            prob = stacking_model.predict_proba(final_features)[0]
            pred = stacking_model.predict(final_features)[0]
            
            r1, r2 = st.columns(2)
            with r1:
                status = "POSITIVE" if pred == 1 else "NEGATIVE"
                color = "#ff7b72" if pred == 1 else "#3fb950"
                st.markdown(f"### Result:<br><span style='color:{color}; font-size:38px; font-weight:800;'>{status}</span>", unsafe_allow_html=True)
                st.metric("Confidence Score", f"{max(prob)*100:.2f}%")
            
            with r2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob[1] * 100,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                           'steps': [{'range': [0, 50], 'color': "#21262d"}, {'range': [50, 100], 'color': "#21262d"}]},
                    title={'text': "Risk Probability %", 'font': {'size': 14}}
                ))
                fig_g.update_layout(height=200, margin=dict(t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_g, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Masukkan data pasien dan klik 'Analyze Diabetes Risk' untuk melihat hasil.")

with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("""
    ### Metodologi Penelitian
    1. **Data Preprocessing**: Pengisian nilai hilang (*missing values*) menggunakan median dari dataset latih.
    2. **Feature Normalization**: Semua fitur klinis ditransformasikan menggunakan `MinMaxScaler` ke rentang [0, 1] agar model neural network bekerja optimal.
    3. **Feature Compensation (DAE)**: Menggunakan *Denoising Autoencoder* untuk memperbaiki kualitas fitur pada kelompok usia dewasa (> 30 tahun).
    4. **Stacking Ensemble**: Prediksi akhir dihasilkan oleh model ensemble yang menggabungkan beberapa algoritma klasifikasi.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
