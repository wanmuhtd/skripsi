import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Diabetes AI Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_resources():
    model_path = 'models'
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    dae_model = load_model(os.path.join(model_path, 'dae_model.h5'), compile=False)
    stacking_model = joblib.load(os.path.join(model_path, 'stacking_model.pkl'))
    return scaler, dae_model, stacking_model

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- 3. SIDEBAR (DOKUMENTASI & PROFIL) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864339.png", width=100)
    st.title("Sistem Pakar Diabetes")
    st.info("""
    **Metode:**
    1. Preprocessing (Median Imputation)
    2. Denoising Autoencoder (Kompensasi Umur > 30)
    3. Stacking Ensemble Classifier
    """)
    st.divider()
    st.markdown("**Pengembang:** Arwan Muhtada")
    st.markdown("**Program Studi:** Teknik Informatika")

# --- 4. HEADER ---
st.title("🏥 Klasifikasi Risiko Diabetes Mellitus")
st.write("Implementasi Model Stacking Ensemble dengan Fitur Terkompensasi (DAE)")

# --- 5. TABS ---
tab1, tab2 = st.tabs(["🔍 Prediksi Pasien", "📚 Informasi Teori"])

with tab1:
    with st.container():
        st.subheader("Input Parameter Klinis")
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
            age = st.number_input("Age (Umur)", 1, 120, 25)

    st.divider()

    if st.button("🚀 ANALISIS SEKARANG"):
        with st.spinner('Sedang memproses data...'):
            # Menyiapkan Data
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]], columns=feature_names)
            
            # --- PROSES IMPUTASI ---
            zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            medians = {'Glucose': 117.0, 'BloodPressure': 72.0, 'SkinThickness': 29.0, 'Insulin': 131.0, 'BMI': 32.05}
            
            was_imputed = False
            for col in zero_cols:
                if input_data[col][0] == 0:
                    input_data[col] = medians[col]
                    was_imputed = True
            
            if was_imputed:
                st.warning("Data mengandung nilai 0 yang tidak valid. Sistem telah melakukan imputasi median.")

            # --- PROSES MODEL ---
            input_scaled = scaler.transform(input_data)
            
            if age > 30:
                features_dae_scaled = dae_model.predict(input_scaled)
                features_dae_original = scaler.inverse_transform(features_dae_scaled)
                
                # Visualisasi Perubahan (Plotly)
                st.subheader("🧬 Hasil Kompensasi Fitur (DAE)")
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Input Asli', x=feature_names, y=input_data.values[0]))
                fig.add_trace(go.Bar(name='Setelah DAE', x=feature_names, y=features_dae_original[0]))
                fig.update_layout(barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                final_features = features_dae_scaled
            else:
                st.info("Kompensasi DAE Tidak Aktif (Umur ≤ 30).")
                final_features = input_scaled

            # PREDIKSI
            prediction = stacking_model.predict(final_features)
            prob = stacking_model.predict_proba(final_features)[0]

            # --- TAMPILAN HASIL ---
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction[0] == 1:
                    st.error("### HASIL: POSITIF")
                else:
                    st.success("### HASIL: NEGATIF")
                
                st.metric("Tingkat Keyakinan", f"{max(prob)*100:.2f}%")

            with res_col2:
                # Gauge Chart untuk probabilitas
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob[1] * 100,
                    title = {'text': "Risiko Diabetes (%)"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "red" if prob[1] > 0.5 else "green"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.header("Metodologi Penelitian")
    st.write("""
    1. **Denoising Autoencoder (DAE)**: Digunakan untuk memperbaiki kualitas fitur dengan cara merekonstruksi input yang dianggap memiliki noise tinggi (pada kelompok usia tertentu).
    2. **Stacking Ensemble**: Menggabungkan kekuatan beberapa algoritma (misal: RF, KNN, XGBoost) untuk meminimalkan kesalahan prediksi masing-masing model.
    3. **Imputasi Median**: Mengatasi masalah *missing value* (angka 0) yang sering ditemukan pada dataset medis.
    """)
