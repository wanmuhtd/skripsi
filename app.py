import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kesehatan ML", layout="wide")

# --- FUNGSI LOAD MODEL ---
# Menggunakan st.cache_resource agar model tidak di-load berulang kali setiap user berinteraksi
@st.cache_resource
def load_resources():
    # Mendefinisikan path folder model
    model_path = 'models'
    
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    dae_model = load_model(os.path.join(model_path, 'dae_model.h5'))
    stacking_model = joblib.load(os.path.join(model_path, 'stacking_model.pkl'))
    
    return scaler, dae_model, stacking_model

# Memanggil resource
try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file berada di folder 'models/'. Error: {e}")

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("Sistem Klasifikasi Kesehatan")
st.write("Aplikasi ini menggunakan integrasi **Denoising Autoencoder** dan **Stacking Classifier**.")

# Layout Kolom untuk Input
col1, col2 = st.columns(2)

with col1:
    st.header("Input Parameter")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 100)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    st.write(" ") # Spacer
    st.write(" ")
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

# Tombol Prediksi
if st.button("Lakukan Prediksi"):
    # 1. Menyiapkan Data
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
    
    # 2. Preprocessing (Scaling)
    input_scaled = scaler.transform(input_data)
    
    # 3. Feature Transformation (DAE)
    features_dae = dae_model.predict(input_scaled)
    
    # 4. Klasifikasi Akhir (Stacking)
    prediction = stacking_model.predict(features_dae)
    probabilitas = stacking_model.predict_proba(features_dae)
    
    # --- TAMPILAN HASIL ---
    st.divider()
    if prediction[0] == 1:
        st.error(f"### Hasil Prediksi: Positif (Probabilitas: {probabilitas[0][1]*100:.2f}%)")
    else:
        st.success(f"### Hasil Prediksi: Negatif (Probabilitas: {probabilitas[0][0]*100:.2f}%)")