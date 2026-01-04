import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kesehatan ML", layout="wide")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_resources():
    model_path = 'models'
    
    # Memuat scaler
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    
    # Memuat DAE dengan compile=False untuk menghindari error metrik
    dae_model = load_model(os.path.join(model_path, 'dae_model.h5'), compile=False)
    
    # Memuat Stacking Classifier
    stacking_model = joblib.load(os.path.join(model_path, 'stacking_model.pkl'))
    
    return scaler, dae_model, stacking_model

# Memanggil resource
try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("Sistem Klasifikasi Kesehatan")
st.markdown("""
Aplikasi ini menggunakan logika **Kompensasi Fitur Otomatis**:
- **Umur > 30**: Fitur diproses melalui **Denoising Autoencoder (DAE)** untuk kompensasi noise.
- **Umur ≤ 30**: Fitur langsung diproses ke tahap klasifikasi tanpa DAE.
""")

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
    # --- INPUT UMUR BARU ---
    age = st.number_input("Age (Umur)", 1, 120, 25)

# Tombol Prediksi
if st.button("Lakukan Prediksi"):
    # 1. Menyiapkan Data (Hanya 7 fitur yang sesuai dengan Scaler/Model)
    # Catatan: Umur digunakan untuk logika IF, bukan input ke model (sesuai metadata scaler Anda)
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
    
    # 2. Preprocessing (Scaling)
    input_scaled = scaler.transform(input_data)
    
    # 3. Logika Kondisional DAE Berdasarkan Umur
    if age > 30:
        st.info(f"💡 Info: Pasien berusia {age} tahun (>30). Mengaktifkan kompensasi fitur DAE.")
        final_features = dae_model.predict(input_scaled)
    else:
        st.info(f"💡 Info: Pasien berusia {age} tahun (≤30). Data langsung diproses tanpa DAE.")
        final_features = input_scaled
    
    # 4. Klasifikasi Akhir (Stacking)
    # Pastikan output DAE memiliki jumlah kolom yang sama dengan input aslinya
    prediction = stacking_model.predict(final_features)
    probabilitas = stacking_model.predict_proba(final_features)
    
    # --- TAMPILAN HASIL ---
    st.divider()
    if prediction[0] == 1:
        st.error(f"### Hasil Prediksi: Positif (Probabilitas: {probabilitas[0][1]*100:.2f}%)")
    else:
        st.success(f"### Hasil Prediksi: Negatif (Probabilitas: {probabilitas[0][0]*100:.2f}%)")

    # Tambahan Informasi Teknis
    with st.expander("Detail Alur Proses"):
        st.write(f"1. Input Data: {input_data.values.tolist()}")
        st.write(f"2. Data Scaled: {input_scaled.tolist()}")
        st.write(f"3. DAE Active: {'Ya' if age > 30 else 'Tidak'}")
