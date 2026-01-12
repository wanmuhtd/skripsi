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
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed" # Sesuai gambar, sidebar tidak dominan
)

# ======================================================
# 2. CUSTOM CSS (MENYELARASKAN DENGAN UI DESIGN)
# ======================================================
st.markdown("""
<style>
/* Background Utama */
.stApp {
    background-color: #1a1d23;
    color: #ffffff;
}

/* Menengahkan Judul & Subjudul */
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 600;
    margin-bottom: 0px;
}
.sub-title {
    text-align: center;
    color: #9aa0a6;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* Styling Tab agar seperti Tombol Pil di Tengah */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 20px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: #2b3036;
    border-radius: 30px;
    padding: 10px 40px;
    color: white;
    border: 1px solid #444c56;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #1a1d23 !important;
}

/* Card Input & Result */
.custom-card {
    background-color: #21252b;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #30363d;
    height: 100%;
}

/* Input Fields */
div[data-baseweb="input"] {
    background-color: #2b3036 !important;
    border-radius: 8px !important;
}

/* Tombol Biru Muda Sesuai Gambar */
.stButton>button {
    width: 100%;
    border-radius: 30px;
    background-color: #a8c7fa; 
    color: #1a1d23;
    border: none;
    padding: 15px;
    font-weight: bold;
    font-size: 1.2rem;
    margin-top: 20px;
}
.stButton>button:hover {
    background-color: #d2e3fc;
    color: #1a1d23;
}

/* Label Input */
label {
    color: #ffffff !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD MODEL
# ======================================================
@st.cache_resource
def load_resources():
    model_path = "models"
    # Pastikan file ini ada di folder models/
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    dae_model = load_model(os.path.join(model_path, "dae_model.h5"), compile=False)
    stacking_model = joblib.load(os.path.join(model_path, "stacking_model.pkl"))
    return scaler, dae_model, stacking_model

try:
    scaler, dae_model, stacking_model = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# 4. HEADER
# ======================================================
st.markdown('<h1 class="main-title">Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Implementasi Model Stacking Ensemble dengan Fitur Terkompensasi (DAE)</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Tool", "Methodology"])

with tab1:
    # Membagi Layout Utama: Kiri (Input), Kanan (Visualisasi & Hasil)
    main_col1, main_col2 = st.columns([1.2, 1], gap="large")

    with main_col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Patient Clinical Parameters")
        
        # Grid 2 Kolom untuk Input (4 baris x 2 kolom = 8 field)
        in_col1, in_col2 = st.columns(2)
        with in_col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            blood_pressure = st.number_input("Blood Pressure", 0, 200, 72)
            insulin = st.number_input("Insulin", 0, 900, 131)
            dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.47)
        with in_col2:
            glucose = st.number_input("Glucose", 0, 300, 117)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 29)
            bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
            age = st.number_input("Age", 1, 120, 25)
        
        btn_predict = st.button("Analyze Diabetes Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with main_col2:
        if btn_predict:
            with st.spinner("Calculating..."):
                # 1. Preprocessing
                feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
                input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]], columns=feature_names)

                # Median Imputation sederhana
                medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
                for col, val in medians.items():
                    if input_data[col][0] == 0: input_data[col] = val

                # 2. Scaling & DAE logic
                input_scaled = scaler.transform(input_data)
                
                # Plotly Chart (Bar Chart Comparison)
                fig = go.Figure()
                
                if age > 30:
                    features_dae_scaled = dae_model.predict(input_scaled, verbose=0)
                    final_features = features_dae_scaled
                    features_dae_original = scaler.inverse_transform(features_dae_scaled)
                    
                    fig.add_bar(name="Input Asli", x=feature_names, y=input_data.values[0], marker_color='#444c56')
                    fig.add_bar(name="Hasil DAE", x=feature_names, y=features_dae_original[0], marker_color='#238636')
                else:
                    final_features = input_scaled
                    fig.add_bar(name="Input Asli", x=feature_names, y=input_data.values[0], marker_color='#444c56')
                    st.info("ℹ️ Kelompok usia ≤ 30: Tanpa kompensasi DAE.")

                fig.update_layout(
                    barmode="group", height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', margin=dict(t=10, b=10, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Kartu Hasil
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

                # 3. Final Prediction
                prediction = stacking_model.predict(final_features)[0]
                prob = stacking_model.predict_proba(final_features)[0]

                res_col_a, res_col_b = st.columns([1, 1])
                with res_col_a:
                    status = "DIABETES POSITIVE" if prediction == 1 else "DIABETES NEGATIVE"
                    color = "#f85149" if prediction == 1 else "#3fb950"
                    st.markdown(f"<h2 style='color:{color}; margin-top:20px;'>{status}</h2>", unsafe_allow_html=True)
                    st.metric("Confidence Score", f"{max(prob)*100:.2f}%")

                with res_col_b:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob[1] * 100,
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "white"},
                            "bar": {"color": "#f85149" if prob[1] > 0.5 else "#3fb950"},
                            "bgcolor": "#22272e",
                            "steps": [{"range": [0, 50], "color": "#238636"}, {"range": [50, 100], "color": "#8b100e"}]
                        }
                    ))
                    fig_gauge.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=30, b=0))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Tampilan default sebelum tombol diklik
            st.info("Masukkan data pasien dan klik 'Analyze Diabetes Risk' untuk melihat hasil.")

with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.write("### Penjelasan Metodologi")
    st.markdown("""
    1. **Denoising Autoencoder (DAE):** Digunakan untuk membersihkan noise atau mengkompensasi fitur pada data pasien usia dewasa (>30 tahun).
    2. **Stacking Ensemble:** Menggabungkan beberapa model klasifikasi untuk mendapatkan prediksi yang lebih kuat.
    3. **Preprocessing:** Meliputi pembersihan data nol menggunakan median dari dataset latih.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
