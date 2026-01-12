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
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. CUSTOM CSS (DARK MODE & SEAMLESS DESIGN)
# ======================================================
st.markdown("""
<style>
/* Mengubah warna background utama */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Card Style untuk Input & Hasil */
.custom-card {
    background-color: #1c2128;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #30363d;
    margin-bottom: 20px;
}

/* Header & Text */
h1, h2, h3 {
    color: #ffffff !important;
}

.stMarkdown {
    color: #adbac7;
}

/* Input Fields Styling */
.stNumberInput div[data-baseweb="input"] {
    background-color: #22272e !important;
    border-color: #444c56 !important;
    color: white !important;
}

/* Button Styling */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    background: linear-gradient(45deg, #238636, #2ea043);
    color: white;
    border: none;
    padding: 10px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(45deg, #2ea043, #3fb950);
    border: none;
    color: white;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #1c2128;
    border-radius: 8px 8px 0px 0px;
    color: #adbac7;
    border: 1px solid #30363d;
}

.stTabs [aria-selected="true"] {
    background-color: #238636 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. LOAD MODEL (Fungsi tetap sama)
# ======================================================
@st.cache_resource
def load_resources():
    model_path = "models"
    # Pastikan file-file ini ada di folder /models
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
# 4. SIDEBAR
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864275.png", width=80)  # Opsional: Icon
    st.markdown("### Diabetes AI System")
    st.caption("Clinical Decision Support System")
    st.divider()

    st.markdown("""
    **Metodologi Skripsi:**
    - Median Imputation
    - Denoising Autoencoder
    - Stacking Ensemble
    """)
    st.divider()
    st.caption(f"Arwan Muhtada\nTeknik Informatika")

# ======================================================
# 5. MAIN CONTENT
# ======================================================
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Implementasi Model **Stacking Ensemble** dengan Fitur Terkompensasi (**DAE**).")

tab1, tab2 = st.tabs(["🔍 Prediction Tool", "📚 Methodology"])

with tab1:
    # Menggunakan container agar background card terlihat
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Patient Clinical Parameters")

        # Layout input dalam kolom agar rapi
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 300, 117)
            blood_pressure = st.number_input("Blood Pressure", 0, 200, 72)
        with col2:
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 29)
            insulin = st.number_input("Insulin", 0, 900, 131)
            bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
        with col3:
            dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.47)
            age = st.number_input("Age", 1, 120, 25)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyze Diabetes Risk"):
        with st.spinner("Calculating..."):
            # Logika Pemrosesan Data
            feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf]], columns=feature_names)

            # Imputasi Median
            medians = {"Glucose": 117.0, "BloodPressure": 72.0, "SkinThickness": 29.0, "Insulin": 131.0, "BMI": 32.05}
            for col, val in medians.items():
                if input_data[col][0] == 0: input_data[col] = val

            # Scaling & DAE
            input_scaled = scaler.transform(input_data)

            if age > 30:
                features_dae_scaled = dae_model.predict(input_scaled, verbose=0)
                final_features = features_dae_scaled

                # Plotly Comparison (Transparan)
                features_dae_original = scaler.inverse_transform(features_dae_scaled)
                fig = go.Figure()
                fig.add_bar(name="Input Asli", x=feature_names, y=input_data.values[0], marker_color='#444c56')
                fig.add_bar(name="Hasil DAE", x=feature_names, y=features_dae_original[0], marker_color='#238636')
                fig.update_layout(
                    barmode="group", height=300,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', margin=dict(t=20, b=20, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                final_features = input_scaled
                st.info("ℹ️ Kelompok usia ≤ 30: Fitur diolah tanpa kompensasi DAE.")

            # Prediksi Akhir
            prediction = stacking_model.predict(final_features)[0]
            prob = stacking_model.predict_proba(final_features)[0]

            # Hasil dalam Card
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                status = "DIABETES POSITIVE" if prediction == 1 else "DIABETES NEGATIVE"
                color = "#f85149" if prediction == 1 else "#3fb950"
                st.markdown(f"<h2 style='color:{color};'>{status}</h2>", unsafe_allow_html=True)
                st.metric("Confidence Score", f"{max(prob)*100:.2f}%")

            with res_col2:
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
                fig_gauge.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=50, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.write("### Penjelasan Metodologi")
    st.markdown("""
    1. **Denoising Autoencoder (DAE):** Berfungsi sebagai *feature compensation* untuk memperbaiki kualitas data pasien usia dewasa ( > 30 tahun).
    2. **Stacking Ensemble:** Menggunakan kombinasi model (misal: RF, SVM, XGB) sebagai *base learners* dan Meta-classifier untuk hasil akhir yang lebih akurat.
    3. **Median Imputation:** Menangani data hilang/nol pada parameter klinis yang krusial.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
