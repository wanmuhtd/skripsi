import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Diabetes Risk Prediction System", layout="wide")

# Tema gelap
st.markdown("""
<style>
    .css-1d391kg {background-color: #1e1e1e;}
    .css-1v0mbdj {color: white;}
    .stButton>button {background-color: #4e9cff; color: white; border-radius: 30px; padding: 10px 40px;}
    .stTextInput>div>div>input {background-color: #333; color: white; border-radius: 10px;}
    .stNumberInput>div>div>input {background-color: #333; color: white; border-radius: 10px;}
    .gauge-text {font-size: 40px; font-weight: bold; color: white;}
    .confidence-text {font-size: 24px; color: #4e9cff;}
</style>
""", unsafe_allow_html=True)

# Judul utama
st.title("Diabetes Risk Prediction System")
st.caption("Implementasi Model Stacking Ensemble dengan Fitur Terkompensasi (DAE)")

# Tabs
tab1, tab2 = st.tabs(["Prediction Tool", "Methodology"])

with tab1:
    # Layout dua kolom utama
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### Patient Clinical Parameters")
        
        # Inputs dua kolom
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)
        row4_col1, row4_col2 = st.columns(2)

        with row1_col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, value=0, step=1)
        with row1_col2:
            glucose = st.number_input("Glucose", min_value=0, value=0, step=1)
        with row2_col1:
            blood_pressure = st.number_input("Blood Pressure", min_value=0, value=0, step=1)
        with row2_col2:
            skin_thickness = st.number_input("Skin Thickness", min_value=0, value=0, step=1)
        with row3_col1:
            insulin = st.number_input("Insulin", min_value=0, value=0, step=1)
        with row3_col2:
            bmi = st.number_input("BMI", min_value=0.0, value=0.0, format="%.2f")
        with row4_col1:
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0, format="%.3f")
        with row4_col2:
            age = st.number_input("Age", min_value=0, value=21, step=1)

        # Tombol
        analyze_btn = st.button("Analyze Diabetes Risk", use_container_width=True)

    with col_right:
        if analyze_btn:
            # Dummy data untuk simulasi (ganti dengan prediksi modelmu nanti)
            features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                       "Insulin", "BMI", "DiabetesPedigreeFunction"]
            input_values = [pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, diabetes_pedigree]
            dae_values = [pregnancies*1.1, glucose*0.95, blood_pressure*1.05, skin_thickness*1.02,
                         insulin*0.98, bmi*1.08, diabetes_pedigree*1.15]  # contoh kompensasi

            df = pd.DataFrame({
                "Feature": features,
                "Input Asli": input_values,
                "Hasil DAE": dae_values
            })

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(features))
            width = 0.35
            ax.bar(x - width/2, df["Input Asli"], width, label="Input Asli", color="#a0a0a0")
            ax.bar(x + width/2, df["Hasil DAE"], width, label="Hasil DAE", color="#4eff8b")
            ax.set_ylabel("Value")
            ax.set_title("")
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45)
            ax.legend()
            st.pyplot(fig)

            # Hasil prediksi (dummy)
            result = "DIABETES NEGATIVE" if np.random.rand() > 0.4 else "DIABETES POSITIVE"
            confidence = round(np.random.uniform(85, 98), 2)
            risk_score = round(np.random.uniform(2.0, 8.0), 1)

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='color: {'#4eff8b' if 'NEGATIVE' in result else '#ff4e4e'}; text-align: center;'>{result}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence-text' style='text-align: center;'>Confidence Score</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='gauge-text' style='text-align: center;'>{confidence}%</p>", unsafe_allow_html=True)
            
            # Gauge sederhana dengan progress bar horizontal + custom HTML
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 50px; font-weight: bold; color: white;">{risk_score}</div>
                <div style="width: 80%; margin: 0 auto; background: linear-gradient(to right, #4eff8b 0%, #ff4e4e 100%); border-radius: 50px; height: 30px; position: relative;">
                    <div style="position: absolute; left: {min(max((risk_score / 10) * 100, 0), 100)}%; top: 50%; transform: translate(-50%, -50%); width: 60px; height: 60px; background: white; border-radius: 50%; border: 8px solid #333;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; width: 80%; margin: 0 auto;">
                    <span style="color: #4eff8b;">0</span>
                    <span style="color: white;">5</span>
                    <span style="color: #ff4e4e;">10</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("#### (Grafik dan hasil akan muncul setelah analisis)")

with tab2:
    st.markdown("### Methodology")
    st.write("Di sini kamu bisa menjelaskan detail model Stacking Ensemble dengan DAE, arsitektur, dataset, performa, dll. Tambahkan gambar, rumus LaTeX, atau tabel sesuai kebutuhan.")
