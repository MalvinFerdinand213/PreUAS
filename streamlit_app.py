import streamlit as st
import pandas as pd
from derma_base import predict_productivity, model

st.title("Prediksi Produktivitas Pabrik")

# --- Bagian Slider sederhana untuk prediksi cepat ---
idle_men = st.slider("Idle Men", 0.0, 500.0, 0.0)
style_changes = st.slider("Number of Style Changes", 0, 50, 0)
num_workers = st.slider("Number of Workers", 1, 100, 25)
month = st.slider("Month", 1, 12, 6)

if st.button("Predict"):
    try:
        result = predict_productivity(idle_men, style_changes, num_workers, month)
        st.success(f"Hasil prediksi produktivitas: {result:.2f}")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Form lengkap untuk prediksi model utama ---
st.title("Prediksi Produktivitas Pekerja Garmen")
st.markdown("Masukkan detail produksi di bawah ini untuk memprediksi actual productivity:")

# Input Form
with st.form("prediction_form"):
    date = st.date_input("Tanggal").strftime('%Y-%m-%d')
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    department = st.selectbox("Departemen", ['sewing', 'finishing'])
    day = st.selectbox("Hari", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday'])
    team = st.number_input("Team", min_value=1, max_value=100, value=1)
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.75)
    smv = st.number_input("SMV", min_value=0.0)
    wip = st.number_input("WIP", min_value=0)
    over_time = st.number_input("Over Time (menit)", min_value=0)
    incentive = st.number_input("Incentive", min_value=0.0)
    idle_time = st.number_input("Idle Time", min_value=0.0)
    idle_men = st.number_input("Idle Men", min_value=0)
    no_of_style_change = st.number_input("Style Changes", min_value=0)
    no_of_workers = st.number_input("Jumlah Pekerja", min_value=1)

    submit = st.form_submit_button("Prediksi")

if submit:
    if model is None:
        st.error("Model belum dimuat.")
    else:
        input_data = pd.DataFrame([[
            date, quarter, department, day, team, targeted_productivity,
            smv, wip, over_time, incentive, idle_time, idle_men,
            no_of_style_change, no_of_workers
        ]], columns=[
            'date', 'quarter', 'department', 'day', 'team', 'targeted_productivity',
            'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
            'no_of_style_change', 'no_of_workers'
        ])

        try:
            prediction = model.predict(input_data)
            st.success(f"Prediksi Produktivitas Aktual: {prediction[0]:.4f}")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
