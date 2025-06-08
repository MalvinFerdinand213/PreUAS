import streamlit as st
import pandas as pd
import pickle

# Load model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Prediksi Aktual Produktivitas")

# Input form
with st.form("prediction_form"):
    date = st.text_input("Tanggal (yyyy-mm-dd)")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    department = st.selectbox("Departemen", ["sweing", "finishing"])
    day = st.selectbox("Hari", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"])
    team = st.number_input("Team", min_value=1)
    targeted_productivity = st.number_input("Targeted Productivity", min_value=0.0)
    smv = st.number_input("SMV", min_value=0.0)
    wip = st.number_input("WIP", min_value=0.0)
    over_time = st.number_input("Over Time", min_value=0)
    incentive = st.number_input("Incentive", min_value=0.0)
    idle_time = st.number_input("Idle Time", min_value=0.0)
    idle_men = st.number_input("Idle Men", min_value=0)
    no_of_style_change = st.number_input("No of Style Change", min_value=0)
    no_of_workers = st.number_input("No of Workers", min_value=1)

    submitted = st.form_submit_button("Prediksi")

# Predict
if submitted:
    input_df = pd.DataFrame([[date, quarter, department, day, team, targeted_productivity,
                              smv, wip, over_time, incentive, idle_time, idle_men,
                              no_of_style_change, no_of_workers]],
                            columns=[
                                'date', 'quarter', 'department', 'day', 'team', 'targeted_productivity',
                                'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
                                'no_of_style_change', 'no_of_workers'
                            ])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediksi Aktual Produktivitas: **{prediction:.2f}**")
