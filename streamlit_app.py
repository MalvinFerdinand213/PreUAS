import streamlit as st
import pandas as pd
import pickle
import os

# Load model function with error handling
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Prediction function
def predict_with_model(model, user_input_df):
    try:
        prediction = model.predict(user_input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

# Streamlit UI config
st.set_page_config(layout="centered")
st.write("Group 2")

# Main function
def main():
    st.title("Garment Worker Productivity Prediction")
    st.info("Predicting the Actual Productivity of Garment Workers")
    st.subheader("Please Input the Data:")

    # User inputs
    department = st.selectbox("Department", ["sewing", "finishing"])
    team = st.slider("Team Number", 1, 12, value=1)
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, value=0.6, step=0.01)
    smv = st.slider("Standard Minute Value (SMV)", 0.0, 100.0, value=20.0, step=0.1)
    wip = st.slider("Work in Progress (WIP)", 0.0, 25000.0, value=0.0, step=100.0)
    over_time = st.slider("Over Time (minutes)", 0, 30000, value=3000, step=60)
    incentive = st.slider("Incentive (BDT)", 0, 4000, value=30, step=10)
    idle_time = st.slider("Idle Time (minutes)", 0.0, 500.0, value=0.0, step=0.1)
    idle_men = st.slider("Idle Men", 0, 50, value=0, step=1)
    no_of_style_change = st.slider("Number of Style Changes", 0, 3, value=0, step=1)
    no_of_workers = st.slider("Number of Workers", 1.0, 100.0, value=25.0, step=0.5)
    month = st.slider("Month", 1, 12, value=1, step=1)

    # Convert input to DataFrame
    user_input_df = pd.DataFrame([{
        "department": department,
        "team": int(team),
        "targeted_productivity": targeted_productivity,
        "smv": smv,
        "wip": wip,
        "over_time": int(over_time),
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_style_change": int(no_of_style_change),
        "no_of_workers": no_of_workers,
        "month": int(month),
    }])

    # Load the model
    model_path = "trained_model.pkl"
    model_pipeline = load_model(model_path)

    # Prediction button
    if model_pipeline is not None:
        if st.button("Predict Productivity"):
            prediction = predict_with_model(model_pipeline, user_input_df)
            if prediction is not None:
                st.success(f"🎯 Predicted Actual Productivity: **{prediction:.4f}**")
                st.caption("*(A higher value indicates higher productivity.)*")

if __name__ == "__main__":
    main()
