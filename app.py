
import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Disease Prediction System")

st.title("Disease Prediction System")
st.write("Enter patient information to predict disease.")

user_data = {}

# Dynamically create input fields
for col in feature_columns:
    user_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

if st.button("Predict Disease"):

    input_df = pd.DataFrame([user_data])

    # Ensure column order matches training
    input_df = input_df[feature_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)

    disease_name = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease_name[0]}")
