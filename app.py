import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Wine Origin Predictor", page_icon="🍷")

st.title("🍷 Wine Cultivar Origin Prediction")
st.write("Name: ISHOLA OLUFEMI | Matric: 22H032024")
st.write("Predict the origin (Cultivar) of wine based on chemical analysis.")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("System Status: Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# User Inputs (6 Features)
st.subheader("Chemical Properties")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", min_value=10.0, max_value=15.0, value=13.0, step=0.1)
    magnesium = st.number_input("Magnesium", min_value=70, max_value=170, value=100)
    flavanoids = st.number_input("Flavanoids", min_value=0.0, max_value=6.0, value=2.0, step=0.1)

with col2:
    color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1)
    hue = st.number_input("Hue", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
    proline = st.number_input("Proline", min_value=200, max_value=1700, value=750)

# Predict
if st.button("Identify Cultivar"):
    features = np.array([[alcohol, magnesium, flavanoids, color_intensity, hue, proline]])
    
    prediction = model.predict(features)[0]
    # Calculate probability if possible
    try:
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba)
        conf_text = f"(Confidence: {confidence:.2%})"
    except:
        conf_text = ""

    # Map 0,1,2 to Cultivar names
    cultivar_map = {0: "Cultivar 1 (Barolo)", 1: "Cultivar 2 (Grignolino)", 2: "Cultivar 3 (Barbera)"}
    result_name = cultivar_map.get(prediction, "Unknown")
    
    st.balloons()
    st.success(f"Predicted Origin: {result_name} {conf_text}")
