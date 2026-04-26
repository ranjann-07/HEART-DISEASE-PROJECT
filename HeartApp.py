import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts whether a patient is likely to have heart disease based on various medical metrics. 
Please enter the patient's information below.
""")

# Input form
with st.form("prediction_form"):
    st.header("Patient Health Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3], 
                          format_func=lambda x: {
                              0: "0: Typical Angina", 
                              1: "1: Atypical Angina", 
                              2: "2: Non-anginal Pain", 
                              3: "3: Asymptomatic"
                          }[x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2],
                               format_func=lambda x: {
                                   0: "0: Normal",
                                   1: "1: ST-T wave abnormality",
                                   2: "2: Probable/definite left ventricular hypertrophy"
                               }[x])

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2],
                             format_func=lambda x: {0: "0: Upsloping", 1: "1: Flat", 2: "2: Downsloping"}[x])
        ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0-4)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3],
                            format_func=lambda x: {
                                0: "0: Normal",
                                1: "1: Fixed Defect",
                                2: "2: Reversable Defect",
                                3: "3: Unknown"
                            }[x])
                            
    submit_button = st.form_submit_button("Predict Heart Disease")

# Inference
if submit_button:
    if model is not None and scaler is not None:
        # Create an array of the inputs
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale the inputs
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)[0]
        
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Heart Disease** (Confidence: {prediction_proba[1]:.2%})")
            st.write("The model predicts that this patient is likely to have heart disease. Please consult with a healthcare professional.")
        else:
            st.success(f"✅ **Low Risk of Heart Disease** (Confidence: {prediction_proba[0]:.2%})")
            st.write("The model predicts that this patient is unlikely to have heart disease. However, regular check-ups are always recommended.")
            
        st.info(f"Model used: Support Vector Machine (SVM)")
    else:
        st.error("Model or scaler not available. Please train the model first.")
