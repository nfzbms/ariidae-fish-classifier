import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ariidae Classification", page_icon="🐟")

st.title("🐟 Ariidae Fish Classification System")
st.markdown("### Hybrid CART-SVM | 95.2% Accuracy")

@st.cache_resource
def load_models():
    selector = joblib.load('feature_selector.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    svm = joblib.load('svm_model.pkl')
    features = joblib.load('selected_cols.pkl')
    return selector, scaler, pca, svm, features

selector, scaler, pca, svm, features = load_models()

col1, col2, col3 = st.columns(3)

with col1:
    head = st.number_input("Head Length (mm)", 0.0, 200.0, 45.0)
    body = st.number_input("Body Depth (mm)", 0.0, 150.0, 28.0)
    eye = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 6.0)

with col2:
    snout = st.number_input("Snout Length (mm)", 0.0, 50.0, 12.0)
    maxillary = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 35.0)
    mandibullary = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 25.0)

with col3:
    mental = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 8.0)
    dorsal = st.number_input("Dorsal Fin Ray", 0, 50, 18)
    anal = st.number_input("Anal Fin Ray", 0, 40, 14)

if st.button("Classify", type="primary", use_container_width=True):
    input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
    X_selected = selector.transform(input_data)
    X_scaled = scaler.transform(X_selected)
    X_pca = pca.transform(X_scaled)
    prediction = svm.predict(X_pca)[0]
    st.success(f"### 🎯 Predicted Species: {prediction}")

st.markdown("---")
st.markdown("🎓 Final Year Project - Hybrid CART-SVM for Ariidae Classification")
