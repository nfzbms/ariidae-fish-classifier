import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Ariidae Fish Classification",
    page_icon="🐟",
    layout="wide"
)

# Title
st.title("🐟 Ariidae Fish Species Classification System")
st.markdown("### Hybrid CART-SVM Approach")
st.markdown("---")

@st.cache_resource
def load_models():
    try:
        selector = joblib.load('feature_selector.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        svm = joblib.load('svm_model.pkl')
        features = joblib.load('selected_cols.pkl')
        classes = joblib.load('classes.pkl')
        return selector, scaler, pca, svm, features, classes
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None, None

selector, scaler, pca, svm, features, classes = load_models()

# Sidebar
with st.sidebar:
    st.header("System Overview")
    st.markdown("Hybrid CART-SVM for fish classification")
    st.markdown("- Automated identification")
    st.markdown("- Quick predictions")
    st.markdown("- High accuracy")

# Tabs
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# Tab 1
with tab1:
    st.header("Enter Fish Measurements")
    
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
    
    if st.button("Classify"):
        if selector is not None:
            input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
            X_selected = selector.transform(input_data)
            X_scaled = scaler.transform(X_selected)
            X_pca = pca.transform(X_scaled)
            prediction = svm.predict(X_pca)[0]
            st.success(f"Predicted Species: {prediction}")

# Tab 2
with tab2:
    st.header("Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            batch_data = pd.read_csv(uploaded_file)
        else:
            batch_data = pd.read_excel(uploaded_file)
        
        st.dataframe(batch_data.head())
        
        if st.button("Classify All"):
            X_batch = batch_data[features].values
            X_selected = selector.transform(X_batch)
            X_scaled = scaler.transform(X_selected)
            X_pca = pca.transform(X_scaled)
            predictions = svm.predict(X_pca)
            batch_data['Predicted_Species'] = predictions
            st.dataframe(batch_data)
            
            csv = batch_data.to_csv(index=False)
            st.download_button("Download CSV", csv, "results.csv")

st.markdown("---")
st.markdown("Final Year Project - Hybrid CART-SVM")
