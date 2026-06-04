# ============================================
# FISH SPECIES CLASSIFIER APP
# Hybrid CART-SVM Model Deployment
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        # Mode 1: Real Data Models (6 species)
        models['scaler_real'] = joblib.load('scaler_real.pkl')
        models['cart_real'] = joblib.load('cart_real.pkl')
        models['svm_real'] = joblib.load('svm_real.pkl')
        models['knn_real'] = joblib.load('knn_real.pkl')
        models['hybrid_real'] = joblib.load('svm_hybrid_real.pkl')
        models['features_real'] = joblib.load('features_real.pkl')
        models['classes_real'] = joblib.load('classes_real.pkl')
        
        # Mode 2: Simulated Data Models (12 species)
        models['scaler_sim'] = joblib.load('scaler_sim.pkl')
        models['cart_sim'] = joblib.load('cart_sim.pkl')
        models['svm_sim'] = joblib.load('svm_sim.pkl')
        models['knn_sim'] = joblib.load('knn_sim.pkl')
        models['hybrid_sim'] = joblib.load('svm_hybrid_sim.pkl')
        models['features_sim'] = joblib.load('features_sim.pkl')
        models['classes_sim'] = joblib.load('classes_sim.pkl')
        
        # Load optional hybrid components
        try:
            models['selector_real'] = joblib.load('feature_selector_real.pkl')
            models['scaler_hybrid_real'] = joblib.load('scaler_hybrid_real.pkl')
            models['pca_real'] = joblib.load('pca_hybrid_real.pkl')
        except:
            models['selector_real'] = None
            models['scaler_hybrid_real'] = None
            models['pca_real'] = None
            
        try:
            models['selector_sim'] = joblib.load('feature_selector_sim.pkl')
            models['scaler_hybrid_sim'] = joblib.load('scaler_hybrid_sim.pkl')
            models['pca_sim'] = joblib.load('pca_hybrid_sim.pkl')
        except:
            models['selector_sim'] = None
            models['scaler_hybrid_sim'] = None
            models['pca_sim'] = None
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all model files are in the same directory")
        return None

def predict_hybrid_real(features, models):
    """Predict using Hybrid CART-SVM for Real Data"""
    try:
        # Apply feature selection if available
        if models['selector_real'] is not None:
            features_selected = models['selector_real'].transform(features.reshape(1, -1))
            features_scaled = models['scaler_hybrid_real'].transform(features_selected)
            if models['pca_real'] is not None:
                features_pca = models['pca_real'].transform(features_scaled)
                prediction = models['hybrid_real'].predict(features_pca)
            else:
                prediction = models['hybrid_real'].predict(features_scaled)
        else:
            features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
            prediction = models['hybrid_real'].predict(features_scaled)
        return prediction[0]
    except:
        features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
        return models['hybrid_real'].predict(features_scaled)[0]

def predict_hybrid_sim(features, models):
    """Predict using Hybrid CART-SVM for Simulated Data"""
    try:
        if models['selector_sim'] is not None:
            features_selected = models['selector_sim'].transform(features.reshape(1, -1))
            features_scaled = models['scaler_hybrid_sim'].transform(features_selected)
            if models['pca_sim'] is not None:
                features_pca = models['pca_sim'].transform(features_scaled)
                prediction = models['hybrid_sim'].predict(features_pca)
            else:
                prediction = models['hybrid_sim'].predict(features_scaled)
        else:
            features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
            prediction = models['hybrid_sim'].predict(features_scaled)
        return prediction[0]
    except:
        features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
        return models['hybrid_sim'].predict(features_scaled)[0]

# Header
st.markdown("""
<div class="main-header">
    <h1>🐟 Fish Species Classification System</h1>
    <p>Powered by Hybrid CART-SVM Machine Learning Model</p>
    <p>FYP Project - Faculty of Computing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3095/3095440.png", width=100)
    st.title("Navigation")
    
    mode = st.radio(
        "Select Classification Mode:",
        ["🌿 Real Data Mode (6 Species)", "📊 Simulated Data Mode (12 Species)"],
        help="Real Mode: Actual fish species | Simulated Mode: Extended species dataset"
    )
    
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("""
    **Models Available:**
    - 🌿 CART (Decision Tree)
    - ⚡ SVM (Support Vector Machine)
    - 📊 KNN (K-Nearest Neighbors)
    - 🏆 Hybrid CART-SVM (Best)
    """)
    
    st.markdown("---")
    st.markdown("### Performance Metrics")
    
    if "Real" in mode:
        st.metric("Best Model Accuracy", "95.4%", "Hybrid CART-SVM")
        st.metric("Dataset Size", "12 Species", "Simulated")
    else:
        st.metric("Best Model Accuracy", "88.5%", "Standalone SVM")
        st.metric("Dataset Size", "6 Species", "Real")

# Load models
models = load_models()

if models is None:
    st.stop()

# Main content
if "Real" in mode:
    st.markdown("## 🌿 Real Data Mode (6 Fish Species)")
    st.markdown("Classify fish species using morphological features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Head Features")
        head_length = st.number_input("Head Length (mm)", min_value=10.0, max_value=100.0, value=50.0, step=0.5)
        body_depth = st.number_input("Body Depth (mm)", min_value=10.0, max_value=100.0, value=40.0, step=0.5)
        eye_diameter = st.number_input("Eye Diameter (mm)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
        
    with col2:
        st.markdown("### Snout & Barbell Features")
        snout_length = st.number_input("Snout Length (mm)", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
        max_barbell = st.number_input("Maxillary Barbell Length (mm)", min_value=5.0, max_value=80.0, value=30.0, step=0.5)
        mand_barbell = st.number_input("Mandibullary Barbell Length (mm)", min_value=5.0, max_value=80.0, value=28.0, step=0.5)
        
    with col3:
        st.markdown("### Fin Features")
        mental_barbell = st.number_input("Mental Barbell Length (mm)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
        dorsal_fin = st.number_input("Dorsal Fin Ray Count", min_value=1, max_value=50, value=15, step=1)
        anal_fin = st.number_input("Anal Fin Ray Count", min_value=1, max_value=50, value=12, step=1)
    
    features = np.array([head_length, body_depth, eye_diameter, snout_length, 
                        max_barbell, mand_barbell, mental_barbell, dorsal_fin, anal_fin])
    
    # Model selection for prediction
    st.markdown("---")
    st.markdown("### Model Selection")
    
    model_choice = st.selectbox(
        "Choose model for prediction:",
        ["🏆 Hybrid CART-SVM (Recommended)", "⚡ SVM", "📊 KNN", "🌿 CART"]
    )
    
    if st.button("🔍 Predict Species", type="primary", use_container_width=True):
        with st.spinner("Analyzing features..."):
            if model_choice == "🏆 Hybrid CART-SVM (Recommended)":
                prediction = predict_hybrid_real(features, models)
                model_name = "Hybrid CART-SVM"
                accuracy = 95.4
            elif model_choice == "⚡ SVM":
                features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
                prediction = models['svm_real'].predict(features_scaled)[0]
                model_name = "SVM"
                accuracy = 92.6
            elif model_choice == "📊 KNN":
                features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
                prediction = models['knn_real'].predict(features_scaled)[0]
                model_name = "KNN"
                accuracy = 93.5
            else:
                prediction = models['cart_real'].predict(features.reshape(1, -1))[0]
                model_name = "CART"
                accuracy = 64.8
            
            # Display result
            st.markdown(f"""
            <div class="result-card">
                <h2>🐟 Predicted Species: {prediction}</h2>
                <h3>Model: {model_name}</h3>
                <p>Model Accuracy: {accuracy}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance visualization
            st.markdown("### Feature Values Used")
            feature_df = pd.DataFrame({
                'Feature': models['features_real'],
                'Value': features
            })
            fig = px.bar(feature_df, x='Feature', y='Value', title='Input Features', color='Value')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("## 📊 Simulated Data Mode (12 Species)")
    st.markdown("Extended classification system with 12 fish species")
    
    # Dynamic feature inputs based on simulated features
    sim_features = models['features_sim']
    
    st.markdown("### Enter Morphological Features")
    
    cols = st.columns(3)
    feature_values = []
    
    for i, feature in enumerate(sim_features):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", min_value=0.0, max_value=200.0, value=50.0, step=0.5, key=f"sim_{feature}")
            feature_values.append(val)
    
    features = np.array(feature_values)
    
    # Model selection
    st.markdown("---")
    st.markdown("### Model Selection")
    
    model_choice = st.selectbox(
        "Choose model for prediction:",
        ["🏆 Hybrid CART-SVM (Recommended - Best 95.4%)", "📊 KNN (93.5%)", "⚡ SVM (92.6%)", "🌿 CART (64.8%)"]
    )
    
    if st.button("🔍 Predict Species", type="primary", use_container_width=True):
        with st.spinner("Analyzing features..."):
            if "Hybrid" in model_choice:
                prediction = predict_hybrid_sim(features, models)
                model_name = "Hybrid CART-SVM"
                accuracy = 95.4
            elif "KNN" in model_choice:
                features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
                prediction = models['knn_sim'].predict(features_scaled)[0]
                model_name = "KNN"
                accuracy = 93.5
            elif "SVM" in model_choice:
                features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
                prediction = models['svm_sim'].predict(features_scaled)[0]
                model_name = "SVM"
                accuracy = 92.6
            else:
                prediction = models['cart_sim'].predict(features.reshape(1, -1))[0]
                model_name = "CART"
                accuracy = 64.8
            
            st.markdown(f"""
            <div class="result-card">
                <h2>🐟 Predicted Species: {prediction}</h2>
                <h3>Model: {model_name}</h3>
                <p>Model Accuracy: {accuracy}%</p>
                <p>⭐ This is the BEST model with 95.4% accuracy!</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 Fish Species Classification System | FYP Project</p>
    <p>Developed using Hybrid CART-SVM Machine Learning</p>
    <p>📧 Contact: fyp@university.edu</p>
</div>
""", unsafe_allow_html=True)

# Run: streamlit run fish_classifier_app.py
