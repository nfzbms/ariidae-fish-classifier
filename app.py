
# ============================================
# FISH SPECIES CLASSIFIER APP
# Hybrid CART-SVM Model Deployment
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
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
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
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
            
        st.success("✅ Models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all model files are in the same directory as this app")
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

def plot_feature_bar(features, feature_names, title):
    """Create bar chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(feature_names, features, color='steelblue', alpha=0.7)
    ax.set_xlabel('Features')
    ax.set_ylabel('Values (mm)')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, features):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

# Header
st.markdown("""
<div class="main-header">
    <h1>🐟 Fish Species Classification System</h1>
    <p>Powered by Hybrid CART-SVM Machine Learning Model</p>
    <p>Final Year Project - Faculty of Computing</p>
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
    st.markdown("### 📊 Model Performance")
    
    if "Simulated" in mode:
        st.markdown("**Mode 2: Simulated Data (12 Species)**")
        st.metric("🏆 Best Model", "Hybrid CART-SVM", "95.4% Accuracy")
        st.metric("🥈 Second Best", "KNN", "93.5% Accuracy")
        st.metric("🥉 Third Best", "SVM", "92.6% Accuracy")
    else:
        st.markdown("**Mode 1: Real Data (6 Species)**")
        st.metric("🏆 Best Model", "SVM", "88.5% Accuracy")
        st.metric("🥈 Second Best", "KNN", "80.8% Accuracy")
        st.metric("🥉 Third Best", "Hybrid", "80.8% Accuracy")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This system uses machine learning to classify fish species based on morphological features.
    
    **Features used:**
    - Head measurements
    - Barbell lengths
    - Fin ray counts
    """)

# Load models
models = load_models()

if models is None:
    st.stop()

# Main content
if "Real" in mode:
    st.markdown("## 🌿 Real Data Mode (6 Fish Species)")
    st.markdown("Enter morphological features to classify among 6 fish species")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🐟 Head Features")
        head_length = st.number_input("Head Length (mm)", min_value=10.0, max_value=100.0, value=50.0, step=0.5)
        body_depth = st.number_input("Body Depth (mm)", min_value=10.0, max_value=100.0, value=40.0, step=0.5)
        eye_diameter = st.number_input("Eye Diameter (mm)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
        
    with col2:
        st.markdown("### 👃 Snout & Barbell")
        snout_length = st.number_input("Snout Length (mm)", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
        max_barbell = st.number_input("Maxillary Barbell (mm)", min_value=5.0, max_value=80.0, value=30.0, step=0.5)
        mand_barbell = st.number_input("Mandibullary Barbell (mm)", min_value=5.0, max_value=80.0, value=28.0, step=0.5)
        
    with col3:
        st.markdown("### 🎣 Fin Features")
        mental_barbell = st.number_input("Mental Barbell (mm)", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
        dorsal_fin = st.number_input("Dorsal Fin Rays", min_value=1, max_value=50, value=15, step=1)
        anal_fin = st.number_input("Anal Fin Rays", min_value=1, max_value=50, value=12, step=1)
    
    features = np.array([head_length, body_depth, eye_diameter, snout_length, 
                        max_barbell, mand_barbell, mental_barbell, dorsal_fin, anal_fin])
    
    # Model selection
    st.markdown("---")
    st.markdown("### 🤖 Select Model for Prediction")
    
    model_choice = st.selectbox(
        "Choose model:",
        ["🏆 Hybrid CART-SVM (Recommended)", "⚡ SVM", "📊 KNN", "🌿 CART"]
    )
    
    if st.button("🔍 Predict Species", type="primary", use_container_width=True):
        with st.spinner("Analyzing features..."):
            if model_choice == "🏆 Hybrid CART-SVM (Recommended)":
                prediction = predict_hybrid_real(features, models)
                model_name = "Hybrid CART-SVM"
                accuracy = 80.8
            elif model_choice == "⚡ SVM":
                features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
                prediction = models['svm_real'].predict(features_scaled)[0]
                model_name = "SVM"
                accuracy = 88.5
            elif model_choice == "📊 KNN":
                features_scaled = models['scaler_real'].transform(features.reshape(1, -1))
                prediction = models['knn_real'].predict(features_scaled)[0]
                model_name = "KNN"
                accuracy = 80.8
            else:
                prediction = models['cart_real'].predict(features.reshape(1, -1))[0]
                model_name = "CART"
                accuracy = 76.9
            
            # Display result
            st.markdown(f"""
            <div class="result-card">
                <h1>🐟 {prediction}</h1>
                <h3>Model: {model_name}</h3>
                <h4>Model Accuracy: {accuracy}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature visualization
            st.markdown("### 📊 Input Features Visualization")
            fig = plot_feature_bar(features, models['features_real'], "Fish Morphological Features")
            st.pyplot(fig)

else:
    st.markdown("## 📊 Simulated Data Mode (12 Fish Species)")
    st.markdown("Extended classification system with 12 fish species - **HYBRID MODEL ACHIEVES 95.4% ACCURACY**")
    
    # Dynamic feature inputs
    sim_features = models['features_sim']
    
    st.markdown("### 📏 Enter Morphological Features")
    
    cols = st.columns(3)
    feature_values = []
    
    for i, feature in enumerate(sim_features):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", min_value=0.0, max_value=200.0, value=50.0, step=0.5, key=f"sim_{feature}")
            feature_values.append(val)
    
    features = np.array(feature_values)
    
    # Model selection
    st.markdown("---")
    st.markdown("### 🤖 Select Model for Prediction")
    
    model_choice = st.selectbox(
        "Choose model:",
        ["🏆 Hybrid CART-SVM (BEST: 95.4%)", "📊 KNN (93.5%)", "⚡ SVM (92.6%)", "🌿 CART (64.8%)"]
    )
    
    if st.button("🔍 Predict Species", type="primary", use_container_width=True):
        with st.spinner("Analyzing features..."):
            if "Hybrid" in model_choice:
                prediction = predict_hybrid_sim(features, models)
                model_name = "Hybrid CART-SVM"
                accuracy = 95.4
                is_best = True
            elif "KNN" in model_choice:
                features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
                prediction = models['knn_sim'].predict(features_scaled)[0]
                model_name = "KNN"
                accuracy = 93.5
                is_best = False
            elif "SVM" in model_choice:
                features_scaled = models['scaler_sim'].transform(features.reshape(1, -1))
                prediction = models['svm_sim'].predict(features_scaled)[0]
                model_name = "SVM"
                accuracy = 92.6
                is_best = False
            else:
                prediction = models['cart_sim'].predict(features.reshape(1, -1))[0]
                model_name = "CART"
                accuracy = 64.8
                is_best = False
            
            # Display result
            best_badge = "🏆 BEST MODEL (95.4% ACCURACY) 🏆" if is_best else ""
            
            st.markdown(f"""
            <div class="result-card">
                <h1>🐟 {prediction}</h1>
                <h3>Model: {model_name}</h3>
                <h4>Model Accuracy: {accuracy}%</h4>
                <h4>{best_badge}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature visualization
            st.markdown("### 📊 Input Features Visualization")
            fig = plot_feature_bar(features, sim_features, "Fish Morphological Features")
            st.pyplot(fig)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 Fish Species Classification System | Final Year Project</p>
    <p>Developed using Hybrid CART-SVM Machine Learning | Faculty of Computing</p>
</div>
""", unsafe_allow_html=True)
