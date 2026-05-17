
# ============================================
# ARIIDAE FISH CLASSIFICATION SYSTEM
# Dual Mode: Hybrid CART-SVM + CNN
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow for CNN
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="Ariidae Fish Classification",
    page_icon="🐟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .mode-selector {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .prediction-card-hybrid {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-card-cnn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-species {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🐟 Ariidae Fish Classification System</h1>
    <p>Hybrid CART-SVM (Measurements) | CNN (Image Recognition)</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_hybrid_model():
    try:
        selector = joblib.load('feature_selector.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        svm = joblib.load('svm_model.pkl')
        features = joblib.load('selected_cols.pkl')
        classes = joblib.load('classes.pkl')
        return selector, scaler, pca, svm, features, classes
    except Exception as e:
        return None, None, None, None, None, None

@st.cache_resource
def load_cnn_model():
    try:
        model = load_model('ariidae_cnn_model.h5', compile=False)
        classes = joblib.load('ariidae_cnn_classes.pkl')
        return model, classes
    except Exception as e:
        return None, None

# Load models
selector, scaler, pca, svm, features, classes = load_hybrid_model()
cnn_model, cnn_classes = load_cnn_model()

# ============================================
# MODE SELECTION
# ============================================

st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.subheader("🎯 Select Classification Mode")

col1, col2 = st.columns(2)

with col1:
    mode_hybrid = st.button(
        "📏 **Mode 1: Measurements**\n\nHybrid CART-SVM\nEnter 9 morphological measurements",
        use_container_width=True,
        type="primary"
    )

with col2:
    mode_cnn = st.button(
        "📸 **Mode 2: Image**\n\nCNN\nUpload Ariidae fish photo",
        use_container_width=True,
        type="secondary"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Session state
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = None

if mode_hybrid:
    st.session_state.selected_mode = "hybrid"
if mode_cnn:
    st.session_state.selected_mode = "cnn"

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    if selector is not None:
        st.success("✅ Mode 1 (Measurements): Ready")
    else:
        st.error("❌ Mode 1: Not Loaded")
    
    if cnn_model is not None:
        st.success(f"✅ Mode 2 (Image): Ready ({len(cnn_classes)} species)")
    else:
        st.error("❌ Mode 2: Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.metric("Hybrid CART-SVM", "95.2%")
    st.metric("CNN", "89.5%")
    st.markdown("---")
    st.caption("Final Year Project")

# ============================================
# MODE 1: HYBRID CART-SVM
# ============================================

if st.session_state.selected_mode == "hybrid":
    st.markdown("## 📏 Mode 1: Ariidae Classification (Measurements)")
    
    if selector is None:
        st.error("⚠️ Models not loaded. Please check files.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            head = st.number_input("Head Length (mm)", 0.0, 200.0, 45.0, 0.1)
            body = st.number_input("Body Depth (mm)", 0.0, 150.0, 28.0, 0.1)
            eye = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 6.0, 0.1)
        
        with col2:
            snout = st.number_input("Snout Length (mm)", 0.0, 50.0, 12.0, 0.1)
            maxillary = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 35.0, 0.1)
            mandibullary = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 25.0, 0.1)
        
        with col3:
            mental = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 8.0, 0.1)
            dorsal = st.number_input("Dorsal Fin Ray", 0, 50, 18, 1)
            anal = st.number_input("Anal Fin Ray", 0, 40, 14, 1)
        
        if st.button("🔍 Identify Species", use_container_width=True):
            input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
            X_selected = selector.transform(input_data)
            X_scaled = scaler.transform(X_selected)
            X_pca = pca.transform(X_scaled)
            prediction = svm.predict(X_pca)[0]
            
            st.markdown(f"""
            <div class="prediction-card-hybrid">
                <div>🎯 Predicted Species</div>
                <div class="prediction-species">{prediction}</div>
                <div>✅ Hybrid CART-SVM | 95.2% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# MODE 2: CNN
# ============================================

elif st.session_state.selected_mode == "cnn":
    st.markdown("## 📸 Mode 2: Ariidae Classification (Image)")
    
    if cnn_model is None:
        st.error("⚠️ CNN model not loaded. Please check 'ariidae_cnn_model.h5'")
    else:
        uploaded_file = st.file_uploader("Upload Ariidae fish image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            if st.button("🔍 Identify Species", use_container_width=True):
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = cnn_model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                predicted_class = cnn_classes[predicted_idx]
                confidence = np.max(predictions[0]) * 100
                
                st.markdown(f"""
                <div class="prediction-card-cnn">
                    <div>🎯 Predicted Species</div>
                    <div class="prediction-species">{predicted_class}</div>
                    <div>Confidence: {confidence:.1f}%</div>
                    <div>✅ CNN-based Classification</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Final Year Project - Hybrid CART-SVM + CNN for Ariidae Fish Classification</p>
</div>
""", unsafe_allow_html=True)
