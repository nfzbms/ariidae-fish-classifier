import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GOOGLE DRIVE FILE ID (GANTI DENGAN ID ANDA)
# ============================================
CNN_MODEL_ID = "1e0KzPs66rGtvmu8VbSic836C9fqjF51X"  # ← TUKAR DENGAN ID ANDA!

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Ariidae Classification", page_icon="🐟", layout="wide")

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

st.markdown("""
<div class="main-header">
    <h1>🐟 Ariidae Fish Classification System</h1>
    <p>Mode 1: Hybrid CART-SVM (Measurements) | Mode 2: CNN (Image)</p>
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
        st.error(f"Error loading hybrid model: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def load_cnn_model():
    """Download CNN model from Google Drive"""
    model_path = 'ariidae_cnn_model.h5'
    classes_path = 'ariidae_cnn_classes.pkl'
    
    # Download model from Google Drive if not exists
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading CNN model (first time only)... This may take a few minutes."):
            try:
                url = f"https://drive.google.com/uc?id={CNN_MODEL_ID}"
                gdown.download(url, model_path, quiet=False)
                st.success("✅ CNN model downloaded!")
            except Exception as e:
                st.error(f"Failed to download CNN model: {e}")
                return None, None
    
    # Load model
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        classes = joblib.load(classes_path) if os.path.exists(classes_path) else None
        return model, classes
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None, None

# Load models
selector, scaler, pca, svm, features, classes = load_hybrid_model()
cnn_model, cnn_classes = load_cnn_model()

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
        st.success(f"✅ Mode 2 (Image): Ready")
    else:
        st.warning("⚠️ Mode 2: Model will download on first use")
    
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.metric("Hybrid CART-SVM", "95.2%")
    st.metric("CNN", "89.5%")
    st.markdown("---")
    st.caption("Final Year Project")

# ============================================
# MODE SELECTION
# ============================================
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.subheader("🎯 Select Classification Mode")

col1, col2 = st.columns(2)

with col1:
    mode_hybrid = st.button("📏 Mode 1: Measurements\n95.2% Accuracy", use_container_width=True, type="primary")

with col2:
    mode_cnn = st.button("📸 Mode 2: Image\n89.5% Accuracy", use_container_width=True, type="secondary")

st.markdown('</div>', unsafe_allow_html=True)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "hybrid"

if mode_hybrid:
    st.session_state.selected_mode = "hybrid"
if mode_cnn:
    st.session_state.selected_mode = "cnn"

# ============================================
# MODE 1: HYBRID CART-SVM
# ============================================
if st.session_state.selected_mode == "hybrid":
    st.markdown("## 📏 Mode 1: Measurements")
    
    if selector is None:
        st.error("⚠️ Models not loaded.")
    else:
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
    st.markdown("## 📸 Mode 2: Image Classification")
    
    if cnn_model is None:
        st.warning("⚠️ CNN model is being downloaded. Please wait or check your Google Drive ID.")
        st.info(f"Current FILE ID: `{CNN_MODEL_ID}`\n\nMake sure this ID is correct.")
    else:
        uploaded_file = st.file_uploader("Upload fish image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            if st.button("🔍 Identify Species", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    import tensorflow as tf
                    predictions = cnn_model.predict(img_array)
                    predicted_idx = np.argmax(predictions[0])
                    predicted_class = cnn_classes[predicted_idx]
                    confidence = np.max(predictions[0]) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-card-cnn">
                        <div>🎯 Predicted Species</div>
                        <div class="prediction-species">{predicted_class}</div>
                        <div>Confidence: {confidence:.1f}%</div>
                        <div>✅ CNN Classification</div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Final Year Project - Hybrid CART-SVM + CNN for Ariidae Fish Classification</p>
</div>
""", unsafe_allow_html=True)
