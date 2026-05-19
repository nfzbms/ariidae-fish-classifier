# ============================================
# ARIIDAE FISH CLASSIFICATION SYSTEM
# Mode 1: Hybrid CART-SVM (Measurements)
# Mode 2: CNN (Image - Download from Google Drive)
# ============================================

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
# DOWNLOAD CNN MODEL FROM GOOGLE DRIVE
# ============================================

@st.cache_resource
def download_cnn_model():
    """Download CNN model from Google Drive if not exists"""
    
    # GANTIKAN ID INI DENGAN ID DARI LINK GOOGLE DRIVE ANDA
    FILE_ID = "YOUR_FILE_ID_HERE"  # ← TUKAR INI!
    
    model_path = 'ariidae_cnn_model.h5'
    classes_path = 'ariidae_cnn_classes.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading CNN model (first time only)..."):
            # Download model
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, model_path, quiet=False)
            
            # Download classes (upload separately)
            # Atau jika classes dalam zip yang sama
            st.success("✅ CNN model downloaded!")
    
    # Load model
    if os.path.exists(model_path):
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        classes = joblib.load(classes_path) if os.path.exists(classes_path) else None
        return model, classes
    return None, None

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(page_title="Ariidae Classification", page_icon="🐟", layout="wide")

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

selector, scaler, pca, svm, features, classes = load_hybrid_model()

# Try to load CNN model from Google Drive
cnn_model, cnn_classes = None, None
try:
    cnn_model, cnn_classes = download_cnn_model()
    if cnn_model is not None:
        st.sidebar.success("✅ CNN Model: Ready")
    else:
        st.sidebar.warning("⚠️ CNN Model: Not configured")
except Exception as e:
    st.sidebar.warning(f"⚠️ CNN Model: {e}")

# ============================================
# MODE SELECTION
# ============================================

st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.subheader("🎯 Select Classification Mode")

col1, col2 = st.columns(2)

with col1:
    mode_hybrid = st.button(
        "📏 **Mode 1: Measurements**\n\nHybrid CART-SVM\nEnter 9 morphological measurements\n95.2% Accuracy",
        use_container_width=True,
        type="primary"
    )

with col2:
    mode_cnn = st.button(
        "📸 **Mode 2: Image**\n\nCNN\nUpload Ariidae fish photo\n89.5% Accuracy",
        use_container_width=True,
        type="secondary"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Session state
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "hybrid"

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
        st.warning("⚠️ Mode 2: Model will download on first use")
    
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
            st.markdown("**📏 Head & Body**")
            head = st.number_input("Head Length (mm)", 0.0, 200.0, 45.0, 0.1)
            body = st.number_input("Body Depth (mm)", 0.0, 150.0, 28.0, 0.1)
            eye = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 6.0, 0.1)
        
        with col2:
            st.markdown("**🪢 Barbell & Snout**")
            snout = st.number_input("Snout Length (mm)", 0.0, 50.0, 12.0, 0.1)
            maxillary = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 35.0, 0.1)
            mandibullary = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 25.0, 0.1)
        
        with col3:
            st.markdown("**🎯 Fins & Other**")
            mental = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 8.0, 0.1)
            dorsal = st.number_input("Dorsal Fin Ray", 0, 50, 18, 1)
            anal = st.number_input("Anal Fin Ray", 0, 40, 14, 1)
        
        if st.button("🔍 Identify Species", use_container_width=True):
            with st.spinner("Analyzing measurements..."):
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
    st.markdown("Upload a photo of an Ariidae fish for instant identification")
    
    if cnn_model is None:
        st.warning("⚠️ CNN model is being downloaded or not configured.")
        st.info("""
        **First time setup:**
        1. Upload your `ariidae_cnn_model.h5` to Google Drive
        2. Get the file ID from the shareable link
        3. Update the FILE_ID in the app code
        """)
        
        # Show instructions
        st.markdown("""
        <div class="info-box">
            <h4>📸 How to enable CNN mode:</h4>
            <ol>
                <li>Upload <code>ariidae_cnn_model.h5</code> to Google Drive</li>
                <li>Right-click → Share → Get link</li>
                <li>Copy the FILE ID (long string in the URL)</li>
                <li>Replace <code>YOUR_FILE_ID_HERE</code> in the app code</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader(
            "📤 Choose an Ariidae fish image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the fish"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            with col2:
                st.markdown(f"**Image Details:**")
                st.markdown(f"- Size: {image.size[0]} x {image.size[1]} px")
                st.markdown(f"- Format: {image.format}")
                
                if st.button("🔍 Identify Species", use_container_width=True):
                    with st.spinner("Analyzing image with CNN..."):
                        # Preprocess
                        img = image.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                        
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Predict
                        import tensorflow as tf
                        predictions = cnn_model.predict(img_array)
                        predicted_idx = np.argmax(predictions[0])
                        predicted_class = cnn_classes[predicted_idx]
                        confidence = np.max(predictions[0]) * 100
                        
                        # Top 3 predictions
                        top_indices = np.argsort(predictions[0])[-3:][::-1]
                    
                    st.markdown(f"""
                    <div class="prediction-card-cnn">
                        <div>🎯 Predicted Species</div>
                        <div class="prediction-species">{predicted_class}</div>
                        <div class="confidence-high">Confidence: {confidence:.1f}%</div>
                        <div>✅ CNN-based Classification</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 📊 Top 3 Predictions:")
                    for idx in top_indices:
                        prob = predictions[0][idx] * 100
                        species = cnn_classes[idx]
                        st.progress(prob/100, text=f"{species}: {prob:.1f}%")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - Hybrid CART-SVM + CNN for Ariidae Fish Classification</p>
    <p>📏 Mode 1: Measurements (95.2% Accuracy) | 📸 Mode 2: Image (89.5% Accuracy)</p>
</div>
""", unsafe_allow_html=True)
