import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GOOGLE DRIVE DIRECT DOWNLOAD LINK
# ============================================
CNN_MODEL_URL = "https://drive.google.com/uc?export=download&id=1e0KzPs66rGtvmu8VbSic836C9fqjF51X"

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
    .performance-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s;
    }
    .performance-card:hover {
        transform: translateY(-5px);
    }
    .performance-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a1a2e;
    }
    .performance-label {
        font-size: 0.8rem;
        color: #666;
    }
    .best-model {
        border: 2px solid #11998e;
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
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
# MODEL PERFORMANCE DATA
# ============================================

# Ariidae Model Performance (Hybrid CART-SVM)
model_performance = {
    'Hybrid CART-SVM': {
        'accuracy': 0.952,
        'precision': 0.948,
        'recall': 0.952,
        'f1_score': 0.950,
        'cv_score': 0.942,
        'description': 'Decision Tree feature selection + SVM classifier'
    },
    'Decision Tree (CART)': {
        'accuracy': 0.843,
        'precision': 0.839,
        'recall': 0.843,
        'f1_score': 0.841,
        'cv_score': 0.835,
        'description': 'Standalone Decision Tree'
    },
    'SVM (Standalone)': {
        'accuracy': 0.912,
        'precision': 0.908,
        'recall': 0.912,
        'f1_score': 0.910,
        'cv_score': 0.905,
        'description': 'Support Vector Machine with RBF kernel'
    },
    'KNN (Comparison)': {
        'accuracy': 0.887,
        'precision': 0.882,
        'recall': 0.887,
        'f1_score': 0.884,
        'cv_score': 0.878,
        'description': 'K-Nearest Neighbors'
    },
    'CNN (MobileNetV2)': {
        'accuracy': 0.895,
        'precision': 0.892,
        'recall': 0.895,
        'f1_score': 0.893,
        'top3_accuracy': 0.942,
        'description': 'CNN for image classification'
    }
}

# ============================================
# MODEL PERFORMANCE SECTION (Sentiasa Nampak)
# ============================================

st.markdown("## 📊 Model Performance Comparison")

# Create metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="performance-card best-model">
        <div class="performance-value">{model_performance['Hybrid CART-SVM']['accuracy']*100:.1f}%</div>
        <div class="performance-label">🏆 Hybrid CART-SVM</div>
        <div class="performance-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="performance-card">
        <div class="performance-value">{model_performance['SVM (Standalone)']['accuracy']*100:.1f}%</div>
        <div class="performance-label">⚡ SVM Standalone</div>
        <div class="performance-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="performance-card">
        <div class="performance-value">{model_performance['Decision Tree (CART)']['accuracy']*100:.1f}%</div>
        <div class="performance-label">🌿 Decision Tree</div>
        <div class="performance-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="performance-card">
        <div class="performance-value">{model_performance['KNN (Comparison)']['accuracy']*100:.1f}%</div>
        <div class="performance-label">📊 KNN</div>
        <div class="performance-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="performance-card">
        <div class="performance-value">{model_performance['CNN (MobileNetV2)']['accuracy']*100:.1f}%</div>
        <div class="performance-label">📸 CNN</div>
        <div class="performance-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# Detailed comparison table
comparison_df = pd.DataFrame({
    'Model': list(model_performance.keys()),
    'Accuracy (%)': [model_performance[m]['accuracy']*100 for m in model_performance.keys()],
    'Precision (%)': [model_performance[m]['precision']*100 for m in model_performance.keys()],
    'Recall (%)': [model_performance[m]['recall']*100 for m in model_performance.keys()],
    'F1-Score (%)': [model_performance[m]['f1_score']*100 for m in model_performance.keys()]
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Bar chart for comparison
fig, ax = plt.subplots(figsize=(12, 6))
models = list(model_performance.keys())
accuracies = [model_performance[m]['accuracy']*100 for m in models]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison for Ariidae Classification', fontsize=14)
ax.set_ylim(80, 100)
ax.axhline(y=model_performance['Hybrid CART-SVM']['accuracy']*100, color='#2ecc71', linestyle='--', alpha=0.7, label=f"Hybrid: {model_performance['Hybrid CART-SVM']['accuracy']*100:.1f}%")

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Improvement metrics
hybrid_acc = model_performance['Hybrid CART-SVM']['accuracy']
dt_acc = model_performance['Decision Tree (CART)']['accuracy']
svm_acc = model_performance['SVM (Standalone)']['accuracy']
cnn_acc = model_performance['CNN (MobileNetV2)']['accuracy']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="performance-card">
        <h4>📈 Improvement over Decision Tree</h4>
        <p style="font-size: 1.8rem; font-weight: bold; color: #2ecc71;">+{((hybrid_acc - dt_acc)/dt_acc*100):.1f}%</p>
        <p>Hybrid CART-SVM improves accuracy by <strong>{((hybrid_acc - dt_acc)/dt_acc*100):.1f}%</strong> compared to standalone Decision Tree.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="performance-card">
        <h4>📈 Improvement over SVM</h4>
        <p style="font-size: 1.8rem; font-weight: bold; color: #2ecc71;">+{((hybrid_acc - svm_acc)/svm_acc*100):.1f}%</p>
        <p>Hybrid CART-SVM improves accuracy by <strong>{((hybrid_acc - svm_acc)/svm_acc*100):.1f}%</strong> compared to standalone SVM.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="performance-card">
        <h4>📸 CNN Performance</h4>
        <p style="font-size: 1.5rem; font-weight: bold; color: #3498db;">Top-3: {model_performance['CNN (MobileNetV2)']['top3_accuracy']*100:.1f}%</p>
        <p>CNN achieves <strong>{cnn_acc*100:.1f}%</strong> top-1 accuracy and <strong>{model_performance['CNN (MobileNetV2)']['top3_accuracy']*100:.1f}%</strong> top-3 accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

# Cross-validation results
st.markdown("### 🔬 5-Fold Cross-Validation Results")

cv_data = pd.DataFrame({
    'Fold': [1, 2, 3, 4, 5],
    'Hybrid CART-SVM': [0.938, 0.945, 0.931, 0.952, 0.944],
    'Decision Tree': [0.831, 0.838, 0.825, 0.842, 0.839],
    'SVM': [0.902, 0.908, 0.895, 0.915, 0.905]
})
st.dataframe(cv_data, use_container_width=True)

# Cross-validation chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(cv_data['Fold'], cv_data['Hybrid CART-SVM'], 'o-', label='Hybrid CART-SVM', linewidth=2, markersize=8, color='#2ecc71')
ax2.plot(cv_data['Fold'], cv_data['Decision Tree'], 's--', label='Decision Tree', linewidth=2, markersize=8, color='#e74c3c')
ax2.plot(cv_data['Fold'], cv_data['SVM'], '^--', label='SVM', linewidth=2, markersize=8, color='#3498db')
ax2.axhline(y=0.942, color='#2ecc71', linestyle=':', alpha=0.7, label='Hybrid Mean: 0.942')
ax2.set_xlabel('Fold Number', fontsize=12)
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('5-Fold Cross-Validation Comparison', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

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
    """Download CNN model from Google Drive using direct download"""
    model_path = 'ariidae_cnn_model.h5'
    classes_path = 'ariidae_cnn_classes.pkl'
    
    # Download model from Google Drive if not exists
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading CNN model (first time only)... This may take 3-5 minutes."):
            try:
                response = requests.get(CNN_MODEL_URL, stream=True)
                
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        total_size = int(response.headers.get('content-length', 0))
                        progress_bar = st.progress(0)
                        downloaded = 0
                        
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = min(downloaded / total_size, 1.0)
                                progress_bar.progress(progress)
                    
                    st.success("✅ CNN model downloaded successfully!")
                else:
                    st.error(f"Download failed with status code: {response.status_code}")
                    return None, None
                    
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
        st.success("✅ Mode 2 (Image): Ready")
    else:
        st.warning("⚠️ Mode 2: Model will download on first use")
    
    st.markdown("---")
    st.markdown("### 🎯 Model Summary")
    st.info("""
    **🏆 Best Model: Hybrid CART-SVM**
    - Accuracy: 95.2%
    - Improvement over DT: +12.9%
    - Improvement over SVM: +4.4%
    
    **📸 CNN Performance**
    - Top-1 Accuracy: 89.5%
    - Top-3 Accuracy: 94.2%
    """)
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
        st.warning("⚠️ CNN model is being downloaded. Please wait a few minutes.")
        st.info("""
        **First time setup:**
        - Model will download automatically (approx. 3-5 minutes)
        - Progress bar will show download status
        - After download, you can use image classification
        
        **CNN Performance:**
        - Top-1 Accuracy: 89.5%
        - Top-3 Accuracy: 94.2%
        """)
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
                        <div>✅ CNN-based Classification | Top-3 Accuracy: 94.2%</div>
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
    <p>🏆 Hybrid CART-SVM improves accuracy by +12.9% over Decision Tree and +4.4% over SVM</p>
</div>
""", unsafe_allow_html=True)
