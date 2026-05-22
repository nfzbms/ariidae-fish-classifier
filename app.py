import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

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
    .prediction-card-real {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-card-sim {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
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
    <p>Mode 1: Real Data (Hybrid CART-SVM) | Mode 2: Simulated Data (Random Forest)</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_real_model():
    try:
        scaler = joblib.load('scaler_real.pkl')
        pca = joblib.load('pca_real.pkl')
        svm = joblib.load('svm_real_model.pkl')
        features = joblib.load('real_features.pkl')
        classes = joblib.load('real_classes.pkl')
        return scaler, pca, svm, features, classes
    except Exception as e:
        st.error(f"Error loading real model: {e}")
        return None, None, None, None, None

@st.cache_resource
def load_sim_model():
    try:
        scaler = joblib.load('scaler_sim.pkl')
        rf = joblib.load('sim_model.pkl')
        features = joblib.load('sim_features.pkl')
        classes = joblib.load('sim_classes.pkl')
        return scaler, rf, features, classes
    except Exception as e:
        st.error(f"Error loading simulated model: {e}")
        return None, None, None, None

# Load models
real_scaler, real_pca, real_svm, real_features, real_classes = load_real_model()
sim_scaler, sim_rf, sim_features, sim_classes = load_sim_model()

# ============================================
# MODE SELECTION
# ============================================

st.markdown("## 🎯 Select Classification Mode")
col1, col2 = st.columns(2)

with col1:
    mode_real = st.button("📏 Mode 1: Real Data\nActual fish measurements\n95.2% Accuracy", use_container_width=True, type="primary")
with col2:
    mode_sim = st.button("📈 Mode 2: Simulated Data\nFrom simulated Excel data\n81.2% Accuracy", use_container_width=True)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "real"
if mode_real:
    st.session_state.selected_mode = "real"
if mode_sim:
    st.session_state.selected_mode = "sim"

# ============================================
# MODE 1: REAL DATA
# ============================================
if st.session_state.selected_mode == "real":
    st.markdown("## 📏 Real Data Mode")
    st.markdown("Enter actual morphological measurements of Ariidae fish")
    
    if real_scaler is None:
        st.error("⚠️ Real data model not loaded.")
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
            X_scaled = real_scaler.transform(input_data)
            X_pca = real_pca.transform(X_scaled)
            prediction = real_svm.predict(X_pca)[0]
            
            st.markdown(f"""
            <div class="prediction-card-real">
                <div>🎯 Predicted Species (Real Data)</div>
                <div class="prediction-species">{prediction}</div>
                <div>✅ Hybrid CART-SVM | 95.2% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# MODE 2: SIMULATED DATA
# ============================================
else:
    st.markdown("## 📈 Simulated Data Mode")
    st.markdown("Enter simulated fish measurements (from your Excel data)")
    
    if sim_scaler is None:
        st.error("⚠️ Simulated data model not loaded.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            head = st.number_input("Head Length (mm) - Sim", 0.0, 200.0, 45.0, 0.1)
            body = st.number_input("Body Depth (mm) - Sim", 0.0, 150.0, 28.0, 0.1)
            eye = st.number_input("Eye Diameter (mm) - Sim", 0.0, 30.0, 6.0, 0.1)
        
        with col2:
            snout = st.number_input("Snout Length (mm) - Sim", 0.0, 50.0, 12.0, 0.1)
            maxillary = st.number_input("Maxillary Barbell (mm) - Sim", 0.0, 100.0, 35.0, 0.1)
            mandibullary = st.number_input("Mandibullary Barbell (mm) - Sim", 0.0, 80.0, 25.0, 0.1)
        
        with col3:
            mental = st.number_input("Mental Barbell (mm) - Sim", 0.0, 50.0, 8.0, 0.1)
            dorsal = st.number_input("Dorsal Fin Ray - Sim", 0, 50, 18, 1)
            anal = st.number_input("Anal Fin Ray - Sim", 0, 40, 14, 1)
        
        if st.button("🔍 Identify Species (Simulated)", use_container_width=True):
            input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
            X_scaled = sim_scaler.transform(input_data)
            prediction = sim_rf.predict(X_scaled)[0]
            proba = sim_rf.predict_proba(X_scaled)[0]
            confidence = max(proba) * 100
            
            st.markdown(f"""
            <div class="prediction-card-sim">
                <div>🎯 Predicted Species (Simulated Data)</div>
                <div class="prediction-species">{prediction}</div>
                <div>Confidence: {confidence:.1f}%</div>
                <div>🌲 Random Forest | 81.2% Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - Hybrid CART-SVM + Random Forest for Ariidae Fish Classification</p>
    <p>📏 Mode 1: Real Data (95.2% Accuracy) | 📈 Mode 2: Simulated Data (81.2% Accuracy)</p>
</div>
""", unsafe_allow_html=True)
