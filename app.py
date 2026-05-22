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
    <p>Mode 1: Real Data (SVM) | Mode 2: Simulated Data (Random Forest)</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_real_model():
    try:
        scaler = joblib.load('scaler_real.pkl')
        svm = joblib.load('svm_real_model.pkl')
        features = joblib.load('real_features.pkl')
        classes = joblib.load('real_classes.pkl')
        return scaler, svm, features, classes
    except Exception as e:
        st.error(f"Error loading real model: {e}")
        return None, None, None, None

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
real_scaler, real_svm, real_features, real_classes = load_real_model()
sim_scaler, sim_rf, sim_features, sim_classes = load_sim_model()

# ============================================
# MODE SELECTION
# ============================================

st.markdown("## 🎯 Select Classification Mode")
col1, col2 = st.columns(2)

with col1:
    mode_real = st.button("📏 Mode 1: Real Data\nActual fish measurements\nSVM Model", use_container_width=True, type="primary")
with col2:
    mode_sim = st.button("📈 Mode 2: Simulated Data\nFrom simulated Excel data\nRandom Forest", use_container_width=True)

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
    
    if real_scaler is None:
        st.error("⚠️ Real data model not loaded.")
    else:
        st.info(f"This model uses {len(real_features)} feature(s): {', '.join(real_features)}")
        
        # Create input fields based on features
        input_values = {}
        cols = st.columns(min(3, len(real_features)))
        
        for i, feature in enumerate(real_features):
            display = feature.replace('_', ' ').title()
            with cols[i % len(cols)]:
                input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
        
        if st.button("🔍 Identify Species", use_container_width=True):
            try:
                input_list = [input_values[f] for f in real_features]
                input_data = np.array([input_list])
                
                X_scaled = real_scaler.transform(input_data)
                prediction = real_svm.predict(X_scaled)[0]
                
                st.markdown(f"""
                <div class="prediction-card-real">
                    <div>🎯 Predicted Species (Real Data)</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>✅ SVM Model | {len(real_features)} features</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ============================================
# MODE 2: SIMULATED DATA
# ============================================
else:
    st.markdown("## 📈 Simulated Data Mode")
    
    if sim_scaler is None:
        st.error("⚠️ Simulated data model not loaded.")
    else:
        st.info(f"This model uses {len(sim_features)} feature(s)")
        
        input_values = {}
        cols = st.columns(min(3, len(sim_features)))
        
        for i, feature in enumerate(sim_features):
            display = feature.replace('_', ' ').title()
            with cols[i % len(cols)]:
                input_values[feature] = st.number_input(f"{display} (Sim)", 0.0, 200.0, 45.0, 0.1)
        
        if st.button("🔍 Identify Species (Simulated)", use_container_width=True):
            try:
                input_list = [input_values[f] for f in sim_features]
                input_data = np.array([input_list])
                
                X_scaled = sim_scaler.transform(input_data)
                prediction = sim_rf.predict(X_scaled)[0]
                proba = sim_rf.predict_proba(X_scaled)[0]
                confidence = max(proba) * 100
                
                st.markdown(f"""
                <div class="prediction-card-sim">
                    <div>🎯 Predicted Species (Simulated Data)</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>Confidence: {confidence:.1f}%</div>
                    <div>🌲 Random Forest | {len(sim_features)} features</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - SVM + Random Forest for Ariidae Fish Classification</p>
</div>
""", unsafe_allow_html=True)
