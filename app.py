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
# LOAD MODELS (SAMA DENGAN CODE TRAINING ANDA)
# ============================================

@st.cache_resource
def load_real_model():
    """Load real data models from training"""
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
    """Load simulated data models from training"""
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

# Show feature info in sidebar
with st.sidebar:
    st.header("📊 Model Info")
    if real_features:
        st.info(f"Real Model: {len(real_features)} features selected by DT")
    if sim_features:
        st.info(f"Sim Model: {len(sim_features)} features")

# ============================================
# MODE SELECTION
# ============================================

st.markdown("## 🎯 Select Classification Mode")
col1, col2 = st.columns(2)

with col1:
    mode_real = st.button("📏 Mode 1: Real Data\nActual fish measurements\nHybrid CART-SVM", use_container_width=True, type="primary")
with col2:
    mode_sim = st.button("📈 Mode 2: Simulated Data\nFrom simulated Excel data\nRandom Forest", use_container_width=True)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "real"
if mode_real:
    st.session_state.selected_mode = "real"
if mode_sim:
    st.session_state.selected_mode = "sim"

# ============================================
# MODE 1: REAL DATA (Hybrid CART-SVM)
# ============================================
if st.session_state.selected_mode == "real":
    st.markdown("## 📏 Real Data Mode")
    st.markdown("Enter the morphological measurements below")
    
    if real_scaler is None:
        st.error("⚠️ Real data model not loaded.")
    else:
        # Show which features are required
        st.info(f"This model uses {len(real_features)} features: {', '.join(real_features)}")
        
        # Create input fields based on the features from training
        input_values = {}
        
        # Map feature names to display names
        display_names = {
            'Head_length': 'Head Length (mm)',
            'Body_depth': 'Body Depth (mm)',
            'Eye_diameter': 'Eye Diameter (mm)',
            'Snout_length': 'Snout Length (mm)',
            'Maxillary_barbell_length': 'Maxillary Barbell (mm)',
            'Mandibullary_barbell_length': 'Mandibullary Barbell (mm)',
            'Mental_barbell_length': 'Mental Barbell (mm)',
            'Dorsal_fin_ray': 'Dorsal Fin Ray',
            'Anal_fin_ray': 'Anal Fin Ray'
        }
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        for i, feature in enumerate(real_features):
            display = display_names.get(feature, feature.replace('_', ' ').title())
            
            if i < 3:
                with col1:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
            elif i < 6:
                with col2:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
            else:
                with col3:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
        
        if st.button("🔍 Identify Species", use_container_width=True):
            try:
                # Create input array in the correct order (same as training)
                input_list = [input_values[feature] for feature in real_features]
                input_data = np.array([input_list])
                
                # Apply same transformations as training
                X_scaled = real_scaler.transform(input_data)
                X_pca = real_pca.transform(X_scaled)
                prediction = real_svm.predict(X_pca)[0]
                
                st.markdown(f"""
                <div class="prediction-card-real">
                    <div>🎯 Predicted Species (Real Data)</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>✅ Hybrid CART-SVM | Feature Selection by DT</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ============================================
# MODE 2: SIMULATED DATA (Random Forest)
# ============================================
else:
    st.markdown("## 📈 Simulated Data Mode")
    st.markdown("Enter the simulated fish measurements below")
    
    if sim_scaler is None:
        st.error("⚠️ Simulated data model not loaded.")
    else:
        st.info(f"This model uses {len(sim_features)} features")
        
        # Create input fields for simulated data
        input_values = {}
        col1, col2, col3 = st.columns(3)
        
        for i, feature in enumerate(sim_features):
            display = feature.replace('_', ' ').title()
            
            if i < 3:
                with col1:
                    input_values[feature] = st.number_input(f"{display} (Sim)", 0.0, 200.0, 45.0, 0.1)
            elif i < 6:
                with col2:
                    input_values[feature] = st.number_input(f"{display} (Sim)", 0.0, 200.0, 45.0, 0.1)
            else:
                with col3:
                    input_values[feature] = st.number_input(f"{display} (Sim)", 0.0, 200.0, 45.0, 0.1)
        
        if st.button("🔍 Identify Species (Simulated)", use_container_width=True):
            try:
                input_list = [input_values[feature] for feature in sim_features]
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
                    <div>🌲 Random Forest | 81.2% Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - Hybrid CART-SVM + Random Forest for Ariidae Fish Classification</p>
</div>
""", unsafe_allow_html=True)
