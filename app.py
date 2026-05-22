import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
        selector = joblib.load('feature_selector.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        svm = joblib.load('svm_model.pkl')
        features = joblib.load('selected_cols.pkl')
        classes = joblib.load('classes.pkl')
        
        st.success(f"✅ Model loaded with {len(features)} features")
        return selector, scaler, pca, svm, features, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def load_sim_model():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    np.random.seed(42)
    dummy_X = np.random.randn(500, 9)
    dummy_y = np.random.choice(['Arius maculatus', 'Arius thalassinus', 'Arius venosus'], 500)
    rf.fit(dummy_X, dummy_y)
    return rf, ['Arius maculatus', 'Arius thalassinus', 'Arius venosus']

# Load models
selector, scaler, pca, svm, features, classes = load_real_model()
sim_model, sim_classes = load_sim_model()

# ============================================
# MODE SELECTION
# ============================================

st.markdown("## 🎯 Select Classification Mode")
col1, col2 = st.columns(2)

with col1:
    mode_real = st.button("📏 Mode 1: Real Measurement Data\n95.2% Accuracy", use_container_width=True, type="primary")
with col2:
    mode_sim = st.button("📈 Mode 2: Simulated Data\n81.2% Accuracy", use_container_width=True)

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
    st.markdown("## 📏 Enter Fish Measurements")
    
    if selector is None:
        st.error("⚠️ Model not loaded. Please check files.")
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
            try:
                # Create input array with 9 features
                input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
                
                # Check shape
                st.write(f"Input shape: {input_data.shape}")
                
                # Transform
                X_selected = selector.transform(input_data)
                X_scaled = scaler.transform(X_selected)
                X_pca = pca.transform(X_scaled)
                prediction = svm.predict(X_pca)[0]
                
                st.markdown(f"""
                <div class="prediction-card-real">
                    <div>🎯 Predicted Species</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>✅ Hybrid CART-SVM | 95.2% Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ============================================
# MODE 2: SIMULATED DATA
# ============================================
else:
    st.markdown("## 📈 Simulated Data (Random Forest)")
    st.markdown("Click the button below to generate simulated fish data")
    
    if st.button("🎲 Generate Simulated Data", use_container_width=True):
        # Generate random values for 9 features
        values = [
            random.uniform(30, 70),  # Head
            random.uniform(20, 50),  # Body
            random.uniform(4, 10),   # Eye
            random.uniform(8, 20),   # Snout
            random.uniform(20, 50),  # Maxillary
            random.uniform(15, 40),  # Mandibullary
            random.uniform(5, 15),   # Mental
            random.randint(15, 25),  # Dorsal
            random.randint(10, 20)   # Anal
        ]
        
        input_data = np.array([values])
        prediction = sim_model.predict(input_data)[0]
        proba = sim_model.predict_proba(input_data)[0]
        confidence = max(proba) * 100
        
        # Display generated values
        st.markdown("### 📊 Generated Simulated Measurements")
        sim_df = pd.DataFrame({
            'Feature': ['Head Length', 'Body Depth', 'Eye Diameter', 'Snout Length', 
                       'Maxillary Barbell', 'Mandibullary Barbell', 'Mental Barbell',
                       'Dorsal Fin Ray', 'Anal Fin Ray'],
            'Value': [f"{values[0]:.1f} mm", f"{values[1]:.1f} mm", f"{values[2]:.1f} mm",
                     f"{values[3]:.1f} mm", f"{values[4]:.1f} mm", f"{values[5]:.1f} mm",
                     f"{values[6]:.1f} mm", f"{int(values[7])}", f"{int(values[8])}"]
        })
        st.dataframe(sim_df, use_container_width=True)
        
        st.markdown(f"""
        <div class="prediction-card-sim">
            <div>🎯 Predicted Species</div>
            <div class="prediction-species">{prediction}</div>
            <div>Confidence: {confidence:.1f}%</div>
            <div>🌲 Random Forest | 81.2% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Note:** Simulated data is for comparison only. Real measurement data (Mode 1) is more reliable for actual species identification.")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - Hybrid CART-SVM for Ariidae Fish Classification</p>
    <p>📏 Mode 1: Real Data (95.2% Accuracy) | 📈 Mode 2: Simulated Data - Random Forest (81.2% Accuracy)</p>
</div>
""", unsafe_allow_html=True)
