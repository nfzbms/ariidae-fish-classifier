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
    .mode-selector {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .prediction-card {
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
    .performance-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .best-model {
        border: 2px solid #11998e;
        background: #e8f5e9;
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
    <p>Final Year Project: Hybrid CART-SVM vs Standalone Models</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_mode1_models():
    try:
        scaler = joblib.load('scaler.pkl')
        dt = joblib.load('cart_model.pkl')
        svm = joblib.load('svm_model.pkl')
        knn = joblib.load('knn_model.pkl')
        dt_selector = joblib.load('dt_selector.pkl')
        selector = joblib.load('feature_selector.pkl')
        scaler_hybrid = joblib.load('scaler_hybrid.pkl')
        pca_hybrid = joblib.load('pca_hybrid.pkl')
        svm_hybrid = joblib.load('svm_hybrid.pkl')
        features = joblib.load('features.pkl')
        classes = joblib.load('classes.pkl')
        return scaler, dt, svm, knn, dt_selector, selector, scaler_hybrid, pca_hybrid, svm_hybrid, features, classes
    except Exception as e:
        st.error(f"Error loading Mode 1 models: {e}")
        return None, None, None, None, None, None, None, None, None, None, None

@st.cache_resource
def load_mode2_models():
    try:
        cart_sim = joblib.load('cart_sim_model.pkl')
        sim_features = joblib.load('sim_features.pkl')
        sim_classes = joblib.load('sim_classes.pkl')
        return cart_sim, sim_features, sim_classes
    except Exception as e:
        return None, None, None

# Load all models
(m1_scaler, m1_dt, m1_svm, m1_knn, m1_dt_selector, m1_selector, 
 m1_scaler_hybrid, m1_pca_hybrid, m1_svm_hybrid, m1_features, m1_classes) = load_mode1_models()

m2_cart, m2_features, m2_classes = load_mode2_models()

# ============================================
# MODEL PERFORMANCE DATA (Display in App)
# ============================================

st.markdown("## 📊 Model Performance Comparison (Real Data)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="performance-card">
        <div style="font-size: 1.8rem; font-weight: bold;">84.3%</div>
        <div>🌿 Decision Tree</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="performance-card">
        <div style="font-size: 1.8rem; font-weight: bold;">91.2%</div>
        <div>⚡ SVM</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="performance-card">
        <div style="font-size: 1.8rem; font-weight: bold;">88.7%</div>
        <div>📊 KNN</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="performance-card best-model">
        <div style="font-size: 1.8rem; font-weight: bold; color: #11998e;">95.2%</div>
        <div>🏆 HYBRID CART-SVM</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# MODE SELECTION
# ============================================

st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.subheader("🎯 Select Classification Mode")

col1, col2 = st.columns(2)

with col1:
    mode1 = st.button("📏 MODE 1: REAL DATA\nEnter 9 morphological measurements\n🏆 Hybrid CART-SVM (Best Model)", use_container_width=True, type="primary")

with col2:
    mode2 = st.button("📈 MODE 2: SIMULATED DATA\nCART Model on Simulated Data\n🌿 For comparison only", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "mode1"
if mode1:
    st.session_state.selected_mode = "mode1"
if mode2:
    st.session_state.selected_mode = "mode2"

# ============================================
# MODE 1: REAL DATA (9 Features Only)
# ============================================
if st.session_state.selected_mode == "mode1":
    st.markdown("## 📏 Mode 1: Real Data - Enter 9 Measurements")
    st.markdown("*Please enter the 9 morphological measurements below*")
    
    if m1_scaler is None:
        st.error("⚠️ Models not loaded")
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
            try:
                input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
                
                # Process through hybrid pipeline
                X_selected = m1_selector.transform(input_data)
                X_scaled = m1_scaler_hybrid.transform(X_selected)
                X_pca = m1_pca_hybrid.transform(X_scaled)
                prediction = m1_svm_hybrid.predict(X_pca)[0]
                
                # Show comparison with other models
                dt_pred = m1_dt.predict(input_data)[0]
                svm_pred = m1_svm.predict(m1_scaler.transform(input_data))[0]
                knn_pred = m1_knn.predict(m1_scaler.transform(input_data))[0]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div>🎯 Predicted Species (HYBRID CART-SVM)</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>✅ Best Model - 95.2% Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Comparison table
                st.markdown("### 📊 Model Comparison for this Input")
                comparison = pd.DataFrame({
                    'Model': ['Decision Tree', 'SVM', 'KNN', '🏆 HYBRID CART-SVM'],
                    'Prediction': [dt_pred, svm_pred, knn_pred, prediction],
                    'Model Accuracy': ['84.3%', '91.2%', '88.7%', '95.2%']
                })
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================
# MODE 2: SIMULATED DATA (CART Only)
# ============================================
else:
    st.markdown("## 📈 Mode 2: Simulated Data - CART Model")
    st.markdown("*Enter values to test CART model on simulated data*")
    
    if m2_cart is None:
        st.info("ℹ️ Simulated data model ready. Accuracy: ~81.2% (for comparison)")
        st.warning("⚠️ Simulated model not loaded. Ensure 'cart_sim_model.pkl' exists.")
    else:
        st.info(f"This model uses {len(m2_features)} features and has ~81.2% accuracy (CART on simulated data)")
        
        # Create input fields based on features
        input_values = {}
        col1, col2, col3 = st.columns(3)
        
        for i, feature in enumerate(m2_features):
            display = feature.replace('_', ' ').title()
            if i < 3:
                with col1:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
            elif i < 6:
                with col2:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
            else:
                with col3:
                    input_values[feature] = st.number_input(display, 0.0, 200.0, 45.0, 0.1)
        
        if st.button("🔍 Identify Species (Simulated)", use_container_width=True):
            try:
                input_list = [input_values[f] for f in m2_features]
                input_data = np.array([input_list])
                prediction = m2_cart.predict(input_data)[0]
                
                st.markdown(f"""
                <div class="prediction-card-sim">
                    <div>🎯 Predicted Species (Simulated Data)</div>
                    <div class="prediction-species">{prediction}</div>
                    <div>🌿 CART Model | 81.2% Accuracy (Comparison Only)</div>
                    <div>⚠️ Note: Simulated data produces lower accuracy than real data</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("💡 **Conclusion:** Real data with Hybrid CART-SVM (Mode 1) achieves **95.2%** accuracy, which is **+14.0%** higher than CART on simulated data (81.2%). This proves that real morphological measurements with hybrid approach are superior!")
                
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================
# SIDEBAR - CONCLUSION
# ============================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=80)
    st.markdown("---")
    
    st.markdown("### 🎯 FYP Conclusion")
    st.success("""
    **🏆 HYBRID CART-SVM is the BEST!**
    
    | Model | Accuracy |
    |-------|----------|
    | Decision Tree | 84.3% |
    | SVM | 91.2% |
    | KNN | 88.7% |
    | **HYBRID CART-SVM** | **95.2%** |
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Key Finding")
    st.info("""
    **Real Data + Hybrid Approach = Best Results**
    
    - Hybrid CART-SVM outperforms DT by **+12.9%**
    - Hybrid CART-SVM outperforms SVM by **+4.4%**
    - Real data (95.2%) > Simulated data (81.2%)
    """)
    
    st.markdown("---")
    st.caption("Final Year Project | Hybrid CART-SVM for Ariidae Classification")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project - Hybrid CART-SVM for Ariidae Fish Classification</p>
    <p>🏆 Best Model: Hybrid CART-SVM (95.2% Accuracy) | Improvement over DT: +12.9%</p>
</div>
""", unsafe_allow_html=True)
