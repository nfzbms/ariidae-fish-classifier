# ============================================
# TAB 1: HOME (HANYA BAHAGIAN INI YANG DIUBAH)
# ============================================
with tab1:
    st.markdown("## Welcome to Ariidae Fish Classification System")
    
    # Training data information
    st.markdown("""
    <div class="info-box">
        <strong>📊 Training Data Information:</strong><br>
        • <strong>Hybrid CART-SVM Model:</strong> Trained and evaluated using <strong>real morphological data</strong> from <strong>6 Ariidae species</strong> with sufficient specimen records<br>
        • <strong>CART Model (Simulated):</strong> Developed using a <strong>simulated dataset</strong> covering <strong>12 Ariidae species</strong><br>
        • The system can classify all 12 species from the library, with higher confidence for the 6 real-trained species
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        
        This system uses **Hybrid CART-SVM** machine learning approach 
        to automatically classify **Ariidae fish species** based on 
        **9 morphological measurements**.
        
        #### Key Features:
        - ✅ **95.2% Accuracy** - Highest among compared models
        - ✅ **6 Real Species** - Trained on actual specimen data
        - ✅ **12 Species Library** - Comprehensive Ariidae coverage
        - ✅ **9 Measurements** - Easy data collection
        - ✅ **Real-time Prediction** - Instant results
        
        #### Model Comparison:
        - 🌿 Decision Tree (CART): 84.3%
        - ⚡ SVM Standalone: 91.2%
        - 📊 KNN: 88.7%
        - 🏆 **Hybrid CART-SVM: 95.2%**
        """)
    
    with col2:
        st.markdown("""
        ### 🐟 Species Training Status
        
        | No | Species Name | Data Source |
        |----|--------------|-------------|
        | 1 | Arius gagora | 📊 Simulated Data |
        | 2 | Arius leptonotacanthus | 📊 Simulated Data |
        | 3 | Arius maculatus | ✅ Real Data |
        | 4 | Arius oetik | 📊 Simulated Data |
        | 5 | Arius venosus | ✅ Real Data |
        | 6 | Cryptarius truncatus | ✅ Real Data |
        | 7 | Hexanematichthys sagor | 📊 Simulated Data |
        | 8 | Nemapteryx macronotacantha | ✅ Real Data |
        | 9 | Nemapteryx nenga | ✅ Real Data |
        | 10 | Osteogeneiosus militaris | ✅ Real Data |
        | 11 | Plicofollis argyropleuron | 📊 Simulated Data |
        | 12 | Plicofollis layardi | 📊 Simulated Data |
        
        **Note:** Hybrid model achieves highest accuracy on 6 real-trained species (✅ Real Data).
        """)
    
    st.markdown("---")
    
    st.markdown("### 🔬 Research Value")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="performance-card">
            <div style="font-size: 1.5rem; font-weight: bold;">95.2%</div>
            <div>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="performance-card">
            <div style="font-size: 1.5rem; font-weight: bold;">6</div>
            <div>Real Species Trained</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="performance-card">
            <div style="font-size: 1.5rem; font-weight: bold;">12</div>
            <div>Species Library</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="performance-card">
            <div style="font-size: 1.5rem; font-weight: bold;">9</div>
            <div>Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📝 About Ariidae Fish</h4>
        <p>Ariidae, commonly known as sea catfishes, are found in coastal waters, 
        estuaries, and freshwater systems. They play important roles in local 
        fisheries and ecosystem health. Accurate species identification is crucial 
        for fisheries management and conservation efforts.</p>
        <p><strong>⚠️ Note:</strong> The Hybrid CART-SVM model was trained on 6 species with sufficient real specimen records. 
        The remaining 6 species are included in the library using simulated data for reference purposes.</p>
    </div>
    """, unsafe_allow_html=True)
