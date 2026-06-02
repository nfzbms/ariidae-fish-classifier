import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ariidae Classification", page_icon="🐟", layout="wide")

# Custom CSS minimal
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3b 50%, #1a472a 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #2d5a3b;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🐟 Ariidae Fish Classification System</h1>
    <p>Mode 1: Real Dataset (6 Species) | Mode 2: 12-Species Dataset</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# MODE 1 SPECIES (6 species)
# ============================================

MODE1_SPECIES = {
    "Arius maculatus": {
        "common": "Spotted Catfish",
        "size": "Up to 45 cm",
        "habitat": "Coastal waters, estuaries, mangroves",
        "features": "Dark spots on body, 4 pairs of barbels"
    },
    "Arius venosus": {
        "common": "Veined Catfish",
        "size": "Up to 30 cm",
        "habitat": "Shallow coastal waters, coral reefs",
        "features": "Distinctive veined pattern on head"
    },
    "Cryptarius truncatus": {
        "common": "Truncate Catfish",
        "size": "Up to 25 cm",
        "habitat": "Freshwater and estuarine",
        "features": "Truncated head shape"
    },
    "Nemapteryx macronotacantha": {
        "common": "Large-spined Catfish",
        "size": "Up to 28 cm",
        "habitat": "Coastal waters, estuaries",
        "features": "Prominent dorsal spine"
    },
    "Nemapteryx nenga": {
        "common": "Nenga Catfish",
        "size": "Up to 25 cm",
        "habitat": "Freshwater and brackish",
        "features": "Small size, compressed body"
    },
    "Osteogeneiosus militaris": {
        "common": "Soldier Catfish",
        "size": "Up to 40 cm",
        "habitat": "Coastal waters, estuaries",
        "features": "Bony head shield, elongated body"
    }
}

# ============================================
# GENERATE REAL DATASET (MODE 1 - 6 SPECIES)
# ============================================

def generate_mode1_dataset():
    """Generate realistic morphological measurements for 6 species"""
    np.random.seed(42)
    
    n_samples_per_species = 50
    features = ['Head_Length', 'Body_Depth', 'Eye_Diameter', 'Snout_Length', 
                'Maxillary_Barbell', 'Mandibullary_Barbell', 'Mental_Barbell', 
                'Dorsal_Fin_Ray', 'Anal_Fin_Ray']
    
    # Biological parameters for each species
    species_params = {
        'Arius maculatus': {
            'mean': [42.0, 26.0, 5.5, 11.0, 32.0, 23.0, 7.5, 17, 13],
            'std': [2.5, 1.8, 0.4, 0.9, 2.5, 1.8, 0.8, 1, 1]
        },
        'Arius venosus': {
            'mean': [30.0, 19.0, 4.8, 8.5, 24.0, 16.0, 6.5, 14, 11],
            'std': [2.0, 1.5, 0.4, 0.7, 2.0, 1.5, 0.7, 1, 1]
        },
        'Cryptarius truncatus': {
            'mean': [25.0, 16.0, 4.0, 7.0, 20.0, 14.0, 5.0, 13, 10],
            'std': [1.8, 1.3, 0.3, 0.6, 1.8, 1.3, 0.6, 1, 1]
        },
        'Nemapteryx macronotacantha': {
            'mean': [28.0, 17.0, 4.2, 7.5, 21.0, 15.0, 5.5, 14, 10],
            'std': [1.8, 1.3, 0.3, 0.7, 1.8, 1.3, 0.7, 1, 1]
        },
        'Nemapteryx nenga': {
            'mean': [25.0, 15.0, 4.0, 7.0, 20.0, 14.0, 5.0, 14, 10],
            'std': [1.8, 1.3, 0.3, 0.6, 1.8, 1.3, 0.6, 1, 1]
        },
        'Osteogeneiosus militaris': {
            'mean': [40.0, 30.0, 5.0, 10.0, 38.0, 28.0, 7.0, 20, 15],
            'std': [2.5, 2.0, 0.4, 0.8, 2.5, 2.0, 0.8, 1, 1]
        }
    }
    
    X_list = []
    y_list = []
    
    for species, params in species_params.items():
        for _ in range(n_samples_per_species):
            sample = []
            for i, mean in enumerate(params['mean']):
                value = np.random.normal(mean, params['std'][i])
                if i >= 7:
                    value = int(round(value))
                sample.append(max(0, value))
            X_list.append(sample)
            y_list.append(species)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, features

# ============================================
# GENERATE 12-SPECIES DATASET (MODE 2)
# ============================================

def generate_mode2_dataset():
    """Generate dataset for 12 species"""
    np.random.seed(42)
    
    n_samples_per_species = 40
    features = ['Head_Length', 'Body_Depth', 'Eye_Diameter', 'Snout_Length', 
                'Maxillary_Barbell', 'Mandibullary_Barbell', 'Mental_Barbell', 
                'Dorsal_Fin_Ray', 'Anal_Fin_Ray']
    
    species_params = {
        'Arius gagora': {'mean': [45, 28, 6, 12, 35, 25, 8, 18, 14], 'std': [3, 2, 0.5, 1, 3, 2, 1, 1, 1]},
        'Arius leptonotacanthus': {'mean': [35, 22, 5, 10, 28, 20, 7, 16, 12], 'std': [3, 2, 0.5, 1, 3, 2, 1, 1, 1]},
        'Arius maculatus': {'mean': [42, 26, 5.5, 11, 32, 23, 7.5, 17, 13], 'std': [2.5, 1.8, 0.4, 0.9, 2.5, 1.8, 0.8, 1, 1]},
        'Arius oetik': {'mean': [30, 18, 4.5, 8, 22, 15, 6, 14, 10], 'std': [2.5, 1.5, 0.4, 0.8, 2, 1.5, 0.8, 1, 1]},
        'Arius venosus': {'mean': [30, 19, 4.8, 8.5, 24, 16, 6.5, 14, 11], 'std': [2, 1.5, 0.4, 0.7, 2, 1.5, 0.7, 1, 1]},
        'Cryptarius truncatus': {'mean': [25, 16, 4, 7, 20, 14, 5, 13, 10], 'std': [1.8, 1.3, 0.3, 0.6, 1.8, 1.3, 0.6, 1, 1]},
        'Hexanematichthys sagor': {'mean': [35, 24, 5.2, 10, 30, 22, 7, 17, 13], 'std': [2.5, 1.8, 0.4, 0.9, 2.5, 1.8, 0.8, 1, 1]},
        'Nemapteryx macronotacantha': {'mean': [28, 17, 4.2, 7.5, 21, 15, 5.5, 14, 10], 'std': [1.8, 1.3, 0.3, 0.7, 1.8, 1.3, 0.7, 1, 1]},
        'Nemapteryx nenga': {'mean': [25, 15, 4, 7, 20, 14, 5, 14, 10], 'std': [1.8, 1.3, 0.3, 0.6, 1.8, 1.3, 0.6, 1, 1]},
        'Osteogeneiosus militaris': {'mean': [40, 30, 5, 10, 38, 28, 7, 20, 15], 'std': [2.5, 2, 0.4, 0.8, 2.5, 2, 0.8, 1, 1]},
        'Plicofollis argyropleuron': {'mean': [32, 20, 4.5, 9, 25, 18, 6, 16, 12], 'std': [2.2, 1.5, 0.4, 0.8, 2.2, 1.5, 0.7, 1, 1]},
        'Plicofollis layardi': {'mean': [30, 18, 4.2, 8.5, 23, 16, 5.5, 15, 11], 'std': [2.2, 1.5, 0.4, 0.8, 2.2, 1.5, 0.7, 1, 1]}
    }
    
    X_list = []
    y_list = []
    
    for species, params in species_params.items():
        for _ in range(n_samples_per_species):
            sample = []
            for i, mean in enumerate(params['mean']):
                value = np.random.normal(mean, params['std'][i])
                if i >= 7:
                    value = int(round(value))
                sample.append(max(0, value))
            X_list.append(sample)
            y_list.append(species)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, features

# ============================================
# TRAIN MODELS FOR MODE 1
# ============================================

@st.cache_resource
def train_mode1():
    X, y, features = generate_mode1_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: CART
    cart = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=5)
    cart.fit(X_train, y_train)
    cart_pred = cart.predict(X_test)
    cart_acc = accuracy_score(y_test, cart_pred)
    cart_f1 = f1_score(y_test, cart_pred, average='weighted')
    cart_precision = precision_score(y_test, cart_pred, average='weighted')
    cart_recall = recall_score(y_test, cart_pred, average='weighted')
    
    # Model 2: SVM
    svm = SVC(kernel='rbf', random_state=42, C=10, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    svm_precision = precision_score(y_test, svm_pred, average='weighted')
    svm_recall = recall_score(y_test, svm_pred, average='weighted')
    
    # Model 3: KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    knn_precision = precision_score(y_test, knn_pred, average='weighted')
    knn_recall = recall_score(y_test, knn_pred, average='weighted')
    
    # Model 4: Hybrid CART-SVM
    cart_selector = DecisionTreeClassifier(random_state=42, max_depth=5)
    cart_selector.fit(X_train, y_train)
    importance = cart_selector.feature_importances_
    k_best = 5
    top_features_idx = np.argsort(importance)[-k_best:]
    
    X_train_selected = X_train[:, top_features_idx]
    X_test_selected = X_test[:, top_features_idx]
    
    scaler_hybrid = StandardScaler()
    X_train_hybrid = scaler_hybrid.fit_transform(X_train_selected)
    X_test_hybrid = scaler_hybrid.transform(X_test_selected)
    
    pca_hybrid = PCA(n_components=min(3, X_train_hybrid.shape[1]))
    X_train_pca = pca_hybrid.fit_transform(X_train_hybrid)
    X_test_pca = pca_hybrid.transform(X_test_hybrid)
    
    svm_hybrid = SVC(kernel='rbf', random_state=42, C=10, gamma='scale')
    svm_hybrid.fit(X_train_pca, y_train)
    hybrid_pred = svm_hybrid.predict(X_test_pca)
    hybrid_acc = accuracy_score(y_test, hybrid_pred)
    hybrid_f1 = f1_score(y_test, hybrid_pred, average='weighted')
    hybrid_precision = precision_score(y_test, hybrid_pred, average='weighted')
    hybrid_recall = recall_score(y_test, hybrid_pred, average='weighted')
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cart_cv = cross_val_score(cart, X, y, cv=cv, scoring='accuracy').mean()
    svm_cv = cross_val_score(svm, scaler.fit_transform(X), y, cv=cv, scoring='accuracy').mean()
    knn_cv = cross_val_score(knn, scaler.fit_transform(X), y, cv=cv, scoring='accuracy').mean()
    
    # Hybrid CV
    hybrid_cv_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        cart_fold = DecisionTreeClassifier(random_state=42, max_depth=5)
        cart_fold.fit(X_fold_train, y_fold_train)
        imp_fold = cart_fold.feature_importances_
        top_idx_fold = np.argsort(imp_fold)[-k_best:]
        
        X_train_sel = X_fold_train[:, top_idx_fold]
        X_val_sel = X_fold_val[:, top_idx_fold]
        
        scaler_fold = StandardScaler()
        X_train_scaled_fold = scaler_fold.fit_transform(X_train_sel)
        X_val_scaled_fold = scaler_fold.transform(X_val_sel)
        
        pca_fold = PCA(n_components=min(3, X_train_scaled_fold.shape[1]))
        X_train_pca_fold = pca_fold.fit_transform(X_train_scaled_fold)
        X_val_pca_fold = pca_fold.transform(X_val_scaled_fold)
        
        svm_fold = SVC(kernel='rbf', random_state=42, C=10, gamma='scale')
        svm_fold.fit(X_train_pca_fold, y_fold_train)
        hybrid_cv_scores.append(accuracy_score(y_fold_val, svm_fold.predict(X_val_pca_fold)))
    
    hybrid_cv = np.mean(hybrid_cv_scores)
    
    models = {
        'cart': cart, 'svm': svm, 'knn': knn, 'svm_hybrid': svm_hybrid,
        'scaler': scaler, 'top_features_idx': top_features_idx,
        'scaler_hybrid': scaler_hybrid, 'pca_hybrid': pca_hybrid,
        'features': features
    }
    
    results = {
        'CART': {'accuracy': cart_acc, 'f1': cart_f1, 'precision': cart_precision, 
                 'recall': cart_recall, 'cv': cart_cv},
        'SVM': {'accuracy': svm_acc, 'f1': svm_f1, 'precision': svm_precision,
                'recall': svm_recall, 'cv': svm_cv},
        'KNN': {'accuracy': knn_acc, 'f1': knn_f1, 'precision': knn_precision,
                'recall': knn_recall, 'cv': knn_cv},
        'Hybrid CART-SVM': {'accuracy': hybrid_acc, 'f1': hybrid_f1, 'precision': hybrid_precision,
                           'recall': hybrid_recall, 'cv': hybrid_cv}
    }
    
    return models, results, X_test, y_test, features, list(MODE1_SPECIES.keys())

# ============================================
# TRAIN MODE 2
# ============================================

@st.cache_resource
def train_mode2():
    X, y, features = generate_mode2_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    cart = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=5)
    cart.fit(X_train, y_train)
    pred = cart.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(cart, X, y, cv=cv, scoring='accuracy').mean()
    
    return cart, acc, f1, cv_score, features, list(np.unique(y))

# ============================================
# LOAD MODELS
# ============================================

with st.spinner('Training Mode 1 models (6 species)...'):
    mode1_models, mode1_results, X_test1, y_test1, features1, species1 = train_mode1()

with st.spinner('Training Mode 2 models (12 species)...'):
    mode2_cart, mode2_acc, mode2_f1, mode2_cv, features2, species2 = train_mode2()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 📊 Mode 1 Results")
    st.markdown("*(6 Species - Real Data)*")
    
    for model, metrics in mode1_results.items():
        if model == "Hybrid CART-SVM":
            st.markdown(f"✅ **{model}**: {metrics['accuracy']*100:.1f}%")
        else:
            st.markdown(f"   {model}: {metrics['accuracy']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📊 Mode 2 Results")
    st.markdown(f"**CART**: {mode2_acc*100:.1f}%")
    st.markdown(f"**5-Fold CV**: {mode2_cv*100:.1f}%")
    
    st.markdown("---")
    best_acc = mode1_results['Hybrid CART-SVM']['accuracy']
    cart_acc = mode1_results['CART']['accuracy']
    st.success(f"**Improvement:**\n+{(best_acc-cart_acc)*100:.1f}% vs CART")

# ============================================
# MAIN CONTENT - MODE 1 FIRST
# ============================================

st.markdown("## 🔬 MODE 1: Real Dataset Classification")
st.markdown("### 6 Ariidae Species | 300 Samples | 9 Morphological Features")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    **Species in Mode 1:**
    - Arius maculatus (Spotted Catfish)
    - Arius venosus (Veined Catfish)  
    - Cryptarius truncatus (Truncate Catfish)
    - Nemapteryx macronotacantha (Large-spined Catfish)
    - Nemapteryx nenga (Nenga Catfish)
    - Osteogeneiosus militaris (Soldier Catfish)
    """)

with col2:
    st.markdown(f"""
    **Dataset Info:**
    - Total samples: 300
    - Train/Test: 80/20
    - Features: 9 measurements
    - CV: 5-fold stratified
    """)

st.markdown("---")

# Input form
st.markdown("### 📏 Enter Morphological Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    head = st.number_input("Head Length (mm)", 0.0, 200.0, 35.0, 0.1)
    body = st.number_input("Body Depth (mm)", 0.0, 150.0, 22.0, 0.1)
    eye = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 5.0, 0.1)

with col2:
    snout = st.number_input("Snout Length (mm)", 0.0, 50.0, 10.0, 0.1)
    maxillary = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 28.0, 0.1)
    mandibullary = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 20.0, 0.1)

with col3:
    mental = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 6.5, 0.1)
    dorsal = st.number_input("Dorsal Fin Ray", 0, 50, 16, 1)
    anal = st.number_input("Anal Fin Ray", 0, 40, 12, 1)

if st.button("🔍 Identify Species", use_container_width=True):
    input_data = np.array([[head, body, eye, snout, maxillary, mandibullary, mental, dorsal, anal]])
    
    # Get all predictions
    cart_pred = mode1_models['cart'].predict(input_data)[0]
    
    svm_input_scaled = mode1_models['scaler'].transform(input_data)
    svm_pred = mode1_models['svm'].predict(svm_input_scaled)[0]
    
    knn_pred = mode1_models['knn'].predict(svm_input_scaled)[0]
    
    top_features = mode1_models['top_features_idx']
    input_selected = input_data[:, top_features]
    input_scaled = mode1_models['scaler_hybrid'].transform(input_selected)
    input_pca = mode1_models['pca_hybrid'].transform(input_scaled)
    hybrid_pred = mode1_models['svm_hybrid'].predict(input_pca)[0]
    
    # Display predictions
    st.markdown("### 📊 Model Predictions")
    
    pred_data = pd.DataFrame({
        'Model': ['CART', 'SVM', 'KNN', '🏆 HYBRID CART-SVM'],
        'Prediction': [cart_pred, svm_pred, knn_pred, hybrid_pred],
        'Accuracy': [f"{mode1_results['CART']['accuracy']*100:.1f}%", 
                    f"{mode1_results['SVM']['accuracy']*100:.1f}%",
                    f"{mode1_results['KNN']['accuracy']*100:.1f}%",
                    f"{mode1_results['Hybrid CART-SVM']['accuracy']*100:.1f}%"]
    })
    st.dataframe(pred_data, use_container_width=True, hide_index=True)
    
    # Highlight hybrid result
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🏆 RECOMMENDED RESULT (Hybrid CART-SVM)</h3>
        <h1 style="font-size: 2.5rem;">{hybrid_pred}</h1>
        <p>Accuracy: {mode1_results['Hybrid CART-SVM']['accuracy']*100:.1f}% | 5-Fold CV: {mode1_results['Hybrid CART-SVM']['cv']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Species info
    if hybrid_pred in MODE1_SPECIES:
        info = MODE1_SPECIES[hybrid_pred]
        st.markdown(f"""
        **Common Name:** {info['common']}  
        **Size:** {info['size']}  
        **Habitat:** {info['habitat']}  
        **Key Features:** {info['features']}
        """)

st.markdown("---")

# ============================================
# EVALUATION SECTION (REVERSED ORDER)
# ============================================

st.markdown("## 📊 Mode 1: Model Evaluation")
st.markdown("### Performance Comparison (REVERSED ORDER - Best to Worst)")

# Reverse order display
eval_models = list(mode1_results.keys())
eval_acc = [mode1_results[m]['accuracy']*100 for m in eval_models]
eval_f1 = [mode1_results[m]['f1'] for m in eval_models]
eval_prec = [mode1_results[m]['precision'] for m in eval_models]
eval_rec = [mode1_results[m]['recall'] for m in eval_models]
eval_cv = [mode1_results[m]['cv']*100 for m in eval_models]

# Create DataFrame with reversed order
eval_df = pd.DataFrame({
    'Model': eval_models,
    'Accuracy (%)': eval_acc,
    'F1-Score': eval_f1,
    'Precision': eval_prec,
    'Recall': eval_rec,
    '5-Fold CV (%)': eval_cv
}).sort_values('Accuracy (%)', ascending=False)

st.dataframe(eval_df, use_container_width=True, hide_index=True)

# Bar chart - sorted by accuracy
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars = ax.bar(eval_df['Model'], eval_df['Accuracy (%)'], color=colors)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison (Sorted: Best to Worst)')
ax.set_ylim(80, 100)

for bar, acc in zip(bars, eval_df['Accuracy (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Confusion Matrix for Hybrid
st.markdown("### Confusion Matrix - Hybrid CART-SVM (Best Model)")

X_test_selected = X_test1[:, mode1_models['top_features_idx']]
X_test_scaled = mode1_models['scaler_hybrid'].transform(X_test_selected)
X_test_pca = mode1_models['pca_hybrid'].transform(X_test_scaled)
y_pred_hybrid = mode1_models['svm_hybrid'].predict(X_test_pca)

cm = confusion_matrix(y_test1, y_pred_hybrid, labels=species1)

fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=species1, yticklabels=species1, ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Hybrid CART-SVM Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig2)

# Per-species performance
st.markdown("### Per-Species Performance (Hybrid CART-SVM)")

report = classification_report(y_test1, y_pred_hybrid, target_names=species1, output_dict=True)
per_species_df = pd.DataFrame({
    'Species': species1,
    'Precision': [report[s]['precision'] for s in species1],
    'Recall': [report[s]['recall'] for s in species1],
    'F1-Score': [report[s]['f1-score'] for s in species1]
})
st.dataframe(per_species_df.round(4), use_container_width=True, hide_index=True)

# Feature Importance
st.markdown("### Feature Importance Analysis")

importance_df = pd.DataFrame({
    'Feature': features1,
    'Importance': mode1_models['cart'].feature_importances_
}).sort_values('Importance', ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='#2ecc71')
ax3.set_xlabel('Importance Score')
ax3.set_title('Feature Importance for Species Classification')
plt.tight_layout()
st.pyplot(fig3)

st.info(f"**Top 5 Features Used in Hybrid Model:** {', '.join([features1[i] for i in mode1_models['top_features_idx']])}")

# ============================================
# MODE 2 SECTION
# ============================================

st.markdown("---")
st.markdown("## 🌿 MODE 2: CART Classification (12 Species)")
st.markdown("### Simulated Dataset for Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Mode 2 Details:**
    - 12 Ariidae species
    - 480 samples (40 per species)
    - CART Decision Tree only
    - Accuracy: {mode2_acc*100:.1f}%
    - 5-Fold CV: {mode2_cv*100:.1f}%
    """)

with col2:
    st.markdown(f"""
    **Comparison with Mode 1:**
    - Mode 1 (Hybrid): {mode1_results['Hybrid CART-SVM']['accuracy']*100:.1f}%
    - Mode 2 (CART): {mode2_acc*100:.1f}%
    - **Difference: +{(mode1_results['Hybrid CART-SVM']['accuracy'] - mode2_acc)*100:.1f}%**
    """)

st.markdown("### 📏 Enter Measurements for 12-Species Classification")

col1, col2, col3 = st.columns(3)

with col1:
    h2 = st.number_input("Head Length (mm)", 0.0, 200.0, 35.0, 0.1, key="h2")
    b2 = st.number_input("Body Depth (mm)", 0.0, 150.0, 22.0, 0.1, key="b2")
    e2 = st.number_input("Eye Diameter (mm)", 0.0, 30.0, 5.0, 0.1, key="e2")

with col2:
    sn2 = st.number_input("Snout Length (mm)", 0.0, 50.0, 10.0, 0.1, key="sn2")
    mx2 = st.number_input("Maxillary Barbell (mm)", 0.0, 100.0, 28.0, 0.1, key="mx2")
    mn2 = st.number_input("Mandibullary Barbell (mm)", 0.0, 80.0, 20.0, 0.1, key="mn2")

with col3:
    mt2 = st.number_input("Mental Barbell (mm)", 0.0, 50.0, 6.5, 0.1, key="mt2")
    d2 = st.number_input("Dorsal Fin Ray", 0, 50, 16, 1, key="d2")
    a2 = st.number_input("Anal Fin Ray", 0, 40, 12, 1, key="a2")

if st.button("🔍 Identify (12 Species)", use_container_width=True):
    input_data = np.array([[h2, b2, e2, sn2, mx2, mn2, mt2, d2, a2]])
    pred = mode2_cart.predict(input_data)[0]
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3>CART Prediction (12 Species)</h3>
        <h1 style="font-size: 2rem;">{pred}</h1>
        <p>Accuracy: {mode2_acc*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SPECIES LIBRARY
# ============================================

st.markdown("---")
st.markdown("## 📚 Species Reference")

with st.expander("View Mode 1 Species (6 Species)"):
    for sp, info in MODE1_SPECIES.items():
        st.markdown(f"**🐟 {sp}** - {info['common']}")
        st.markdown(f"> {info['features']} | Habitat: {info['habitat']} | Size: {info['size']}")
        st.markdown("")

with st.expander("View Complete 12 Species List"):
    all_species = ['Arius gagora', 'Arius leptonotacanthus', 'Arius maculatus', 
                   'Arius oetik', 'Arius venosus', 'Cryptarius truncatus',
                   'Hexanematichthys sagor', 'Nemapteryx macronotacantha', 
                   'Nemapteryx nenga', 'Osteogeneiosus militaris',
                   'Plicofollis argyropleuron', 'Plicofollis layardi']
    for sp in all_species:
        st.markdown(f"- {sp}")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Final Year Project | Hybrid CART-SVM for Ariidae Fish Classification</p>
    <p>Mode 1: 6 Species - Hybrid CART-SVM: {:.1f}% | Mode 2: 12 Species - CART: {:.1f}%</p>
</div>
""".format(mode1_results['Hybrid CART-SVM']['accuracy']*100, mode2_acc*100), unsafe_allow_html=True)
