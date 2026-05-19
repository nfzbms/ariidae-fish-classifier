# ============================================
# GOOGLE DRIVE DIRECT DOWNLOAD
# ============================================
CNN_MODEL_URL = "https://drive.google.com/uc?export=download&id=1e0KzPs66rGtvmu8VbSic836C9fqjF51X"

@st.cache_resource
def load_cnn_model():
    """Download CNN model from Google Drive using direct link"""
    model_path = 'ariidae_cnn_model.h5'
    classes_path = 'ariidae_cnn_classes.pkl'
    
    # Download model from Google Drive if not exists
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading CNN model (first time only)... This may take 3-5 minutes."):
            try:
                # Try direct download first
                import requests
                response = requests.get(CNN_MODEL_URL, stream=True)
                
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success("✅ CNN model downloaded!")
                else:
                    # Alternative: use gdown with confirm token
                    import gdown
                    url = f"https://drive.google.com/uc?id={CNN_MODEL_ID}"
                    gdown.download(url, model_path, quiet=False)
                    st.success("✅ CNN model downloaded via gdown!")
                    
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
