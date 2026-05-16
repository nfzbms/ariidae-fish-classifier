# ============================================
# KUMPUL SEMUA FILE UNTUK DEPLOY
# ============================================
import os
import zipfile
from google.colab import files

print("="*50)
print("MENGUMPUL FILE UNTUK GITHUB & STREAMLIT")
print("="*50)

# Senarai semua file yang perlu diupload ke GitHub
files_to_collect = []

# Ariidae Mode Files (Hybrid CART-SVM)
ariidae_files = [
    'dt_selector.pkl',
    'feature_selector.pkl', 
    'scaler.pkl',
    'pca.pkl',
    'svm_model.pkl',
    'selected_cols.pkl',
    'classes.pkl'
]

# Freshwater Mode Files (CNN)
freshwater_files = [
    'freshwater_fish_cnn_model.h5',
    'freshwater_classes.pkl'
]

# Check Ariidae files
print("\n📁 Checking Ariidae model files:")
for file in ariidae_files:
    if os.path.exists(file):
        files_to_collect.append(file)
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} (not found - run Ariidae training first)")

# Check Freshwater files
print("\n📁 Checking Freshwater CNN files:")
for file in freshwater_files:
    if os.path.exists(file):
        files_to_collect.append(file)
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} (not found - run CNN training first)")

# Create zip file
if files_to_collect:
    with zipfile.ZipFile('complete_fish_system.zip', 'w') as zipf:
        for file in files_to_collect:
            zipf.write(file)
    
    print(f"\n✅ Zip file 'complete_fish_system.zip' siap!")
    print(f"📦 Ada {len(files_to_collect)} file dalam zip")
    
    # Download
    files.download('complete_fish_system.zip')
    print("\n📥 File downloaded to your computer!")
else:
    print("\n❌ No files found! Please run training first.")

print("\n" + "="*50)
print("READY FOR GITHUB UPLOAD!")
print("="*50)
