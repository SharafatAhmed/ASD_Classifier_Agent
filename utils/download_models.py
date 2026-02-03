import gdown
import zipfile
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure the config module is correctly imported
try:
    from config import MODEL_DIR, XGBOOST_MODEL_PATH, BERT_MODEL_PATH
except ImportError as e:
    print(f"❌ Import error: {str(e)}. Please check if the config module exists.")
    sys.exit(1)

def download_models():
    """Download models from Google Drive"""
    print("📥 Downloading models from Google Drive...")
    
    # Create directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Google Drive folder URL
    url = "https://drive.google.com/drive/folders/1zNUQbmaNz-UhS0Mkxi8XcuclOd4PuxFJ"
    
    try:
        # Download folder
        gdown.download_folder(url, output=MODEL_DIR, quiet=False)
        print("✅ Download completed!")
        
        # Check for zip file and extract
        zip_path = os.path.join(MODEL_DIR, "asd_classifier_model.zip")
        if os.path.exists(zip_path):
            print("📦 Extracting BERT model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            print("✅ Extraction completed!")
            
            # Clean up zip file
            os.remove(zip_path)
        
        # Check what was downloaded
        print("\n📁 Downloaded files:")
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"❌ Error downloading models: {str(e)}")

if __name__ == "__main__":
    download_models()