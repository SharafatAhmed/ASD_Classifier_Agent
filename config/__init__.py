import os
import sys
from pathlib import Path

# Get the absolute path to the config directory
CONFIG_DIR = Path(__file__).parent.absolute()

# Define model paths relative to config directory
XGBOOST_MODEL_PATH = str(CONFIG_DIR / "xgboost_asd_model.pkl")
BERT_MODEL_PATH = str(CONFIG_DIR / "asd_classifier_model")

# Your API key
groq_api_key = "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"

# Verify files exist
def verify_model_paths():
    """Check if model files exist at specified paths."""
    print(f"🔍 Checking model paths...")
    print(f"Config directory: {CONFIG_DIR}")
    
    # Check XGBoost model
    if os.path.exists(XGBOOST_MODEL_PATH):
        print(f"✅ XGBoost model found: {XGBOOST_MODEL_PATH}")
    else:
        print(f"❌ XGBoost model NOT found: {XGBOOST_MODEL_PATH}")
        print(f"   Current files in config dir: {os.listdir(CONFIG_DIR)}")
    
    # Check BERT model directory
    if os.path.exists(BERT_MODEL_PATH):
        print(f"✅ BERT model directory found: {BERT_MODEL_PATH}")
        print(f"   Files in BERT dir: {os.listdir(BERT_MODEL_PATH)}")
    else:
        print(f"❌ BERT model directory NOT found: {BERT_MODEL_PATH}")

# Run verification on import
if __name__ == "__main__":
    verify_model_paths()