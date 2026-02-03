import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import XGBOOST_MODEL_PATH, BERT_MODEL_PATH

class ModelLoader:
    """Load and manage ML models"""
    
    def __init__(self):
        self.xgboost_model = None
        self.bert_model = None
        self.tokenizer = None
        
    def load_xgboost_model(self):
        """Load XGBoost model for questionnaire analysis"""
        try:
            if os.path.exists(XGBOOST_MODEL_PATH):
                with open(XGBOOST_MODEL_PATH, 'rb') as f:
                    self.xgboost_model = pickle.load(f)
                print(f"✅ XGBoost model loaded from {XGBOOST_MODEL_PATH}")
                return True
            else:
                print(f"❌ XGBoost model not found at {XGBOOST_MODEL_PATH}")
                return False
        except Exception as e:
            print(f"❌ Error loading XGBoost model: {str(e)}")
            return False
    
    def load_bert_model(self):
        """Load BERT model for text analysis"""
        try:
            if os.path.exists(BERT_MODEL_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
                self.bert_model.eval()
                print(f"✅ BERT model loaded from {BERT_MODEL_PATH}")
                return True
            else:
                print(f"❌ BERT model not found at {BERT_MODEL_PATH}")
                return False
        except Exception as e:
            print(f"❌ Error loading BERT model: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all models"""
        xgboost_loaded = self.load_xgboost_model()
        bert_loaded = self.load_bert_model()
        return xgboost_loaded and bert_loaded
    
    def get_models(self):
        """Return loaded models"""
        return {
            'xgboost': self.xgboost_model,
            'bert': self.bert_model,
            'tokenizer': self.tokenizer
        }

# Singleton instance
model_loader = ModelLoader()