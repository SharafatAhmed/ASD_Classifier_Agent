import numpy as np
from models.model_loader import model_loader

class QuestionnaireAgent:
    """Agent for processing Q-CHAT-10 questionnaire predictions"""
    
    def __init__(self):
        self.model = model_loader.xgboost_model
    
    def predict(self, features: list) -> str:
        """Make prediction from questionnaire features"""
        try:
            if not self.model:
                return "❌ Error: XGBoost model not loaded."
            
            features_array = np.array([features])
            prediction = self.model.predict(features_array)[0]
            
            label = "ASD" if prediction == 1 else "Non-ASD"
            
            response = (
                f"📋 **QUESTIONNAIRE PREDICTION RESULTS**\n"
                f"═"*40 + "\n"
                f"🔍 **Assessment**: {label}\n"
                f"🔢 **Binary Output**: {prediction}\n"
                f"   - 0 = Non-ASD (No Autism Spectrum Disorder traits)\n"
                f"   - 1 = ASD (Autism Spectrum Disorder traits detected)\n\n"
                f"📝 **Input Features**: {features}\n\n"
                f"⚠️ **IMPORTANT**: This is a screening tool only, not a medical diagnosis."
            )
            
            return response
            
        except Exception as e:
            return f"❌ Error in prediction: {str(e)}"