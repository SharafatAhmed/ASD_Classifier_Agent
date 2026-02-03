import torch
from models.model_loader import model_loader

class TextAnalysisAgent:
    """Agent for processing text descriptions"""
    
    def __init__(self):
        self.model = model_loader.bert_model
        self.tokenizer = model_loader.tokenizer
    
    def predict(self, text: str) -> str:
        """Make prediction from text description"""
        try:
            if not self.model or not self.tokenizer:
                return "❌ Error: BERT model not loaded."
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(predictions, dim=1).item()
            
            label = "ASD" if prediction == 1 else "Non-ASD"
            
            response = (
                f"📝 **TEXT ANALYSIS PREDICTION RESULTS**\n"
                f"═"*40 + "\n"
                f"🔍 **Assessment**: {label}\n"
                f"🔢 **Binary Output**: {prediction}\n"
                f"   - 0 = Non-ASD (No Autism Spectrum Disorder traits)\n"
                f"   - 1 = ASD (Autism Spectrum Disorder traits detected)\n\n"
                f"📄 **Analyzed Text**:\n"
                f"   \"{text[:150]}{'...' if len(text) > 150 else ''}\"\n\n"
                f"⚠️ **IMPORTANT**: This is a screening tool only, not a medical diagnosis."
            )
            
            return response
            
        except Exception as e:
            return f"❌ Error in prediction: {str(e)}"