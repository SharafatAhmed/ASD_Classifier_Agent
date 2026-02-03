import streamlit as st
import pickle
import numpy as np
import torch
from config import XGBOOST_MODEL_PATH, BERT_MODEL_PATH, groq_api_key
from transformers import BertTokenizer, BertForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="ASD Detection AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model loading functions
@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model with error handling."""
    try:
        with open(XGBOOST_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"XGBoost model file not found at: {XGBOOST_MODEL_PATH}"
    except Exception as e:
        return None, f"Error loading XGBoost model: {str(e)}"

@st.cache_resource
def load_bert_model():
    """Load BERT model with error handling."""
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        return tokenizer, model, None
    except Exception as e:
        return None, None, f"Error loading BERT model: {str(e)}"

# Load models
xgboost_model, xgb_error = load_xgboost_model()
bert_tokenizer, bert_model, bert_error = load_bert_model()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "prediction_type": "none",  # "none", "greeting", "questionnaire", "text"
        "feature_values": [],
        "text_input": "",
        "awaiting_input": False
    }

# Conversation handling function
def handle_user_input(user_input):
    """Process user input and return appropriate response."""
    user_input = user_input.lower().strip()
    state = st.session_state.agent_state
    
    # Initial greeting
    if user_input in ["hello", "hi", "hey", "hlw", "hola", "greetings"]:
        greeting = (
            "👋 **Welcome to the Autism Behavioral Trait Detection System**\n\n"
            "I'm an AI agent specialized in detecting Autism Spectrum Disorder (ASD) traits in children "
            "using pretrained machine learning models.\n\n"
            "I can help you assess potential ASD traits through two methods:\n"
            "1. **Structured Questionnaire** (Q-CHAT-10 format)\n"
            "2. **Natural Language Description** of behavior\n\n"
            "Would you like to make a prediction? (yes/no)"
        )
        state["prediction_type"] = "greeting"
        return greeting
    
    # Handle identity queries
    identity_queries = ["who are you", "what are you", "what is your purpose", "what do you do"]
    if any(query in user_input for query in identity_queries):
        response = (
            "🤖 **About Me:**\n\n"
            "I'm a specialized AI agent for **Autism Spectrum Disorder (ASD) trait detection** in children.\n\n"
            "**My Capabilities:**\n"
            "• Screen for ASD traits using Q-CHAT-10 questionnaire\n"
            "• Analyze natural language descriptions of behavior\n"
            "• Provide preliminary assessments (not diagnoses)\n\n"
            "**Important:** I'm a screening tool, not a diagnostic tool.\n"
            "Always consult healthcare professionals for clinical assessments.\n\n"
            "Would you like to make a prediction? (yes/no)"
        )
        state["prediction_type"] = "identity"
        return response
    
    # Handle "yes" to prediction
    if user_input in ["yes", "y", "yeah", "sure", "ok", "okay", "yes please"]:
        response = (
            "✅ **Great! Let's get started.**\n\n"
            "I offer two assessment methods:\n\n"
            "📋 **1. Questionnaire Method**\n"
            "   • Based on Q-CHAT-10 screening tool\n"
            "   • 11 specific questions about behavior\n"
            "   • Answers: 0 (No) or 1 (Yes)\n\n"
            "📝 **2. Text Description Method**\n"
            "   • Describe behavior in your own words\n"
            "   • Example: 'My child has difficulty with eye contact'\n\n"
            "Which method would you prefer?\n"
            "Type 'questionnaire' or 'text'"
        )
        state["prediction_type"] = "method_choice"
        return response
    
    # Handle "no" response
    if user_input in ["no", "n", "nah", "not now", "no thanks"]:
        response = "👌 Okay. I'm here if you change your mind. Type 'yes' when you're ready to make a prediction."
        state["prediction_type"] = "none"
        return response
    
    # Handle questionnaire choice
    if user_input in ["questionnaire", "q", "1", "qchat", "question"]:
        state["prediction_type"] = "questionnaire"
        state["feature_values"] = []
        state["text_input"] = ""
        state["awaiting_input"] = True
        
        response = (
            "📋 **Q-CHAT-10 Questionnaire Selected**\n\n"
            "Please enter 11 comma-separated values (0 or 1):\n"
            "Here 0 means NO, 1 means Yes"
            "Format: 0,1,0,1,1,0,0,0,1,0,0\n\n"
            "**Answer The Following Questions:**\n"
            "1. A9 - Does your child use simple gestures? (e.g. wave goodbye)\n"
            "2. A6 - Does your child follow where you’re looking?\n"
            "3. A5 - Does your child pretend? (e.g. care for dolls, talk on a toy phone)\n"
            "4. A7 - If you or someone else in the family is visibly upset, does your child show signs of wantng to comfort them?\n"
            "5. A4 - Does your child point to share interest with you?(\n"
            "6. A1 - Does your child look at you when you call his/her name?\n"
            "7. A2 - Does your child make easy eye contact?\n"
            "8. A8 - Your child’s first words are normal?\n"
            "9. A3 - Does your child point to indicate that s/he wants something?\n"
            "10. A10 - Does your child stare at nothing with no apparent purpose?\n"
            "11. Sex (0=Female, 1=Male)\n\n"
            "Your answers:"
        )
        return response
    
    # Handle text analysis choice
    if user_input in ["text", "t", "2", "describe", "description"]:
        state["prediction_type"] = "text"
        state["text_input"] = ""
        state["feature_values"] = []
        state["awaiting_input"] = True
        
        response = (
            "📝 **Text Description Method Selected**\n\n"
            "Please describe the child's behavior in natural language.\n"
            "Be as detailed as possible about:\n"
            "• Social interactions\n"
            "• Communication patterns\n"
            "• Repetitive behaviors\n"
            "• Response to surroundings\n\n"
            "**Examples:**\n"
            "• 'My 3-year-old rarely makes eye contact and doesn't respond to his name.'\n"
            "• 'She has delayed speech and lines up toys repetitively.'\n"
            "• 'He doesn't point at things and seems uninterested in other children.'\n\n"
            "Your description:"
        )
        return response
    
    # Handle questionnaire answers
    if state["prediction_type"] == "questionnaire" and state["awaiting_input"]:
        # Clean input
        cleaned_input = user_input.replace(" ", "")
        
        if "," in cleaned_input:
            try:
                values = [int(x) for x in cleaned_input.split(",")]
                
                if len(values) != 11:
                    return f"❌ Need exactly 11 values, but got {len(values)}. Please enter 11 comma-separated numbers."
                
                # Validate 0 or 1
                invalid_values = [i for i, val in enumerate(values) if val not in [0, 1]]
                if invalid_values:
                    return f"❌ Values at positions {[pos+1 for pos in invalid_values]} are not 0 or 1."
                
                # Store values and make prediction
                state["feature_values"] = values
                state["awaiting_input"] = False
                
                # Make XGBoost prediction
                if xgboost_model:
                    try:
                        prediction = xgboost_model.predict([values])[0]
                        proba = xgboost_model.predict_proba([values])[0]
                        
                        result_msg = (
                            f"✅ **Analysis Complete**\n\n"
                            f"**Prediction:** {'ASD traits detected' if prediction == 1 else 'No ASD traits detected'}\n"
                            f"**Confidence:** {max(proba)*100:.1f}%\n"
                            f"**Scores:** ASD={proba[1]*100:.1f}%, Non-ASD={proba[0]*100:.1f}%\n\n"
                            f"**Disclaimer:** This is a screening tool only. Please consult a healthcare professional for formal assessment."
                        )
                    except Exception as e:
                        result_msg = f"❌ Prediction error: {str(e)}"
                else:
                    result_msg = "❌ XGBoost model not loaded. Cannot make prediction."
                
                state["prediction_type"] = "none"
                return result_msg
                
            except ValueError:
                return "❌ Invalid format. Please enter numbers only (0 or 1) separated by commas.\nExample: 0,1,0,1,1,0,0,0,1,0,0"
        else:
            return "❌ Please enter 11 comma-separated numbers (0 or 1).\nExample: 0,1,0,1,1,0,0,0,1,0,0"
    
    # Handle text input analysis
    if state["prediction_type"] == "text" and state["awaiting_input"]:
        if len(user_input.split()) >= 3:
            state["text_input"] = user_input
            state["awaiting_input"] = False
            
            # Make BERT prediction
            if bert_model and bert_tokenizer:
                try:
                    inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                        prediction = torch.argmax(outputs.logits, dim=1).item()
                        proba = torch.softmax(outputs.logits, dim=1)[0]
                    
                    result_msg = (
                        f"✅ **Analysis Complete**\n\n"
                        f"**Description:** \"{user_input[:100]}{'...' if len(user_input) > 100 else ''}\"\n\n"
                        f"**Prediction:** {'ASD traits detected' if prediction == 1 else 'No ASD traits detected'}\n"
                        f"**Confidence:** {max(proba)*100:.1f}%\n\n"
                        f"**Disclaimer:** This is a screening tool only. Please consult a healthcare professional for formal assessment."
                    )
                except Exception as e:
                    result_msg = f"❌ Text analysis error: {str(e)}"
            else:
                result_msg = "❌ BERT model not loaded. Cannot analyze text."
            
            state["prediction_type"] = "none"
            return result_msg
        else:
            return "❌ Please provide a more detailed description (at least 3 words)."
    
    # Handle exit/quit
    if user_input in ["exit", "quit", "bye", "goodbye", "stop"]:
        state["prediction_type"] = "none"
        return "👋 Thank you for using the ASD Detection System. Goodbye!"
    
    # Default: handle irrelevant queries
    response = (
        "❓ **Topic Alert**\n\n"
        "I specialize only in **Autism Spectrum Disorder trait detection** in children.\n\n"
        "I cannot answer queries on other topics.\n\n"
        "If you'd like to assess ASD traits, I can help with:\n"
        "• Q-CHAT-10 questionnaire screening\n"
        "• Text description analysis\n\n"
        "Would you like to make a prediction? (yes/no)"
    )
    state["prediction_type"] = "basic"
    return response

def main():
    # Sidebar
    with st.sidebar:
        st.title("🧠 ASD Detection System")
        st.markdown("---")
        
        # Model status
        st.subheader("Model Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if xgboost_model:
                st.success("✅ XGBoost")
            else:
                st.error(f"❌ XGBoost")
                if xgb_error:
                    st.caption(f"Error: {xgb_error[:50]}...")
        
        with col2:
            if bert_model:
                st.success("✅ BERT")
            else:
                st.error(f"❌ BERT")
                if bert_error:
                    st.caption(f"Error: {bert_error[:50]}...")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Questionnaire", use_container_width=True):
                response = handle_user_input("questionnaire")
                st.session_state.messages.append(("assistant", response))
                st.rerun()
        
        with col2:
            if st.button("📝 Text Analysis", use_container_width=True):
                response = handle_user_input("text")
                st.session_state.messages.append(("assistant", response))
                st.rerun()
        
        if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_state = {
                "prediction_type": "none",
                "feature_values": [],
                "text_input": "",
                "awaiting_input": False
            }
            st.rerun()
        
        st.markdown("---")
        
        # Information
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Start** with 'hi' or ask about the system
            2. **Choose** questionnaire or text method
            3. **Provide** input as requested
            4. **Get** prediction with confidence score
            """)
        
        with st.expander("📋 Q-CHAT-10 Features"):
            st.markdown("""
            **11 Questions (answer in 0=No or 1=Yes):**
            "1. A9 - Does your child use simple gestures? (e.g. wave goodbye)\n"
            "2. A6 - Does your child follow where you’re looking?\n"
            "3. A5 - Does your child pretend? (e.g. care for dolls, talk on a toy phone)\n"
            "4. A7 - If you or someone else in the family is visibly upset, does your child show signs of wantng to comfort them?\n"
            "5. A4 - Does your child point to share interest with you?(\n"
            "6. A1 - Does your child look at you when you call his/her name?\n"
            "7. A2 - Does your child make easy eye contact?\n"
            "8. A8 - Your child’s first words are normal?\n"
            "9. A3 - Does your child point to indicate that s/he wants something?\n"
            "10. A10 - Does your child stare at nothing with no apparent purpose?\n"
             11. **Sex**: 0=Female, 1=Male
            """)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This is a screening tool only, not a diagnostic tool.
        Always consult healthcare professionals for clinical assessments.
        """)
    
    # Main content
    st.title("Autism Behavioral Trait Detection AI Agent")
    st.markdown("---")
    
    # Chat container
    chat_container = st.container(height=500)
    
    # Display chat messages
    with chat_container:
        for role, message in st.session_state.messages:
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append(("user", prompt))
        
        # Get agent response
        with st.spinner("Thinking..."):
            response = handle_user_input(prompt)
        
        # Add agent response
        st.session_state.messages.append(("assistant", response))
        
        # Rerun to update display
        st.rerun()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("Built with ❤️ using Streamlit | For screening purposes only")

if __name__ == "__main__":
    main()
