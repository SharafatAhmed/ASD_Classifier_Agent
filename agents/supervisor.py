from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import groq_api_key

class State(TypedDict):
    messages: Annotated[list, "add_messages"]
    prediction: str
    prediction_type: Literal["questionnaire", "text", "none", "basic"]
    feature_values: list
    text_input: str

class SupervisorAgent:
    """Supervisor agent for handling user queries and routing"""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=groq_api_key,
            temperature=0.1
        )
    
    def process(self, state: State) -> State:
        """Process user input and determine routing"""
        
        # Initialize if no messages
        if not state.get("messages"):
            greeting = (
                "👋 **AUTISM BEHAVIORAL TRAIT DETECTION SYSTEM**\n"
                "═"*50 + "\n"
                "Welcome! I'm an AI agent specialized in detecting Autism Spectrum \n"
                "Disorder (ASD) traits in children using trained machine learning models.\n\n"
                "✨ **Features**:\n"
                "• Q-CHAT-10 Questionnaire Screening (11 questions)\n"
                "• Natural Language Behavior Analysis\n\n"
                "📊 **Output Format**:\n"
                "• 0 = Non-ASD (No ASD traits detected)\n"
                "• 1 = ASD (ASD traits detected)\n\n"
                "Would you like to make a prediction? (yes/no)"
            )
            return {
                "messages": [greeting],
                "prediction": "",
                "prediction_type": "basic",
                "feature_values": [],
                "text_input": ""
            }
        
        last_message = state["messages"][-1]
        user_input = str(last_message).lower().strip()
        
        # Handle basic greetings
        if user_input in ["hi", "hello", "hey"]:
            response = "👋 Hello! Ready to assess ASD traits? Type 'yes' to begin."
            state["messages"].append(response)
            state["prediction_type"] = "basic"
            return state
        
        # Handle identity queries
        if any(q in user_input for q in ["who are you", "what are you", "what do you do"]):
            response = (
                "🤖 **I am an ASD Screening Assistant**\n\n"
                "• **Purpose**: Screen for Autism Spectrum Disorder traits in children\n"
                "• **Methods**: Q-CHAT-10 questionnaire & text behavior analysis\n"
                "• **Models**: XGBoost classifier (questionnaire) & BERT (text)\n"
                "• **Output**: Binary prediction (0=Non-ASD, 1=ASD)\n\n"
                "Type 'yes' to start a screening assessment."
            )
            state["messages"].append(response)
            state["prediction_type"] = "basic"
            return state
        
        # Handle "yes" to start prediction
        if user_input in ["yes", "y", "yeah", "sure"]:
            response = (
                "✅ **Great! Choose your assessment method:**\n\n"
                "📋 **1. Questionnaire Method**\n"
                "   • Based on Q-CHAT-10 screening tool\n"
                "   • 11 specific behavioral questions\n"
                "   • Answers: 0 (No) or 1 (Yes)\n\n"
                "📝 **2. Text Description Method**\n"
                "   • Describe behavior in natural language\n"
                "   • Example: 'My child rarely makes eye contact'\n\n"
                "Type 'questionnaire' or 'text' to continue:"
            )
            state["messages"].append(response)
            state["prediction_type"] = "none"
            return state
        
        # Handle "no" response
        if user_input in ["no", "n", "nah", "not now"]:
            response = "👌 Okay. I'm here if you change your mind. Type 'yes' when ready."
            state["messages"].append(response)
            state["prediction_type"] = "none"
            return state
        
        # Route to questionnaire
        if user_input in ["questionnaire", "q", "1", "qchat"]:
            state["prediction_type"] = "questionnaire"
            state["feature_values"] = []
            state["text_input"] = ""
            
            questions = [
                ("A9", "Does your child use simple gestures? (e.g., wave goodbye)"),
                ("A6", "Does your child follow where you're looking?"),
                ("A5", "Does your child pretend? (e.g., care for dolls, talk on toy phone)"),
                ("A7", "If someone is visibly upset, does your child show signs of wanting to comfort them?"),
                ("A4", "Does your child point to share interest with you?"),
                ("A1", "Does your child look at you when you call his/her name?"),
                ("A2", "How easy is it to get eye contact with your child?"),
                ("A8", "Would you describe your child's first words as normal?"),
                ("A3", "Does your child point to indicate they want something?"),
                ("A10", "Does your child stare at nothing with no apparent purpose?"),
                ("Sex", "Child's biological sex (0=Female, 1=Male)")
            ]
            
            response = "📋 **Q-CHAT-10 Questionnaire Selected**\n\n"
            response += "Please enter 11 comma-separated values (0 or 1):\n"
            response += "Format: 0,1,0,1,1,0,0,0,1,0,0\n\n"
            response += "Your answers:"
            
            state["messages"].append(response)
            return state
        
        # Route to text analysis
        if user_input in ["text", "t", "2", "describe", "description"]:
            state["prediction_type"] = "text"
            state["text_input"] = ""
            state["feature_values"] = []
            
            response = (
                "📝 **Text Description Method Selected**\n\n"
                "Please describe the child's behavior in natural language.\n"
                "Be as detailed as possible about social interactions, communication,\n"
                "and any repetitive behaviors.\n\n"
                "Your description:"
            )
            
            state["messages"].append(response)
            return state
        
        # Handle questionnaire answers
        if state.get("prediction_type") == "questionnaire":
            cleaned_input = user_input.replace(" ", "")
            
            if "," in cleaned_input:
                try:
                    values = [int(x) for x in cleaned_input.split(",")]
                    
                    if len(values) != 11:
                        error_msg = f"❌ Need exactly 11 values, got {len(values)}. Please try again."
                        state["messages"].append(error_msg)
                        return state
                    
                    for val in values:
                        if val not in [0, 1]:
                            error_msg = f"❌ Values must be 0 or 1. Please try again."
                            state["messages"].append(error_msg)
                            return state
                    
                    state["feature_values"] = values
                    state["messages"].append(f"✅ Answers received: {values}")
                    return state
                    
                except ValueError:
                    error_msg = "❌ Invalid format. Use comma-separated numbers (0 or 1)."
                    state["messages"].append(error_msg)
                    return state
            else:
                error_msg = "❌ Please enter 11 comma-separated numbers (0 or 1)."
                state["messages"].append(error_msg)
                return state
        
        # Handle text input
        if state.get("prediction_type") == "text":
            if len(user_input.split()) >= 3:
                state["text_input"] = user_input
                state["messages"].append("✅ Description received.")
                return state
            else:
                error_msg = "❌ Please provide a more detailed description (at least 3 words)."
                state["messages"].append(error_msg)
                return state
        
        # Handle exit
        if user_input in ["exit", "quit", "bye", "goodbye"]:
            response = "👋 Thank you for using the ASD Detection System. Stay well!"
            state["messages"].append(response)
            return state
        
        # Default: handle irrelevant queries
        response = (
            "❓ **Topic Alert**\n\n"
            "I specialize only in Autism Spectrum Disorder trait detection.\n\n"
            "If you'd like to assess ASD traits, type 'yes' to begin."
        )
        state["messages"].append(response)
        state["prediction_type"] = "basic"
        return state
    