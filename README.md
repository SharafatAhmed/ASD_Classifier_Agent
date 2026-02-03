# 🧠 ASD Detection AI Agent

A multimodal AI agent framework for Autism Behavioral Trait Detection in children using LangGraph and Streamlit.

## Features
- **Q-CHAT-10 Questionnaire Analysis** using XGBoost model
- **Natural Language Behavior Analysis** using fine-tuned BERT
- **Supervisor Agent** with Groq LLM for intelligent routing
- **Streamlit Web Interface** for user-friendly interaction

## Project Structure
ASD_Agent/
├── app.py # Streamlit main app
├── config/ # Configuration management
├── models/ # ML model loading and management
├── agents/ # LangGraph agent definitions
├── utils/ # Utility functions
└── requirements.txt # Python dependencies