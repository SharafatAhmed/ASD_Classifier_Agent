from langgraph.graph import StateGraph, START, END
from agents.supervisor import State, SupervisorAgent
from agents.questionnaire import QuestionnaireAgent
from agents.text_analysis import TextAnalysisAgent
from typing import Literal

class WorkflowBuilder:
    """Build LangGraph workflow for the agent system"""
    
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.questionnaire_agent = QuestionnaireAgent()
        self.text_agent = TextAnalysisAgent()
    
    def supervisor_node(self, state: State) -> State:
        """Supervisor agent node"""
        return self.supervisor.process(state)
    
    def questionnaire_node(self, state: State) -> State:
        """Questionnaire agent node"""
        if "feature_values" in state and state["feature_values"]:
            result = self.questionnaire_agent.predict(state["feature_values"])
            state["prediction"] = result
            state["messages"].append(result)
        return state
    
    def text_analysis_node(self, state: State) -> State:
        """Text analysis agent node"""
        if "text_input" in state and state["text_input"]:
            result = self.text_agent.predict(state["text_input"])
            state["prediction"] = result
            state["messages"].append(result)
        return state
    
    def routing_logic(self, state: State) -> Literal["questionnaire", "text", "end"]:
        """Routing logic based on state"""
        prediction_type = state.get("prediction_type", "none")
        
        if prediction_type == "questionnaire" and "feature_values" in state and state["feature_values"]:
            return "questionnaire"
        elif prediction_type == "text" and "text_input" in state and state["text_input"]:
            return "text"
        else:
            return "end"
    
    def build(self):
        """Build and compile the workflow graph"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("questionnaire", self.questionnaire_node)
        workflow.add_node("text_analysis", self.text_analysis_node)
        
        # Add edges
        workflow.add_edge(START, "supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self.routing_logic,
            {
                "questionnaire": "questionnaire",
                "text": "text_analysis",
                "end": END
            }
        )
        
        workflow.add_edge("questionnaire", END)
        workflow.add_edge("text_analysis", END)
        
        # Compile
        return workflow.compile()

# Global workflow instance
workflow_builder = WorkflowBuilder()
agents_workflow = workflow_builder.build()