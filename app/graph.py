import logging
from langgraph.graph import StateGraph, START, END
from .functions import State
# Import each agent function from its individual file
from .agents.lead_bot import Lead_Bot
from .agents.conversational_bot import conversational_bot
from .agents.coordinator_bot import Coordinator_Bot
from .agents.analyst_bot import Analyst_Bot
from .agents.data_engineer_bot import Data_Engineer_Bot
from .agents.refiner_bot import Refiner_Bot

logger = logging.getLogger(__name__)

def router(state: State):
    """Route between analytical and conversational flows based on query content."""
    analytical_keywords = [
        "analyze", "data", "csv", "database", "visualization",
        "calculate", "trend", "report", "statistics", "chart"
    ]
    query = state.messages[0].content
    
    if state.files is not None or any(keyword in query.lower() for keyword in analytical_keywords):
        logger.info("Routing to analytical flow")
        return "analytical flow"
    else:
        logger.info("Routing to conversational flow")
        return "conversational flow"

def coordinator_router(state: State):
    """Route tasks to appropriate specialized bots."""
    current_queue = state.queue
    
    if not current_queue:
        logger.info("No tasks in queue, ending workflow")
        return "END"
        
    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)
    
    if current_task.assigned_bot == "python":
        logger.info(f"Routing task {current_task_id} to Analyst Bot")
        return "Analyst Bot"
    else:
        logger.info(f"Routing task {current_task_id} to Data Engineer Bot")
        return "Data Engineer Bot"

def refiner_router(state: State):
    """Route based on refiner validation results."""
    current_queue = state.queue
    if not current_queue:
        logger.info("Refiner routing to Coordinator Bot - no tasks in queue")
        return "Coordinator Bot"

    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)
    current_output = state.task_outputs.get(current_task_id)

    if not current_output or current_output.validation_passed is None:
        logger.info("Refiner routing to Coordinator Bot - no validation result")
        return "Coordinator Bot"

    if current_output.validation_passed:
        logger.info("Validation passed, routing to Coordinator Bot")
        return "Coordinator Bot"
    else:
        if current_task.status == "retry_needed":
            if current_task.assigned_bot == "python":
                logger.info("Validation failed, retrying with Analyst Bot")
                return "Analyst Bot"
            else:
                logger.info("Validation failed, retrying with Data Engineer Bot")
                return "Data Engineer Bot"
        else:
            logger.info("Max retries reached, routing to Coordinator Bot")
            return "Coordinator Bot"

# Build the graph
logger.info("Building LangGraph workflow")

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("Lead_Bot", Lead_Bot)
graph_builder.add_node("Conversational_Bot", conversational_bot)
graph_builder.add_node("Coordinator_Bot", Coordinator_Bot)
graph_builder.add_node("Analyst_Bot", Analyst_Bot)
graph_builder.add_node("Data_Engineer_Bot", Data_Engineer_Bot)
graph_builder.add_node("Refiner_Bot", Refiner_Bot)

# Add edges
graph_builder.add_conditional_edges(START, router, {
    "analytical flow": "Lead_Bot",
    "conversational flow": "Conversational_Bot"
})

graph_builder.add_edge("Lead_Bot", "Coordinator_Bot")
graph_builder.add_edge("Conversational_Bot", END)

graph_builder.add_conditional_edges("Coordinator_Bot", coordinator_router, {
    "Analyst Bot": "Analyst_Bot",
    "Data Engineer Bot": "Data_Engineer_Bot",
    "END": END
})

graph_builder.add_edge("Analyst_Bot", "Refiner_Bot")
graph_builder.add_edge("Data_Engineer_Bot", "Refiner_Bot")

graph_builder.add_conditional_edges("Refiner_Bot", refiner_router, {
    "Coordinator Bot": "Coordinator_Bot",
    "Analyst Bot": "Analyst_Bot",
    "Data Engineer Bot": "Data_Engineer_Bot"
})

graph = graph_builder.compile()

logger.info("LangGraph workflow compiled successfully")
