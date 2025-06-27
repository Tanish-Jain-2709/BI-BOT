import logging
from langchain_core.messages import SystemMessage
from ..functions import Create_Agent, State

logger = logging.getLogger(__name__)

def conversational_bot(state: State):
    """Friendly assistant for general conversational queries."""
    logger.info("Conversational bot processing query")
    
    system_prompt = "You are a helpful friendly assistant and you will help everyone in solving their queries and do everything your power to give them all the information that they need, if you do not know you will politely decline"
    
    convo_bot = Create_Agent(system_prompt)
    conversation = [SystemMessage(content=system_prompt)] + state.messages
    
    logger.debug("Invoking conversational model")
    response = convo_bot.llm.invoke(conversation)
    
    logger.info("Conversational response generated successfully")
    return {"messages": response}
