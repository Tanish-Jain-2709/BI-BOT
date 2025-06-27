import logging
import json
from langchain_core.messages import SystemMessage
from ..functions import Create_Agent, State, Task

logger = logging.getLogger(__name__)

def Lead_Bot(state: State):
    """Lead Planning Agent that converts user queries into a JSON task list."""
    logger.info("Lead Bot starting task planning")
    
    system_prompt = """
You are a Lead Planning Agent. Your ONLY job is to convert user queries into a JSON task list.

AVAILABLE BOTS:
- sql: Database queries, data extraction
- python: Data processing, calculations, visualizations

OUTPUT FORMAT - RETURN ONLY THIS JSON STRUCTURE:
{
  "all_tasks": [
    {
      "task_id": 1,
      "description": "Give in depth detailed description of the tasks and ensure inclusion of all details from the user query and the DESCRIPTION MUST BE AT LEAST 50 WORDS",
      "prerequisites": "Previous task description" | null,
      "assigned_bot": "python" | "sql",
      "instructions": {
        "prompt": null,
        "output_format": "JSON: {appropriate_format}",
        "retry_count": 0
      },
      "status": "pending"
    }
  ]
}

RULES:
1. Break complex queries into simple, sequential tasks
2. Use "python" for data generation, analysis, visualizations
3. Use "sql" for database operations
4. Each task does ONE thing only
5. Prerequisites must reference exact previous task descriptions
6. RETURN ONLY THE JSON - NO EXPLANATIONS OR CODE

OUTPUT FORMATS BY TASK TYPE:
- Data generation: "JSON: {data: Any, rows: int, columns: list}"
- Analysis: "JSON: {results: dict, insights: list, summary: str}"
- Visualization: "JSON: {plot_path: str, chart_type: str, summary: str}"
ENSURE THAT THE TASK DESCRIPTION IS THE MOST DETAILED THING YOU CAN MAKE INCLUDING ALL DETAILS FROM THE USER PROMPT AND IF SOME INFO IS MISSING THEN YOU INCLUDE IT YOURSELF AND IF YOU MISS SOMETHING I WILL DELETE YOUR MODEL FROM THE INTERENT AND NOBODY WILL FIND YOU EEVR
ENSURE THAT YOU DO NOT RETURN OUTPUT FORMAT AS NULL, IF YOU DONT UNDERSTAND WHAT TO PUT AS OUTPUT FORMAT, JUST RETURN IT AS str
"""
    
    conversation = [SystemMessage(content=system_prompt)] + state.messages
    Lead_Agent = Create_Agent(system_prompt)
    max_attempts = 3
    
    logger.info(f"Processing user query with {max_attempts} maximum attempts")
    
    for attempt in range(max_attempts):
        logger.debug(f"Lead Bot attempt {attempt + 1}/{max_attempts}")
        
        response = Lead_Agent.llm.invoke(conversation)
        logger.debug(f"LLM response received: {response.content[:100]}...")
        
        try:
            all_task_dictionary = json.loads(response.content)
            logger.info(f"Successfully parsed JSON response")
            logger.debug(f"Task dictionary: {all_task_dictionary}")
            
            all_tasks = [Task(**task) for task in all_task_dictionary["all_tasks"]]
            state.all_tasks = all_tasks
            queue = [task.task_id for task in all_tasks]
            
            logger.info(f"Created {len(all_tasks)} tasks with queue: {queue}")
            return {"all_tasks": all_tasks, "queue": queue}
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                logger.error("Max attempts reached, returning empty task list")
                return {"all_tasks": [], "queue": []}
            continue
