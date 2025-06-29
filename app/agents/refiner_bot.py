import logging
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from ..functions import Create_Agent, State, perform_basic_validation

logger = logging.getLogger(__name__)

# Define static output directory
OUTPUT_DIR = os.path.join('/app', 'outputs')

def Refiner_Bot(state: State):
    """Refines and validates output from agents, diagnoses errors and suggests fixes."""
    logger.info("Refiner Bot starting validation")
    
    current_queue = state.queue
    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)
    current_output = state.task_outputs.get(current_task_id)

    if not current_output:
        logger.error("No current output found for validation")
        return state
    
    # Get prerequisite task IDs
    prerequisite_ids = []
    if current_task.prerequisites:
        if isinstance(current_task.prerequisites, list):
            prerequisite_ids = [int(x) for x in current_task.prerequisites if str(x).isdigit()]
        elif isinstance(current_task.prerequisites, str):
            prerequisite_ids = [int(x.strip()) for x in current_task.prerequisites.split(',') if x.strip().isdigit()]

    completed_tasks = []
    expected_format = current_task.instructions.output_format
    output_data = current_output.output_data
    task_success = current_output.success
    error_message = current_output.error_messsage
    
    for task in state.all_tasks:
        if task.task_id in prerequisite_ids and task.status == "completed":
            task_output = state.task_outputs.get(task.task_id, {})
            output_filename = f"output_{task.task_id}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Get task output data preview
            output_preview = ""
            if task_output and hasattr(task_output, 'output_data') and task_output.output_data:
                output_str = str(task_output.output_data)
                first_100 = output_str[:100]
                last_100 = output_str[-100:] if len(output_str) > 100 else ''
                output_preview = f"Data preview - First 100: {first_100} ... Last 100: {last_100}"
            
            completed_tasks.append({
                "task_id": task.task_id,
                "description": task.description,
                "output_file": output_path,
                "output_preview": output_preview
            })

    # Initialize previous_tasks_context
    previous_tasks_context = ""
    if completed_tasks:
        previous_tasks_context = "\nAvailable data files:\n"
        for task_info in completed_tasks:
            previous_tasks_context += f"- {task_info['output_file']}: {task_info['description']}\n"
            if task_info['output_preview']:
                previous_tasks_context += f"  {task_info['output_preview']}\n"

    logger.info(f"Analyzing task {current_task_id} - Success: {task_success}")
    logger.debug(f"Output preview: {str(output_data)[:200]}...")

    system_prompt = """
You are a Python Code Refiner Bot. Your job is to analyze Analyst Bot output and provide corrections.

YOUR ANALYSIS JOBS:
1. If execution failed: FIND THE EXACT ERRORS IN THE SCRIPT AND THEN GIVE COMPREHENSIVE SOLUTION TO IT IN FEEDBACK
2. If execution succeeded THEN JUST PASS IT ONWARDS, VALIDATION PASSED
3. If output is correct: Validate and approve
4. GIVE DETAILED SOLUTIONS INCLUDING CODE SUGGESTIONS TO FIX THE PROBLEM

RESPONSE FORMAT (JSON only):
{{
    "validation_passed": true/false,
    "feedback": "DETAILED DESCRIPTION OF THE EXACT PROBLEM AND THE SOLUTION USING THE PREVIOUS TASK OUTPUTS AND THE SCRIPT"
}}

ALWAYS RETURN VALID JSON AS GIVEN IN THE FORMAT"""

    refiner_agent = Create_Agent(system_prompt)
    user_query = f"""
    CHECK THE FOLLOWING DETAILS AND PROVIDE RESONSE IN EXACTLY AS THE FORMAT, RETURN ONLY VALID JSON FORMAT NO CONTROL CHARACTERS, NO TEXT, ONLY THE JSON FORMAT :
  - The generated script is : {current_task.script}
  - Execution Success: {task_success}
  - Error Message: {error_message}
  PLEAASE I BEG YOU MY LIFE DEPENDS ON IT, ONLY RETURN VALID JSON DATA
"""

    try:
        logger.debug("Invoking refiner agent for validation")
        response = refiner_agent.llm.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(content=user_query)])
        logger.info(f"Refiner response: {repr(response.content)}")

        validation_result = json.loads(response.content.strip())
        logger.info("Refiner analysis completed successfully")

    except Exception as e:
        logger.warning(f"Refiner failed, using fallback validation: {e}")
        validation_result = perform_basic_validation(output_data, expected_format)

    logger.info(f"Validation result: {validation_result}")

    # Update task output
    current_output.validation_passed = validation_result["validation_passed"]

    if validation_result["validation_passed"]:
        # Move to next task
        state.queue = state.queue[1:]
        state.refiner_feedback = None
        logger.info(f"Task {current_task_id} validation PASSED - moving to next task")
    else:
        # Handle retry with specific feedback
        max_retries = 3
        current_retry_count = current_task.instructions.retry_count

        if current_retry_count < max_retries:
            state.all_tasks[current_task_id-1].instructions.retry_count += 1
            state.refiner_feedback = validation_result["feedback"]
            state.all_tasks[current_task_id-1].status = "retry_needed"
            logger.warning(f"Task {current_task_id} validation FAILED. Retry {current_retry_count + 1}/{max_retries}")
            logger.info(f"Feedback: {validation_result['feedback']}")
        else:
            state.all_tasks[current_task_id-1].status = "failed"
            state.queue = state.queue[1:]
            state.refiner_feedback = None
            logger.error(f"Task {current_task_id} FAILED. Max retries reached.")

    return state
