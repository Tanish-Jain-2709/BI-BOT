import logging
import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from ..functions import Create_Agent, State, TaskOutput, python_executor

logger = logging.getLogger(__name__)

# Define static output directory
OUTPUT_DIR = os.path.join('/app', 'outputs')

def ensure_output_directory():
    """Ensure the outputs directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.debug(f"Output directory ensured at: {OUTPUT_DIR}")

def Analyst_Bot(state: State):
    """Analyst Bot that generates and executes Python scripts for data analysis."""
    logger.info("Analyst Bot starting analysis")
    
    # Ensure output directory exists
    ensure_output_directory()
    
    current_queue = state.queue
    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)

    if not current_task or current_task.assigned_bot != "python":
        logger.error(f"Invalid task for Analyst Bot: {current_task}")
        return state

    logger.info(f"Processing Python task {current_task_id}")

    # Get prerequisite task IDs
    prerequisite_ids = []
    if current_task.prerequisites:
        if isinstance(current_task.prerequisites, list):
            prerequisite_ids = [int(x) for x in current_task.prerequisites if str(x).isdigit()]
        elif isinstance(current_task.prerequisites, str):
            prerequisite_ids = [int(x.strip()) for x in current_task.prerequisites.split(',') if x.strip().isdigit()]

    # Build context from previous tasks with full paths
    previous_tasks_context = ""
    completed_tasks = []

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
                "output_file": output_path,  # Full path for script access
                "output_preview": output_preview
            })

    if completed_tasks:
        previous_tasks_context = "\nAvailable data files:\n"
        for task_info in completed_tasks:
            previous_tasks_context += f"- {task_info['output_file']}: {task_info['description']}\n"
            if task_info['output_preview']:
                previous_tasks_context += f"  {task_info['output_preview']}\n"

    # Include refiner feedback and previous script if this is a retry
    retry_context = ""
    if state.refiner_feedback:
        if hasattr(current_task, 'script') and current_task.script:
            retry_context += f"FAILED SCRIPT:\n{current_task.script}\n\nFix the above script to resolve the error.\n"
            logger.info(f"THE RETRY CONTEXT IS {retry_context}")

    # Define output filename and full path for current task
    output_filename = f"output_{current_task_id}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    system_prompt = f"""You are a Python data analyst. Write executable Python code to complete the task.

OUTPUT FILE: {output_path}


RULES:
1. Import all needed libraries
2. Read data from previous files when needed (use full paths provided above)
3. Save final result as JSON: with open('{output_path}', 'w') as f: f.write(json.dumps(result))
4. Return ONLY executable Python code - no explanations, no markdown, no code blocks
5. If the retry context shows a failed script then do not return the same script make changes to the script as indicated by the user prompt to make the code correct
7. Save the data directly in the output file, do not save it unde
6. RETURN ONLY THE SCRIPT AS RAW TEXT

Task: """

    user_query = state.all_tasks[current_task_id-1].instructions.prompt+f" {retry_context}{previous_tasks_context}"
    Script_Agent = Create_Agent(system_prompt)

    conversation = [SystemMessage(content=system_prompt)] + [HumanMessage(content=user_query)]

    logger.debug("Generating Python script")
    response = Script_Agent.llm.invoke(conversation)

    if hasattr(response, 'content'):
        generated_script = response.content.strip()
    else:
        generated_script = str(response).strip()

    # Clean up any potential code block markers
    if generated_script.startswith('```python'):
        generated_script = generated_script[9:]
    if generated_script.startswith('```'):
        generated_script = generated_script[3:]
    if generated_script.endswith('```'):
        generated_script = generated_script[:-3]
    
    generated_script = generated_script.strip()

    logger.info(f"Generated script preview: {generated_script}...")

    # Store script in task
    state.all_tasks[current_task_id-1].script = generated_script
    state.all_tasks[current_task_id-1].status = "script_generated"

    # Execute the script with full output path
    logger.info(f"Executing script for task {current_task_id}, saving to {output_path}")
    result = python_executor(generated_script, output_path)
    logger.debug(f"Execution result: {result}")

    # Check if execution was successful by trying to parse JSON
    try:
        result_data = json.loads(result)
        logger.info("Script execution successful - valid JSON result")

        task_output = TaskOutput(
            output_data=result_data,
            success=True,
            error_messsage=None,
            validation_passed=None
        )
        state.all_tasks[current_task_id-1].status = "completed"

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Script execution failed: {result}")

        task_output = TaskOutput(
            output_data=None,
            success=False,
            error_messsage=result,
            validation_passed=None
        )
        state.all_tasks[current_task_id-1].status = "failed"

    state.task_outputs[current_task_id] = task_output
    return state
