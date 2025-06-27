import logging
import os
from langchain_core.messages import SystemMessage
from ..functions import Create_Agent, State

logger = logging.getLogger(__name__)

# Define static output directory
OUTPUT_DIR = os.path.join('/app', 'outputs')

def Coordinator_Bot(state: State):
    """Coordinator Agent that creates execution prompts for assigned bots."""
    logger.info("Coordinator Bot starting task coordination")
    
    current_queue = state.queue
    
    # Check if queue is empty (all tasks completed)
    if not current_queue:
        logger.info("All tasks completed - ending workflow")
        return state
    
    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)

    if not current_task:
        logger.error(f"Task with ID {current_task_id} not found")
        return state

    logger.info(f"Coordinating task {current_task_id}: {current_task.description[:50]}...")

    # Get prerequisite task IDs
    prerequisite_ids = []
    if current_task.prerequisites:
        if isinstance(current_task.prerequisites, list):
            prerequisite_ids = [int(x) for x in current_task.prerequisites if str(x).isdigit()]
        elif isinstance(current_task.prerequisites, str):
            prerequisite_ids = [int(x.strip()) for x in current_task.prerequisites.split(',') if x.strip().isdigit()]

    # Collect completed prerequisite tasks with full paths
    completed_tasks = []
    for task in state.all_tasks:
        if task.task_id in prerequisite_ids and task.status == "completed":
            output_filename = f"output_{task.task_id}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            completed_tasks.append({
                "task_id": task.task_id,
                "description": task.description,
                "output_file": output_path,  # Full path
                "output_format": task.instructions.output_format
            })

    logger.debug(f"Found {len(completed_tasks)} completed prerequisite tasks")

    # Build previous tasks context with full paths
    previous_tasks_context = ""
    if completed_tasks:
        previous_tasks_context = "PREVIOUS COMPLETED TASKS:\n"
        for task_info in completed_tasks:
            previous_tasks_context += f"Task {task_info['task_id']}: {task_info['description']}\n"
            previous_tasks_context += f"Data available at: {task_info['output_file']}\n\n"

    file_info = f"File paths: {state.files}"
    outputs_summary = []
    for task_id, task_output in state.task_outputs.items():
        if task_id in prerequisite_ids:
            output_data = task_output.output_data
            output_str = str(output_data)
            first_100 = output_str[:100]
            last_100 = output_str[-100:] if len(output_str) > 100 else ''
            outputs_summary.append(f"Task {task_id}: First 100 chars: {first_100} ... Last 100 chars: {last_100}")

    output_info = f"Task outputs summary: {outputs_summary}"

    # Add refiner feedback if available
    refiner_feedback = ""
    if state.refiner_feedback:
        refiner_feedback = f"\nREFINER FEEDBACK: {state.refiner_feedback}\nAddress these issues."
        logger.info("Including refiner feedback in coordination")

    # Define current task output path
    current_output_path = os.path.join(OUTPUT_DIR, f"output_{current_task_id}.txt")
    script = current_task.script

    system_prompt = f"""You are the Coordinator Agent. Create a concise execution prompt for the assigned bot.




PROMPT RULES:
- Use FULL PATHS for all file references (as shown above)
- The output should be saved in a file {current_output_path}
- KEEP PROMPT UNDER 100-200 WORDS
- IF INFORMATION FROM THE PREVIOUS TASKS IS REQUIRED THEN PROVIDE EXACT INFORMATION ON HOW TO ACCESS THEM, E.G. IF A DATASET IS STORED IN THE TXT FILE OUTPUT_1.TXT UNDER THE KEY 'data' THEN MENTION THAT IN THE PROMPT
- ANALYSE THE PRPEREQUISITES AND THEN MAKE THE PROMPT AND INCLUDE TELLING THE MODEL ABOUT ACCESSING THOSE PRE REQUISITES, CLEARLY STATE THE LOCATION OF THE OUTPUT FILES OF PREVIOUS TASKS AND HOW TO ACCESS THEM
- THE PROMPT SHOULD INCLUDE THE EXACT OUTPUT FORMAT IN WHICH THE OUTPUT SHOULD BE E.G. A DATASET CAN BE IF ITS A DICTIONARY THEN WHAT KEYS SHOULD BE THERE WHAT SHOULD BE ITS VALUES ETC.
- ALL FILES ARE STORED IN: {OUTPUT_DIR}
- DO NOT REPEAT YOURSELF IN THE PROMPT


Generate ONLY the working prompt text:"""

    task_info = f"Task ID: {current_task.task_id}\nDescription: {current_task.description}\nAssigned Bot: {current_task.assigned_bot}\nPrerequisites: {current_task.prerequisites}\nRetry count: {current_task.instructions.retry_count}\n Refiner Feedback : {refiner_feedback}\nPrevious tasks context : {previous_tasks_context}"

    conversation = [SystemMessage(content=system_prompt)] + [task_info] + [file_info] + [output_info]
    coordinator_bot = Create_Agent(system_prompt)
    
    logger.debug("Generating coordination prompt")
    response = coordinator_bot.llm.invoke(conversation)

    logger.info(f"Generated coordination prompt: {response.content}...")
    state.all_tasks[current_task_id-1].instructions.prompt = response.content.strip()
    state.current_task_index = current_task_id
    state.refiner_feedback = None
    
    return state
