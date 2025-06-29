import logging
import os
from pathlib import Path
import subprocess
import tempfile
import requests
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, Dict, Any
from pydantic import BaseModel, Field

# Setup logging
def setup_logging():
    """Configure logging for the entire application"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Create logger for this module
    return logging.getLogger(__name__)

logger = setup_logging()

# Models
class Instruction(BaseModel):
    prompt: Optional[str] = None
    output_format: str
    retry_count: Optional[int] = 0

class Task(BaseModel):
    task_id: int = None
    description: str = None
    prerequisites: Optional[str | list[str]] = None
    assigned_bot: Optional[str] = None
    instructions: Optional[Instruction] = None
    status: str = None
    script: Optional[str] = None

class TaskOutput(BaseModel):
    output_data: Any
    success: bool
    error_messsage: Optional[str] = None
    validation_passed: Optional[bool] = None

class State(BaseModel):
    all_tasks: list[Task] = Field(default_factory=list)
    messages: Annotated[list[AnyMessage], add_messages]
    files: Optional[dict] = None
    queue: list[int] = Field(default_factory=list)
    current_task_index: Optional[int] = None
    task_outputs: dict[int, TaskOutput] = Field(default_factory=dict)
    refiner_feedback: Optional[str | list[str]] = None

# Agent Classes
class Create_Agent():
    def __init__(self, prompt):
        self.llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        )
        self.prompt = prompt
        logger.debug(f"Created agent with model llama3.1:8b")

class Create_Coding_Agent():
    def __init__(self, prompt):
        self.llm = ChatOllama(
            model="codeqwen:7b",
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        )
        self.system_prompt = prompt
        logger.debug(f"Created coding agent with model codeqwen:7b")

# Utility Functions
import logging
import json
import subprocess
import os

logger = logging.getLogger(__name__)

def python_executor(script: str, output_filename: str = "output.txt") -> str:
    """Execute a Python script in a persistent directory and return results from specified output file."""
    logger.info(f"Executing Python script, output file: {output_filename}")
    
    script_path = "script.py"
    output_path = output_filename

    try:
        # Write the script
        with open(script_path, 'w') as f:
            f.write(script)
        logger.debug(f"Script written to {script_path}")

        # Execute the script
        result = subprocess.run(
            ['python', '-u', script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        logger.debug(f"Script execution completed with return code: {result.returncode}")

        # Read output from file
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                output_data = f.read().strip()
                logger.info(f"Successfully read output data from {output_path}")
                logger.info(f"Output data type: {type(output_data)}")
                if output_data:
                    return output_data

        # If no output file or empty, check execution results
        if result.returncode == 0:
            message = f"Script executed successfully but produced no output in {output_filename}"
            logger.warning(message)
            return message
        else:
            error_msg = f"Script execution failed: {result.stderr.strip()}"
            logger.error(error_msg)
            return error_msg

    except Exception as e:
        error_msg = f"Execution failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def perform_basic_validation(output_data, expected_format):
    """Fallback validation logic when LLM fails to respond properly"""
    logger.info("Performing basic validation on output data")
    
    # Convert output_data to string for analysis
    output_str = str(output_data).lower()

    # Check for common error indicators
    error_indicators = ['error', 'exception', 'traceback', 'failed', 'none']
    has_error = any(indicator in output_str for indicator in error_indicators)

    # Check if output is empty or None
    is_empty = not output_data or output_str.strip() in ['', 'none', 'null']

    if has_error:
        logger.warning("Output contains error messages or exceptions")
        return {
            "validation_passed": False,
            "feedback": "Output contains error messages or exceptions"
        }

    if is_empty:
        logger.warning("Output is empty or None")
        return {
            "validation_passed": False,
            "feedback": "Output is empty or None"
        }

    # Check if JSON is expected and try parsing
    if "json" in expected_format.lower():
        try:
            if isinstance(output_data, str):
                json.loads(output_data)
            logger.info("JSON validation passed")
            return {
                "validation_passed": True,
                "feedback": None
            }
        except:
            logger.error("Expected JSON format but output is not valid JSON")
            return {
                "validation_passed": False,
                "feedback": "Expected JSON format but output is not valid JSON"
            }

    # Basic validation passed
    logger.info("Basic validation passed")
    return {
        "validation_passed": True,
        "feedback": None
    }
