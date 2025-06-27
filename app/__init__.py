"""LangGraph Multi-Agent Service Package

This package provides an advanced multi-agent system for data analysis and conversational AI.
It includes:

CORE AGENTS:
- Lead Bot: Task decomposition and planning agent
- Coordinator Bot: Task coordination and prompt generation  
- Analyst Bot: Python data analysis and visualization
- Data Engineer Bot: SQL query generation and data extraction
- Refiner Bot: Output validation and error correction
- Conversational Bot: General purpose chat assistant

WORKFLOW COMPONENTS:
- Router: Intelligent request routing (analytical vs conversational flow)
- State Management: Pydantic models for task and state handling
- Error Handling: Retry mechanisms and validation
- Task Execution: Sequential processing with dependency management

FEATURES:
- Multi-agent collaboration with shared state
- Automatic task dependency resolution
- Error correction and retry mechanisms
- Real-time execution monitoring
- Support for both data analysis and conversational queries
"""

import logging
import os
from pathlib import Path

# Setup logging first - before importing other modules
def setup_package_logging():
    """Configure logging for the entire package"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(log_dir / "main.log")
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Initialize logging
_logger = setup_package_logging()

# Import core components after logging setup
from .functions import (
    State, 
    Task, 
    TaskOutput, 
    Instruction,
    Create_Agent,
    Create_Coding_Agent,
    python_executor,
    perform_basic_validation
)

from .graph import (
    graph,
    router,
    coordinator_router,
    refiner_router
)

# Import all agents - CORRECTED INDIVIDUAL IMPORTS
from .agents.lead_bot import Lead_Bot
from .agents.conversational_bot import conversational_bot
from .agents.coordinator_bot import Coordinator_Bot
from .agents.analyst_bot import Analyst_Bot
from .agents.data_engineer_bot import Data_Engineer_Bot
from .agents.refiner_bot import Refiner_Bot

# Package metadata
__version__ = "2.0.0"
__author__ = "BI BOT Team"
__description__ = "Advanced multi-agent system for data analysis and conversational AI"

# Public API exports
__all__ = [
    # Core graph and state
    "graph",
    "State", 
    "Task", 
    "TaskOutput",
    "Instruction",
    
    # Agent classes
    "Create_Agent",
    "Create_Coding_Agent",
    
    # Agent functions
    "Lead_Bot",
    "Coordinator_Bot", 
    "Analyst_Bot",
    "Data_Engineer_Bot",
    "Refiner_Bot",
    "conversational_bot",
    
    # Router functions
    "router",
    "coordinator_router", 
    "refiner_router",
    
    # Utility functions
    "python_executor",
    "perform_basic_validation"
]

# Package-level configuration
_logger.info(f"LangGraph Multi-Agent service package v{__version__} initialized")

# Validate environment
_ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
_logger.info(f"Configured Ollama URL: {_ollama_url}")

# Agent configuration summary
_logger.info("Available agents: Lead, Coordinator, Analyst, Data Engineer, Refiner, Conversational")
_logger.info("Supported models: llama3.1:8b, codeqwen:7b")
