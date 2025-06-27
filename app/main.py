import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uvicorn
import time
from pathlib import Path

# Import your graph components - updated to work with new structure
from app import graph, State, Task, Instruction, TaskOutput

# Configure logging to write to logs/main.log
def setup_main_logging():
    """Configure logging for main.py to write to logs/main.log"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "main.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_main_logging()

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "llama3.1:8b"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None

class ChatChoice(BaseModel):
    message: ChatMessage
    finish_reason: str
    index: int = 0

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ollama_url: str
    gpu_available: bool
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting LangGraph Multi-Agent service...")
    
    # Startup tasks
    await startup_tasks()
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down LangGraph service...")

async def startup_tasks():
    """Initialize connections and test services"""
    try:
        # Test Ollama connection
        from langchain_ollama import ChatOllama
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        test_llm = ChatOllama(
            model="llama3.1:8b",
            base_url=ollama_url,
            temperature=0
        )
        
        # Simple connectivity test
        test_message = HumanMessage(content="test connection")
        await asyncio.to_thread(test_llm.invoke, [test_message])
        logger.info("Ollama connection successful")
        
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {e}")
        logger.info("Service will still start, but may fail on first request")

# Create FastAPI app
app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="Advanced multi-agent system with Lead Bot, Coordinator, Analyst, Data Engineer, Refiner, and Conversational agents",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_messages_to_langchain(messages: List[ChatMessage]) -> List[Any]:
    """Convert OpenWebUI messages to LangChain format"""
    langchain_messages = []
    
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
    
    return langchain_messages

def format_task_execution_response(result: Dict[str, Any]) -> str:
    """Format complete task execution results into readable response"""
    
    # Check if we have completed tasks with outputs
    if "task_outputs" in result and result["task_outputs"]:
        response = "**Analysis Complete!** Here are your results:\n\n"
        
        # Sort task outputs by task_id
        sorted_outputs = sorted(result["task_outputs"].items(), key=lambda x: int(x[0]))
        
        for task_id, task_output in sorted_outputs:
            task_id = int(task_id)
            
            # Find corresponding task
            task = None
            if "all_tasks" in result:
                for t in result["all_tasks"]:
                    if isinstance(t, dict) and t.get("task_id") == task_id:
                        task = t
                        break
                    elif hasattr(t, "task_id") and t.task_id == task_id:
                        task = t
                        break
            
            if task:
                task_desc = task.get("description", f"Task {task_id}") if isinstance(task, dict) else task.description
                response += f"**Task {task_id}:** {task_desc[:100]}{'...' if len(task_desc) > 100 else ''}\n"
                
                # Extract task output data
                if isinstance(task_output, dict):
                    output_data = task_output.get("output_data")
                    success = task_output.get("success", False)
                    error_msg = task_output.get("error_messsage")
                elif hasattr(task_output, "output_data"):
                    output_data = task_output.output_data
                    success = task_output.success
                    error_msg = task_output.error_messsage
                else:
                    output_data = str(task_output)
                    success = True
                    error_msg = None
                
                if success and output_data:
                    # Format output data
                    if isinstance(output_data, dict):
                        if "insights" in output_data:
                            response += f"   **Key Insights:**\n"
                            insights = output_data["insights"]
                            if isinstance(insights, list):
                                for insight in insights[:3]:  # Show top 3 insights
                                    response += f"{insight}\n"
                            else:
                                response += f"{insights}\n"
                        
                        if "summary" in output_data:
                            response += f"   **Summary:** {output_data['summary']}\n"
                        
                        if "results" in output_data:
                            response += f"   **Results:** {str(output_data['results'])[:200]}{'...' if len(str(output_data['results'])) > 200 else ''}\n"
                    else:
                        # Handle string output
                        output_str = str(output_data)
                        if len(output_str) > 300:
                            response += f"   **Output:** {output_str[:300]}...\n"
                        else:
                            response += f"   **Output:** {output_str}\n"
                else:
                    response += f"   **Error:** {error_msg or 'Task failed'}\n"
                
                response += "\n"
        
        response += "**Execution Summary:**\n"
        total_tasks = len(result.get("all_tasks", []))
        completed_tasks = len([t for t in result.get("all_tasks", []) if 
                             (isinstance(t, dict) and t.get("status") == "completed") or 
                             (hasattr(t, "status") and t.status == "completed")])
        
        response += f"   Total Tasks: {total_tasks}\n"
        response += f"   Completed: {completed_tasks}\n"
        response += f"   Success Rate: {(completed_tasks/total_tasks*100):.1f}%\n"
        
        return response
    
    # Check if we have task planning (Lead Bot output)
    elif "all_tasks" in result and result["all_tasks"]:
        tasks = []
        for task_dict in result["all_tasks"]:
            if isinstance(task_dict, dict):
                tasks.append(Task(**task_dict))
            elif isinstance(task_dict, Task):
                tasks.append(task_dict)
        
        response = "**Task Planning Complete!** I've created your execution plan:\n\n"
        
        for task in tasks:
            response += f"**Task {task.task_id}:** {task.description[:100]}{'...' if len(task.description) > 100 else ''}\n"
            response += f"   *Agent:* {task.assigned_bot.upper()}\n"
            response += f"   *Format:* {task.instructions.output_format}\n"
            if task.prerequisites:
                response += f"   *Dependencies:* {task.prerequisites}\n"
            response += "\n"
        
        response += f"**Ready for execution!** {len(tasks)} tasks planned.\n"
        return response
    
    # Conversational response
    elif "messages" in result and result["messages"]:
        last_message = result["messages"]
        if isinstance(last_message, list) and last_message:
            last_message = last_message[-1]
        
        if hasattr(last_message, 'content'):
            return last_message.content
        elif hasattr(last_message, 'text'):
            return last_message.text
        else:
            return str(last_message)
    
    # Fallback response
    return "Request processed successfully. How can I assist you further?"

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Main chat endpoint compatible with OpenAI API"""
    try:
        logger.info(f"Processing chat request with {len(request.messages)} messages")
        logger.debug(f"Request model: {request.model}, temperature: {request.temperature}")
        
        if not request.messages:
            logger.warning("Empty messages in request")
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Convert messages to LangChain format
        langchain_messages = convert_messages_to_langchain(request.messages)
        logger.debug(f"Converted {len(langchain_messages)} messages to LangChain format")
        
        # Get the latest user message for processing
        latest_message = None
        for msg in reversed(langchain_messages):
            if isinstance(msg, HumanMessage):
                latest_message = msg
                break
        
        if not latest_message:
            logger.warning("No user message found in request")
            raise HTTPException(status_code=400, detail="No user message found")
        
        logger.info(f"Processing user message: {latest_message.content[:100]}...")
        
        # Create initial state
        initial_state = State(messages=[latest_message])
        logger.debug("Created initial state")
        
        # Execute the graph with increased recursion limit for complex workflows
        logger.info("Executing LangGraph multi-agent workflow...")
        result = await asyncio.to_thread(
            graph.invoke, 
            initial_state.model_dump(),
            config={"recursion_limit": 75}
        )
        logger.info("LangGraph workflow completed")
        
        # Format response based on result content
        response_content = format_task_execution_response(result)
        
        # Log execution summary
        if "task_outputs" in result:
            logger.info(f"Completed {len(result['task_outputs'])} tasks")
        elif "all_tasks" in result:
            logger.info(f"Planned {len(result['all_tasks'])} tasks")
        else:
            logger.info("Conversational response generated")
        
        # Calculate token usage (approximate)
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = len(response_content.split())
        total_tokens = prompt_tokens + completion_tokens
        
        # Format response for OpenWebUI
        chat_response = ChatResponse(
            id=f"chatcmpl-{os.urandom(12).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
        
        logger.info("Chat request processed successfully")
        return chat_response
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        import subprocess
        
        logger.info("Health check requested")
        
        # Check GPU availability
        gpu_available = False
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            gpu_available = result.returncode == 0
        except:
            pass
        
        health_response = HealthResponse(
            status="healthy",
            service="LangGraph Multi-Agent System",
            version="2.0.0",
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            gpu_available=gpu_available,
            timestamp=str(int(time.time()))
        )
        
        logger.info("Health check completed successfully")
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "LangGraph Multi-Agent API",
        "version": "2.0.0",
        "status": "running",
        "agents": [
            "Lead Bot - Task planning and decomposition",
            "Coordinator Bot - Task coordination and prompt generation", 
            "Analyst Bot - Python data analysis and visualization",
            "Data Engineer Bot - SQL query generation and data extraction",
            "Refiner Bot - Output validation and error correction",
            "Conversational Bot - General purpose chat assistant"
        ],
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
            "docs": "/docs"
        },
        "description": "Advanced multi-agent system for data analysis and conversational AI"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))  # Correct port number
    workers = int(os.getenv("WORKERS", 1))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting LangGraph Chat API server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False  # Set to True for development
    )
