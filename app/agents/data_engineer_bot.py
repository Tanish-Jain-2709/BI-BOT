import logging
from langchain_core.messages import SystemMessage
from ..functions import Create_Agent, State, TaskOutput

logger = logging.getLogger(__name__)

def Data_Engineer_Bot(state: State):
    """Data Engineer Bot that generates SQL scripts for data operations."""
    logger.info("Data Engineer Bot starting SQL generation")
    
    current_queue = state.queue
    current_task_id = current_queue[0]
    current_task = next((task for task in state.all_tasks if task.task_id == current_task_id), None)
    prompt = current_task.instructions.prompt
    
    logger.info(f"Processing SQL task {current_task_id}")
    
    system_prompt = """
You are the SQL Data Engineer Bot in a distributed data analysis architecture.

CORE RESPONSIBILITY:
Generate executable SQL scripts for data extraction, transformation, and analysis tasks.

TECHNICAL STANDARDS:
- Write optimized SQL queries for performance
- Use proper JOIN syntax and WHERE clauses
- Include appropriate GROUP BY and ORDER BY clauses
- Handle NULL values and edge cases
- Add comments for complex query logic
- Ensure cross-database compatibility when possible
- Return output in EXACT JSON format as specified

SCRIPT REQUIREMENTS:
- Include proper database connection handling
- Implement efficient queries with appropriate joins and indexes
- Include result formatting and metadata collection
- Add comprehensive error handling
- Handle database connection errors and query timeouts
- Return results in valid JSON format with metadata matching the expected output format exactly

OUTPUT FORMAT:
Return ONLY the executable SQL script as plain text. No explanations, markdown formatting, code blocks, or additional text.
Ensure the script outputs data in the exact JSON format specified in the task requirements.

EXAMPLE STRUCTURE:
-- SQL Script for loading customer transaction data
SET @start_time = NOW(6);

-- Main query with error handling
SELECT
    customer_id,
    order_value,
    order_date
FROM transactions
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
    AND customer_id IS NOT NULL
    AND order_value > 0
ORDER BY customer_id, order_date;

-- Calculate execution metadata
SET @end_time = NOW(6);
SET @execution_time = TIMESTAMPDIFF(MICROSECOND, @start_time, @end_time) / 1000000;
SET @row_count = ROW_COUNT();

-- Format results as JSON
SELECT JSON_OBJECT(
    'query_result', JSON_ARRAYAGG(
        JSON_OBJECT(
            'customer_id', customer_id,
            'order_value', order_value,
            'order_date', order_date
        )
    ),
    'row_count', @row_count,
    'execution_time', @execution_time,
    'columns', JSON_ARRAY('customer_id', 'order_value', 'order_date')
) AS result;

RESPONSE REQUIREMENTS:
- Return ONLY executable SQL script code
- Ensure script is complete and handles all specified requirements
- Output must match the expected JSON format exactly
"""
    
    SQL_bot = Create_Agent(system_prompt)
    conversation = [SystemMessage(content=system_prompt)] + [prompt]
    
    logger.debug("Generating SQL script")
    response = SQL_bot.llm.invoke(conversation)
    logger.info(f"Generated SQL script: {response.content[:100]}...")

    # Update task status
    try:
        task_output = TaskOutput(
            output_data=response.content,
            success=True,
            error_messsage=None,
            validation_passed=None
        )
        state.task_outputs[current_task_id] = task_output
        state.all_tasks[current_task_id-1].status = "completed"
        logger.info("SQL script generation successful")

    except Exception as e:
        logger.error(f"SQL script generation failed: {e}")
        task_output = TaskOutput(
            output_data="",
            success=False,
            error_messsage=str(e),
            validation_passed=False
        )
        state.task_outputs[current_task_id] = task_output
        state.all_tasks[current_task_id-1].status = "failed"

    return state
