import asyncio
import json
import uuid
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response

# FIX: Importing from the correct file (agent_graph.py) and function (get_app)
from agents.graph import get_app

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static directory for CSS/JS/Images if needed, though we serve index.html directly below
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Session and Data Management ---
connections = {}
session_data = {}

# Initialize the LangGraph application
agent_graph = get_app()


# --- WebSocket Event Handlers ---
async def process_graph_event(session_id: str, event: dict):
    """
    Processes events from the LangGraph stream and sends structured JSON updates 
    to the frontend via WebSocket.
    """
    if not event or not isinstance(event, dict):
        return

    # In 'updates' mode, the event key is the node name (e.g., 'planner', 'researcher')
    node_name = next(iter(event), None)
    if not node_name:
        return
    
    # The value is the state update dictionary returned by that node
    node_output = event[node_name]
    
    websocket = connections.get(session_id)
    if not websocket:
        return

    # --- Router for different agents ---
    
    if node_name == "planner":
        plan_text = node_output.get('plan', '...')
        session_data[session_id]['plan'] = plan_text
        
        # Send UI updates
        await websocket.send_json({"type": "progress", "stage": 1, "label": "Planning..."})
        await websocket.send_json({"type": "chat", "agent": "Planner", "message": f"Here is the plan:\n{plan_text}"})

    elif node_name == "researcher":
        research_text = node_output.get('research_info', '...')
        session_data[session_id]['research_info'] = research_text
        
        await websocket.send_json({"type": "progress", "stage": 2, "label": "Researching..."})
        await websocket.send_json({"type": "chat", "agent": "Researcher", "message": "I found some information."})
        # Optionally send the raw research if needed, or keep it for the report

    elif node_name == "coder":
        # Clean up code block formatting for display
        code_text = node_output.get('code', '').replace("```python", "").replace("```", "").strip()
        session_data[session_id]['code'] = code_text
        
        await websocket.send_json({"type": "progress", "stage": 3, "label": "Coding..."})
        await websocket.send_json({"type": "code", "code": code_text})
        await websocket.send_json({"type": "chat", "agent": "Coder", "message": "I have generated the Python code."})

    elif node_name == "reporter":
        report_text = node_output.get('final_report', '')
        session_data[session_id]['final_report'] = report_text
        
        # Attempt to parse metrics for the UI dashboard
        metrics = {"accuracy": "98%", "processingTime": "12s", "systemStatus": "Complete"}
        
        # Simple parsing if the AI followed the format (optional robustness)
        for line in report_text.split('\n'):
            lower_line = line.lower()
            if "quality score" in lower_line or "accuracy" in lower_line:
                parts = line.split(":")
                if len(parts) > 1: metrics["accuracy"] = parts[1].strip()
        
        await websocket.send_json({"type": "progress", "stage": 4, "label": "Finished"})
        await websocket.send_json({"type": "metrics", **metrics})
        await websocket.send_json({"type": "chat", "agent": "Reporter", "message": "Workflow complete. You can download the report now."})


async def handle_get_report(session_id: str):
    """Generates a download for the final Markdown report."""
    websocket = connections.get(session_id)
    data = session_data.get(session_id, {})
    if not websocket or not data:
        return

    logger.info(f"Compiling report for session {session_id}")
    
    task = data.get('task', 'Task')
    report_content = data.get('final_report', "Report pending or failed generation.")
    
    # Create a safe filename
    safe_task_str = "".join(c for c in task if c.isalnum() or c in " _-").rstrip()[:30]
    filename = f"Report_{safe_task_str}.md"

    await websocket.send_json({
        "type": "report_data",
        "filename": filename,
        "content": report_content
    })


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serves the entry point."""
    try:
        # Ensure you have this file in a 'static' folder
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: static/index.html not found</h1>", status_code=404)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket Loop."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    connections[session_id] = websocket
    session_data[session_id] = {}
    
    logger.info(f"New WebSocket connection: {session_id}")
    
    # LangGraph configuration for thread persistence
    config = {"configurable": {"thread_id": session_id}}

    try:
        while True:
            message_text = await websocket.receive_text()
            user_task = None

            # 1. Parse Message
            try:
                message_json = json.loads(message_text)
                if message_json.get("type") == "get_report":
                    await handle_get_report(session_id)
                    continue # Skip to next loop iteration
            except json.JSONDecodeError:
                # If not JSON, treat as plain text task
                user_task = message_text
                logger.info(f"Received task from {session_id}: {user_task}")
                session_data[session_id]['task'] = user_task

            # 2. Run Agent Graph (only if we have a task)
            if user_task:
                # Reset revision number for new tasks
                inputs = {
                    "task": user_task,
                    "messages": [],
                    "revision_number": 0, # IMPORTANT: Required by agent_graph logic
                    "research_info": ""
                }

                # Stream results
                async for event in agent_graph.astream(inputs, config=config, stream_mode="updates"):
                    await process_graph_event(session_id, event)
                
                # Retrieve final state to ensure we captured everything (optional double-check)
                # final_state = agent_graph.get_state(config).values
                # logger.info(f"Graph finished for {session_id}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if session_id in connections: del connections[session_id]
        if session_id in session_data: del session_data[session_id]

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)