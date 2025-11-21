# 🤖 AI Multi-Agent Workflow Dashboard

This project is a real-time, web-based dashboard that visualizes a team of **Self-Correcting AI Agents** collaborating to solve complex tasks.

Built with **FastAPI**, **LangGraph**, and **Google Gemini**, this system demonstrates an "Agentic" workflow where agents don't just follow a straight line—they plan, research, check their own work, and loop back if necessary before writing code.

## ✨ Key Features

- **🧠 Self-Healing Workflow**: The unique "Quality Check" node analyzes research results. If the data is insufficient, the system automatically loops back to the Planner to try a different strategy.
- **🔄 Real-Time Visualization**: A futuristic, neon-styled dashboard (Tailwind CSS) that shows exactly which agent is working, what they are thinking, and their current status.
- **🤝 Interactive WebSocket Stream**: Watch the agents chat, exchange data, and update progress bars in real-time without page reloads.
- **🛠️ Robust Tooling**: Integrated with **Tavily Search** for high-accuracy web results and **Gemini 1.5 Flash** for fast reasoning.
- **📄 Downloadable Reports**: One-click generation of a comprehensive Markdown report containing the plan, research citations, and generated code.

## 🏛️ Architecture Overview

The application follows a modern micro-architecture:

1. **Frontend**: A single, responsive `index.html` using **Tailwind CSS**. It infers agent states via logic-driven UI updates based on WebSocket events.
2. **Backend**: **FastAPI** handles the WebSocket lifecycle and serves static assets.
3. **The Brain (LangGraph)**: A state machine defined in `agents/Graph.py`. It manages the state object (Task, Plan, Research, Code) as it passes between nodes.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Google Gemini API Key](https://aistudio.google.com/)
- [Tavily Search API Key](https://tavily.com/)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd AI-Multi-Agent-Workflow
2. Install Dependencies
It is highly recommended to use a virtual environment.
```

```Bash

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
3. Configure Environment Variables
```
# Create a file named .env in the root directory and paste your keys:

```
GOOGLE_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here
```
# 4. Run the Application
Start the server using Uvicorn. The app checks for the .env file on startup.

```Bash

uvicorn main:app --reload
```
# 5. Open the Dashboard
# Navigate to http://127.0.0.1:8000 in your browser.

# 📁 File Structure


```.
├── .env                 # API Keys (Not committed to Git)
├── main.py              # FastAPI entry point & WebSocket router
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html       # Frontend Dashboard
└── agents/              # Logic Package
    ├── __init__.py      # Package marker
    ├── Graph.py         # LangGraph definition & State Machine
    └── Tools.py         # Search tools configuration
```
# ⚙️ The Agentic Workflow
Unlike simple linear chains, this project uses a Graph with conditional edges:

Planner Agent: Analyzes the user request. If a previous attempt failed, it reads the context and generates a new strategy.

Researcher Agent: Executes web searches using Tavily to gather live data.

# ⚖️ Quality Check (Conditional Edge):

The system evaluates the research output.

If Bad: It loops BACK to the Planner (increments revision count).

If Good: It proceeds to the Coder.

Safety: Max 3 loops allowed to prevent infinite costs.

Coder Agent: Writes Python code based only on the verified research.

Reporter Agent: Compiles the Task, Plan, Research, and Code into a final summary.

# 🛠️ Tech Stack
Orchestration: LangGraph

LLM: Google Gemini 1.5 Flash

Web Search: Tavily AI

Backend: FastAPI, Uvicorn

Frontend: HTML5, JavaScript, Tailwind CSS
