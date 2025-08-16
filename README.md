# AI Multi-Agent Workflow Dashboard

This project is a real-time, web-based dashboard that allows users to input a task and watch a team of AI agents collaborate to solve it. The system uses a sophisticated agentic workflow built with LangChain and LangGraph, with a FastAPI backend for WebSocket communication and a vanilla HTML/JS frontend with Tailwind CSS for the user interface.

## ✨ Features

- **Real-Time Collaboration**: Watch AI agents communicate and see their work in real-time through a live chat interface.
- **Dynamic Workflow**: The system intelligently routes tasks to the appropriate agents, skipping unnecessary steps (e.g., research) to improve efficiency.
- **Versatile Task Handling**: Capable of handling both knowledge-based questions (which are formatted into tables) and coding tasks (which generate Python scripts).
- **Interactive UI**: A sleek, modern interface with a progress bar, code display, and final metrics.
- **Downloadable Reports**: Users can download a complete markdown report of the entire workflow, including the plan, research, and final output.

## 🏛️ Architecture Overview

The application is composed of three main parts:

1.  **Frontend**: A single `index.html` file styled with **Tailwind CSS**. It communicates with the backend over a WebSocket connection to send tasks and receive real-time updates.
2.  **Backend**: A **FastAPI** server that manages WebSocket connections and serves the frontend. It acts as the bridge between the user and the agentic workflow.
3.  **Agentic Workflow**: Built with **LangGraph**, this is the core of the application. It's a graph-based system where different AI agents (Planner, Researcher, Coder, etc.) are defined as "nodes." The system passes the task from one agent to the next, with conditional logic to decide the best path.

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- An active internet connection

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a file named `.env` in the root directory of the project and add your API keys:

```
GOOGLE_API_KEY="your_google_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 4. Run the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

The `--reload` flag will automatically restart the server whenever you make changes to the code.

### 5. Open the Dashboard

Open your web browser and navigate to **http://127.0.0.1:8000**. You should see the AI Multi-Agent Workflow Dashboard.

## 📁 File Structure

```
.
├── .env                # Stores API keys and other secrets
├── main.py             # The FastAPI backend server
├── requirements.txt    # A list of all the Python dependencies
├── static/
│   └── index.html      # The frontend user interface
└── agents/
    ├── __init__.py
    ├── graph.py        # Defines the LangGraph agentic workflow
    └── tools.py        # Configures external tools like web search
```

## ⚙️ How It Works

The agentic workflow in `agents/graph.py` is the core of this project. When a user submits a task, it goes through the following steps:

1.  **Planner**: The first agent creates a high-level, multi-step plan to address the user's request.
2.  **Conditional Routing (Research)**: The system checks the plan for keywords like "research" or "explain."
    - If found, the task is sent to the **Researcher**.
    - If not, the research step is skipped to save time.
3.  **Researcher**: If called, this agent uses the Tavily search tool to gather information from the web.
4.  **Conditional Routing (Code or Format)**: After the research step (or if it was skipped), the system again checks the plan for keywords like "code" or "script."
    - If found, the task is sent to the **Coder**.
    - Otherwise, it's sent to the **Formatter**.
5.  **Coder / Formatter**:
    - The **Coder** writes a Python script.
    - The **Formatter** organizes the research into a clean markdown table.
6.  **Reporter**: The final agent gathers all the information from the previous steps (the plan, research, and either the code or the formatted data) and compiles a comprehensive final report, including mock performance metrics.
7.  **Frontend Update**: Each step of this process sends real-time updates to the frontend via the WebSocket, allowing the user to watch the collaboration unfold.

## 🛠️ Technologies Used

- **Backend**: FastAPI, Uvicorn, Python-dotenv
- **AI & Machine Learning**: LangChain, LangGraph, Google Gemini, Tavily Search
- **Frontend**: HTML, JavaScript, Tailwind CSS
