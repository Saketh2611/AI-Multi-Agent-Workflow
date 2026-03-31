# 🤖 AI Multi-Agent Workflow Dashboard

> **Live Demo:** [ai-multi-agent-workflow.onrender.com](https://ai-multi-agent-workflow.onrender.com)

A real-time, web-based dashboard that visualizes a team of **self-correcting, self-debugging AI Agents** collaborating to solve complex tasks. Built with **FastAPI**, **LangGraph**, and **Google Gemini 2.5 Flash**.

Agents don't follow a straight line — they plan, research the web, read uploaded PDFs, **write and execute code in a sandbox**, verify the output, loop back on failure, and only finalize a report when the code actually runs.

---

## ✨ Key Features

- **🧠 Self-Healing Research Loop** — A quality-check node evaluates research results. If data is insufficient, the system automatically loops back to the Planner for a revised strategy (up to 3 attempts).
- **⚙️ Self-Debugging Code Loop** — After code is generated, it runs in an isolated sandbox. If execution fails, the error is fed back to the Coder to fix and re-run (up to 3 attempts).
- **🧪 Sandboxed Code Execution** — Generated Python code runs in an [E2B](https://e2b.dev) cloud sandbox (set `E2B_API_KEY`) with an automatic local subprocess fallback — no untrusted code runs on your machine.
- **🔍 Tool-Equipped Coder** — The Coder agent has access to both `web_search` (to look up fixes and docs) and `execute_python_code` (to test its own output), enabling a write → test → debug inner loop.
- **📄 Document Intelligence (RAG)** — Upload PDFs directly. Content is chunked, embedded via Google Generative AI, and stored in **FAISS + PostgreSQL**, allowing agents to cite your documents.
- **🔄 Real-Time Visualization** — A live WebSocket dashboard shows each agent's status, the generated code, sandbox output (with exit codes), and retry counts as they happen.
- **📄 Downloadable Reports** — One-click Markdown report containing the plan, research, verified code, and execution result.

---

## 🏛️ Architecture

```
User Task
    │
    ▼
┌─────────┐     ┌────────────┐
│ Planner │◄────┤ (retry if  │
│  Chain  │     │  bad data) │
└────┬────┘     └────────────┘
     │                 ▲
     ▼                 │
┌────────────┐   ┌─────┴──────────┐
│ Researcher │──►│ Quality Check  │
│   Agent    │   │ (conditional)  │
└────────────┘   └───────┬────────┘
 tools: web_search               │ good
        pdf_search               ▼
                        ┌────────────────┐
                        │  Coder Agent   │◄──────────┐
                        │                │           │ (retry on
                        │  write → exec  │           │  fail, ≤3x)
                        │  → debug loop  │           │
                        └───────┬────────┘           │
 tools: web_search              │                    │
        execute_python_code     ▼                    │
                        ┌───────────────┐            │
                        │   Executor    │────────────┘
                        │  (clean run)  │
                        └───────┬───────┘
                                │ pass
                                ▼
                        ┌───────────────┐
                        │   Reporter    │
                        │    Chain      │
                        └───────┬───────┘
                                │
                                ▼
                           Final Report
```

**Component breakdown:**
- **Frontend** — Single `index.html` (Tailwind CSS + vanilla JS). Manages WebSocket events, renders the terminal output panel, code editor, agent status cards, and progress bar.
- **Backend** — FastAPI handles the WebSocket lifecycle, PDF uploads, and static assets.
- **Agent Graph** — `agents/graph.py` — LangGraph `StateGraph` with 5 nodes and 2 conditional edges.
- **Agents** — Researcher and Coder use `create_agent` (LangChain v1). Planner and Reporter are simple `prompt | llm` chains (no tools needed).
- **Vector Store** — FAISS (in-memory search) + PostgreSQL (metadata) for PDF embeddings.

---

## 🔄 The Agentic Workflow (Step-by-Step)

1. **Planner** — Analyzes the user task and any prior research context. Produces a 3-step plan. Re-runs if the researcher returns poor data.
2. **Researcher** — Runs `web_search` (Tavily) for live web data and `pdf_search` against uploaded PDFs. Feeds findings to the quality check.
3. **⚖️ Quality Check** *(conditional edge)* — If research is thin or returns no results, loops back to Planner (max 3 iterations). Otherwise proceeds to Coder.
4. **Coder** — Has an *internal* agent loop: writes code → calls `execute_python_code` → reads errors → calls `web_search` for fixes → rewrites → re-executes. Repeats until the code passes or it gives up.
5. **Executor** — Runs the Coder's final output in a *fresh* sandbox pass. This is the ground-truth execution result that goes into the report.
6. **⚖️ Execution Check** *(conditional edge)* — If the executor run fails and retries remain, sends the error back to the Coder. Otherwise proceeds to Reporter.
7. **Reporter** — Compiles task, plan, research, code, and execution result (including whether it passed) into a final Markdown report with a quality score.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent Orchestration | LangGraph 1.x (`StateGraph`) |
| Agent API | LangChain 1.x (`create_agent`) |
| Web Search | Tavily AI |
| Code Sandbox | E2B cloud (+ subprocess fallback) |
| Vector DB | FAISS + PostgreSQL |
| Embeddings | Google Generative AI (`embedding-001`) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5, Vanilla JS, Tailwind CSS |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Google Gemini API Key](https://aistudio.google.com/)
- [Tavily Search API Key](https://tavily.com/)
- PostgreSQL database (for PDF vector storage)
- *(Optional)* [E2B API Key](https://e2b.dev) for cloud sandbox execution

### 1. Clone

```bash
git clone <your-repo-url>
cd AI-Multi-Agent-Workflow
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
GOOGLE_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here

# PostgreSQL — required for PDF/RAG features
# Format: postgresql://USER:PASSWORD@HOST:PORT/DATABASE_NAME
DATABASE_URL=postgresql://postgres:password@localhost:5432/vector_db

# Optional: E2B cloud sandbox for safer code execution
# If not set, code runs in a local subprocess with a 30s timeout
E2B_API_KEY=your_e2b_key_here
```

### 4. Run

```bash
uvicorn main:app --reload
# or
uv run main.py
```

### 5. Open

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📁 Project Structure

```
.
├── .env                    # API keys (not committed)
├── main.py                 # FastAPI server, WebSocket router, PDF upload
├── requirements.txt        # Python dependencies (langchain>=1.0, langgraph>=1.0)
├── static/
│   └── index.html          # Frontend dashboard
└── agents/
    ├── __init__.py
    ├── graph.py             # LangGraph StateGraph — 5 nodes, 2 conditional edges
    └── tools.py             # web_search, pdf_search, execute_python_code, FAISS/PG
```

---

## 📡 WebSocket Event Reference

The frontend communicates over a single `/ws` connection. Here are all message types:

| `type` | Direction | Key Fields | Description |
|---|---|---|---|
| `progress` | server → client | `stage: 1–4` | Advances the progress bar |
| `chat` | server → client | `agent, message, revision?` | Log panel message |
| `code` | server → client | `code, revision` | Updates the code editor panel |
| `execution` | server → client | `success: bool, output: str` | Terminal panel with sandbox result |
| `metrics` | server → client | `accuracy, processingTime, systemStatus` | Results panel |
| `report_data` | server → client | `filename, content` | Triggers Markdown download |
| `error` | server → client | `message` | Displays error in log panel |
| `get_report` | client → server | `type: "get_report"` | Requests report download |

---

## 📦 Key Dependency Versions

This project requires **LangChain v1** and **LangGraph v1**, which introduced breaking changes from the `0.x` series. Ensure your `requirements.txt` specifies:

```
langchain>=1.0
langchain-core>=1.0
langgraph>=1.0
```

The `create_agent` API (`from langchain.agents import create_agent`) is the v1 replacement for the deprecated `create_tool_calling_agent` + `AgentExecutor` pattern. `create_react_agent` from `langgraph.prebuilt` is also deprecated as of LangGraph v1.
