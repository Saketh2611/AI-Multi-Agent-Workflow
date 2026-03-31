import os
import sys
import tempfile
import subprocess
import textwrap
from typing import TypedDict, Annotated, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from agents.tools import web_search, pdf_search

load_dotenv()


# ─────────────────────────────────────────────────────────────
# 1. SANDBOX TOOL
#    Primary  : E2B cloud sandbox (set E2B_API_KEY in .env)
#    Fallback : local subprocess with 30 s timeout
# ─────────────────────────────────────────────────────────────

def _run_with_e2b(code: str) -> tuple[bool, str]:
    """Returns (success, output). Raises ImportError if e2b not installed."""
    from e2b_code_interpreter import Sandbox  # pip install e2b-code-interpreter
    with Sandbox(api_key=os.getenv("E2B_API_KEY")) as sbx:
        execution = sbx.run_code(code)
        if execution.error:
            msg = (
                f"ERROR: {execution.error.name}: {execution.error.value}\n"
                f"{execution.error.traceback}"
            )
            return False, msg
        output = "\n".join(str(r) for r in execution.results)
        output = output or "".join(execution.logs.stdout) or "(no output)"
        return True, output


def _run_with_subprocess(code: str) -> tuple[bool, str]:
    """Local fallback: runs code in a subprocess with a 30 s hard timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False, f"ERROR:\n{result.stderr.strip()}"
        return True, result.stdout.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return False, "ERROR: Execution timed out (30 s limit)"
    finally:
        os.unlink(tmpfile)


@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a sandbox and return stdout / errors.
    Use this to test your code before finalising it.
    Strip markdown fences before passing — plain Python only.
    """
    # Strip accidental markdown fences the LLM might include
    clean = code.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    if os.getenv("E2B_API_KEY"):
        try:
            success, output = _run_with_e2b(clean)
            prefix = "SUCCESS\n" if success else "FAILED\n"
            return prefix + output
        except ImportError:
            pass  # fall through to subprocess

    success, output = _run_with_subprocess(clean)
    prefix = "SUCCESS\n" if success else "FAILED\n"
    return prefix + output


# ─────────────────────────────────────────────────────────────
# 2. STATE
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task: str
    plan: str
    research_info: str
    code: str                   # final extracted code block
    execution_result: str       # stdout / errors from final clean run
    execution_success: bool     # did the final run pass?
    final_report: str
    revision_number: int        # research retry counter
    code_revision_number: int   # code retry counter
    messages: Annotated[list, lambda x, y: x + y]


# ─────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ─────────────────────────────────────────────────────────────
# 4. CHAINS & AGENTS
# ─────────────────────────────────────────────────────────────

# -- Planner (no tools -> simple chain) --
planner_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a Tech Lead. Create a clear 3-step plan to solve the user's request. Output strictly the plan."),
    ("human", "Task: {task}\n\nCurrent Research Context: {research_info}"),
]) | llm

# -- Researcher (web_search + pdf_search -> create_agent) --
researcher_agent = create_agent(
    llm,
    tools=[web_search, pdf_search],
    system_prompt=(
        "You are a Senior Researcher. "
        "Use web_search to find current information relevant to the plan, "
        "and pdf_search for content inside any uploaded PDFs. "
        "Be thorough - the coder depends entirely on your findings."
    ),
)

# -- Coder (web_search + execute_python_code -> create_agent) --
# The agent loop lets it: write -> execute -> debug -> search docs -> fix -> re-execute
coder_agent = create_agent(
    llm,
    tools=[web_search, execute_python_code],
    system_prompt=(
        "You are a Python Expert. Your job is to write correct, working Python code.\n"
        "Workflow you MUST follow:\n"
        "1. Write the code based on the research provided.\n"
        "2. Call execute_python_code to test it.\n"
        "3. If it errors, diagnose the issue. Use web_search to look up fixes if needed.\n"
        "4. Fix and re-execute until the code runs successfully.\n"
        "5. Output ONLY the final working code as a markdown ```python block.\n"
        "Do not stop until the code executes without errors."
    ),
)

# -- Reporter (no tools -> simple chain) --
report_chain = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Technical Writer. Compile a professional report containing:\n"
        "1. Task summary\n2. Plan\n3. Research findings\n4. Final code\n"
        "5. Execution result\n6. Quality score (0-10) with justification."
    )),
    ("human", (
        "Task: {task}\nPlan: {plan}\nResearch: {research_info}\n"
        "Code:\n{code}\n\nExecution Result:\n{execution_result}"
    )),
]) | llm


# ─────────────────────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────────────────────

def _extract_code_block(text) -> str:
    """Pull the first ```python ... ``` block; fall back to the raw text."""
    import re
    import textwrap
    
    # FIX: Flatten the output into a single string if LangChain returns a list
    if isinstance(text, list):
        text = "".join(
            chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk) 
            for chunk in text
        )
    else:
        text = str(text)

    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1)).strip()
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 6. NODES
# ─────────────────────────────────────────────────────────────

async def planner_node(state: AgentState):
    result = await planner_chain.ainvoke({
        "task": state["task"],
        "research_info": state.get("research_info", "None yet"),
    })
    return {
        "plan": result.content,
        "revision_number": state.get("revision_number", 0) + 1,
    }


async def researcher_node(state: AgentState):
    result = await researcher_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"Task: {state['task']}\nPlan: {state['plan']}",
        }]
    })
    
    raw_content = result["messages"][-1].content
    
    # FIX: Flatten the output into a single string so len() counts characters, not list items.
    if isinstance(raw_content, list):
        content_str = "".join(
            chunk.get("text", str(chunk)) if isinstance(chunk, dict) else str(chunk) 
            for chunk in raw_content
        )
    else:
        content_str = str(raw_content)
        
    return {"research_info": content_str}


async def coder_node(state: AgentState):
    # On retry, inject the previous error so the agent knows what broke
    content = (
        f"Task: {state['task']}\n\n"
        f"Research:\n{state['research_info']}"
    )
    prev_error = state.get("execution_result", "")
    if prev_error and not state.get("execution_success", True):
        content += f"\n\nPrevious attempt failed with:\n{prev_error}\nPlease fix it."

    result = await coder_agent.ainvoke({
        "messages": [{"role": "user", "content": content}]
    })
    raw_output = result["messages"][-1].content
    code = _extract_code_block(raw_output)
    return {
        "code": code,
        "code_revision_number": state.get("code_revision_number", 0) + 1,
    }


async def executor_node(state: AgentState):
    """
    Runs the final code block in a clean sandbox pass.
    Result drives the execution_check edge and appears in the report.
    """
    output = execute_python_code.invoke(state["code"])
    success = output.startswith("SUCCESS")
    return {
        "execution_result": output,
        "execution_success": success,
    }


async def reporter_node(state: AgentState):
    result = await report_chain.ainvoke({
        "task": state["task"],
        "plan": state["plan"],
        "research_info": state["research_info"],
        "code": state["code"],
        "execution_result": state.get("execution_result", "Not executed"),
    })
    return {"final_report": result.content}


# ─────────────────────────────────────────────────────────────
# 7. CONDITIONAL EDGES
# ─────────────────────────────────────────────────────────────

def research_quality_check(state: AgentState) -> Literal["coder", "planner"]:
    research = state.get("research_info", "")
    revision = state.get("revision_number", 1)
    if revision > 3:
        return "coder"  # give up retrying, proceed anyway
    if len(research) < 50 or "no results" in research.lower():
        return "planner"
    return "coder"


def execution_check(state: AgentState) -> Literal["reporter", "coder"]:
    """Retry coder up to 3 times if the final execution failed."""
    if state.get("execution_success", False):
        return "reporter"
    if state.get("code_revision_number", 0) >= 3:
        return "reporter"  # max retries hit, report what we have
    return "coder"


# ─────────────────────────────────────────────────────────────
# 8. BUILD GRAPH
# ─────────────────────────────────────────────────────────────

def get_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner",    planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder",      coder_node)
    workflow.add_node("executor",   executor_node)
    workflow.add_node("reporter",   reporter_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_conditional_edges(
        "researcher",
        research_quality_check,
        {"planner": "planner", "coder": "coder"},
    )
    workflow.add_edge("coder", "executor")
    workflow.add_conditional_edges(
        "executor",
        execution_check,
        {"reporter": "reporter", "coder": "coder"},
    )
    workflow.add_edge("reporter", END)

    return workflow.compile(checkpointer=MemorySaver())