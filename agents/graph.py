import os
from typing import TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.agent_executor import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# --- FIX: Relative import because Tools.py is in the same folder ---
from agents.tools import web_search
from agents.tools import pdf_search

load_dotenv()

# --- 1. State Definition ---
class AgentState(TypedDict):
    task: str
    plan: str
    research_info: str
    code: str
    final_report: str
    revision_number: int
    messages: Annotated[list, lambda x, y: x + y]

# --- 2. Model Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 3. Agent Prompts & Executors ---
# Planner
# FIX APPLIED: Added ("placeholder", "{agent_scratchpad}") to prevent ValueError
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Tech Lead. Create a 3-step plan to solve the user's request. Output strictly the plan."),
    ("human", "Task: {task}\n\nCurrent Research Context: {research_info}"),
    ("placeholder", "{agent_scratchpad}") 
])
planner_agent = create_tool_calling_agent(llm, [], planner_prompt)
planner_executor = AgentExecutor(agent=planner_agent, tools=[], verbose=True)

# Researcher
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Researcher. Use the web_search tool to find info for the plan and pdf_search for information found inside uploaded PDFs."),
    ("human", "Task: {task}\nPlan: {plan}"),
    ("placeholder", "{agent_scratchpad}")
])
research_agent = create_tool_calling_agent(llm, [web_search,pdf_search], researcher_prompt)
research_executor = AgentExecutor(agent=research_agent, tools=[web_search,pdf_search], verbose=True)

# Coder
coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python Expert. Write functional code based on the research. Output ONLY the markdown code block."),
    ("human", "Task: {task}\nResearch: {research_info}"),
    ("placeholder", "{agent_scratchpad}")
])
coder_agent = create_tool_calling_agent(llm, [], coder_prompt)
coder_executor = AgentExecutor(agent=coder_agent, tools=[], verbose=True)

# Reporter
report_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Technical Writer. Compile a report with: Task, Plan, Research, Code, and a mock Quality Score."),
    ("human", "Task: {task}\nPlan: {plan}\nResearch: {research_info}\nCode: {code}"),
    ("placeholder", "{agent_scratchpad}")
])
report_agent = create_tool_calling_agent(llm, [], report_prompt)
report_executor = AgentExecutor(agent=report_agent, tools=[], verbose=True)


# --- 4. Nodes ---
async def planner_node(state: AgentState):
    res_info = state.get("research_info", "None yet")
    result = await planner_executor.ainvoke({"task": state["task"], "research_info": res_info})
    current_rev = state.get("revision_number", 0) + 1
    return {"plan": result["output"], "revision_number": current_rev}

async def researcher_node(state: AgentState):
    result = await research_executor.ainvoke({"task": state["task"], "plan": state["plan"]})
    return {"research_info": result["output"]}

async def coder_node(state: AgentState):
    result = await coder_executor.ainvoke({"task": state["task"], "research_info": state["research_info"]})
    return {"code": result["output"]}

async def reporter_node(state: AgentState):
    result = await report_executor.ainvoke({
        "task": state["task"], 
        "plan": state["plan"], 
        "research_info": state["research_info"],
        "code": state["code"]
    })
    return {"final_report": result["output"]}

# --- 5. Conditional Logic ---
def research_quality_check(state: AgentState) -> Literal["coder", "planner"]:
    research = state.get("research_info", "")
    revision = state.get("revision_number", 1)
    
    if revision > 3: return "coder"
    if len(research) < 50 or "no results" in research.lower(): return "planner"
    return "coder"

# --- 6. Build Graph ---
def get_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reporter", reporter_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_conditional_edges("researcher", research_quality_check, {"planner": "planner", "coder": "coder"})
    workflow.add_edge("coder", "reporter")
    workflow.add_edge("reporter", END)

    return workflow.compile(checkpointer=MemorySaver())