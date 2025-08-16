import os
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langgraph.graph import StateGraph, START, END
from langgraph.pregel import Pregel
from langgraph.checkpoint.memory import MemorySaver

from .tools import web_search

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --- Graph State ---
class AgentState(TypedDict):
    task: str
    plan: str
    research_info: str
    code: str
    final_report: str
    messages: Annotated[list, lambda x, y: x + y]

# --- Agent Definitions ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 1. Planner Agent
planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert project planner. Your job is to create a concise, step-by-step plan to address the user's task. "
         "The plan should have a maximum of 4 steps. "
         "Do NOT write any code or provide detailed explanations. "
         "Simply create a high-level plan that the other agents can follow. "
         "Start by saying 'Here is my plan:'. "
         "Example: 1. Research topic X. 2. Write Python script for Y. 3. Create a summary report."),
        ("human", "{task}"),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
planner_agent = create_tool_calling_agent(llm, [], planner_prompt)
planner_executor = AgentExecutor(agent=planner_agent, tools=[], verbose=False)

# 2. Research Agent
# FIXED: Made the prompt stricter to prevent API rate limit errors.
researcher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert researcher. Your job is to use the web search tool to gather information based on the provided plan. "
         "You must perform a maximum of 2 web searches. "
         "After searching, synthesize your findings into a single, coherent response to pass to the next agent and no need to write any code and write short."),
        ("human", "Research the following task based on this plan:\n\nTASK: {task}\n\nPLAN: {plan}"),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
research_agent = create_tool_calling_agent(llm, [web_search], researcher_prompt)
research_executor = AgentExecutor(agent=research_agent, tools=[web_search], verbose=False)

# 3. Coder Agent
coder_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Python programmer. Your job is to write clean, efficient Python code based on the provided research and plan. The code should be fully functional. Add comments to explain complex parts. Enclose the final code in a single markdown code block (```python ... ```)."),
        ("human", "Write Python code for the following task, using the provided plan and research:\n\nPLAN: {plan}\n\nRESEARCH: {research_info}"),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
coder_agent = create_tool_calling_agent(llm, [], coder_prompt)
coder_executor = AgentExecutor(agent=coder_agent, tools=[], verbose=False)

# 4. Report Agent
report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert report writer. Your job is to create a final summary report of the entire workflow. Include the initial task, the plan, a summary of the research, and the generated code. Also, create a mock 'Accuracy Score' (e.g., 95%), 'Processing Time' (e.g., 12.5s), and set 'System Status' to 'Complete'."),
        ("human", "Create the final report based on the following information:\n\nTASK: {task}\n\nPLAN: {plan}\n\nRESEARCH: {research_info}\n\nCODE:\n{code}"),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
report_agent = create_tool_calling_agent(llm, [], report_prompt)
report_executor = AgentExecutor(agent=report_agent, tools=[], verbose=False)


# --- Node Functions ---
async def planner_node(state: AgentState, config):
    print("--- NODE: Planner ---")
    result = await planner_executor.ainvoke({"task": state["task"], "messages": []})
    return {"plan": result["output"]}

async def researcher_node(state: AgentState, config):
    print("--- NODE: Researcher ---")
    result = await research_executor.ainvoke({"task": state["task"], "plan": state["plan"], "messages": []})
    output = result.get("output", "")
    if not output:
        output = "No research results found."
    return {"research_info": output}


async def coder_node(state: AgentState, config):
    print("--- NODE: Coder ---")
    result = await coder_executor.ainvoke({"task": state["task"], "plan": state["plan"], "research_info": state["research_info"]})
    return {"code": result["output"]}

async def report_node(state: AgentState, config):
    print("--- NODE: Reporter ---")
    plan = state.get("plan", "")
    research = state.get("research_info", "")
    code = state.get("code", "")
    report = f"""
    ### Task
    {state.get("task", "")}

    ### Plan
    {plan}

    ### Research Findings
    {research}

    ### Generated Code
    {code}

    ### Final Report
    The above plan, research, and code together form the solution.
    """
    return {"final_report": report}




# --- Graph Definition ---
def get_graph() -> Pregel:
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reporter", report_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "reporter")
    workflow.add_edge("reporter", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
