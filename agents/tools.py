from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


@tool
def web_search(query: str):
    """
    Search the web for accurate information. 
    """
    try:
        search_tool = TavilySearchResults(max_results=3)
        results = search_tool.invoke({"query": query})
        
        output = []
        for res in results:
            output.append(f"Source: {res.get('url')}\nContent: {res.get('content')}")
            
        return "\n\n".join(output)
    except Exception as e:
        return f"Error executing search: {str(e)}"