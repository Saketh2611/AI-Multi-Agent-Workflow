from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch

# Load environment variables from .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY not found. Please set it in your .env file.")

# Create Tavily search tool (already LangChain-compatible)
web_search = TavilySearch(
    tavily_api_key=api_key,
    max_results=3  # You can change this as needed
)
