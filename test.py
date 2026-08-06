from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi3")

response = llm.invoke("Explain quicksort in simple terms")

print(response.content)