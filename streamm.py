from typing import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.config import get_stream_writer

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    writer = get_stream_writer()  # Access the stream writer
    writer({"progress": "Starting to process query"})  # Emit custom data
    # Simulate some processing
    result = f"Answer to '{state['query']}'"
    writer({"progress": "Finished processing query"})  # Emit more custom data
    return {"answer": result}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "What is LangGraph?"}

# Stream the custom data emitted by the node
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
