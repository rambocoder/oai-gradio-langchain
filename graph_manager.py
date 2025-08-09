from datetime import datetime
from typing import Optional
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, SystemMessage
from init_system_prompt import InitSystemPromptNode, MyMessagesState


def should_init_system(state: MyMessagesState):
    """Conditional edge: check if we need to initialize system prompt"""
    if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
        return "init_system_prompt"
    return "set_timestamp"


def set_timestamp(state: MyMessagesState):
    """Set LastUpdatedDateTime for new conversations"""
    last_updated = datetime.now().isoformat()
    return {"messages": state.get("messages", []), "LastUpdatedDateTime": last_updated}


async def stream_node(state: MyMessagesState, config: dict = None):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    print("Config in streaming_llm: {config}")
    content = ""
    async for chunk in llm.astream(state["messages"]):
        if chunk.content:
            content += chunk.content
            print(f"Chunk in streaming_llm: {chunk.content}")
            yield chunk.content

    yield {"messages": state["messages"] + [AIMessage(content=content)]}


# Create the persistent graph with conditional initialization
memory = MemorySaver()
builder = StateGraph(MyMessagesState)
builder.add_node("set_timestamp", set_timestamp)
builder.add_node("init_system_prompt", InitSystemPromptNode())
builder.add_node("stream_node", stream_node)
builder.add_conditional_edges(START, should_init_system)
builder.add_edge("init_system_prompt", "set_timestamp")
builder.add_edge("set_timestamp", "stream_node")
builder.set_finish_point("stream_node")
persistent_graph = builder.compile(checkpointer=memory)

img = persistent_graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img)


ascii_art = persistent_graph.get_graph().draw_ascii()
print(ascii_art)
