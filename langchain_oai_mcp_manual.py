import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
import os

import httpx

# To disable SSL verification, use verify=False when creating a client or making requests
# Example: client = httpx.Client(verify=False)

import asyncio
from typing import cast
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage

# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-4.1")

# https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools

# openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the ChatOpenAI model
os.environ["HTTPX_VERIFY"] = "0"  # Disable SSL verification globally for httpx
os.environ["HTTPX_VERIFY"] = "false"  # Disable SSL verification globally for httpx
http_client = httpx.Client(verify=False)

model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", http_client=http_client)

client = MultiServerMCPClient(
    {
        # "math": {
        #     "command": "python",
        #     # Make sure to update to the full absolute path to your math_server.py file
        #     "args": ["./math_server.py"],
        #     "transport": "stdio",
        # },
        "weather": {
            # make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    }
)

STATE_FILE = "state.json"

def save_state(state):
    # Convert MessagesState to a serializable dict
    with open(STATE_FILE, "w") as f:
        json.dump(
            {"messages": [msg.model_dump() for msg in state["messages"]]},
            f,
            indent=2
        )

def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
        # Reconstruct MessagesState with proper message objects
        from langchain.schema import AIMessage, HumanMessage, SystemMessage
        messages = []
        for msg in data["messages"]:
            if msg["type"] == "human":
                messages.append(HumanMessage(**msg))
            elif msg["type"] == "ai":
                messages.append(AIMessage(**msg))
            elif msg["type"] == "system":
                messages.append(SystemMessage(**msg))
            elif msg["type"] == "tool":
                # Assuming tool messages are handled as AI messages
                messages.append(ToolMessage(**msg))
        return {"messages": messages}

async def build_graph():
    tools = await client.get_tools()

    async def call_model(state: MessagesState, config=None):
        stream_callback = None
        if config and "stream_callback" in config['configurable']:
            stream_callback = config["configurable"]["stream_callback"]
        streamed_content = ""
        # Use the async streaming API
        async for chunk in model.bind_tools(tools).astream(state["messages"]):
            print(f"Chunk in async def call_model: {chunk.content}")  # Debugging line
            if stream_callback:
                await stream_callback(chunk.content)  # Send chunk to FastAPI or queue
            streamed_content += chunk.content
        return {"messages": state["messages"] + [AIMessage(content=streamed_content)]}



    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    return builder.compile()


async def main():
    graph = await build_graph()

    print("Welcome to the CLI Chat Application! Type 'exit' to quit.")

    # Try to load previous state, or start fresh
    # state = checkpointer.get("my_chat_session")
    # if state is None:
    state = load_state()
    if state is None:
        state = MessagesState(messages=[])

    # When invoking your graph, provide a thread_id for persistence
    config = {"configurable": {"thread_id": "conversation_1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        state["messages"].append(HumanMessage(content=user_input))
        final_state = None
        async for current_state in graph.astream(
            state, stream_mode="values", config=config
        ):
            final_state = current_state

        if final_state is not None:
            state = cast(MessagesState, final_state)
            print("Bot:", state["messages"][-1].content)
            # Save state after each turn
            save_state(state)
        else:
            print("Bot: (no response)")

if __name__ == "__main__":
    asyncio.run(main())
