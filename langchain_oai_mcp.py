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

async def main():
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    conn_string = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")

    # Set up MongoCheckpointer (replace with your MongoDB URI and collection)
    async with AsyncMongoDBSaver.from_conn_string(
        conn_string=conn_string,
        db_name="langgraph_db",
        checkpoint_collection_name="chat_checkpoints_aio",
        writes_collection_name="chat_writes_aio",
    ) as checkpointer:

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_node(ToolNode(tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            tools_condition,
        )
        builder.add_edge("tools", "call_model")
        graph = builder.compile(checkpointer=checkpointer)  # <-- Pass checkpointer here

        print("Welcome to the CLI Chat Application! Type 'exit' to quit.")

        # Try to load previous state, or start fresh
        # state = checkpointer.get("my_chat_session")
        # if state is None:
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
                # await checkpointer.put("my_chat_session", state)
            else:
                print("Bot: (no response)")


asyncio.run(main())
