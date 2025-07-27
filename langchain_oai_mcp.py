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
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["./math_server.py"],
            "transport": "stdio",
        },
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

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    from langchain.schema import HumanMessage
    # math_response = await graph.ainvoke(MessagesState(messages=[HumanMessage(content="what's (3 + 5) x 12?")]))

    # print("Math response:", math_response)
    # weather_response = await graph.ainvoke({"messages": "what is the weather in nyc?"})
    # print("Weather response:", weather_response)

    print("Welcome to the CLI Chat Application! Type 'exit' to quit.")
    

    state = MessagesState(messages=[])

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        state["messages"].append(HumanMessage(content=user_input))
        # Use astream to process the conversation and collect the final state
        final_state = None
        # async for message, config in graph.astream(state, stream_mode="messages"):
        #     final_state = message  # message is the updated state at each step

        # if final_state is not None:
        #     # issue is that final_state is a AIMessageChunk
        #     state = cast(MessagesState, final_state)
        #     print("Bot:", state["messages"][-1].content)
        # else:
        #     print("Bot: (no response)")
        async for current_state in graph.astream(state, stream_mode="values"):
            final_state = current_state  # This is the complete state
        
        if final_state is not None:
            state = cast(MessagesState, final_state)
            print("Bot:", state["messages"][-1].content)
        else:
            print("Bot: (no response)")


asyncio.run(main())