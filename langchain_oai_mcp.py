from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import os

import asyncio

from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4.1")

# https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools

# openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the ChatOpenAI model
model = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

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
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        conversation_history.append({"role": "user", "content": user_input})
        weather_response = await graph.ainvoke({"messages": conversation_history})
        print("Bot:", weather_response["messages"][-1].content)

asyncio.run(main())