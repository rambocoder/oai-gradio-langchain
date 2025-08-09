import asyncio
import json
from typing import Annotated, List, Optional, TypedDict

# from typing import list
from fastapi.params import Depends
from langchain_openai import ChatOpenAI
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import (
    StreamingResponse,
)
from langgraph.graph import StateGraph, MessagesState, START
from langchain_oai_mcp_manual import build_graph, load_state
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessageChunk

from langgraph.checkpoint.memory import MemorySaver

from langgraph.config import get_stream_writer

from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Access the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()
app = FastAPI()


class Message(BaseModel):
    id: str
    role: str
    content: str


class ChatPayload(BaseModel):
    messages: List[Message]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"messages": [{"id": "1", "role": "user", "content": "Hello"}]}
            ]
        }
    }


async def send_completion_events(messages, chat: ChatOpenAI):
    async for patch in chat.astream_log(messages):
        for op in patch.ops:
            if op["op"] == "add" and op["path"] == "/streamed_output/-":
                content = (
                    op["value"] if isinstance(op["value"], str) else op["value"].content
                )
                json_dict = {"type": "llm_chunk", "content": content}
                json_str = json.dumps(json_dict)

                yield f"data: {json_str}\n\n"
        # yield f"data: {json.dumps(patch.to_dict())}\n\n"
    # for message in messages:
    #     completion = chat.send_message(message["content"])
    #     yield f"data: {json.dumps({'id': message['id'], 'role': 'bot', 'content': completion})}\n\n"


@app.post("/api/completion")
async def stream(request: Request, payload: ChatPayload):
    messages = [
        {"id": message.id, "role": message.role, "content": message.content}
        for message in payload.messages
    ]
    chat = ChatOpenAI()
    return StreamingResponse(
        send_completion_events(messages, chat=chat),
        media_type="text/event-stream",
    )


async def streaming_ainvoke(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    aiMessage = await llm.ainvoke(state["messages"])
    print(f"aiMessage in streaming_ainvoke: {aiMessage}")  # Debugging line
    return {"messages": state["messages"] + [aiMessage]}


@app.post("/api/ainvoke")
async def chat_ainvoke(request: Request, payload: ChatPayload):
    messages = [HumanMessage(content=message.content) for message in payload.messages]

    # Create a LangGraph with a streaming node
    builder = StateGraph(MessagesState)
    builder.add_node("stream_node", streaming_ainvoke)
    builder.set_entry_point("stream_node")
    builder.set_finish_point("stream_node")
    graph = builder.compile()

    async def stream_llm():
        # Change stream_mode to "messages"
        async for chunk, _ in graph.astream(
            MessagesState(messages=messages), stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                print(f"AIMessageChunk received: {chunk}")
                yield chunk.content
            else:
                print(f"Non-AIMessageChunk received: {chunk}")
                print(f"Output in /api/ainvoke: {chunk}")  # Debugging line
                yield chunk.content

    return StreamingResponse(stream_llm(), media_type="text/event-stream")


@app.post("/api/ainvoke-with-history")
async def chat_ainvoke_with_history(request: Request, payload: ChatPayload):
    messages = [HumanMessage(content=message.content) for message in payload.messages]

    # Create a LangGraph with a streaming node
    builder = StateGraph(MessagesState)
    builder.add_node("stream_node", streaming_ainvoke)
    builder.set_entry_point("stream_node")
    builder.set_finish_point("stream_node")
    graph = builder.compile()

    # First, stream the response
    async def stream_llm():
        async for chunk, _ in graph.astream(
            MessagesState(messages=messages), stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                yield chunk.content

    # Then, get the final state to print history
    async def get_final_state():
        final_state = None
        async for state in graph.astream(
            MessagesState(messages=messages), stream_mode="values"
        ):
            final_state = state

        # Print the full conversation history
        if final_state:
            print("=== FULL CONVERSATION HISTORY ===")
            for i, msg in enumerate(final_state["messages"]):
                print(f"{i+1}. {msg.__class__.__name__}: {msg.content}")
            print("=== END HISTORY ===")

        return final_state

    # Run both operations
    async def combined():
        # Start the history task
        history_task = asyncio.create_task(get_final_state())

        # Stream the response
        async for chunk in stream_llm():
            yield chunk

        # Wait for history to complete
        await history_task

    return StreamingResponse(combined(), media_type="text/event-stream")


async def streaming_llm(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    content = ""
    async for chunk in llm.astream(state["messages"]):
        # Each chunk is a ChatGenerationChunk, extract content/token
        if chunk.content:
            content += chunk.content
            print(f"Chunk in streaming_llm: {chunk.content}")  # Debugging line
            yield chunk.content

    yield {"messages": state["messages"] + [AIMessage(content=content)]}


async def writer_llm(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    writer = get_stream_writer()
    content = ""
    async for chunk in llm.astream(state["messages"]):
        # Each chunk is a ChatGenerationChunk, extract content/token
        if chunk.content:
            content += chunk.content
            print(f"Chunk in writer_llm: {chunk.content}")  # Debugging line
            writer({"progress": "Streaming chunk", "content": chunk.content})
            yield chunk.content

    yield {"messages": state["messages"] + [AIMessage(content=content)]}


@app.post("/api/writer")
async def chat_writer(request: Request, payload: ChatPayload):
    messages = [HumanMessage(content=message.content) for message in payload.messages]

    # Create a LangGraph with a streaming node
    builder = StateGraph(MessagesState)
    builder.add_node("stream_node", writer_llm)
    builder.set_entry_point("stream_node")
    builder.set_finish_point("stream_node")
    graph = builder.compile()

    async def stream_llm():
        # Change stream_mode to "messages"
        async for output in graph.astream(
            MessagesState(messages=messages), stream_mode="custom"
        ):
            print(f"Output in stream_llm: {output}")  # Debugging line
            yield f"data: {output["content"]}\n\n"

    return StreamingResponse(stream_llm(), media_type="text/event-stream")


@app.post("/api/messages")
async def chat_messages(request: Request, payload: ChatPayload):
    messages = [HumanMessage(content=message.content) for message in payload.messages]

    # Create a LangGraph with a streaming node
    memory = MemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("stream_node", streaming_llm)
    builder.set_entry_point("stream_node")
    builder.set_finish_point("stream_node")
    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    async def stream_llm():
        # Change stream_mode to "messages"
        async for chunk, metadata in graph.astream(
            MessagesState(messages=messages), config=config, stream_mode="messages"
        ):
            # The output will be a tuple of (message_chunk, metadata)
            if isinstance(chunk, AIMessageChunk):
                print(f"AIMessageChunk received: {chunk.content}")  # Debugging line
                yield chunk.content
            # else:
            #     print(f"Non-AIMessageChunk received: {chunk}")
            #     print(f"Non-AIMessageChunk metadata: {metadata}")
            #     if hasattr(chunk, "content"):
            #         yield chunk.content

        # After streaming completes, get full state (including full message history)
        state = graph.get_state(config)
        full_messages = state.values.get("messages", [])
        print(f"Full messages length: {len(full_messages)}")
        print(f"Full messages: {full_messages}")

        # full_messages now contains the complete conversation history
        for msg in full_messages:
            print(msg.type, msg.content)

    return StreamingResponse(
        stream_llm(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data["message"]

    # Load or initialize state as needed
    state = load_state() or MessagesState(messages=[])
    state["messages"].append(HumanMessage(content=user_message))

    graph = await build_graph()

    async def event_stream():
        queue = asyncio.Queue()

        async def stream_callback(chunk: str):
            await queue.put(chunk)

        config = {
            "configurable": {"thread_id": "conversation_1"},
            "stream_callback": stream_callback,
        }

        # Run the graph in the background
        async def run_graph():
            async for _ in graph.astream(state, stream_mode="values", config=config):
                pass
            await queue.put(None)  # Sentinel to signal end

        task = asyncio.create_task(run_graph())

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        await task

    return StreamingResponse(event_stream(), media_type="text/plain")


class HelloWorldParams(BaseModel):
    content: Optional[str] = "Default parameter"


@app.get("/hello")
async def hello(params: HelloWorldParams = Depends()):
    return {"params": params.model_dump()}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9090, reload=True)
